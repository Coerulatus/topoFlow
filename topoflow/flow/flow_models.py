"""Flow-based generative models."""

import os
import numpy as np
import lightning.pytorch as pl
import torch
from torch_geometric.data import Data, Batch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
from rdkit import Chem  # Added for molecule saving

from topoflow.flow.simulator import TorchDiffEqSimulator
from topoflow.flow.utils_plot import plot_results
from topoflow.flow.utils import CustomScheduler
from topoflow.utils.loss import UFFBoundGuidance
from topoflow.evaluators import RDKitEvaluator, GATEvaluator
from topoflow.utils.metrics import (
    wasserstein_1d,
    qed_scorer,
    logp_scorer,
    sas_scorer,
    lipinski_scorer,
    compute_connectivity,
    compute_uniqueness,
    compute_novelty,
    scorer_dict,
)
from topoflow.utils.molecules import data_to_rdmol, data_to_list


def global_score_mol_worker(mol_or_pyg, metrics, scorer_dict):
    """Standalone, top-level function that pickle can safely reference."""
    mol_results = {}

    if "gnina" in metrics:
        rdmol, pyg_mol = mol_or_pyg
    else:
        rdmol = mol_or_pyg
        pyg_mol = None

    for m in metrics:
        if m == "gnina":
            scorer = scorer_dict.get(m)
            if scorer is not None:
                try:
                    # Move to cpu explicitly within the isolated worker process
                    res = scorer(pyg_mol.to("cpu"))
                    mol_results["vina"] = res[0]
                    mol_results["cnn_affinity"] = res[1]
                except Exception:
                    mol_results["vina"] = float(0)
                    mol_results["cnn_affinity"] = float(0)
        else:
            scorer = scorer_dict.get(m)
            if scorer is not None and rdmol is not None:
                try:
                    res = scorer(rdmol)
                    mol_results[m] = res.item() if hasattr(res, "item") else res
                except Exception:
                    mol_results[m] = float(0)
            else:
                mol_results[m] = float(0)
    return mol_results


class FlowModel(pl.LightningModule):
    """A PyTorch Lightning module for training a vector field model in a flow-based generative model.

    This class handles the training, validation, and sampling procedures for a
    model that learns the vector field of a continuous normalizing flow.
    """

    def __init__(
        self,
        vector_field_model,
        path,
        evaluator=None,
        n_val_samples=0,
        max_batched_samples=16,
        num_workers=8,
        val_distribution=None,
        sample_every=1,
        atom_decoder=None,
        norm_path=None,
        device="cpu",
        start_lr=1e-3,
        max_lr=1e-2,
        min_lr=1e-5,
        max_epochs=1000,
        max_frac=0.1,
        time_factor=None,
        w_loss_last_step=0.2,
        w_loss_atom=0.02,
        velocity_scale=1.0,
        metrics=["qed", "logp", "sas", "lipinski", "gnina"],
        val_p_data=None,
    ):
        """Initializes the FlowModel.

        Args:
            vector_field_model: The neural network model that estimates the vector field.
            path: An object representing the probability path, providing methods for
                sampling and conditional vector fields.
            evaluator: An optional evaluator for assessing generated samples.
            n_val_samples: Number of samples to generate during validation for visualization.
                Defaults to 0.
            max_batched_samples: Maximum number of samples to process in a single batch.
                Defaults to 16.
            num_workers: Number of parallel workers to use for scoring generated samples during validation. Defaults to 8.
            val_distribution: The expected distribution of metrics for validation.
            sample_every: Frequency of sampling during validation. Defaults to 1.
            atom_decoder: Optional atom decoder for converting samples to RDKit molecules.
            norm_path: Path to the normalization parameters file.
            device: The device on which to run computations ('cpu' or 'cuda'). Defaults to 'cpu'.
            start_lr: Initial learning rate for the optimizer. Defaults to 1e-3.
            max_lr: Maximum learning rate for the scheduler. Defaults to 1e-2.
            min_lr: Minimum learning rate for the scheduler. Defaults to 1e-5.
            max_epochs: Maximum number of epochs for training. Defaults to 1000.
            max_frac: Fraction of the maximum learning rate to use. Defaults to 0.1.
            time_factor: Optional function to scale the loss by time.
            w_loss_last_step: Weight for the loss component at the last time step. Defaults to 0.2.
            w_loss_atom: Weight for the atom type loss component. Defaults to 0.02.
            velocity_scale: Scaling factor for the velocity in the loss computation. Defaults to 1.0.
            metrics: List of metrics to evaluate during validation.
            val_p_data: Optional validation DataLoaderSampleable for validation steps.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["path", "val_p_data"])
        self.vector_field_model = vector_field_model
        self.path = path
        self.val_p_data = val_p_data
        self.path.device = device
        if hasattr(path.p_data, "unguided_label"):
            self.unguided_label = path.p_data.unguided_label
        else:
            self.unguided_label = None
        self.val_samples = []
        self.n_val_samples = n_val_samples
        self.num_workers = num_workers
        self.val_distribution = val_distribution
        self.atom_decoder = atom_decoder
        self.loss_function = torch.nn.MSELoss(reduction="none")
        self.loss_function_last_step = UFFBoundGuidance(atom_decoder=atom_decoder)
        self.train_loss = []
        self.start_lr = start_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.max_epochs = max_epochs
        self.max_frac = max_frac
        if evaluator is not None:
            if evaluator == "rdkit":
                self.evaluator = RDKitEvaluator()
            elif os.path.exists(evaluator):
                self.evaluator = GATEvaluator.load_from_checkpoint(evaluator)
            else:
                raise ValueError(
                    "Evaluator must be 'rdkit' or a valid checkpoint path for GATEvaluator."
                )
        self.n_epoch = 0
        self.sample_every = sample_every
        self.unnorm = False
        if norm_path is not None:
            if not os.path.exists(norm_path):
                raise FileNotFoundError(
                    f"Normalization parameters not found at {norm_path}."
                )
            norm_params = np.load(norm_path, allow_pickle=True)
            self.normalization_params = {
                "mean": norm_params["mean"],
                "std": norm_params["std"],
            }
            self.unnorm = True
        self.time_factor = time_factor
        self.t_last_step = 0.9
        self.w_loss_last_step = w_loss_last_step
        self.w_loss_atom = w_loss_atom
        self.sample_steps = 80
        self.max_batched_samples = max_batched_samples
        self.velocity_scale = velocity_scale

        self.metrics = metrics

    def _get_loss(
        self,
        u_t_theta_lig,
        u_t_ref_lig,
        t=None,
        x_lig=None,
        batch_lig=None,
        verbose=True,
    ):
        """Calculates the loss only on the ligand vector field."""
        time_mask = (t > self.t_last_step).float()

        if (
            x_lig is not None
            and self.w_loss_last_step > 0
            and bool((t > self.t_last_step).any())
        ):
            raw_loss_last_step = self.loss_function_last_step(
                u_t_theta_lig,
                x_lig,
                t,
                batch_lig,
                verbose=verbose,
                mean=torch.tensor(
                    self.normalization_params["mean"],
                    device=x_lig.device,
                    dtype=x_lig.dtype,
                ),
                std=torch.tensor(
                    self.normalization_params["std"],
                    device=x_lig.device,
                    dtype=x_lig.dtype,
                ),
                velocity_scale=self.velocity_scale,
                t_threshold=self.t_last_step,
            )

            loss_last_step = (raw_loss_last_step * time_mask.unsqueeze(-1)).mean(
                dim=-1
            ) * self.w_loss_last_step
        else:
            loss_last_step = torch.tensor(0.0, device=u_t_theta_lig.device)

        loss_mse = self.loss_function(u_t_theta_lig, u_t_ref_lig)
        loss_pos = loss_mse[:, :3].mean()
        loss_atom = loss_mse[:, 3:].mean()
        loss_mse[:, 3:] *= self.w_loss_atom
        loss_mse = loss_mse.mean(dim=-1)
        loss = loss_mse + loss_last_step
        loss_without_time = loss.clone().mean().detach()

        if self.time_factor is not None and t is not None:
            loss *= self.time_factor(t).view(-1)
        loss = loss.mean()
        return {
            "loss": loss,
            "loss_mse": loss_mse.mean(),
            "loss_last_step": loss_last_step.mean(),
            "loss_without_time": loss_without_time.mean(),
            "loss_pos": loss_pos.mean(),
            "loss_atom": loss_atom.mean() * self.w_loss_atom,
        }

    def _get_data(self, batch, use_val=False):
        """Extracts data from p_data (or val_p_data when use_val=True)."""
        batch_size = batch.shape[0]
        orig_p_data = self.path.p_data
        if use_val and self.val_p_data is not None:
            self.path.p_data = self.val_p_data
        try:
            z, batch_all, batch_lig, batch_pocket, batch_size = (
                self.path.sample_conditioning_variable(batch_size)
            )
            t, t_batched = self.path.sample_time(batch_size, batch_all)
            x = self.path.sample_conditional_path(z, t_batched)
            u_t_ref = self.path.conditional_vector_field(x, z, t_batched)
        finally:
            self.path.p_data = orig_p_data

        # Keep the conditioning pocket uncorrupted (clean z_pocket) during training,
        # matching inference where drift_coefficient receives clean z_pocket.
        is_poc = batch_pocket != -1
        x = x.clone()
        u_t_ref = u_t_ref.clone()
        x[is_poc] = z[is_poc]
        u_t_ref[is_poc] = 0.0

        return {
            "z": z,
            "batch_all": batch_all,
            "batch_lig": batch_lig,
            "batch_pocket": batch_pocket,
            "t": t,
            "t_batched": t_batched,
            "x": x,
            "u_t_ref": u_t_ref,
        }

    def forward(self, batch, return_diagnostics=False):
        """Forward pass for the FlowModel.

        Samples from the path, computes the reference vector field, and predicts
        the vector field using the `vector_field_model`.

        Args:
            batch: The input batch of data.
            return_diagnostics: Whether to return diagnostics from the vector field model.

        Returns:
            A tuple containing:
                - u_t_theta_lig: The predicted ligand vector field by the model.
                - u_t_ref_lig: The reference ligand vector field from the path.
                - t_batched_lig: The time values corresponding to ligand atoms.
                - x_lig: The ligand coordinates.
                - diagnostics: (Optional) Diagnostics from the model.
        """
        batch_all = batch.get("batch_all")
        batch_lig = batch.get("batch_lig")
        batch_pocket = batch.get("batch_pocket")
        t = batch.get("t")
        t_batched = batch.get("t_batched")
        x = batch.get("x")
        z = batch.get("z")
        u_t_ref = batch.get("u_t_ref")
        u_t_ref = u_t_ref * self.velocity_scale

        lig_indices = torch.where(batch_lig != -1)[0]
        pocket_indices = torch.where(batch_pocket != -1)[0]

        x_lig = x[lig_indices, :]
        u_t_ref_lig = u_t_ref[lig_indices, :]
        x_pocket = z[pocket_indices, :] if z is not None else x[pocket_indices, :]

        diagnostics = None
        num_graphs = batch.get(
            "num_graphs", int(batch_lig.max().item() if len(batch_lig) > 0 else 0) + 1
        )
        if return_diagnostics and hasattr(
            self.vector_field_model, "format_diagnostics"
        ):
            u_t_theta_lig, diagnostics = self.vector_field_model(
                x_lig, t, x_pocket, batch_lig, batch_pocket, return_diagnostics=True
            )
        else:
            u_t_theta_lig = self.vector_field_model(
                x_lig, t, x_pocket, batch_lig, batch_pocket, num_graphs=num_graphs
            )

        t_batched_lig = t_batched[lig_indices]
        return u_t_theta_lig, u_t_ref_lig, t_batched_lig, x_lig, diagnostics

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Calculates the MSE loss between the predicted and reference vector fields.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The training loss for the current step.
        """
        batch = self._get_data(batch)
        batch_lig = batch["batch_lig"]
        batch_lig = batch_lig[batch_lig != -1]
        u_t_theta, u_t_ref, t, x_lig, _ = self.forward(batch)
        losses = self._get_loss(u_t_theta, u_t_ref, t, x_lig, batch_lig, verbose=False)

        for name, value in losses.items():
            self.log(
                f"train/{name}", value, on_epoch=True, prog_bar=False, on_step=True
            )
        self.log(
            "train/lr",
            self.optimizers().param_groups[0]["lr"],
            on_epoch=True,
            on_step=False,
        )
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Calculates the MSE loss and optionally samples and logs generated data.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The validation loss for the current step.
        """
        batch = self._get_data(batch, use_val=True)
        batch_lig = batch["batch_lig"]
        batch_lig = batch_lig[batch_lig != -1]

        return_diagnostics = batch_idx == 0

        u_t_theta, u_t_ref, t, x_lig, diagnostics = self.forward(
            batch, return_diagnostics=return_diagnostics
        )
        losses = self._get_loss(u_t_theta, u_t_ref, t, x_lig, batch_lig)

        self.log("val/loss", losses["loss"], on_epoch=True, prog_bar=False)
        self.log(
            "val/loss_without_time",
            losses["loss_without_time"],
            on_epoch=True,
            prog_bar=False,
        )

        if diagnostics is not None and "summary" in diagnostics:
            summary = diagnostics["summary"]

            # SANITIZATION: Re-route via helper to capture structural nan tensors
            def safe_val(v):
                if isinstance(v, torch.Tensor):
                    return torch.where(
                        torch.isnan(v) | torch.isinf(v), torch.zeros_like(v), v
                    )
                return 0.0 if (np.isnan(v) or np.isinf(v)) else v

            self.log(
                "val/mean_entropy",
                safe_val(summary["mean_entropy"]),
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/max_entropy",
                safe_val(summary["max_entropy"]),
                on_epoch=True,
                prog_bar=False,
            )
            self.log(
                "val/min_entropy",
                safe_val(summary["min_entropy"]),
                on_epoch=True,
                prog_bar=False,
            )

        is_sanity = getattr(self, "trainer", None) is not None and getattr(
            self.trainer, "sanity_checking", False
        )
        if (
            not is_sanity
            and self.val_distribution is not None
            and batch_idx == 0
            and self.sample_every > 0
            and self.n_epoch % self.sample_every == 0
        ):
            save_path = None
            if hasattr(self, "current_run_dir"):
                save_path = os.path.join(
                    self.current_run_dir, f"valid_mols_epoch_{self.current_epoch}.sdf"
                )

            distances, other_metrics = self.test_distribution(
                test_distribution=self.val_distribution,
                metrics=self.metrics,
                atom_decoder=self.atom_decoder,
                keep_biggest_fragment=True,
                save_path=save_path,
            )

            # SANITIZATION: Clean up structural dictionary metrics
            for metric, distance in distances.items():
                clean_dist = (
                    0.0 if (np.isnan(distance) or np.isinf(distance)) else distance
                )
                self.log(f"val/{metric}", clean_dist, on_epoch=True, prog_bar=False)

            for metric, value in other_metrics.items():
                if isinstance(value, torch.Tensor):
                    clean_val = torch.where(
                        torch.isnan(value) | torch.isinf(value),
                        torch.zeros_like(value),
                        value,
                    )
                else:
                    clean_val = 0.0 if (np.isnan(value) or np.isinf(value)) else value
                self.log(f"val/{metric}", clean_val, on_epoch=True, prog_bar=False)

        if not is_sanity and batch_idx == 0:
            self.n_epoch += 1
        return losses["loss"]

    @torch.no_grad()
    def test_distribution(
        self,
        test_distribution,
        metrics=["qed", "logp", "sas", "lipinski", "gnina"],
        atom_decoder=None,
        keep_biggest_fragment=False,
        save_path=None,
    ):
        """Test is the molecules produced follow the expected distribution.

        This method uses the Wasserstein distance to compare the distribution of
        the generated samples with the expected distribution.

        Args:
            test_distribution: Dictionary containing the expected distribution
                of metrics (e.g., {'qed': [0.5, 0.6], 'logp': [-2, 2]}).
            metrics: List of metrics to evaluate. Defaults to ['qed', 'logp', 'sas'].
            atom_decoder: Optional atom decoder for converting samples to RDKit molecules.
                Defaults to None to use the model's default atom decoder.
            keep_biggest_fragment: If True, only the biggest fragment of each molecule is considered. Defaults to False.
            save_path: Path to folder to save generated molecules.

        Returns:
            A tuple containing:
                - distances: A dictionary with the Wasserstein distances for each metric.
                - results: A dictionary with the computed metrics for the generated samples.
        """
        if atom_decoder is None:
            atom_decoder = self.atom_decoder
        num_samples = len(test_distribution[metrics[0]])

        n_batches = (
            num_samples + self.max_batched_samples - 1
        ) // self.max_batched_samples

        mols = []
        pyg_mols = []

        # Sequential sampling because vector_field_model and simulator
        # are likely on GPU and not easily shareable across processes
        # without complex management.
        # However, we can still benefit from parallelism in the evaluation/docking phase
        # if we want, or just keep sampling sequential and parallelize scoring.

        orig_p_data = self.path.p_data
        if self.val_p_data is not None:
            self.path.p_data = self.val_p_data
        try:
            for i in tqdm(
                range(n_batches),
                desc="Sampling molecules",
                unit="batch",
                ncols=100,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
            ):
                samples = self.sample(
                    min(
                        self.max_batched_samples, num_samples - i * self.max_batched_samples
                    ),
                    n_steps=self.sample_steps,
                )
                rd_mols = data_to_rdmol(samples, atom_decoder=atom_decoder, sanitize=True)
                if not isinstance(rd_mols, list):
                    rd_mols = [rd_mols]
                mols += rd_mols
                pyg_mols += data_to_list(samples)
        finally:
            self.path.p_data = orig_p_data

        mols_all = mols
        connectivity_score, connected_mols, connected_smiles, kept_indices = (
            compute_connectivity(mols)
        )

        pyg_mols = [pyg_mols[i] for i in kept_indices]
        if keep_biggest_fragment:
            mols = connected_mols
        if len(mols) == 0:
            print("Warning: No valid molecules generated.")
            return {metric: float("0") for metric in metrics}, {}

        if save_path is not None and len(mols) > 0:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # 1. Save RDKit molecules (SDF)
            writer = Chem.SDWriter(save_path)
            for m in mols:
                if m is not None:
                    writer.write(m)
            writer.close()

            # 2. Save PyTorch Geometric Data objects (.pt)
            pyg_save_path = os.path.splitext(save_path)[0] + ".pt"
            torch.save(pyg_mols, pyg_save_path)
            print(f"Saved SDF to {save_path} and PyG data to {pyg_save_path}")

        results = {m: [] for m in metrics}

        # Scoring can be parallelized
        if self.num_workers > 1:
            import concurrent.futures
            import multiprocessing as mp

            ctx = mp.get_context("spawn")

            # Prepare data for workers
            worker_data = []
            if "gnina" in metrics:
                for i in range(len(mols)):
                    worker_data.append((mols[i], pyg_mols[i]))
            else:
                worker_data = mols

            print(
                f"Scoring {len(worker_data)} molecules with {self.num_workers} workers..."
            )
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=self.num_workers, mp_context=ctx
            ) as executor:
                futures = [
                    executor.submit(
                        global_score_mol_worker,
                        d,
                        metrics,
                        scorer_dict,
                    )
                    for d in worker_data
                ]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc="Scoring",
                ):
                    mol_res = future.result()
                    for m, val in mol_res.items():
                        if m in results:
                            results[m].append(val)
                        elif m in ["vina", "cnn_affinity"]:
                            if m not in results:
                                results[m] = []
                            results[m].append(val)
        else:
            # Sequential scoring
            for idx in tqdm(range(len(mols)), desc="Scoring molecules"):
                mol = mols[idx]
                pyg_mol = pyg_mols[idx]
                for metric in metrics:
                    scorer = scorer_dict.get(metric)
                    if scorer is not None:
                        if metric == "gnina":
                            res = scorer(pyg_mol.to("cpu"))
                            if "vina" not in results:
                                results["vina"] = []
                            if "cnn_affinity" not in results:
                                results["cnn_affinity"] = []
                            results["vina"].append(res[0])
                            results["cnn_affinity"].append(res[1])
                        else:
                            results[metric].append(scorer(mol))

        metrics_calculated = list(results.keys())
        validity_score = min(len(connected_mols) / max(len(mols_all), 1), 1.0)
        uniqueness_ratio, connected_smiles_unique = compute_uniqueness(connected_smiles)

        res = {}
        res["validity"] = torch.tensor(validity_score, dtype=torch.float)
        res["connectivity"] = torch.tensor(connectivity_score, dtype=torch.float)
        res["uniqueness"] = torch.tensor(uniqueness_ratio, dtype=torch.float)

        for metric in metrics_calculated:
            if metric in results:
                # Filter out raw nan inputs from array blocks entirely before calculating stats
                clean_arr = [
                    v
                    for v in results[metric]
                    if v is not None and not np.isnan(v) and not np.isinf(v)
                ]

                if len(clean_arr) > 0:
                    m_val = np.mean(clean_arr)
                    s_val = np.std(clean_arr)
                    res[f"{metric}_mean"] = torch.tensor(
                        0.0 if np.isnan(m_val) else m_val, dtype=torch.float
                    )
                    res[f"{metric}_std"] = torch.tensor(
                        0.0 if np.isnan(s_val) else s_val, dtype=torch.float
                    )
                else:
                    res[f"{metric}_mean"] = torch.tensor(0.0, dtype=torch.float)
                    res[f"{metric}_std"] = torch.tensor(0.0, dtype=torch.float)

        distances = {}
        try:
            for metric in metrics_calculated:
                if metric in test_distribution:
                    expected = test_distribution[metric]
                    expected = [
                        e for e in expected if e is not None and not np.isnan(e)
                    ]
                    actual = results[metric]
                    # Filter matching validation evaluations
                    actual = [
                        a
                        for a in actual
                        if a is not None and not np.isnan(a) and not np.isinf(a)
                    ]

                    if len(actual) > 0 and len(expected) > 0:
                        expected_t = torch.tensor(expected)
                        actual_t = torch.tensor(actual, device=expected_t.device)
                        distance = wasserstein_1d(expected_t, actual_t)
                        dist_val = distance.item()
                        distances[metric + "_wl_distance"] = (
                            0.0 if np.isnan(dist_val) else dist_val
                        )
                    else:
                        distances[metric + "_wl_distance"] = 0.0

            return distances, res
        except Exception as e:
            print(f"Error calculating distances: {e}")
            return {
                metric + "_wl_distance": float(0)
                for metric in metrics
                if metric in test_distribution
            }, res

    def drift_coefficient(self, x, t, batch_all, batch_lig, batch_pocket, z):
        """Computes the drift coefficient (vector field) at time t.

        Args:
            x: Input tensor. Shape: [num_atoms, *dims].
            t: Time at which to compute the drift coefficient. Shape: [batch_size].
            batch_all: Tensor containing the combined batch indices for all samples.
            batch_lig: Tensor containing ligands associated with the samples.
            batch_pocket: Tensor containing pockets associated with the samples.
            z: Tensor containing the target pocket.

        Returns:
            The drift coefficient. Shape: [num_atoms, *dims].
        """
        is_lig = batch_lig != -1
        x_lig = x[is_lig, :]
        x_pocket = z
        u_t_theta_lig = self.vector_field_model(
            x_lig=x_lig,
            t=t,
            x_pocket=x_pocket,
            batch_lig=batch_lig,
            batch_pocket=batch_pocket,
        )
        guided_vector_field = torch.zeros_like(x)
        guided_vector_field[is_lig, :] = u_t_theta_lig
        guided_vector_field /= self.velocity_scale
        return guided_vector_field

    def unnorm_x(self, x_lig, x_pocket):
        """Unnormalizes the input tensor x using the stored normalization parameters."""
        if not self.unnorm:
            raise ValueError(
                "Normalization parameters are not set. Cannot unnormalize."
            )
        mean = torch.tensor(self.normalization_params["mean"]).to(x_pocket.device)
        std = torch.tensor(self.normalization_params["std"]).to(x_pocket.device)
        x_lig = x_lig * std + mean
        x_pocket = x_pocket * std + mean
        return x_lig, x_pocket

    @torch.no_grad()
    def sample(self, n_samples, n_steps=20, pocket=None, recenter=True):
        """Samples data by simulating the learned flow from the simple distribution.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.
            pocket: Optional pocket data to condition the sampling.
            recenter: If True, recenters the generated ligand to the pocket's original position. Defaults to True.

        Returns:
            The generated samples after simulating the flow.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")
        if pocket is not None:
            if hasattr(pocket, "batch_pocket"):
                num_pockets = pocket.batch_pocket.max().item() + 1
            else:
                num_pockets = 1

            if num_pockets == 1 and n_samples > 1:
                z_pocket_single = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)
                z_pocket = z_pocket_single.repeat(n_samples, 1)
                num_atoms_pocket = pocket.x_pocket.shape[0]
                batch_pocket = torch.arange(
                    n_samples, device=z_pocket.device
                ).repeat_interleave(num_atoms_pocket)
            else:
                z_pocket = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)
                batch_pocket = pocket.batch_pocket
            z_pocket = z_pocket.to(self.path.device)
            batch_pocket = batch_pocket.to(self.path.device)
        else:
            z_pocket, _, _, batch_pocket, _ = self.path.sample_conditioning_variable(
                n_samples, pocket_only=True, force_batch_size=True
            )
        starting_points, batch_all, batch_lig, batch_pocket = (
            self.path.sample_simple_distribution(n_samples, batch_pocket=batch_pocket)
        )
        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        starting_points[is_poc] = z_pocket
        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        output = simulator.simulate(
            starting_points, t_eval, batch_all, batch_lig, batch_pocket, z=z_pocket
        )
        output = output[:, -1]

        x_lig = output[is_lig, :3]
        h_lig = output[is_lig, 3:]
        x_pocket = output[is_poc, :3]
        h_pocket = output[is_poc, 3:]
        if self.unnorm:
            x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)

        if recenter and pocket is not None and hasattr(pocket, "original_centroid"):
            # Ensure centroids are on the correct device
            centroids = pocket.original_centroid.to(x_lig.device)
            if centroids.shape[0] == 1 and n_samples > 1:
                x_lig = x_lig + centroids[0]
                x_pocket = x_pocket + centroids[0]
            else:
                x_lig = x_lig + centroids[batch_lig[is_lig]]
                x_pocket = x_pocket + centroids[batch_pocket[is_poc]]

        data = Data(
            x_lig=x_lig,
            h_lig=h_lig,
            x_pocket=x_pocket,
            h_pocket=h_pocket,
            batch_all=batch_all,
            batch_lig=batch_lig[is_lig],
            batch_pocket=batch_pocket[is_poc],
        )
        return data

    @torch.no_grad()
    def sample_with_entropy(
        self,
        n_samples,
        n_steps=20,
        pocket=None,
        entropy_times=None,
        n_atoms=None,
        recenter=True,
    ):
        """Samples data by simulating the learned flow and returns entropy at specified times.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.
            pocket: Optional pocket data to condition the sampling.
            entropy_times: List of times at which to calculate the entropy.
            n_atoms: Optional number of atoms in the ligand to consider.
            recenter: If True, recenters the generated ligand to the pocket's original position. Defaults to True.

        Returns:
            The generated samples after simulating the flow, and a dictionary of entropies.
        """
        if entropy_times is None:
            entropy_times = []

        simulator = TorchDiffEqSimulator(self, solver="rk4")
        if pocket is not None:
            if hasattr(pocket, "batch_pocket"):
                num_pockets = pocket.batch_pocket.max().item() + 1
            else:
                num_pockets = 1

            if num_pockets == 1 and n_samples > 1:
                z_pocket_single = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)
                z_pocket = z_pocket_single.repeat(n_samples, 1)
                num_atoms_pocket = pocket.x_pocket.shape[0]
                batch_pocket = torch.arange(
                    n_samples, device=z_pocket.device
                ).repeat_interleave(num_atoms_pocket)
            else:
                z_pocket = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)
                batch_pocket = pocket.batch_pocket
            z_pocket = z_pocket.to(self.path.device)
            batch_pocket = batch_pocket.to(self.path.device)
        else:
            z_pocket, _, _, batch_pocket, _ = self.path.sample_conditioning_variable(
                n_samples, pocket_only=True, force_batch_size=True
            )

        if n_atoms is not None:
            if isinstance(n_atoms, int):
                size_ligand = torch.tensor(
                    [n_atoms] * n_samples, device=self.path.device
                )
            else:
                size_ligand = n_atoms.to(self.path.device)
        else:
            size_ligand = None

        starting_points, batch_all, batch_lig, batch_pocket = (
            self.path.sample_simple_distribution(
                n_samples, batch_pocket=batch_pocket, size_ligand=size_ligand
            )
        )
        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        starting_points[is_poc] = z_pocket

        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        if len(entropy_times) > 0:
            entropy_times_t = torch.tensor(entropy_times, device=self.path.device)
            t_eval = torch.cat([t_eval, entropy_times_t])
            t_eval = torch.unique(t_eval).sort()[0]

        output = simulator.simulate(
            starting_points, t_eval, batch_all, batch_lig, batch_pocket, z=z_pocket
        )

        entropies = {}
        for et in entropy_times:
            idx = (t_eval - et).abs().argmin().item()
            state = output[:, idx]
            x_lig = state[is_lig, :]
            t = torch.tensor([t_eval[idx]], device=self.path.device)
            _, diagnostics = self.vector_field_model(
                x_lig, t, z_pocket, batch_lig, batch_pocket, return_diagnostics=True
            )
            entropies[et] = diagnostics

        last_output = output[:, -1]
        x_lig = last_output[is_lig, :3]
        h_lig = last_output[is_lig, 3:]
        x_pocket = last_output[is_poc, :3]
        h_pocket = last_output[is_poc, 3:]
        if self.unnorm:
            x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)

        if recenter and pocket is not None and hasattr(pocket, "original_centroid"):
            # Ensure centroids are on the correct device
            centroids = pocket.original_centroid.to(x_lig.device)
            if centroids.shape[0] == 1 and n_samples > 1:
                x_lig = x_lig + centroids[0]
                x_pocket = x_pocket + centroids[0]
            else:
                x_lig = x_lig + centroids[batch_lig[is_lig]]
                x_pocket = x_pocket + centroids[batch_pocket[is_poc]]

        data = Data(
            x_lig=x_lig,
            h_lig=h_lig,
            x_pocket=x_pocket,
            h_pocket=h_pocket,
            batch_all=batch_all,
            batch_lig=batch_lig[is_lig],
            batch_pocket=batch_pocket[is_poc],
        )
        return data, entropies

    def sample_with_trajectory(
        self,
        n_samples,
        n_steps=20,
        force_connected=False,
        max_attempts=10,
        pocket=None,
        n_atoms=None,
        recenter=True,
    ):
        """Samples data by simulating the learned flow from the simple distribution and returns the trajectory.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.
            force_connected: If True, generate molecules until it is connected. Defaults to False.
            max_attempts: Maximum number of attempts to generate a connected molecule. Defaults to 10.
            pocket: Optional pocket data to condition the sampling.
            n_atoms: Optional number of atoms in the ligand to consider.
            recenter: If True, recenters the generated ligand to the pocket's original position. Defaults to True.

        Returns:
            The generated samples after simulating the flow, along with the trajectory.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")

        if pocket is not None:
            # Check if pocket is a single sample or a batch
            if hasattr(pocket, "batch_pocket"):
                num_pockets = pocket.batch_pocket.max().item() + 1
            else:
                num_pockets = 1

            if num_pockets == 1 and n_samples > 1:
                # Repeat the single pocket for each sample
                z_pocket_single = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)
                z_pocket = z_pocket_single.repeat(n_samples, 1)
                num_atoms_pocket = pocket.x_pocket.shape[0]
                batch_pocket = torch.arange(
                    n_samples, device=z_pocket.device
                ).repeat_interleave(num_atoms_pocket)
            else:
                z_pocket = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)
                batch_pocket = pocket.batch_pocket

            z_pocket = z_pocket.to(self.path.device)
            batch_pocket = batch_pocket.to(self.path.device)
        else:
            z_pocket, _, _, batch_pocket, _ = self.path.sample_conditioning_variable(
                n_samples, pocket_only=True, force_batch_size=True
            )

        if n_atoms is not None:
            if isinstance(n_atoms, int):
                size_ligand = torch.tensor(
                    [n_atoms] * n_samples, device=self.path.device
                )
            else:
                size_ligand = n_atoms.to(self.path.device)
        else:
            size_ligand = None

        starting_points, batch_all, batch_lig, batch_pocket = (
            self.path.sample_simple_distribution(
                n_samples, batch_pocket=batch_pocket, size_ligand=size_ligand
            )
        )

        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        starting_points[is_poc] = z_pocket

        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)

        output = simulator.simulate(
            starting_points, t_eval, batch_all, batch_lig, batch_pocket, z=z_pocket
        )

        attempts = 0
        while force_connected:
            final_output = output[:, -1]
            final_mol = Data(
                x_lig=final_output[is_lig, :3],
                h_lig=final_output[is_lig, 3:],
                x_pocket=final_output[is_poc, :3],
                h_pocket=final_output[is_poc, 3:],
                batch_all=batch_all,
                batch_lig=batch_lig[is_lig],
                batch_pocket=batch_pocket[is_poc],
            )
            final_rdmol = data_to_rdmol(
                final_mol, atom_decoder=self.atom_decoder, sanitize=True
            )
            connected = compute_connectivity(final_rdmol)[0]
            if connected > 0 or attempts > max_attempts:
                break
            else:
                print(
                    f"Regenerating sample to get a connected molecule ({attempts}/{max_attempts})"
                )
                starting_points, batch_all, batch_lig, batch_pocket = (
                    self.path.sample_simple_distribution(
                        n_samples, batch_pocket=batch_pocket, size_ligand=size_ligand
                    )
                )
                is_lig = batch_lig != -1
                is_poc = batch_pocket != -1
                starting_points[is_poc] = z_pocket
                output = simulator.simulate(
                    starting_points,
                    t_eval,
                    batch_all,
                    batch_lig,
                    batch_pocket,
                    z=z_pocket,
                )
                attempts += 1

        if attempts > max_attempts:
            print(
                f"Failed to generate a connected molecule after {max_attempts} attempts."
            )
            return []

        data_trajectory = []
        for i in range(output.shape[1]):
            x_lig = output[is_lig, i, :3]
            h_lig = output[is_lig, i, 3:]
            x_pocket = output[is_poc, i, :3]
            h_pocket = output[is_poc, i, 3:]
            if self.unnorm:
                x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)

            if recenter and pocket is not None and hasattr(pocket, "original_centroid"):
                centroids = pocket.original_centroid.to(x_lig.device)
                # For trajectories, we need to handle batch replication if n_samples > 1
                if centroids.shape[0] == 1 and n_samples > 1:
                    x_lig = x_lig + centroids[0]
                    x_pocket = x_pocket + centroids[0]
                else:
                    x_lig = x_lig + centroids[batch_lig[is_lig]]
                    x_pocket = x_pocket + centroids[batch_pocket[is_poc]]

            data = Data(
                batch_all=batch_all,
                x_lig=x_lig,
                h_lig=h_lig,
                batch_lig=batch_lig[is_lig],
                x_pocket=x_pocket,
                h_pocket=h_pocket,
                batch_pocket=batch_pocket[is_poc],
            )
            data_trajectory.append(data)
        return data_trajectory

    @torch.no_grad()
    def inpainting(
        self,
        ligand,
        pocket,
        n_atoms_to_add,
        batch_size=1,
        n_steps=80,
        t_max_inpainting=1.0,
        force_scaffold=False,
        recenter=True,
    ):
        """Performs inpainting by adding atoms to a given ligand within a pocket.

        Args:
            ligand: Data object representing the initial ligand.
            pocket: Data object representing the pocket.
            n_atoms_to_add: Number of atoms to add to the ligand. Can be int or torch.Tensor.
            batch_size: Number of samples to generate in parallel. Defaults to 1.
            n_steps: Number of integration steps for the simulator. Defaults to 80.
            t_max_inpainting: Maximum time to consider inpainting. Defaults to 1.0.
            force_scaffold: If True, fixes the scaffold atoms to their original values.
            recenter: If True, recenters the generated ligand to the pocket's original position. Defaults to True.

        Returns:
            The inpainted molecule after simulating the flow.
        """
        trajectory = self.inpainting_with_trajectory(
            ligand,
            pocket,
            n_atoms_to_add,
            batch_size,
            n_steps,
            t_max_inpainting,
            force_scaffold,
            recenter=recenter,
        )
        return trajectory[-1]

    @torch.no_grad()
    def inpainting_with_trajectory(
        self,
        ligand,
        pocket,
        n_atoms_to_add,
        batch_size=1,
        n_steps=80,
        t_max_inpainting=1.0,
        force_scaffold=False,
        recenter=True,
    ):
        """Performs inpainting and returns the full trajectory.

        Args:
            ligand: Data object representing the initial ligand.
            pocket: Data object representing the pocket.
            n_atoms_to_add: Number of atoms to add to the ligand. Can be int or torch.Tensor.
            batch_size: Number of samples to generate in parallel. Defaults to 1.
            n_steps: Number of integration steps for the simulator. Defaults to 80.
            t_max_inpainting: Maximum time to consider inpainting. Defaults to 1.0.
            force_scaffold: If True, fixes the scaffold atoms to their original values.
            recenter: If True, recenters the generated ligand to the pocket's original position. Defaults to True.

        Returns:
            List of Data objects representing the trajectory.
        """
        ligand = ligand.to(self.path.device)
        pocket = pocket.to(self.path.device)

        # 1. Handle batch replication if ligand/pocket are single graphs but batch_size > 1
        if batch_size > 1:
            from torch_geometric.data import Batch

            if (
                not hasattr(ligand, "num_graphs") or ligand.num_graphs < batch_size
            ) and ligand.x_lig.numel() > 0:
                ligand = Batch.from_data_list([ligand] * batch_size)
            if not hasattr(pocket, "num_graphs") or pocket.num_graphs < batch_size:
                pocket = Batch.from_data_list([pocket] * batch_size)

        # 2. Determine indices
        if hasattr(pocket, "batch_pocket") and pocket.batch_pocket is not None:
            batch_pocket_orig = pocket.batch_pocket
        elif hasattr(pocket, "batch") and pocket.batch is not None:
            batch_pocket_orig = pocket.batch
        else:
            batch_pocket_orig = torch.zeros(
                pocket.x_pocket.shape[0], dtype=torch.long, device=self.path.device
            )

        if hasattr(ligand, "batch_lig") and ligand.batch_lig is not None:
            batch_lig_orig = ligand.batch_lig
        elif hasattr(ligand, "batch") and ligand.batch is not None:
            batch_lig_orig = ligand.batch
        else:
            batch_lig_orig = torch.zeros(
                ligand.x_lig.shape[0], dtype=torch.long, device=self.path.device
            )

        # 3. Handle n_atoms_to_add
        if isinstance(n_atoms_to_add, int):
            n_atoms_to_add = torch.tensor(
                [n_atoms_to_add] * batch_size, device=self.path.device
            )
        else:
            n_atoms_to_add = torch.as_tensor(n_atoms_to_add, device=self.path.device)

        simulator = TorchDiffEqSimulator(self, solver="rk4")

        z_ligand = torch.cat([ligand.x_lig, ligand.h_lig], dim=-1)
        z_pocket = torch.cat([pocket.x_pocket, pocket.h_pocket], dim=-1)

        num_nodes_lig_existing = torch.bincount(batch_lig_orig, minlength=batch_size)
        num_nodes_pocket = torch.bincount(batch_pocket_orig, minlength=batch_size)
        n_total_lig = num_nodes_lig_existing + n_atoms_to_add

        # 4. Sample starting points (noise at time 0)
        starting_points, batch_all, batch_lig, batch_pocket_new = (
            self.path.sample_simple_distribution(
                batch_size,
                batch_pocket=batch_pocket_orig,
                size_ligand=n_total_lig,
            )
        )

        is_lig = batch_lig != -1
        is_poc = batch_pocket_new != -1

        # Fill pocket features in starting points
        starting_points[is_poc] = z_pocket

        # 5. Create mask for known nodes (scaffold) and end points (data at time 1)
        mask = torch.zeros(
            starting_points.shape[0], dtype=torch.bool, device=self.path.device
        )
        end_points = torch.zeros_like(starting_points)

        current_idx = 0
        current_lig_idx = 0
        current_pocket_idx = 0
        for i in range(batch_size):
            # Known scaffold mask (first num_nodes_lig_existing[i] atoms of the ligand part)
            mask[current_idx : current_idx + num_nodes_lig_existing[i]] = True

            # Known scaffold data in end_points
            end_points[current_idx : current_idx + num_nodes_lig_existing[i]] = (
                z_ligand[current_lig_idx : current_lig_idx + num_nodes_lig_existing[i]]
            )

            # Added nodes (remain zero in end_points)

            # Pocket data in end_points
            end_points[
                current_idx
                + n_total_lig[i] : current_idx
                + n_total_lig[i]
                + num_nodes_pocket[i]
            ] = z_pocket[current_pocket_idx : current_pocket_idx + num_nodes_pocket[i]]

            current_idx += n_total_lig[i] + num_nodes_pocket[i]
            current_lig_idx += num_nodes_lig_existing[i]
            current_pocket_idx += num_nodes_pocket[i]

        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)

        # 6. Inverse trajectory and Forward simulation
        if mask.any():
            inverse_n_steps = int(n_steps * t_max_inpainting)
            t_eval_inverse = torch.linspace(
                t_max_inpainting, 0, max(2, inverse_n_steps), device=self.path.device
            )

            # Inverse trajectory from end_points (data) to starting_points (noise)
            inverse_traj = simulator.simulate(
                end_points,
                t_eval_inverse,
                batch_all,
                batch_lig,
                batch_pocket_new,
                z=z_pocket,
            ).flip(
                1
            )  # Flip to have time increasing

            # Update scaffold parts of starting points with values from inverse trajectory at time 0
            starting_points[mask] = inverse_traj[mask, 0, :]

            # Forward simulation with inpainting
            output = simulator.simulate_inpaint(
                starting_points,
                inverse_traj,
                t_eval,
                t_max_inpainting,
                batch_all,
                batch_lig,
                batch_pocket_new,
                z=z_pocket,
                mask=mask,
                end_points=end_points,
                force_scaffold=force_scaffold,
                ts_inverse=t_eval_inverse.flip(0),
            )
        else:
            # If no scaffold is provided, just run regular forward simulation
            output = simulator.simulate(
                starting_points,
                t_eval,
                batch_all,
                batch_lig,
                batch_pocket_new,
                z=z_pocket,
            )

        # 8. Pack results into a trajectory of Data objects
        data_trajectory = []
        for i in range(output.shape[1]):
            x_lig = output[is_lig, i, :3]
            h_lig = output[is_lig, i, 3:]
            x_pocket = output[is_poc, i, :3]
            h_pocket = output[is_poc, i, 3:]
            if self.unnorm:
                x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)

            if recenter and pocket is not None and hasattr(pocket, "original_centroid"):
                centroids = pocket.original_centroid.to(x_lig.device)
                if centroids.shape[0] == 1 and batch_size > 1:
                    x_lig = x_lig + centroids[0]
                    x_pocket = x_pocket + centroids[0]
                else:
                    x_lig = x_lig + centroids[batch_lig[is_lig]]
                    x_pocket = x_pocket + centroids[batch_pocket_new[is_poc]]

            data = Data(
                x_lig=x_lig,
                h_lig=h_lig,
                x_pocket=x_pocket,
                h_pocket=h_pocket,
                batch_all=batch_all,
                batch_lig=batch_lig[is_lig],
                batch_pocket=batch_pocket_new[is_poc],
            )
            # Add scaffold mask to data for later coloring
            data.scaffold_mask = mask[is_lig]
            data_trajectory.append(data)

        return data_trajectory

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for training."""
        # 1. Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr, foreach=True)

        # 2. Define the scheduler
        scheduler = CustomScheduler(
            optimizer,
            start_lr=self.start_lr,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
            max_epochs=self.max_epochs,
            max_frac=self.max_frac,
        )

        # 3. Return the optimizer and scheduler configuration
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Scheduler is called after each epoch
                "frequency": 1,
            },
        }

    def _test_integration(self, n_steps=20):

        class ode:
            def __init__(self, model):
                self.model = model

            def drift_coefficient(self, x, t, batch_all, batch_lig, batch_pocket, z):
                return self.model.path.conditional_vector_field(x, z, t)

        simulator = TorchDiffEqSimulator(ode(self), solver="rk4")

        z, batch_all, batch_lig, batch_pocket, batch_size = (
            self.path.sample_conditioning_variable(1, force_batch_size=True)
        )
        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        z_lig = z[is_lig]
        z_pocket = z[is_poc]

        # t, t_batched = self.path.sample_time(batch_size, batch_all)
        ts = torch.linspace(0, 1, n_steps, device=self.path.device)

        starting_points = torch.randn_like(z)
        starting_points[is_poc] = z_pocket

        # starting_points[is_lig] = z_lig

        output = simulator.simulate(
            starting_points, ts, batch_all, batch_lig, batch_pocket, z=z
        )

        data_trajectory = []
        for i in range(output.shape[1]):
            x_lig = output[is_lig, i, :3]
            h_lig = output[is_lig, i, 3:]
            x_pocket = output[is_poc, i, :3]
            h_pocket = output[is_poc, i, 3:]
            if self.unnorm:
                x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)
            data = Data(
                batch_all=batch_all,
                x_lig=x_lig,
                h_lig=h_lig,
                batch_lig=batch_lig[is_lig],
                x_pocket=x_pocket,
                h_pocket=h_pocket,
                batch_pocket=batch_pocket[is_poc],
            )
            data_trajectory.append(data)
        return data_trajectory
