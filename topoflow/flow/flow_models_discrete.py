"""Flow-based generative models."""

import os
import numpy as np
import lightning.pytorch as pl
import torch
from torch_geometric.data import Data
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

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
)
from topoflow.utils.molecules import data_to_rdmol


class FlowModel(pl.LightningModule):
    """A PyTorch Lightning module for training a vector field model in a flow-based generative model.

    This class handles the training, validation, and sampling procedures for a
    model that learns the vector field of a continuous normalizing flow.
    """

    def __init__(
        self,
        vector_field_model,
        path,
        discrete_path,
        evaluator=None,
        n_val_samples=0,
        max_batched_samples=16,
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
        w_loss_last_step=0.2,
        w_loss_atom=0.2,
        time_factor=None,
    ):
        """Initializes the FlowModel.

        Args:
            vector_field_model: The neural network model that estimates the vector field.
            path: An object representing the probability path, providing methods for
                sampling and conditional vector fields.
            discrete_path: An object representing the discrete probability path.
            evaluator: An optional evaluator for assessing generated samples.
            n_val_samples: Number of samples to generate during validation for visualization.
                Defaults to 0.
            max_batched_samples: Maximum number of samples to process in a single batch.
                Defaults to 16.
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
            w_loss_last_step: Weight for the loss at the last time step. Defaults to 0.2.
            w_loss_atom: Weight for the atom type loss. Defaults to 0.2.
            time_factor: Optional function to scale the loss by time.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["path", "discrete_path"])
        self.vector_field_model = vector_field_model
        self.path = path
        self.discrete_path = discrete_path
        self.path.device = device
        if hasattr(path.p_data, "unguided_label"):
            self.unguided_label = path.p_data.unguided_label
        else:
            self.unguided_label = None
        self.val_samples = []
        self.n_val_samples = n_val_samples
        self.val_distribution = val_distribution
        self.atom_decoder = atom_decoder
        self.loss_function = torch.nn.MSELoss(reduction="none")
        # self.loss_function_atoms = torch.nn.CrossEntropyLoss(reduction="none")
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
        self.gumbel_temp = 1.0

        # self.forward = torch.compile(self.forward)

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
        last_step_loss_idxs = (t > self.t_last_step).squeeze()
        if torch.any(last_step_loss_idxs) and x_lig is not None:
            loss_last_step = self.loss_function_last_step(
                u_t_theta_lig[last_step_loss_idxs],
                x_lig[last_step_loss_idxs],
                t[last_step_loss_idxs],
                batch_lig[last_step_loss_idxs],
                verbose=verbose,
                mean=torch.tensor(self.normalization_params["mean"]).to(x_lig.device),
                std=torch.tensor(self.normalization_params["std"]).to(x_lig.device),
            )
            loss_last_step = loss_last_step.mean(dim=-1) * self.w_loss_last_step
        else:
            loss_last_step = torch.tensor(0, dtype=torch.float32)

        loss_mse = self.loss_function(u_t_theta_lig[:, :3], u_t_ref_lig[:, :3])
        loss_atoms = self.loss_function(u_t_theta_lig[:, 3:], u_t_ref_lig[:, 3:])

        loss_mse = loss_mse.mean(dim=-1)
        loss_atoms = loss_atoms.mean(dim=-1) * self.w_loss_atom

        loss = loss_mse + loss_atoms + loss_last_step
        loss_without_time = loss.clone().mean().detach()

        if self.time_factor is not None and t is not None:
            loss *= self.time_factor(t).view(-1)
        loss = loss.mean()
        return {
            "loss": loss,
            "loss_mse": loss_mse.mean(),
            "loss_atoms": loss_atoms.mean(),
            "loss_last_step": loss_last_step.mean(),
            "loss_without_time": loss_without_time.mean(),
        }

    def _get_data(self, batch):
        """Extracts data from p_data."""
        batch_size = batch.shape[0]
        z, batch_all, batch_lig, batch_pocket, batch_size = (
            self.path.sample_conditioning_variable(batch_size)
        )
        t, t_batched = self.path.sample_time(batch_size, batch_all)

        z_coords = z[:, :3]
        z_atoms = z[:, 3:]

        x_coords = self.path.sample_conditional_path(z_coords, t_batched)
        x_discrete = self.discrete_path.sample_conditional_path(
            z_atoms, t_batched, temperature=self.gumbel_temp
        )
        u_t_ref_coords = self.path.conditional_vector_field(
            x_coords, z_coords, t_batched
        )
        u_t_ref_atoms = self.discrete_path.conditional_vector_field(
            x_discrete, z_atoms, t_batched
        )

        x = torch.cat([x_coords, x_discrete], dim=-1)
        u_t_ref = torch.cat([u_t_ref_coords, u_t_ref_atoms], dim=-1)

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

    def forward(self, batch):
        """Forward pass for the FlowModel.

        Samples from the path, computes the reference vector field, and predicts
        the vector field using the `vector_field_model`.

        Args:
            batch: The input batch of data.

        Returns:
            A tuple containing:
                - u_t_theta_lig: The predicted ligand vector field by the model.
                - u_t_ref_lig: The reference ligand vector field from the path.
                - t_batched_lig: The time values corresponding to ligand atoms.
        """
        batch_all = batch.get("batch_all")
        batch_lig = batch.get("batch_lig")
        batch_pocket = batch.get("batch_pocket")
        t = batch.get("t")
        t_batched = batch.get("t_batched")
        x = batch.get("x")
        u_t_ref = batch.get("u_t_ref")

        # Get ligand and pocket indices
        lig_indices = torch.where(batch_lig != -1)[0]
        pocket_indices = torch.where(batch_pocket != -1)[0]

        # Extract the ligand
        x_lig = x[lig_indices, :]
        u_t_ref_lig = u_t_ref[lig_indices, :]

        # Extract the pocket positions instead of velocities
        x_pocket = x[pocket_indices, :]

        # Only predict ligand velocities, conditioned on pocket positions
        u_t_theta_lig = self.vector_field_model(
            x_lig, t, x_pocket, batch_lig, batch_pocket
        )

        t_batched_lig = t_batched[lig_indices]

        return u_t_theta_lig, u_t_ref_lig, t_batched_lig, x_lig

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
        u_t_theta, u_t_ref, t, x_lig = self.forward(batch)
        losses = self._get_loss(u_t_theta, u_t_ref, t, x_lig, batch_lig, verbose=False)

        for name, value in losses.items():
            self.log(
                f"train/{name}", value, on_epoch=True, prog_bar=False, on_step=False
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
        batch = self._get_data(batch)
        batch_lig = batch["batch_lig"]
        batch_lig = batch_lig[batch_lig != -1]
        u_t_theta, u_t_ref, t, x_lig = self.forward(batch)
        losses = self._get_loss(u_t_theta, u_t_ref, t, x_lig, batch_lig)

        self.log("val/loss", losses["loss"], on_epoch=True, prog_bar=False)
        self.log(
            "val/loss_without_time",
            losses["loss_without_time"],
            on_epoch=True,
            prog_bar=False,
        )

        if (
            self.val_distribution is not None
            and batch_idx == 0
            and self.n_epoch % self.sample_every == 0
        ):
            distances, other_metrics = self.test_distribution(
                test_distribution=self.val_distribution,
                metrics=[
                    "qed",
                    "logp",
                    "sas",
                ],  # TODO: add lipinski, but also in the val_dist
                atom_decoder=self.atom_decoder,
            )
            for metric, distance in distances.items():
                self.log(
                    f"val/{metric}_wasserstein_distance",
                    distance,
                    on_epoch=True,
                    prog_bar=False,
                )
            for metric, value in other_metrics.items():
                self.log(f"val/{metric}", value, on_epoch=True, prog_bar=False)

        self.n_epoch += 1
        return losses["loss"]

    def on_after_backward(self):
        """Logs the gradients after the backward pass."""
        # TODO: remove this after debugging
        for name, param in self.named_parameters():
            if param.grad is not None:
                # Log the L2 norm of the gradient
                self.log(
                    f"gradients/{name}_norm",
                    param.grad.norm(),
                    on_epoch=True,
                    on_step=False,
                )

                # Log the average absolute value of the gradient
                # self.log(
                #     f"gradients/{name}_mean_abs",
                #     param.grad.abs().mean(),
                #     on_epoch=True,
                #     on_step=False,
                # )
        pass

    @torch.no_grad()
    def test_distribution(
        self, test_distribution, metrics=["qed", "logp", "sas"], atom_decoder=None
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
        """
        if atom_decoder is None:
            atom_decoder = self.atom_decoder
        num_samples = len(test_distribution[metrics[0]])

        n_batches = (
            num_samples + self.max_batched_samples - 1
        ) // self.max_batched_samples
        mols = []
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
            mols += data_to_rdmol(samples, atom_decoder=atom_decoder, sanitize=True)
        results = {}
        for metric in metrics:
            if metric not in test_distribution:
                raise ValueError(f"Metric {metric} not found in test distribution.")
            elif metric == "qed":
                results[metric] = [qed_scorer(mol) for mol in mols]
                results[metric] = [s for s in results[metric] if s is not None]
            elif metric == "logp":
                results[metric] = [logp_scorer(mol) for mol in mols]
                results[metric] = [s for s in results[metric] if s is not None]
            elif metric == "sas":
                results[metric] = [sas_scorer(mol) for mol in mols]
                results[metric] = [s for s in results[metric] if s is not None]
            elif metric == "lipinski":
                results[metric] = [lipinski_scorer(mol) for mol in mols]
                results[metric] = [s for s in results[metric] if s is not None]
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        # Check connectivity, uniqueness and novelty
        validity_score = num_samples / len(mols)
        connectivity_score, connected = compute_connectivity(mols)
        uniqueness_ratio, unique_smiles = compute_uniqueness(connected)
        # novelty_ratio, novel_smiles = compute_novelty(
        #     connected, self.path.p_data.dataset.smiles_list
        # )
        res = {}
        res["validity"] = torch.tensor(validity_score, dtype=torch.float)
        res["connectivity"] = torch.tensor(connectivity_score, dtype=torch.float)
        res["uniqueness"] = torch.tensor(uniqueness_ratio, dtype=torch.float)
        # res["novelty"] = torch.tensor(novelty_ratio, dtype=torch.float)

        distances = {}
        try:
            for metric in metrics:
                expected = test_distribution[metric]
                expected = [e for e in expected if e is not None]
                actual = results[metric]
                if isinstance(expected, list):
                    expected = torch.tensor(expected)
                if isinstance(actual, list):
                    actual = torch.tensor(actual, device=expected.device)
                distance = wasserstein_1d(expected, actual)
                distances[metric] = distance.item()

            return distances, res
        except Exception as e:
            print(f"Error calculating distances: {e}")
            return {metric: float("nan") for metric in metrics}, res

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
            x_lig, t, x_pocket, batch_lig, batch_pocket
        )

        # For the pocket we already have the final positions, so the velocity is zero
        guided_vector_field = torch.zeros_like(x)
        guided_vector_field[is_lig, :] = u_t_theta_lig

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
    def sample(self, n_samples, n_steps=20):
        """Samples data by simulating the learned flow from the simple distribution.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.

        Returns:
            The generated samples after simulating the flow.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")

        z_pocket, _, _, batch_pocket, _ = self.path.sample_conditioning_variable(
            n_samples, pocket_only=True, force_batch_size=True
        )

        starting_points, batch_all, batch_lig, batch_pocket = (
            self.path.sample_simple_distribution(n_samples, batch_pocket=batch_pocket)
        )
        starting_points[:, 3:] = self.discrete_path.sample_simple_distribution(
            starting_points.shape[0]
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
        data = Data(
            x_lig=x_lig,
            h_lig=h_lig,
            x_pocket=x_pocket,
            h_pocket=h_pocket,
            batch_all=batch_all,
            batch_lig=batch_lig,
            batch_pocket=batch_pocket,
        )
        return data

    def sample_with_trajectory(
        self, n_samples, n_steps=20, force_connected=False, max_attempts=10
    ):
        """Samples data by simulating the learned flow from the simple distribution and returns the trajectory.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.
            force_connected: If True, generate molecules until it is connected. Defaults to False.
            max_attempts: Maximum number of attempts to generate a connected molecule. Defaults to 10.

        Returns:
            The generated samples after simulating the flow, along with the trajectory.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")

        z_pocket, _, _, batch_pocket, _ = self.path.sample_conditioning_variable(
            n_samples, pocket_only=True, force_batch_size=True
        )

        starting_points, batch_all, batch_lig, batch_pocket = (
            self.path.sample_simple_distribution(n_samples, batch_pocket=batch_pocket)
        )
        starting_points[:, 3:] = self.discrete_path.sample_simple_distribution(
            starting_points.shape[0]
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
                batch_lig=batch_lig,
                batch_pocket=batch_pocket,
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
                        n_samples, batch_pocket=batch_pocket
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
            x_lig = output[torch.where(batch_lig != -1)[0], i, :3]
            h_lig = output[torch.where(batch_lig != -1)[0], i, 3:]
            x_pocket = output[torch.where(batch_pocket != -1)[0], i, :3]
            h_pocket = output[torch.where(batch_pocket != -1)[0], i, 3:]
            if self.unnorm:
                x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)
            data = Data(
                batch_all=batch_all,
                x_lig=x_lig,
                h_lig=h_lig,
                batch_lig=batch_lig,
                x_pocket=x_pocket,
                h_pocket=h_pocket,
                batch_pocket=batch_pocket,
            )
            data_trajectory.append(data)
        return data_trajectory

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler for training."""
        # 1. Define the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)

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

        simulator = TorchDiffEqSimulator(self, solver="rk4")

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
            x_lig = output[torch.where(batch_lig != -1)[0], i, :3]
            h_lig = output[torch.where(batch_lig != -1)[0], i, 3:]
            x_pocket = output[torch.where(batch_pocket != -1)[0], i, :3]
            h_pocket = output[torch.where(batch_pocket != -1)[0], i, 3:]
            if self.unnorm:
                x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)
            data = Data(
                batch_all=batch_all,
                x_lig=x_lig,
                h_lig=h_lig,
                batch_lig=batch_lig,
                x_pocket=x_pocket,
                h_pocket=h_pocket,
                batch_pocket=batch_pocket,
            )
            data_trajectory.append(data)
        return data_trajectory
