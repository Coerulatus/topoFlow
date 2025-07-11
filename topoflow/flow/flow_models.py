"""Flow-based generative models."""

import os
import numpy as np
import lightning.pytorch as pl
import torch
from torch_geometric.data import Data
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from topoflow.flow.simulator import TorchDiffEqSimulator
from topoflow.flow.utils_plot import plot_results
from topoflow.evaluators import RDKitEvaluator, GATEvaluator
from topoflow.utils.metrics import wasserstein_1d, qed_scorer, logp_scorer, sas_scorer
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
        evaluator=None,
        n_val_samples=0,
        val_distribution=None,
        sample_every=1,
        atom_decoder=None,
        norm_path=None,
        device="cpu",
        lr=1e-3,
    ):
        """Initializes the FlowModel.

        Args:
            vector_field_model: The neural network model that estimates the vector field.
            path: An object representing the probability path, providing methods for
                sampling and conditional vector fields.
            evaluator: An optional evaluator for assessing generated samples.
            n_val_samples: Number of samples to generate during validation for visualization.
                Defaults to 0.
            val_distribution: The expected distribution of metrics for validation.
            sample_every: Frequency of sampling during validation. Defaults to 1.
            atom_decoder: Optional atom decoder for converting samples to RDKit molecules.
            norm_path: Path to the normalization parameters file.
            device: The device on which to run computations ('cpu' or 'cuda'). Defaults to 'cpu'.
            lr: Learning rate for the optimizer. Defaults to 1e-3.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["path"])
        self.vector_field_model = vector_field_model
        self.path = path
        self.path.device = device
        if hasattr(path.p_data, "unguided_label"):
            self.unguided_label = path.p_data.unguided_label
        else:
            self.unguided_label = None
        self.val_samples = []
        self.n_val_samples = n_val_samples
        self.val_distribution = val_distribution
        self.atom_decoder = atom_decoder
        self.loss_function = torch.nn.MSELoss()
        self.train_loss = []
        self.lr = lr
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
                "mean_lig": norm_params["mean_lig"],
                "std_lig": norm_params["std_lig"],
                "mean_pocket": norm_params["mean_pocket"],
                "std_pocket": norm_params["std_pocket"],
            }
            self.unnorm = True

    def forward(self, batch):
        """Forward pass for the FlowModel.

        Samples from the path, computes the reference vector field, and predicts
        the vector field using the `vector_field_model`.

        Args:
            batch: The input batch of data.

        Returns:
            A tuple containing:
                - u_t_theta: The predicted vector field by the model.
                - u_t_ref: The reference vector field from the path.
        """
        batch_size = batch.shape[0]
        # We override the batch size to ensure it matches the expected size
        # path.p_data.sample() returns always the same batch_size even if the input batch is smaller.
        z, batch_lig, batch_pocket, batch_size = self.path.sample_conditioning_variable(
            batch_size
        )
        t, t_batched = self.path.sample_time(batch_size, batch_lig, batch_pocket)
        x = self.path.sample_conditional_path(z, t_batched)
        u_t_ref = self.path.conditional_vector_field(x, z, t_batched)
        u_t_theta = self.vector_field_model(x, t, batch_lig, batch_pocket)
        return u_t_theta, u_t_ref

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Calculates the MSE loss between the predicted and reference vector fields.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The training loss for the current step.
        """
        u_t_theta, u_t_ref = self.forward(batch)
        loss = self.loss_function(u_t_theta, u_t_ref)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.train_loss.append(loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Calculates the MSE loss and optionally samples and logs generated data.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The validation loss for the current step.
        """
        u_t_theta, u_t_ref = self.forward(batch)
        loss = self.loss_function(u_t_theta, u_t_ref)

        self.log("val/loss", loss, on_epoch=True, prog_bar=False)
        if (
            self.n_val_samples > 0
            and batch_idx == 0
            and self.n_epoch % self.sample_every == 0
        ):
            samples = self.sample(self.n_val_samples, n_steps=50)
            # self.val_samples.append(samples)

            if self.evaluator is not None:
                results = self.evaluator.evaluate(samples)
                prefixed = {f"val/{k}": v for k, v in results.items()}
                self.log_dict(prefixed, on_epoch=True, prog_bar=False)
        self.n_epoch += 1
        if (
            self.val_distribution is not None
            and batch_idx == 0
            and self.n_epoch % self.sample_every == 0
        ):
            distances = self.test_distribution(
                test_distribution=self.val_distribution,
                metrics=["qed", "logp", "sas"],
                atom_decoder=self.atom_decoder,
            )
            for metric, distance in distances.items():
                self.log(
                    f"val/{metric}_wasserstein_distance",
                    distance,
                    on_epoch=True,
                    prog_bar=False,
                )
        return loss

    def test_step(self, batch, batch_idx):
        """Performs a single test step.

        Calculates the MSE loss.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The test loss for the current step.
        """
        u_t_theta, u_t_ref = self.forward(batch)
        loss = self.loss_function(u_t_theta, u_t_ref)

        self.log("test/loss", loss)

        return loss

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
        MAX_BATCH_SIZE = 16
        if atom_decoder is None:
            atom_decoder = self.atom_decoder
        num_samples = len(test_distribution[metrics[0]])

        n_batches = (num_samples + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE
        mols = []
        for i in range(n_batches):
            samples = self.sample(
                min(MAX_BATCH_SIZE, num_samples - i * MAX_BATCH_SIZE),
                n_steps=50,
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
            else:
                raise ValueError(f"Unsupported metric: {metric}")

        distances = {}
        try:
            for metric in metrics:
                expected = test_distribution[metric]
                actual = results[metric]
                if isinstance(expected, list):
                    expected = torch.tensor(expected)
                if isinstance(actual, list):
                    actual = torch.tensor(actual, device=expected.device)
                distance = wasserstein_1d(expected, actual)
                distances[metric] = distance.item()

            return distances
        except Exception as e:
            print(f"Error calculating distances: {e}")
            return {metric: float("nan") for metric in metrics}

    def drift_coefficient(self, x, t, batch_lig, batch_pocket):
        """Computes the drift coefficient (vector field) at time t.

        Applies classifier-free guidance if `guidance_scale` is greater than 0
        and conditioning is enabled.

        Args:
            x: Input tensor. Shape: [batch_size, *dims].
            t: Time at which to compute the drift coefficient. Shape: [batch_size, 1].
            batch_lig: Tensor containing ligands associated with the samples.
                Shape: [batch_size, *dims].
            batch_pocket: Tensor containing pockets associated with the samples.
                Shape: [batch_size, *dims].

        Returns:
            The drift coefficient. Shape: [batch_size, *dims].

        Raises:
            ValueError: If `unguided_label` is not provided when `guidance_scale` > 0.
        """
        guided_vector_field = self.vector_field_model(x, t, batch_lig, batch_pocket)
        return guided_vector_field

    def unnorm_x(self, x_lig, x_pocket):
        """Unnormalizes the input tensor x using the stored normalization parameters."""
        if not self.unnorm:
            raise ValueError(
                "Normalization parameters are not set. Cannot unnormalize."
            )
        mean_lig = torch.tensor(self.normalization_params["mean_lig"]).to(x_lig.device)
        std_lig = torch.tensor(self.normalization_params["std_lig"]).to(x_lig.device)
        mean_pocket = torch.tensor(self.normalization_params["mean_pocket"]).to(
            x_pocket.device
        )
        std_pocket = torch.tensor(self.normalization_params["std_pocket"]).to(
            x_pocket.device
        )

        x_lig = x_lig * std_lig + mean_lig
        x_pocket = x_pocket * std_pocket + mean_pocket
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
        starting_points, batch_lig, batch_pocket = self.path.sample_simple_distribution(
            n_samples
        )
        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        output = simulator.simulate(starting_points, t_eval, batch_lig, batch_pocket)
        output = output[:, -1]
        x_lig = output[: batch_lig.shape[0], :3]
        h_lig = output[: batch_lig.shape[0], 3:]
        x_pocket = output[batch_lig.shape[0] :, :3]
        h_pocket = output[batch_lig.shape[0] :, 3:]
        if self.unnorm:
            x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)
        data = Data(
            x_lig=x_lig,
            h_lig=h_lig,
            batch_lig=batch_lig,
            x_pocket=x_pocket,
            h_pocket=h_pocket,
            batch_pocket=batch_pocket,
        )
        return data

    def sample_with_trajectory(self, n_samples, n_steps=20):
        """Samples data by simulating the learned flow from the simple distribution and returns the trajectory.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.

        Returns:
            The generated samples after simulating the flow, along with the trajectory.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")
        starting_points, batch_lig, batch_pocket = self.path.sample_simple_distribution(
            n_samples
        )
        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        output = simulator.simulate(starting_points, t_eval, batch_lig, batch_pocket)
        data_trajectory = []
        for i in range(output.shape[1]):
            x_lig = output[: batch_lig.shape[0], i, :3]
            h_lig = output[: batch_lig.shape[0], i, 3:]
            x_pocket = output[batch_lig.shape[0] :, i, :3]
            h_pocket = output[batch_lig.shape[0] :, i, 3:]
            if self.unnorm:
                x_lig, x_pocket = self.unnorm_x(x_lig, x_pocket)
            data = Data(
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
        """Configures the optimizer and learning rate scheduler.

        Returns:
            A tuple containing a list of optimizers and a list of schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return [optimizer]


class LatentFlowModel(FlowModel):
    """A FlowModel specifically designed for training in a latent space, incorporating an autoencoder.

    This model extends FlowModel by adding autoencoder-related losses
    (reconstruction, LERP, contrastive, KL divergence) to train the flow in the
    latent space learned by the autoencoder.
    """

    def __init__(
        self,
        vector_field_model,
        path,
        autoencoder,
        l2_vf_loss_weight=0,
        recon_loss_weight=0,
        lerp_loss_weight=0,
        contrastive_loss_weight=0,
        kl_loss_weight=0,
        n_val_samples=0,
        eta=0,
        val_y_target=None,
        val_guidance_scale=0.0,
        device="cpu",
        conditioned=True,
    ):
        """Initializes the LatentFlowModel.

        Args:
            vector_field_model: The neural network model that estimates the vector field in latent space.
            path: An object representing the probability path in latent space.
            autoencoder: The autoencoder model used for encoding data into and decoding from
                the latent space.
            l2_vf_loss_weight: Weight for the L2 norm of the vector field loss. Defaults to 0.
            recon_loss_weight: Weight for the reconstruction loss from the autoencoder. Defaults to 0.
            lerp_loss_weight: Weight for the LERP loss on latent embeddings. Defaults to 0.
            contrastive_loss_weight: Weight for the contrastive loss on latent embeddings. Defaults to 0.
            kl_loss_weight: Weight for the KL divergence loss on latent embeddings. Defaults to 0.
            n_val_samples: Number of samples to generate during validation. Defaults to 0.
            eta: Probability of replacing the true conditioning label with an
                unguided label during training. Defaults to 0.
            val_y_target: The target conditioning variable for validation samples. Defaults to None.
            val_guidance_scale: The guidance scale for validation sampling. Defaults to 0.0.
            device: The device to run computations on. Defaults to 'cpu'.
            conditioned: A boolean indicating whether the model uses conditioning. Defaults to True.
        """
        super().__init__(
            vector_field_model,
            path,
            n_val_samples,
            eta,
            val_y_target,
            val_guidance_scale,
            device,
            conditioned,
        )
        self.save_hyperparameters()
        self.autoencoder = autoencoder
        self.l2_vf_loss_weight = l2_vf_loss_weight
        self.recon_loss_weight = recon_loss_weight
        self.lerp_loss_weight = lerp_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.train_losses = {
            "loss": [],
            "flow_loss": [],
            "l2_vf_loss": [],
            "recon_loss": [],
            "lerp_loss": [],
            "contrastive_loss": [],
            "kl_loss": [],
        }

    def forward(self, batch, training=False):
        """Forward pass for the LatentFlowModel.

        Encodes the input data, samples from the latent path, computes reference
        and predicted vector fields in latent space, and decodes the latent
        representation.

        Args:
            batch: The input batch of data.
            training: A boolean indicating if the model is in training mode. Defaults to False.

        Returns:
            A tuple containing:
                - u_t_theta: The predicted vector field in latent space.
                - u_t_ref: The reference vector field in latent space.
                - original_z: The original input data for autoencoder loss.
                - encoded_z: The encoded latent representation.
                - decoded_z: The decoded data from the latent representation.
                - original_y: The original conditioning label.
        """
        batch_size = batch.shape[0]
        original_z, original_y = self.path.sample_conditioning_variable(batch_size)
        y = original_y.clone() if self.conditioned else None
        encoded_z = self.autoencoder.encode(original_z)
        encoded_z = encoded_z.view(batch_size, -1)
        t = self.path.sample_time(batch_size)
        x = self.path.sample_conditional_path(encoded_z, t)
        u_t_ref = self.path.conditional_vector_field(x, encoded_z, t)

        if self.eta > 0 and training and y is not None:
            unguided_idxs = torch.where(
                torch.rand_like(y, dtype=torch.float) < self.eta
            )[0]
            y[unguided_idxs] = self.unguided_label

        u_t_theta = self.vector_field_model(x, t, y)
        encoded_z = encoded_z.view(
            batch_size, *self.autoencoder.autoencoder.bottleneck_shape
        )
        decoded_z = self.autoencoder.decode(encoded_z)

        return u_t_theta, u_t_ref, original_z, encoded_z, decoded_z, original_y

    @torch.no_grad()
    def sample(self, n_samples, n_steps=20, y=None, guidance_scale=0.0):
        """Samples data by simulating the learned flow in latent space and then decoding.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.
            y: Conditioning variable for the samples. Defaults to None.
            guidance_scale: Guidance scale for sampling. Defaults to 0.0.

        Returns:
            The generated samples in original data space after decoding.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")
        starting_points = self.path.p_simple.sample(n_samples)[0]
        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        output = simulator.simulate(
            starting_points, t_eval, y=y, guidance_scale=guidance_scale
        )
        output = output[:, -1]
        output = output.view(-1, *self.autoencoder.autoencoder.bottleneck_shape)
        output = self.autoencoder.decode(output)
        return output

    def training_step(self, batch, batch_idx):
        """Performs a single training step for the LatentFlowModel.

        Calculates flow loss and various autoencoder-related losses.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The total training loss for the current step.
        """
        u_t_theta, u_t_ref, original_z, encoded_z, decoded_z, y = self.forward(
            batch, training=True
        )

        loss_flow = self.loss_function(u_t_theta, u_t_ref)

        if self.l2_vf_loss_weight > 0:
            loss_l2_vf = self.l2_vf_loss_weight * torch.norm(u_t_theta, p=2)
        else:
            loss_l2_vf = torch.tensor(0.0, device=self.path.device)

        loss_recon, loss_lerp, loss_contrastive, loss_kl = self.autoencoder.get_loss(
            original_z,
            y,
            self.recon_loss_weight,
            self.lerp_loss_weight,
            self.contrastive_loss_weight,
            self.kl_loss_weight,
        )

        self.log("flow_loss", loss_flow, on_epoch=False, prog_bar=True)
        self.log("l2_vf_loss", loss_l2_vf, on_epoch=False, prog_bar=True)
        self.log("recon_loss", loss_recon, on_epoch=False, prog_bar=True)
        self.log("lerp_loss", loss_lerp, on_epoch=False, prog_bar=True)
        self.log("contrastive_loss", loss_contrastive, on_epoch=False, prog_bar=True)
        self.log("kl_loss", loss_kl, on_epoch=False, prog_bar=True)
        loss = (
            loss_flow + loss_l2_vf + loss_recon + loss_lerp + loss_contrastive + loss_kl
        )
        self.train_losses["loss"].append(loss.item())
        self.train_losses["flow_loss"].append(loss_flow.item())
        self.train_losses["l2_vf_loss"].append(loss_l2_vf.item())
        self.train_losses["recon_loss"].append(loss_recon.item())
        self.train_losses["lerp_loss"].append(loss_lerp.item())
        self.train_losses["contrastive_loss"].append(loss_contrastive.item())
        self.train_losses["kl_loss"].append(loss_kl.item())
        return loss


class DebugLatentFlowModel(LatentFlowModel):
    """A subclass of LatentFlowModel for debugging and analysis.

    Provides additional methods for testing the simulator's integration error
    and visualizing latent space trajectories using PCA.
    """

    def __init__(self, **kwargs):
        """Initializes the DebugLatentFlowModel.

        Args:
            **kwargs: Arbitrary keyword arguments passed to the `LatentFlowModel` constructor.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

    @torch.no_grad()
    def test_simulator(self, n_samples, n_steps=100, y=None, guidance_scale=0.0):
        """Tests the simulator's integration error.

        Args:
            n_samples: Number of samples to use for testing.
            n_steps: Number of integration steps. Defaults to 100.
            y: Conditioning variable. Defaults to None.
            guidance_scale: Guidance scale. Defaults to 0.0.

        Returns:
            The integration error.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")
        starting_points = self.path.p_simple.sample(n_samples)[0]
        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        error = simulator.integration_error(
            starting_points, t_eval, y=y, guidance_scale=guidance_scale
        )
        return error

    def inspect_latent_trajectories(
        self, n_samples=100, guidance_scale=0.0, n_steps=10, pca_dims=2
    ):
        """Visualizes latent space trajectories using PCA.

        Simulates trajectories in the latent space and plots them, optionally
        colored by their conditioning labels.

        Args:
            n_samples: Number of trajectories to simulate. Defaults to 100.
            guidance_scale: Guidance scale for trajectory simulation. Defaults to 0.0.
            n_steps: Number of steps for trajectory simulation. Defaults to 10.
            pca_dims: Number of principal components for visualization (2 or 3). Defaults to 2.
        """
        batch, y = self.path.p_data.sample(n_samples)
        if guidance_scale == 0:
            y = None
        B = batch.shape[0]
        t = torch.linspace(0, 1, n_steps, device=self.path.device)
        simulator = TorchDiffEqSimulator(self, solver="rk4")
        encoded_batch = self.autoencoder.encode(batch)
        encoded_batch = encoded_batch.view(B, -1)
        trajectory = simulator.simulate_trajectory(
            encoded_batch, t, y=y, guidance_scale=guidance_scale
        )
        ends_trajectory = trajectory[:, -1]
        ends_trajectory = ends_trajectory.view(
            B, *self.autoencoder.autoencoder.bottleneck_shape
        )
        decoded_trajectory = self.autoencoder.decode(ends_trajectory)
        plot_results([decoded_trajectory.detach()])
        latent_dims = encoded_batch.shape[-1]
        np_trajectories = trajectory.reshape(-1, latent_dims).detach().cpu().numpy()
        if pca_dims == 3:
            ax = plt.axes(projection="3d")
        pca = PCA(n_components=pca_dims)
        trajectories_nd = pca.fit_transform(np_trajectories)
        trajectories_nd = trajectories_nd.reshape(B, -1, pca_dims)
        labels = [None for _ in range(10)]
        for i in range(B):
            if y is None:
                color = np.random.rand(
                    3,
                )
            else:
                color = plt.cm.tab10(y[i].item())
            if pca_dims == 2:
                if y is not None and labels[y[i].item()] is None:
                    labels[y[i].item()] = f"y={y[i].item()}"
                    label = labels[y[i].item()]
                else:
                    label = None
                plt.scatter(
                    trajectories_nd[i, 0, 0],
                    trajectories_nd[i, 0, 1],
                    color=color,
                    label=label,
                )
                plt.plot(
                    trajectories_nd[i, :, 0], trajectories_nd[i, :, 1], color=color
                )
            elif pca_dims == 3:
                ax.scatter(
                    trajectories_nd[i, 0, 0],
                    trajectories_nd[i, 0, 1],
                    trajectories_nd[i, 0, 2],
                    color=color,
                )
                ax.plot(
                    trajectories_nd[i, :, 0],
                    trajectories_nd[i, :, 1],
                    trajectories_nd[i, :, 2],
                    color=color,
                )
        plt.title("Latent Trajectories")
        plt.xlabel("Latent Dimension 1")
        plt.ylabel("Latent Dimension 2")
        plt.legend()
        plt.show()


class LatentFlowModelWithClassifier(LatentFlowModel):
    """A LatentFlowModel that integrates a separate classifier for additional guidance.

    This model adds a classifier loss to the training objective, aiming to improve
    the disentanglement or control over the latent space with respect to
    conditioning labels.
    """

    def __init__(self, classifier=None, classifier_loss_weight=0.0, **kwargs):
        """Initializes the LatentFlowModelWithClassifier.

        Args:
            classifier: An optional classifier model. Defaults to None.
            classifier_loss_weight: Weight for the classifier loss. Defaults to 0.0.
            **kwargs: Arbitrary keyword arguments passed to the `LatentFlowModel` constructor.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.classifier = classifier
        if self.classifier is not None:
            self.classifier.to(self.path.device)
        self.classifier_loss_weight = classifier_loss_weight
        self.train_losses = {
            "loss": [],
            "flow_loss": [],
            "l2_vf_loss": [],
            "recon_loss": [],
            "lerp_loss": [],
            "contrastive_loss": [],
            "kl_loss": [],
            "classifier_loss": [],
        }

    def training_step(self, batch, batch_idx):
        """Performs a single training step for the LatentFlowModelWithClassifier.

        Calculates flow loss, autoencoder-related losses, and a classifier loss.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The total training loss for the current step.
        """
        u_t_theta, u_t_ref, original_z, encoded_z, decoded_z, y = self.forward(
            batch, training=True
        )

        loss_flow = self.loss_function(u_t_theta, u_t_ref)
        if self.l2_vf_loss_weight > 0:
            loss_l2_vf = self.l2_vf_loss_weight * torch.norm(u_t_theta, p=2)
        else:
            loss_l2_vf = torch.tensor(0.0, device=self.path.device)
        loss_recon, loss_lerp, loss_contrastive, loss_kl = self.autoencoder.get_loss(
            original_z,
            y,
            self.recon_loss_weight,
            self.lerp_loss_weight,
            self.contrastive_loss_weight,
            self.kl_loss_weight,
        )
        if self.classifier is not None:
            classifier_loss = self.classifier.get_loss(encoded_z, y)
            classifier_loss *= self.classifier_loss_weight
        else:
            classifier_loss = torch.tensor(0.0, device=self.path.device)

        self.log("flow_loss", loss_flow, on_epoch=False, prog_bar=True)
        self.log("l2_vf_loss", loss_l2_vf, on_epoch=False, prog_bar=True)
        self.log("recon_loss", loss_recon, on_epoch=False, prog_bar=True)
        self.log("lerp_loss", loss_lerp, on_epoch=False, prog_bar=True)
        self.log("contrastive_loss", loss_contrastive, on_epoch=False, prog_bar=True)
        self.log("kl_loss", loss_kl, on_epoch=False, prog_bar=True)
        self.log("classifier_loss", classifier_loss, on_epoch=False, prog_bar=True)
        loss = (
            loss_flow
            + loss_l2_vf
            + loss_recon
            + loss_lerp
            + loss_contrastive
            + loss_kl
            + classifier_loss
        )
        self.train_losses["loss"].append(loss.item())
        self.train_losses["flow_loss"].append(loss_flow.item())
        self.train_losses["l2_vf_loss"].append(loss_l2_vf.item())
        self.train_losses["recon_loss"].append(loss_recon.item())
        self.train_losses["lerp_loss"].append(loss_lerp.item())
        self.train_losses["contrastive_loss"].append(loss_contrastive.item())
        self.train_losses["kl_loss"].append(loss_kl.item())
        self.train_losses["classifier_loss"].append(classifier_loss.item())
        return loss
