"""Flow-based generative models."""

import numpy as np
import pytorch_lightning as pl
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

from topoflow.flow.simulator import TorchDiffEqSimulator
from topoflow.flow.utils_plot import plot_results


class FlowModel(pl.LightningModule):
    """A PyTorch Lightning module for training a vector field model in a flow-based generative model.

    This class handles the training, validation, and sampling procedures for a
    model that learns the vector field of a continuous normalizing flow.
    """

    def __init__(
        self,
        vector_field_model,
        path,
        n_val_samples=0,
        eta=0,
        val_y_target=None,
        val_guidance_scale=0.0,
        device="cpu",
        conditioned=True,
    ):
        """Initializes the FlowModel.

        Args:
            vector_field_model: The neural network model that estimates the vector field.
            path: An object representing the probability path, providing methods for
                sampling and conditional vector fields.
            n_val_samples: Number of samples to generate during validation for visualization.
                Defaults to 0.
            eta: Probability of replacing the true conditioning label with an
                unguided label during training for classifier-free guidance. Defaults to 0.
            val_y_target: The target conditioning variable for validation samples. Can be
                an int, a torch.Tensor, or None. Defaults to None.
            val_guidance_scale: The guidance scale to use during validation sampling.
                Defaults to 0.0.
            device: The device on which to run computations ('cpu' or 'cuda'). Defaults to 'cpu'.
            conditioned: A boolean indicating whether the model uses conditioning. Defaults to True.
        """
        super().__init__()
        self.vector_field_model = vector_field_model
        self.path = path
        self.path.device = device
        if hasattr(path.p_data, "unguided_label"):
            self.unguided_label = path.p_data.unguided_label
        else:
            self.unguided_label = None
        self.val_samples = []
        self.val_guidance_scale = val_guidance_scale
        self.val_y_target = val_y_target
        if (
            isinstance(self.val_y_target, torch.Tensor)
            and self.val_y_target.shape[0] != 1
        ):
            self.n_val_samples = self.val_y_target.shape[0]
        else:
            self.n_val_samples = n_val_samples
        self.eta = eta
        self.loss_function = torch.nn.MSELoss()
        self.train_loss = []
        self.conditioned = conditioned

    def forward(self, batch, training=False):
        """Forward pass for the FlowModel.

        Samples from the path, computes the reference vector field, and predicts
        the vector field using the `vector_field_model`.

        Args:
            batch: The input batch of data.
            training: A boolean indicating if the model is in training mode. Defaults to False.

        Returns:
            A tuple containing:
                - u_t_theta: The predicted vector field by the model.
                - u_t_ref: The reference vector field from the path.
        """
        batch_size = batch.shape[0]
        z, y = self.path.sample_conditioning_variable(batch_size)
        if not self.conditioned:
            y = None
        t = self.path.sample_time(batch_size)
        x = self.path.sample_conditional_path(z, t)
        u_t_ref = self.path.conditional_vector_field(x, z, t)

        if self.eta > 0 and training and y is not None:
            unguided_idxs = torch.where(
                torch.rand_like(y, dtype=torch.float) < self.eta
            )[0]
            y[unguided_idxs] = self.unguided_label

        u_t_theta = self.vector_field_model(x, t, y)
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
        u_t_theta, u_t_ref = self.forward(batch, training=True)
        loss = self.loss_function(u_t_theta, u_t_ref)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
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
        res = self.forward(batch)
        u_t_theta, u_t_ref = res[0], res[1]
        loss = self.loss_function(u_t_theta, u_t_ref)

        self.log("val_loss", loss, on_epoch=True, prog_bar=False)
        if self.n_val_samples > 0 and batch_idx == 0:
            if self.val_y_target is None or not self.conditioned:
                y = None
            elif isinstance(self.val_y_target, torch.Tensor):
                y = self.val_y_target.to(u_t_ref.device)
            elif isinstance(self.val_y_target, int):
                y = torch.tensor(self.val_y_target, dtype=torch.int).to(u_t_ref.device)
            self.val_samples.append(
                self.sample(
                    self.n_val_samples, y=y, guidance_scale=self.val_guidance_scale
                ).cpu()
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

        self.log("test_loss", loss)
        return loss

    def drift_coefficient(self, x, t, y=None, guidance_scale=0.0):
        """Computes the drift coefficient (vector field) at time t.

        Applies classifier-free guidance if `guidance_scale` is greater than 0
        and conditioning is enabled.

        Args:
            x: Input tensor. Shape: [batch_size, *dims].
            t: Time at which to compute the drift coefficient. Shape: [batch_size, 1].
            y: Conditioning variable. Shape: [batch_size, 1]. Defaults to None.
            guidance_scale: Scale for guidance. Defaults to 0.0.

        Returns:
            The drift coefficient. Shape: [batch_size, *dims].

        Raises:
            ValueError: If `unguided_label` is not provided when `guidance_scale` > 0.
        """
        if guidance_scale > 0 and self.unguided_label is None and self.conditioned:
            raise ValueError(
                "Unguided label must be provided when guidance scale is greater than 0."
            )
        if self.conditioned and y is None:
            y = torch.ones(x.shape[0], dtype=torch.int) * self.unguided_label
        guided_vector_field = self.vector_field_model(x, t, y)
        if guidance_scale > 0 and self.conditioned:
            unguided_y = torch.ones_like(y) * self.unguided_label
            unguided_vector_field = self.vector_field_model(x, t, unguided_y)
            result = (
                guidance_scale * guided_vector_field
                + (1 - guidance_scale) * unguided_vector_field
            )
        else:
            result = guided_vector_field
        return result

    @torch.no_grad()
    def sample(self, n_samples, n_steps=20, y=None, guidance_scale=0.0):
        """Samples data by simulating the learned flow from the simple distribution.

        Args:
            n_samples: Number of samples to generate.
            n_steps: Number of integration steps for the simulator. Defaults to 20.
            y: Conditioning variable for the samples. Defaults to None.
            guidance_scale: Guidance scale for sampling. Defaults to 0.0.

        Returns:
            The generated samples after simulating the flow.
        """
        simulator = TorchDiffEqSimulator(self, solver="rk4")
        starting_points = self.path.p_simple.sample(n_samples)[0]
        t_eval = torch.linspace(0, 1, n_steps, device=self.path.device)
        output = simulator.simulate(
            starting_points, t_eval, y=y, guidance_scale=guidance_scale
        )
        return output[:, -1]

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler.

        Returns:
            A tuple containing a list of optimizers and a list of schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=3, eta_min=1e-6
        )
        return [optimizer], [scheduler]


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
