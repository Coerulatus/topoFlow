"""PyTorch Lightning modules for Autoencoders and Gaussian Autoencoders."""

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from topoflow.flow.vector_field_models.unet import MNISTUNet


class AutoEncoder(pl.LightningModule):
    """A PyTorch Lightning module for a basic Autoencoder.

    This class encapsulates an autoencoder model and defines its training and
    validation logic, including reconstruction and optional latent space
    interpolation (LERP) loss.
    """

    def __init__(self, autoencoder, n_val_samples=0, lambda_lerp=0):
        """Initializes the AutoEncoder.

        Args:
            autoencoder: The underlying autoencoder model (e.g., a UNet).
            n_val_samples: Number of samples to save for visualization during validation.
                Defaults to 0.
            lambda_lerp: Weight for the linear interpolation (LERP) loss in the
                latent space. If 0, LERP loss is not used. Defaults to 0.
        """
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = autoencoder
        self.loss_fn = torch.nn.MSELoss()
        self.n_val_samples = n_val_samples
        self.val_samples = []
        self.lambda_lerp = lambda_lerp

    def forward(self, batch):
        """Forward pass of the autoencoder.

        Args:
            batch: The input data tensor.

        Returns:
            The reconstructed output from the autoencoder.
        """
        x = self.autoencoder(batch, t=None, y=None)
        return x

    def encode(self, batch):
        """Encodes the input data into the latent space.

        Args:
            batch: The input data tensor.

        Returns:
            The latent representation of the input.
        """
        x = self.autoencoder.encode(batch)
        return x[0]

    def decode(self, batch):
        """Decodes a latent representation back into data space.

        Args:
            batch: The latent space tensor.

        Returns:
            The reconstructed data.
        """
        x = self.autoencoder.decode(batch, t=None, y=None, residuals=[])
        return x

    def training_step(self, batch, batch_idx):
        """Performs a single training step.

        Calculates reconstruction loss and optionally LERP loss.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The total training loss for the current step.
        """
        x = batch["x"]
        z = self.encode(x)
        x_recon = self.decode(z)
        loss_recon = self.loss_fn(x_recon, x)
        loss = loss_recon
        if self.lambda_lerp > 0:
            idx_perm = torch.randperm(x.shape[0])
            z_perm = z[idx_perm]
            x_perm = x[idx_perm]
            target_shape = [z.shape[0]] + [1] * (len(z.shape) - 1)
            alpha = torch.rand(target_shape, device=x.device)
            z_lerp = torch.lerp(z, z_perm, alpha)
            x_recon_lerp = self.decode(z_lerp)
            x_lerp = torch.lerp(x, x_perm, alpha)
            loss_lerp = self.loss_fn(x_recon_lerp, x_lerp)
            loss += self.lambda_lerp * loss_lerp

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Calculates reconstruction loss and logs validation samples for visualization.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The validation loss for the current step.
        """
        x = batch["x"]
        x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.val_samples.append(x_recon[: self.n_val_samples])
        return loss

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler.

        Returns:
            A tuple containing a list of optimizers and a list of schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]


class GaussianAutoEncoder(pl.LightningModule):
    """A PyTorch Lightning module for a Gaussian Autoencoder (Variational Autoencoder).

    This class extends the basic autoencoder by introducing a probabilistic
    latent space, encouraging it to follow a Gaussian distribution, and
    includes a KL divergence loss term. It also supports reconstruction, LERP,
    and supervised contrastive losses.
    """

    def __init__(
        self,
        autoencoder,
        n_val_samples=0,
        recon_weight=2.0,
        lerp_weight=2.0,
        contrastive_weight=0.5,
        kl_weight=0.1,
        free_bits=0.0,
    ):
        """Initializes the GaussianAutoEncoder.

        Args:
            autoencoder: The underlying autoencoder model (e.g., a UNet) that
                will be used as the backbone.
            n_val_samples: Number of samples to save for visualization during validation.
                Defaults to 0.
            recon_weight: Weight for the reconstruction loss. Defaults to 2.0.
            lerp_weight: Weight for the linear interpolation (LERP) loss in the
                latent space. If 0, LERP loss is not used. Defaults to 2.0.
            contrastive_weight: Weight for the supervised contrastive loss. If 0,
                contrastive loss is not used. Defaults to 0.5.
            kl_weight: Weight for the KL divergence loss. If 0, KL loss is not used.
                Defaults to 0.1.
            free_bits: Amount of "free bits" for the KL divergence loss, preventing
                the encoder from collapsing too early. Defaults to 0.0.
        """
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = autoencoder
        self.encoder_out_dim = self.autoencoder.bottleneck_dim
        self.encoder_out_shape = self.autoencoder.bottleneck_shape
        self.fc_mu = torch.nn.Linear(self.encoder_out_dim, self.encoder_out_dim)
        self.fc_logvar = torch.nn.Linear(self.encoder_out_dim, self.encoder_out_dim)
        self.loss_fn = torch.nn.MSELoss()
        self.n_val_samples = n_val_samples
        self.val_samples = []
        self.recon_weight = recon_weight
        self.lerp_weight = lerp_weight
        self.contrastive_weight = contrastive_weight
        self.kl_weight = kl_weight
        self.free_bits = free_bits

    def forward(self, batch):
        """Forward pass of the Gaussian Autoencoder.

        Encodes the input, samples from the latent distribution, and then decodes.

        Args:
            batch: The input data tensor.

        Returns:
            The reconstructed output.
        """
        z = self.encode(batch)
        x = self.decode(z.view(-1, *self.encoder_out_shape))
        return x

    def encode(self, batch, get_z=True):
        """Encodes the input data into the parameters of the latent Gaussian distribution.

        Args:
            batch: The input data tensor.
            get_z: If True, samples and returns a latent vector `z`. If False,
                returns the mean `mu` and log-variance `logvar` of the latent distribution.
                Defaults to True.

        Returns:
            A latent vector `z` if `get_z` is True, otherwise a tuple `(mu, logvar)`.
        """
        x = self.autoencoder.encode(batch)[0]
        x = x.view(-1, self.encoder_out_dim)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        if get_z:
            z = self.get_z(mu, logvar)
            z = z.view(-1, *self.encoder_out_shape)
            return z
        return mu, logvar

    def decode(self, batch):
        """Decodes a latent representation back into data space.

        Args:
            batch: The latent space tensor.

        Returns:
            The reconstructed data.
        """
        x = self.autoencoder.decode(batch, t=None, y=None, residuals=[])
        return x

    def get_z(self, mu, logvar):
        """Samples a latent vector `z` from the given mean and log-variance.

        Uses the reparameterization trick for sampling.

        Args:
            mu: Mean of the latent Gaussian distribution.
            logvar: Log-variance of the latent Gaussian distribution.

        Returns:
            A sampled latent vector `z`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def training_step(self, batch, batch_idx):
        """Performs a single training step, calculating various loss components.

        Args:
            batch: The current batch of data, potentially including 'x' and 'y' (labels).
            batch_idx: The index of the current batch.

        Returns:
            The total training loss for the current step.
        """
        x = batch["x"]
        y = batch.get("y", None)
        losses = self.get_loss(
            x,
            y,
            self.recon_weight,
            self.lerp_weight,
            self.contrastive_weight,
            self.kl_weight,
        )
        loss = sum(losses)
        self.log("loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("recon_loss", losses[0], on_step=False, on_epoch=True, prog_bar=True)
        self.log("lerp_loss", losses[1], on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "contrastive_loss", losses[2], on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("kl_loss", losses[3], on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def get_loss(
        self,
        x,
        y=None,
        recon_weight=1,
        lerp_weight=1,
        contrastive_weight=0.5,
        kl_weight=0.1,
    ):
        """Calculates all loss components for the Gaussian Autoencoder.

        Args:
            x: Input data tensor.
            y: Optional conditioning labels. Defaults to None.
            recon_weight: Weight for the reconstruction loss. Defaults to 1.
            lerp_weight: Weight for the LERP loss. Defaults to 1.
            contrastive_weight: Weight for the supervised contrastive loss. Defaults to 0.5.
            kl_weight: Weight for the KL divergence loss. Defaults to 0.1.

        Returns:
            A tuple containing: (recon_loss, lerp_loss, contrastive_loss, kl_loss).
        """
        mu, logvar = self.encode(x, get_z=False)
        z = self.get_z(mu, logvar)
        z = z.view(-1, *self.encoder_out_shape)
        x_recon = self.decode(z)
        loss_recon = self.loss_fn(x_recon, x)
        loss_recon *= recon_weight
        if lerp_weight > 0:
            idx_perm = torch.randperm(x.shape[0])
            z_perm = z[idx_perm]
            x_perm = x[idx_perm]
            target_shape = [z.shape[0]] + [1] * (len(z.shape) - 1)
            alpha = torch.rand(target_shape, device=x.device)
            z_lerp = torch.lerp(z, z_perm, alpha)
            x_recon_lerp = self.decode(z_lerp)
            x_lerp = torch.lerp(x, x_perm, alpha)
            loss_lerp = self.loss_fn(x_recon_lerp, x_lerp)
            loss_lerp *= lerp_weight
        else:
            loss_lerp = torch.tensor(0.0, device=x.device)
        if contrastive_weight > 0 and y is not None:
            loss_contrastive = self.get_supervised_contrastive_loss(z, y)
            loss_contrastive *= contrastive_weight
        else:
            loss_contrastive = torch.tensor(0.0, device=x.device)
        if kl_weight > 0:
            loss_kl_per_sample = -0.5 * torch.sum(
                1 + logvar - mu.pow(2) - logvar.exp(), dim=1
            )

            if self.free_bits > 0:
                loss_kl_per_sample = torch.clamp(
                    loss_kl_per_sample - self.free_bits, min=0
                )

            loss_kl = torch.mean(loss_kl_per_sample)
            loss_kl *= kl_weight
        else:
            loss_kl = torch.tensor(0.0, device=x.device)
        return loss_recon, loss_lerp, loss_contrastive, loss_kl

    def get_supervised_contrastive_loss(self, z, labels, temperature=0.05):
        """Computes supervised contrastive loss on latent embeddings.

        Args:
            z (torch.Tensor): Latent vectors of shape (B, D).
            labels (torch.Tensor): Class labels of shape (B,).
            temperature (float): Temperature parameter for scaling similarities.

        Returns:
            torch.Tensor: Scalar contrastive loss.
        """
        batch_size = labels.size(0)

        z = z.view(batch_size, -1)
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.T) / temperature

        logits_mask = torch.ones(
            (batch_size, batch_size), dtype=torch.bool, device=z.device
        )
        logits_mask.fill_diagonal_(False)

        labels = labels.contiguous().view(-1, 1)
        positive_mask = (labels == labels.T) & logits_mask

        logits = similarity_matrix.masked_fill(~logits_mask, float("-inf"))

        log_prob = F.log_softmax(logits, dim=1)
        diagonal = torch.eye(log_prob.size(0), device=log_prob.device).bool()
        log_prob = log_prob.masked_fill(diagonal, 0)

        loss = -(log_prob * positive_mask).sum(1) / positive_mask.sum(1).clamp(min=1)
        return loss.mean()

    def validation_step(self, batch, batch_idx):
        """Performs a single validation step.

        Calculates reconstruction loss and logs validation samples for visualization.

        Args:
            batch: The current batch of data.
            batch_idx: The index of the current batch.

        Returns:
            The validation loss for the current step.
        """
        x = batch["x"]
        x_recon = self(x)
        loss = self.loss_fn(x_recon, x)
        self.log("val_loss", loss)
        if batch_idx == 0:
            self.val_samples.append(x_recon[: self.n_val_samples])
        return loss

    def configure_optimizers(self):
        """Configures the optimizer and learning rate scheduler.

        Returns:
            A tuple containing a list of optimizers and a list of schedulers.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]
