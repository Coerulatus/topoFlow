"""Gaussian Conditional Probability Path."""

import torch

from topoflow.flow.paths.abstract_paths import ConditionalProbabilityPath
from topoflow.flow.utils import IsotropicGaussian, DistributionNodes


class GaussianConditionalProbabilityPath(ConditionalProbabilityPath):
    """Implements the conditional probability path for a Gaussian distribution.

    This class defines the specific functions required for a conditional
    probability path where the underlying distributions are Gaussian, enabling
    the calculation of samples, vector fields, and scores.

    Attributes:
        alpha (callable): Function defining the time-dependent scaling of the
            conditioning variable.
        beta (callable): Function defining the time-dependent scaling of the
            noise.
    """

    def __init__(
        self,
        p_data,
        alpha,
        beta,
        alpha_atom=None,
        beta_atom=None,
        n_coords=3,
        device="cpu",
        target_dim=None,
        distribution_hist=None,
    ):
        """Initializes the GaussianConditionalProbabilityPath.

        Args:
            p_data: The data distribution.
            alpha: Function that defines the time-dependent scaling of the
                conditioning variable.
            beta: Function that defines the time-dependent scaling of the noise.
            alpha_atom: Optional function for scaling of the
                conditioning variable for the atom type. If None, `alpha` is used. Defaults to None.
            beta_atom: Optional function for scaling of the noise for the atom type. If
                None, `beta` is used. Defaults to None.
            n_coords: Number of coordinates per atom (e.g., 3 for 3D data).
            device: The device to perform computations on (e.g., 'cpu' or 'cuda').
                Defaults to 'cpu'.
            target_dim: The target dimension of the data. If None, it's inferred
                from `p_data.dim`. Defaults to None.
            distribution_hist: Optional size distribution for the data. If not
                provided, it will be inferred from `p_data.dataset.distribution_hist`.
        """
        if target_dim is None:
            target_dim = p_data.dim
        p_simple = IsotropicGaussian(target_dim, 1.0)
        super().__init__(p_simple, p_data, device)
        self.alpha = alpha
        self.alpha.device = device
        self.beta = beta
        self.beta.device = device
        self.alpha_atom = alpha_atom if alpha_atom is not None else alpha
        self.alpha_atom.device = device
        self.beta_atom = beta_atom if beta_atom is not None else beta
        self.beta_atom.device = device
        self.n_coords = n_coords
        if distribution_hist is not None:
            self.size_distribution = DistributionNodes(distribution_hist)
        else:
            try:
                self.size_distribution = DistributionNodes(
                    p_data.dataset.distribution_hist
                )
            except AttributeError:
                raise ValueError(
                    "The provided `p_data.dataset` must have a `size_distribution` attribute."
                )

    def sample_conditioning_variable(
        self, batch_size: int, pocket_only=False, force_batch_size=False
    ):
        """Samples the conditioning variable `z`.

        Args:
            batch_size: The number of samples to generate.
            pocket_only: If True, only samples pocket atoms.
            force_batch_size: If True, ensures exactly `batch_size` samples are returned.

        Returns:
            A tuple containing:
                - `z`: The conditioning variable sampled from the data distribution.
                  [n_atoms, *dims]
                - `batch_lig`: The ligands associated with the samples.
                  [n_atoms, *dims]
                - `batch_pocket`: The pockets associated with the samples.
                  [n_atoms, *dims]
        """
        z, batch_all, batch_lig, batch_pocket, batch_size = self.p_data.sample(
            batch_size, force_batch_size
        )
        if pocket_only:
            z = z[torch.where(batch_pocket != -1)[0], :]  # Only keep pocket atoms
            batch_lig = None
        else:
            batch_lig = batch_lig.to(self.device)
        z = z.to(self.device)
        batch_all = batch_all.to(self.device)
        batch_pocket = batch_pocket.to(self.device)

        return z, batch_all, batch_lig, batch_pocket, batch_size

    def sample_conditional_path(
        self, z: torch.Tensor, t: torch.Tensor, start_points: torch.Tensor = None
    ) -> torch.Tensor:
        """Samples from the conditional distribution p_t(x|z).

        Args:
            z: Conditioning variable. [batch_size, *dims]
            t: Time at which to sample the conditional distribution. [batch_size, 1]
            start_points: (Optional) Starting points for the sampling.

        Returns:
            Samples `x` from p_t(x|z). [batch_size, *dims]
        """
        batch_size = z.shape[0]

        if start_points is None:
            w = self.p_simple.sample(num_samples=batch_size).to(self.device)
            w_coords = w[:, : self.n_coords]
            w_atoms = w[:, self.n_coords :]
        else:
            # If start_points are provided, they replace the noise sampling (w)
            w_coords = start_points[:, : self.n_coords]
            w_atoms = start_points[:, self.n_coords :]

        # Apply the scaling factors
        x_coords = self.alpha(t) * z[:, : self.n_coords] + self.beta(t) * w_coords
        x_atoms = (
            self.alpha_atom(t) * z[:, self.n_coords :] + self.beta_atom(t) * w_atoms
        )

        x = torch.cat([x_coords, x_atoms], dim=-1)

        return x.to(self.device)

    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the conditional vector field u_t(x|z).

        Args:
            x: Sample. [batch_size, *dims]
            z: Conditioning variable. [batch_size, *dims]
            t: Time. [batch_size, 1]

        Returns:
            The conditional vector field. [batch_size, *dims]
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        dt_alpha_t = self.alpha.dt(t)
        dt_beta_t = self.beta.dt(t)

        alpha_t_atom = self.alpha_atom(t)
        beta_t_atom = self.beta_atom(t)
        dt_alpha_t_atom = self.alpha_atom.dt(t)
        dt_beta_t_atom = self.beta_atom.dt(t)

        u_coords = (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z[
            :, : self.n_coords
        ] + dt_beta_t / beta_t * x[:, : self.n_coords]
        u_atoms = (dt_alpha_t_atom - dt_beta_t_atom / beta_t_atom * alpha_t_atom) * z[
            :, self.n_coords :
        ] + dt_beta_t_atom / beta_t_atom * x[:, self.n_coords :]
        u = torch.cat([u_coords, u_atoms], dim=-1)
        return u.to(self.device)

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Deprecated. Here for compatibility.

        Evaluates the conditional score function of p_t(x|z).

        Args:
            x: Sample. [batch_size, *dims]
            z: Conditioning variable. [batch_size, *dims]
            t: Time. [batch_size, 1]

        Returns:
            The conditional score function. [batch_size, *dims]
        """
        alpha_t = self.alpha(t)
        beta_t = self.beta(t)
        s = (z * alpha_t - x) / beta_t**2
        return s.to(self.device)
