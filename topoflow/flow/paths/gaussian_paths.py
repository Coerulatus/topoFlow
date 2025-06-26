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
        self, p_data, alpha, beta, device="cpu", target_dim=None, histogram=None
    ):
        """Initializes the GaussianConditionalProbabilityPath.

        Args:
            p_data: The data distribution, expected to be a Gaussian.
            alpha: Function that defines the time-dependent scaling of the
                conditioning variable.
            beta: Function that defines the time-dependent scaling of the noise.
            device: The device to perform computations on (e.g., 'cpu' or 'cuda').
                Defaults to 'cpu'.
            target_dim: The target dimension of the data. If None, it's inferred
                from `p_data.dim`. Defaults to None.
            histogram: Optional histogram with number of nodes for ligand and pockets.
        """
        if target_dim is None:
            target_dim = p_data.dim
        p_simple = IsotropicGaussian(target_dim, 1.0)
        super().__init__(p_simple, p_data, device)
        self.alpha = alpha
        # Note: Directly assigning a 'device' attribute to the callable might
        # not be standard practice for all callables. Ensure `alpha` and `beta`
        # objects are designed to handle this or pass device explicitly to their
        # calls if needed.
        self.alpha.device = device
        self.beta = beta
        self.beta.device = device
        if histogram is not None:
            self.nodes_distribution = DistributionNodes(histogram)
        else:
            self.nodes_distribution = None

    def sample_conditioning_variable(self, batch_size: int):
        """Samples the conditioning variable `z`.

        Args:
            batch_size: The number of samples to generate.

        Returns:
            A tuple containing:
                - `z`: The conditioning variable sampled from the data distribution.
                  [n_atoms, *dims]
                - `batch_lig`: The ligands associated with the samples.
                  [n_atoms, *dims]
                - `batch_pocket`: The pockets associated with the samples.
                  [n_atoms, *dims]
        """
        z, batch_lig, batch_pocket = self.p_data.sample(batch_size)
        z = z.to(self.device)
        batch_lig = batch_lig.to(self.device)
        batch_pocket = batch_pocket.to(self.device)
        return z, batch_lig, batch_pocket

    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Samples from the conditional distribution p_t(x|z).

        Args:
            z: Conditioning variable. [batch_size, *dims]
            t: Time at which to sample the conditional distribution. [batch_size, 1]

        Returns:
            Samples `x` from p_t(x|z). [batch_size, *dims]
        """
        x = self.alpha(t) * z + self.beta(t) * torch.randn_like(z)
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
        u = (dt_alpha_t - dt_beta_t / beta_t * alpha_t) * z + dt_beta_t / beta_t * x
        return u.to(self.device)

    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Evaluates the conditional score function of p_t(x|z).

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
