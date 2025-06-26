"""Utility functions and classes for sampling and density estimation in PyTorch."""

from abc import ABC, abstractmethod
import numpy as np

import torch


class Sampleable(ABC):
    """Abstract class for sampleable distributions."""

    @property
    @abstractmethod
    def dim(self):
        """Dimensionality of the distribution."""
        pass

    @abstractmethod
    def sample(self, num_samples):
        """Sample from the distribution.

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Samples from the distribution.
            None: Placeholder for labels, not used in this case.
        """
        pass


class Density(ABC):
    """Distribution with density function."""

    @abstractmethod
    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the log density of the distribution at x."""
        pass


class Gaussian(torch.nn.Module, Sampleable, Density):
    """Multivariate Gaussian distribution."""

    def __init__(self, mean: torch.Tensor, cov: torch.Tensor):
        """Initialize the Gaussian distribution.

        Args:
            mean (torch.Tensor): Mean of the Gaussian distribution. Shape: [data_dim].
            cov (torch.Tensor): Covariance matrix of the Gaussian distribution. Shape: [data_dim, data_dim].
        """
        super().__init__()
        self.mean = mean
        self.cov = cov
        self.dim = mean.shape[0]

    @property
    def dim(self):
        """Return the dimensionality of the Gaussian distribution."""
        return self.mean.shape[0]

    @property
    def distribution(self):
        """Return the Gaussian distribution as a MultivariateNormal distribution."""
        return torch.distributions.MultivariateNormal(self.mean, self.cov)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self.distribution.sample((num_samples,)), None

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log density of the Gaussian distribution at x."""
        return self.distribution.log_prob(x).view(-1, 1)

    @classmethod
    def isotropic(cls, dim: int, std: float) -> "Gaussian":
        """Create an isotropic Gaussian distribution."""
        mean = torch.zeros(dim)
        cov = torch.eye(dim) * std**2
        return cls(mean, cov)

    def __len__(self):
        """Returns -1 since the distribution length is not defined."""
        return -1  # Distribution length is not defined


class IsotropicGaussian(torch.nn.Module, Sampleable):
    """Create an isotropic Gaussian distribution."""

    def __init__(self, shape: list, std: float):
        """Initialize the isotropic Gaussian distribution.

        Args:
            shape (list[int] or int): Shape of the distribution. If an int is provided, it will be converted to a single-element list.
            std (float): Standard deviation of the distribution.
        """
        super().__init__()
        if isinstance(shape, int):
            self.shape = [shape]
        else:
            self.shape = shape
        self.std = std
        self.dummy = torch.nn.Buffer(torch.zeros(1))

    @property
    def dim(self):
        """Return the dimensionality of the isotropic Gaussian distribution."""
        return self.shape

    def sample(self, num_samples):
        """Sample from the isotropic Gaussian distribution.

        Args:
            num_samples (int): Number of samples to draw.

        Returns:
            torch.Tensor: Samples from the distribution. Shape: [num_samples, *self.shape].
            None: Placeholder for labels, not used in this case.
        """
        return (torch.randn(num_samples, *self.shape) * self.std).to(self.dummy), None


class GaussianMixture(torch.nn.Module, Sampleable, Density):
    """Two-dimensional Gaussian mixture model, and is a Density and a Sampleable.

    Wrapper around torch.distributions.MixtureSameFamily.
    """

    def __init__(
        self,
        means: torch.Tensor,  # nmodes x data_dim
        covs: torch.Tensor,  # nmodes x data_dim x data_dim
        weights: torch.Tensor,  # nmodes
    ):
        """Initialize the Gaussian mixture model.

        Args:
            means (torch.Tensor): Means of the Gaussian components. Shape: [nmodes, data_dim].
            covs (torch.Tensor): Covariance matrices of the Gaussian components. Shape: [nmodes, data_dim, data_dim].
            weights (torch.Tensor): Weights of the Gaussian components. Shape: [nmodes].
        """
        super().__init__()
        self.nmodes = means.shape[0]
        self.register_buffer("means", means)
        self.register_buffer("covs", covs)
        self.register_buffer("weights", weights)

    @property
    def dim(self) -> int:
        """Return the dimensionality of the Gaussian mixture model."""
        return self.means.shape[1]

    @property
    def distribution(self):
        """Return the Gaussian mixture model as a MixtureSameFamily distribution."""
        return torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(
                probs=self.weights, validate_args=False
            ),
            component_distribution=torch.distributions.MultivariateNormal(
                loc=self.means,
                covariance_matrix=self.covs,
                validate_args=False,
            ),
            validate_args=False,
        )

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        """Return the log density of the Gaussian mixture model at x."""
        return self.distribution.log_prob(x).view(-1, 1)

    def sample(self, num_samples: int) -> torch.Tensor:
        """Sample from the Gaussian mixture model."""
        return self.distribution.sample(torch.Size((num_samples,))), None

    @classmethod
    def random_2D(
        cls,
        nmodes: int,
        std: float,
        scale: float = 10.0,
        x_offset: float = 0.0,
        seed=0.0,
    ) -> "GaussianMixture":
        """Create a random 2D Gaussian mixture model."""
        torch.manual_seed(seed)
        means = (torch.rand(nmodes, 2) - 0.5) * scale + x_offset * torch.Tensor(
            [1.0, 0.0]
        )
        covs = torch.diag_embed(torch.ones(nmodes, 2)) * std**2
        weights = torch.ones(nmodes)
        return cls(means, covs, weights)

    @classmethod
    def symmetric_2D(
        cls, nmodes: int, std: float, scale: float = 10.0, x_offset: float = 0.0
    ) -> "GaussianMixture":
        """Create a symmetric 2D Gaussian mixture model."""
        angles = torch.linspace(0, 2 * np.pi, nmodes + 1)[:nmodes]
        means = (
            torch.stack([torch.cos(angles), torch.sin(angles)], dim=1) * scale
            + torch.Tensor([1.0, 0.0]) * x_offset
        )
        covs = torch.diag_embed(torch.ones(nmodes, 2) * std**2)
        weights = torch.ones(nmodes) / nmodes
        return cls(means, covs, weights)


class DistributionNodes:
    """Distribution of number of nodes.

    This class models the distribution of the number of nodes in a histogram format.
    """

    def __init__(self, histogram):
        """Initialize the distribution of number of nodes."""
        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {
            tuple(x.tolist()): i for i, x in enumerate(self.idx_to_n_nodes)
        }

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1), validate_args=True)

        self.n1_given_n2 = [
            torch.distributions.Categorical(prob[:, j], validate_args=True)
            for j in range(prob.shape[1])
        ]
        self.n2_given_n1 = [
            torch.distributions.Categorical(prob[i, :], validate_args=True)
            for i in range(prob.shape[0])
        ]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        entropy = self.m.entropy()
        print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        """Sample from the distribution of number of nodes."""
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        """Sample from the conditional distribution."""
        assert (n1 is None) ^ (n2 is None), "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        """Compute log probabilities of (n1, n2) pairs."""
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [
                self.n_nodes_to_idx[(n1, n2)]
                for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())
            ]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        """Compute log probabilities of n1 given n2."""
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack(
            [self.n1_given_n2[c].log_prob(i.cpu()) for i, c in zip(n1, n2)]
        )
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        """Compute the log probability of n2 given n1."""
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack(
            [self.n2_given_n1[c].log_prob(i.cpu()) for i, c in zip(n2, n1)]
        )
        return log_probs.to(n2.device)
