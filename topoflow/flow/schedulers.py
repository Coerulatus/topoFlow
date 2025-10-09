"""Schedulers for flows."""

from abc import ABC, abstractmethod

import torch


class Scheduler(ABC):
    """Abstract base class for schedulers."""

    def __init__(self, t_max=1):
        """Initializes the scheduler and performs validation."""
        self.t_max = t_max
        self.validate()

    @abstractmethod
    def validate(self):
        """Validates the scheduler's properties.

        Raises:
            AssertionError: If the scheduler is not valid (e.g., does not meet
                expected boundary conditions).
        """
        pass

    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the scheduler's value at a given time t.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The scheduler's value at time t. [batch_size, *1s]
        """
        pass

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the derivative of the scheduler's value with respect to t.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The derivative of the scheduler's value with respect to t.
            [batch_size, *1s]
        """
        t = t.unsqueeze(1)
        # Using torch.func.vmap and torch.func.jacrev to compute the Jacobian
        # (derivative) for each element in the batch.
        dt = torch.func.vmap(torch.func.jacrev(self))(t)
        return dt.view(-1, 1).to(t.device)


class Alpha(Scheduler):
    """Abstract base class for Alpha schedulers.

    Alpha schedulers are expected to start at 0 at t=0 and end at 1 at t=1.
    """

    def validate(self):
        """Validates that Alpha(0) is 0 and Alpha(1) is 1."""
        assert torch.allclose(
            self(torch.zeros(1, 1)), torch.zeros(1, 1), atol=1e-6
        ), "Alpha(0) should be 0"
        assert torch.allclose(
            self(torch.ones(1, 1)), torch.ones(1, 1), atol=1e-6
        ), "Alpha(1) should be 1"


class Beta(Scheduler):
    """Abstract base class for Beta schedulers.

    Beta schedulers are expected to start at 1 at t=0 and end at 0 at t=1.
    """

    def validate(self):
        """Validates that Beta(0) is 1 and Beta(1) is 0."""
        assert torch.allclose(
            self(torch.zeros(1, 1)), torch.ones(1, 1), atol=1e-6
        ), "Beta(0) should be 1"
        assert torch.allclose(
            self(torch.ones(1, 1)), torch.zeros(1, 1), atol=1e-6
        ), "Beta(1) should be 0"


class LinearAlpha(Alpha):
    """Linear alpha scheduler, where Alpha(t) = t."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Computes Alpha(t) = t.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The input time t itself. [batch_size, *1s]
        """
        return t

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the derivative of LinearAlpha(t) = t, which is 1.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            A tensor of ones with the same shape as t. [batch_size, *1s]
        """
        return torch.ones_like(t)


class CosineAlpha(Alpha):
    """Cosine-based alpha scheduler, where Alpha(t) = 0.5 * (1- cos(pi*t/2)).

    This schedule ensures the noise term goes from 0 to 1, and in combination
    with CosineAlpha, it maintains a constant variance.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Compute Alpha(t).

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            Alpha(t) in [0,1].
        """
        t_scaled = (t / self.t_max).clamp(max=1.0)
        return 0.5 * (1 - torch.cos(t_scaled * torch.pi))

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Compute derivative dAlpha/dt.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            dAlpha/dt.
        """
        t_scaled = (t / self.t_max).clamp(max=1.0)
        # derivative wrt original t, so chain rule factor 1/t_max
        d_alpha = (0.5 * torch.pi / self.t_max) * torch.sin(torch.pi * t_scaled)
        # zero out derivative after t_max
        return torch.where(t <= self.t_max, d_alpha, torch.zeros_like(t))


class SquareRootBeta(Beta):
    """Square root beta scheduler, where Beta(t) = sqrt(1 - t)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Computes Beta(t) = sqrt(1 - t).

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The square root of (1 - t). [batch_size, *1s]
        """
        # Added .to(t.device) for consistency, although torch.sqrt typically
        # maintains device.
        return torch.sqrt(1 - t).to(t.device)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the derivative of SquareRootBeta(t) = sqrt(1 - t).

        The derivative is -0.5 / sqrt(1 - t). A small epsilon is added to
        the denominator to prevent division by zero when t approaches 1.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The derivative of SquareRootBeta(t). [batch_size, *1s]
        """
        # Ensure the operation is on the correct device
        return (-0.5 / ((1 - t) ** 0.5 + 1e-4)).to(t.device)


class LinearBeta(Beta):
    """Linear beta scheduler, where Beta(t) = 1 - t."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Computes Beta(t) = 1 - t.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            One minus the input time t. [batch_size, *1s]
        """
        # Added .to(t.device) for consistency.
        return (1 - t).to(t.device)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the derivative of LinearBeta(t) = 1 - t, which is -1.

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            A tensor of negative ones with the same shape as t. [batch_size, *1s]
        """
        return -torch.ones_like(t)


class CosineBeta(Beta):
    """Cosine-based beta scheduler, where Beta(t) = cos(pi*t/(2*t_max)).

    This schedule ensures the signal term goes from 1 (at t=0) to 0 (at t=t_max).
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """Computes Beta(t) = cos(pi*t/(2*t_max)).

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The value of the cosine schedule at time t. [batch_size, *1s]
        """
        t_scaled = (t / self.t_max).clamp(max=1.0)

        # Beta(t) = cos(pi/2 * t/t_max)
        return torch.cos(t_scaled * torch.pi / 2.0).to(t.device)

    def dt(self, t: torch.Tensor) -> torch.Tensor:
        """Computes the derivative of CosineBeta(t).

        The derivative is -(pi/(2*t_max)) * sin(pi*t/(2*t_max)).

        Args:
            t: Time. [batch_size, *1s]

        Returns:
            The derivative of the cosine schedule at time t. [batch_size, *1s]
        """
        t_scaled = (t / self.t_max).clamp(max=1.0)

        # Derivative pre-factor: -(pi/2) * (1/t_max)
        pre_factor = -torch.pi / (2.0 * self.t_max)

        # dBeta/dt (Derivative wrt original t)
        d_beta = pre_factor * torch.sin(t_scaled * torch.pi / 2.0)
        # Zero out derivative after t_max
        return torch.where(t <= self.t_max, d_beta, torch.zeros_like(t))
