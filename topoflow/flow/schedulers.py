"""Schedulers for flows."""

from abc import ABC, abstractmethod

import torch


class Scheduler(ABC):
    """Abstract base class for schedulers."""

    def __init__(self):
        """Initializes the scheduler and performs validation."""
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
            self(torch.zeros(1, 1)), torch.zeros(1, 1)
        ), "Alpha(0) should be 0"
        assert torch.allclose(
            self(torch.ones(1, 1)), torch.ones(1, 1)
        ), "Alpha(1) should be 1"


class Beta(Scheduler):
    """Abstract base class for Beta schedulers.

    Beta schedulers are expected to start at 1 at t=0 and end at 0 at t=1.
    """

    def validate(self):
        """Validates that Beta(0) is 1 and Beta(1) is 0."""
        assert torch.allclose(
            self(torch.zeros(1, 1)), torch.ones(1, 1)
        ), "Beta(0) should be 1"
        assert torch.allclose(
            self(torch.ones(1, 1)), torch.zeros(1, 1)
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
