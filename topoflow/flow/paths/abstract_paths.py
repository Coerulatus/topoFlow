"""Abstract base class for defining conditional probability paths."""

from abc import ABC, abstractmethod
import numpy as np
import torch


class ConditionalProbabilityPath(torch.nn.Module, ABC):
    """Abstract class for conditional probability paths.

    A conditional probability path defines a continuous transformation between
    a simple distribution (p_simple) and a more complex data distribution (p_data)
    conditioned on a variable `z`.

    Attributes:
        p_simple: The simple probability distribution (e.g., a standard Gaussian).
        p_data: The data probability distribution.
        device: The device (e.g., 'cpu' or 'cuda') on which tensors will be
            processed.
        t_target_shape: The desired shape for time tensors, enabling broadcasting
            across sample dimensions.
    """

    def __init__(self, p_simple, p_data, device="cpu"):
        """Initializes the ConditionalProbabilityPath.

        Args:
            p_simple: The simple probability distribution.
            p_data: The data probability distribution.
            device: The device to perform computations on (e.g., 'cpu' or 'cuda').
                Defaults to 'cpu'.
        """
        super().__init__()
        self.p_simple = p_simple
        self.p_data = p_data
        self.device = device
        # Determine the target shape for time tensors based on the simple distribution's samples.
        x = p_simple.sample(1)
        n_dim = x.dim()
        self.t_target_shape = [-1] + [1] * (n_dim - 1)

    def __len__(self):
        """Returns the length of the path, typically inferred from the data distribution.

        Returns:
            int: Length of the path, usually corresponding to the size of the
                data distribution if it supports such a concept (e.g., a dataset).
        """
        return len(self.p_data)

    def sample_marginal_path(self, t: torch.Tensor) -> torch.Tensor:
        """Samples from the marginal distribution p_t(x) = p_t(x|z) p(z).

        This method first samples a conditioning variable `z` from the data
        distribution and then uses it to sample `x` from the conditional path.

        Args:
            t: Time at which to sample the marginal distribution. [batch_size, 1]

        Returns:
            Samples from p_t(x). [batch_size, *dims]
        """
        batch_size = t.shape[0]
        z, _ = self.sample_conditioning_variable(batch_size)
        x = self.sample_conditional_path(z, t)
        return x.to(self.device)

    def sample_time(self, batch_size: int, batch=None) -> torch.Tensor:
        """Samples time `t` uniformly from [0, 1).

        Args:
            batch_size: Number of time samples to generate.
            batch: Optional tensor indicating the samples associated with the batches.

        Returns:
            Samples from a uniform distribution U[0,1). [batch_size, *1s]
        """
        if torch.max(batch) + 1 != batch_size:
            batch_size = torch.max(batch) + 1
        t = torch.rand(batch_size, device=self.device).view(self.t_target_shape)
        if batch is None:
            return t, None
        t_batched = t[batch]
        return t, t_batched

    def create_mask(self, lengths_lig, lengths_pocket):
        """Creates a tensor that gives to each sample an index in the batch."""
        total_length = lengths_lig + lengths_pocket
        all_segment_ids = torch.repeat_interleave(
            torch.arange(len(lengths_lig), device=lengths_lig.device), total_length
        ).long()
        lig_segment_ids = all_segment_ids.clone()
        pocket_segment_ids = all_segment_ids.clone()

        current_idx = 0
        for i in range(len(lengths_lig)):
            lig_segment_ids[
                current_idx
                + lengths_lig[i] : current_idx
                + lengths_lig[i]
                + lengths_pocket[i]
            ] = -1
            pocket_segment_ids[current_idx : current_idx + lengths_lig[i]] = -1
            current_idx += lengths_lig[i] + lengths_pocket[i]

        return all_segment_ids, lig_segment_ids, pocket_segment_ids

    def sample_simple_distribution(
        self, batch_size: int, batch_pocket=None, size_ligand=None
    ):
        """Samples from the simple distribution `p_simple`.

        Args:
            batch_size: Number of samples.
            batch_pocket: Optional tensor indicating the samples associated with the pockets.
            size_ligand: Optional sizes for the ligands.

        Returns:
            A tuple containing:
                - torch.Tensor: Samples from p_simple(z). [batch_size, *dims]
                - torch.Tensor or None: Labels for the samples. [batch_size, label_dim].
                    Returns None if no labels are associated with the simple distribution.
        """
        if size_ligand is not None:
            num_nodes_lig = size_ligand
            if batch_pocket is not None:
                num_nodes_pocket = torch.tensor(
                    [torch.sum(batch_pocket == i).item() for i in range(batch_size)],
                    device=self.device,
                )
            else:
                _, num_nodes_pocket = self.size_distribution.sample(batch_size)
        elif hasattr(self, "size_distribution") and self.size_distribution is not None:
            if batch_pocket is not None:
                num_nodes_lig, _ = self.size_distribution.sample(batch_size)
                num_nodes_pocket = torch.tensor(
                    [torch.sum(batch_pocket == i).item() for i in range(batch_size)],
                )
            else:
                num_nodes_lig, num_nodes_pocket = self.size_distribution.sample(
                    batch_size
                )
        else:
            raise ValueError(
                "The simple distribution must have a size distribution to sample variable-sized data."
            )

        batch_all, batch_lig, batch_pocket = self.create_mask(
            num_nodes_lig, num_nodes_pocket
        )

        z = self.p_simple.sample(torch.sum(num_nodes_lig) + torch.sum(num_nodes_pocket))
        return (
            z.to(self.device),
            batch_all.to(self.device),
            batch_lig.to(self.device),
            batch_pocket.to(self.device),
        )

    @abstractmethod
    def sample_conditioning_variable(self, batch_size: int):
        """Abstract method to sample the conditioning variable `z`.

        This method defines how the conditioning variable `z` is sampled,
        typically from the `p_data` distribution.

        Args:
            batch_size: Number of samples.

        Returns:
            A tuple containing:
                - torch.Tensor: Samples from p(z). [batch_size, *dims]
                - torch.Tensor or None: Labels for the samples. [batch_size, label_dim].
                    Returns None if no labels are associated.
        """
        pass

    @abstractmethod
    def sample_conditional_path(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Abstract method to sample from the conditional distribution p_t(x|z).

        Args:
            z: Conditioning variable. [batch_size, *dims]
            t: Time at which to sample the conditional distribution. [batch_size, 1]

        Returns:
            Samples from p_t(x|z). [batch_size, *dims]
        """
        pass

    @abstractmethod
    def conditional_vector_field(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Abstract method to compute the conditional vector field u_t(x|z).

        The conditional vector field describes the deterministic evolution of
        samples along the path.

        Args:
            x: Input samples. [batch_size, *dims]
            z: Conditioning variable. [batch_size, *dims]
            t: Time at which to compute the vector field. [batch_size, 1]

        Returns:
            Conditional vector field. [batch_size, *dims]
        """
        pass

    @abstractmethod
    def conditional_score(
        self, x: torch.Tensor, z: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Abstract method to compute the conditional score function of p_t(x|z).

        The conditional score function is the gradient of the log-probability
        density with respect to `x`.

        Args:
            x: Input samples. [batch_size, *dims]
            z: Conditioning variable. [batch_size, *dims]
            t: Time at which to compute the score function. [batch_size, 1]

        Returns:
            Conditional score function. [batch_size, *dims]
        """
        pass
