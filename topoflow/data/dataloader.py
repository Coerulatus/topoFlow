"""Samplelable DataLoader."""

import torch
from torch_geometric.data import DataLoader


class DataLoaderSampleable(DataLoader):
    """A sampleable DataLoader.

    Repeatedly yields batches, restarting at the end of the dataset.
    """

    def __init__(self, dataset, batch_size=32, shuffle=True, num_workers=0, **kwargs):
        """Initialize the DataLoader.

        Args:
            dataset (Dataset): The CrossDocked dataset to sample from.
            batch_size (int): Number of samples per batch.
            shuffle (bool): Whether to shuffle the dataset at every epoch.
            num_workers (int): Number of subprocesses to use for data loading.
            **kwargs: Additional keyword arguments for DataLoader.
        """
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **kwargs,
        )
        self._dataloader_iter = iter(self)
        x_lig_dim = dataset.x_lig.shape[1]
        h_lig_dim = dataset.h_lig.shape[1]
        x_pocket_dim = dataset.x_pocket.shape[1]
        h_pocket_dim = dataset.h_pocket.shape[1]
        # Same total dimensions are needed when generating initial noise. Could be changed if needed.
        assert (
            x_lig_dim + h_lig_dim == x_pocket_dim + h_pocket_dim
        ), "Ligand and pocket features must have the same total dimension."
        self.total_dim = x_lig_dim + h_lig_dim

    @property
    def dim(self):
        """Returns the dimension of the data."""
        return self.total_dim

    def sample(self, num_samples=None):
        """Returns the next batch.

        If the end of the dataset is reached, it restarts and begins a new epoch automatically.

        Args:
            num_samples (int, optional): Not used directly. Included for interface consistency.

        Returns:
            torch_geometric.data.Data: A batch of data from the dataset.
        """
        try:
            batch = next(self._dataloader_iter)
        except StopIteration:
            # Restart the iterator (new epoch)
            self._dataloader_iter = iter(self)
            batch = next(self._dataloader_iter)
        f_lig = torch.cat([batch.x_lig, batch.h_lig], dim=1)
        f_pocket = torch.cat([batch.x_pocket, batch.h_pocket], dim=1)
        f_all = torch.cat([f_lig, f_pocket], dim=0)
        return f_all, batch.batch_lig, batch.batch_pocket, self.batch_size
