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

    def sample(self, num_samples=None, force_size=False):
        """Returns the next batch.

        If the end of the dataset is reached, it restarts and begins a new epoch automatically.

        Args:
            num_samples (int, optional): The number of samples to return if force_size is True.
            force_size (bool): If True, ensures exactly num_samples are returned.

        Returns:
            tuple: A tuple containing the combined features, ligand batch indices, pocket batch indices, and the number of samples.
        """
        if not force_size:
            # Standard sampling: return one batch of the size specified at initialization.
            try:
                batch = next(self._dataloader_iter)
            except StopIteration:
                self._dataloader_iter = iter(self)
                batch = next(self._dataloader_iter)

            f_lig = torch.cat([batch.x_lig, batch.h_lig], dim=1)
            f_pocket = torch.cat([batch.x_pocket, batch.h_pocket], dim=1)
            f_all = torch.cat([f_lig, f_pocket], dim=0)

            return f_all, batch.batch_lig, batch.batch_pocket, self.batch_size

        else:
            # Force size sampling: collect batches until the target number of samples is met.
            if num_samples is None:
                raise ValueError(
                    "num_samples must be specified when force_size is True."
                )

            all_lig_features = []
            all_pocket_features = []
            all_lig_batch_indices = []
            all_pocket_batch_indices = []
            current_samples = 0

            while current_samples < num_samples:
                try:
                    batch = next(self._dataloader_iter)
                except StopIteration:
                    self._dataloader_iter = iter(self)
                    batch = next(self._dataloader_iter)

                batch_size_actual = (
                    batch.batch_lig.max().item() + 1
                    if batch.batch_lig.numel() > 0
                    else 0
                )

                # If adding the whole batch exceeds the target, take only what's needed.
                if current_samples + batch_size_actual > num_samples:
                    needed = num_samples - current_samples

                    # Identify the indices for the needed samples
                    lig_indices = torch.where(batch.batch_lig < needed)[0]
                    pocket_indices = torch.where(batch.batch_pocket < needed)[0]

                    # Subset the data
                    f_lig_subset = torch.cat([batch.x_lig, batch.h_lig], dim=1)[
                        lig_indices
                    ]
                    f_pocket_subset = torch.cat(
                        [batch.x_pocket, batch.h_pocket], dim=1
                    )[pocket_indices]

                    all_lig_features.append(f_lig_subset)
                    all_pocket_features.append(f_pocket_subset)

                    # Update batch indices to be contiguous, starting from 0
                    lig_batch_subset = batch.batch_lig[lig_indices]
                    pocket_batch_subset = batch.batch_pocket[pocket_indices]

                    all_lig_batch_indices.append(lig_batch_subset + current_samples)
                    all_pocket_batch_indices.append(
                        pocket_batch_subset + current_samples
                    )

                    current_samples = num_samples
                else:
                    # Take the whole batch
                    f_lig = torch.cat([batch.x_lig, batch.h_lig], dim=1)
                    f_pocket = torch.cat([batch.x_pocket, batch.h_pocket], dim=1)

                    all_lig_features.append(f_lig)
                    all_pocket_features.append(f_pocket)

                    all_lig_batch_indices.append(batch.batch_lig + current_samples)
                    all_pocket_batch_indices.append(
                        batch.batch_pocket + current_samples
                    )

                    current_samples += batch_size_actual

            # Concatenate all collected data
            f_lig_final = torch.cat(all_lig_features, dim=0)
            f_pocket_final = torch.cat(all_pocket_features, dim=0)
            f_all_final = torch.cat([f_lig_final, f_pocket_final], dim=0)

            lig_batch_final = torch.cat(all_lig_batch_indices, dim=0)
            pocket_batch_final = torch.cat(all_pocket_batch_indices, dim=0)

            return f_all_final, lig_batch_final, pocket_batch_final, num_samples
