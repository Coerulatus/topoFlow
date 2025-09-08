"""A sampleable DataLoader for molecular data in TopoFlow."""

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

    def _process_batch(self, batch):
        """Process a single batch, merging and re-indexing data without using a for loop."""
        f_lig = torch.cat([batch.x_lig, batch.h_lig], dim=1)
        f_pocket = torch.cat([batch.x_pocket, batch.h_pocket], dim=1)

        # Determine the device
        device = f_lig.device

        # Concatenate all features and batch indices
        f_all = torch.cat([f_lig, f_pocket], dim=0)
        batch_all = torch.cat([batch.batch_lig, batch.batch_pocket], dim=0)

        # Create a tensor to distinguish between ligand (0) and pocket (1) atoms
        type_all = torch.cat(
            [
                torch.zeros(f_lig.shape[0], dtype=torch.long, device=device),
                torch.ones(f_pocket.shape[0], dtype=torch.long, device=device),
            ],
            dim=0,
        )

        # Create a unique sorting key that orders by batch index first, then by type
        # Multiplying by a large number ensures batch_all is the primary sorting key
        # We can use the max batch index + 1 as the multiplier
        num_unique_batches = batch_all.max().item() + 1
        sorting_key = batch_all * 2 + type_all

        # Get the sorted indices
        sorted_indices = torch.argsort(sorting_key)

        # Apply the sorted indices to reorder all tensors
        f_all_merged = f_all[sorted_indices]
        batch_all_merged = batch_all[sorted_indices]
        type_all_merged = type_all[sorted_indices]

        # Create the new batch tensors with -1 for the "other" atom type
        batch_lig_merged = torch.where(type_all_merged == 0, batch_all_merged, -1)
        batch_pocket_merged = torch.where(type_all_merged == 1, batch_all_merged, -1)

        return (
            f_all_merged,
            batch_all_merged,
            batch_lig_merged,
            batch_pocket_merged,
            num_unique_batches,
        )

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

            f_all, batch_all, batch_lig, batch_pocket, batch_size = self._process_batch(
                batch
            )
            return f_all, batch_all, batch_lig, batch_pocket, batch_size

        else:
            # Force size sampling: collect batches until the target number of samples is met.
            if num_samples is None:
                raise ValueError(
                    "num_samples must be specified when force_size is True."
                )

            all_f_all = []
            all_batch_all = []
            all_batch_lig = []
            all_batch_pocket = []
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
                    unique_batches = torch.arange(needed, device=batch.batch_lig.device)
                    lig_indices = torch.where(batch.batch_lig < needed)[0]
                    pocket_indices = torch.where(batch.batch_pocket < needed)[0]

                    # Create a new, partial batch object with only the needed data
                    sub_batch = type(batch)()
                    sub_batch.x_lig = batch.x_lig[lig_indices]
                    sub_batch.h_lig = batch.h_lig[lig_indices]
                    sub_batch.batch_lig = batch.batch_lig[lig_indices]
                    sub_batch.x_pocket = batch.x_pocket[pocket_indices]
                    sub_batch.h_pocket = batch.h_pocket[pocket_indices]
                    sub_batch.batch_pocket = batch.batch_pocket[pocket_indices]

                    # Re-index the batches to start from 0
                    for i, b in enumerate(unique_batches):
                        sub_batch.batch_lig[sub_batch.batch_lig == b] = i
                        sub_batch.batch_pocket[sub_batch.batch_pocket == b] = i

                    f_all, batch_all, batch_lig, batch_pocket, _ = self._process_batch(
                        sub_batch
                    )

                    all_f_all.append(f_all)
                    all_batch_all.append(batch_all + current_samples)
                    all_batch_lig.append(batch_lig + current_samples)
                    all_batch_pocket.append(batch_pocket + current_samples)
                    current_samples = num_samples
                else:
                    # Take the whole batch
                    f_all, batch_all, batch_lig, batch_pocket, _ = self._process_batch(
                        batch
                    )
                    all_f_all.append(f_all)
                    all_batch_all.append(batch_all + current_samples)
                    all_batch_lig.append(batch_lig + current_samples)
                    all_batch_pocket.append(batch_pocket + current_samples)
                    current_samples += batch_size_actual

            # Concatenate all collected data
            f_all_final = torch.cat(all_f_all, dim=0)
            batch_all_final = torch.cat(all_batch_all, dim=0)
            batch_lig_final = torch.cat(all_batch_lig, dim=0)
            batch_pocket_final = torch.cat(all_batch_pocket, dim=0)

            return (
                f_all_final,
                batch_all_final,
                batch_lig_final,
                batch_pocket_final,
                num_samples,
            )
