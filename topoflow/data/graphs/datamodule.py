"""A PyTorch Lightning DataModule for handling molecular datasets."""

import torch
from torch.utils.data import random_split
from torch_geometric.data import DataLoader
import lightning.pytorch as pl


class MolecularDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for handling molecular datasets."""

    def __init__(
        self,
        dataset,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        batch_size=32,
        num_workers=0,
    ):
        """Initialize the MolecularDataModule.

        Args:
            dataset: Your full dataset object (e.g., QM9WithRDKitMetrics instance).
            train_ratio (float): Proportion of data for the training set.
            val_ratio (float): Proportion of data for the validation set.
            test_ratio (float): Proportion of data for the test set.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of subprocesses to use for data loading.
                                0 means the data will be loaded in the main process.
        """
        super().__init__()
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Ensure ratios sum to 1 (or close enough due to float precision)
        total_ratio = train_ratio + val_ratio + test_ratio
        if not torch.isclose(torch.tensor(total_ratio), torch.tensor(1.0)):
            raise ValueError(
                f"Train, validation, and test ratios must sum to 1.0. Current sum: {total_ratio}"
            )

        # These will be set in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage=None):
        """Splits the dataset into train, validation, and test sets.

        This method is called by the Lightning Trainer.
        """
        total_size = len(self.dataset)
        train_size = int(self.train_ratio * total_size)
        val_size = int(self.val_ratio * total_size)
        test_size = total_size - train_size - val_size  # Ensure all samples are used

        print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        """Returns the DataLoader for the training set."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """Returns the DataLoader for the validation set."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """Returns the DataLoader for the test set."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
