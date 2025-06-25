"""A PyTorch Lightning DataModule for handling molecular datasets."""

import torch
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader
import lightning.pytorch as pl


class MolecularDataModule(pl.LightningDataModule):
    """A PyTorch Lightning DataModule for handling molecular datasets."""

    def __init__(
        self,
        dataset,
        val_dataset=None,
        test_dataset=None,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=32,
        num_workers=0,
    ):
        """Initialize the MolecularDataModule.

        Args:
            dataset: Your full dataset object (e.g., QM9WithRDKitMetrics instance).
            val_dataset: Optional validation dataset. If provided, it will be used instead of splitting.
            test_dataset: Optional test dataset. If provided, it will be used instead of splitting.
            train_ratio (float): Proportion of data for the training set.
            val_ratio (float): Proportion of data for the validation set.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of subprocesses to use for data loading.
                                0 means the data will be loaded in the main process.
        """
        super().__init__()
        self.dataset = dataset
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        # These will be set in setup()
        self.train_dataset = None

        if val_dataset is not None:
            self.val_dataset = val_dataset
        else:
            self.val_dataset = None
        if test_dataset is not None:
            self.test_dataset = test_dataset
        else:
            self.test_dataset = None

    def setup(self, stage=None):
        """Splits the dataset into train, validation, and test sets.

        This method is called by the Lightning Trainer.
        """
        if self.val_dataset is not None and self.test_dataset is not None:
            self.train_dataset = self.dataset
            return
        total_size = len(self.dataset)
        train_size = int(self.train_ratio * total_size)
        if self.val_dataset is not None:
            val_size = 0
        else:
            val_size = int(self.val_ratio * total_size)

        if self.test_dataset is not None:
            test_size = 0
        else:
            test_size = total_size - train_size - val_size

        print(f"Dataset sizes: Train={train_size}, Val={val_size}, Test={test_size}")

        train_dataset, val_dataset, test_dataset = random_split(
            self.dataset, [train_size, val_size, test_size]
        )
        self.train_dataset = train_dataset
        if self.val_dataset is None:
            self.val_dataset = val_dataset
        if self.test_dataset is None:
            self.test_dataset = test_dataset

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
