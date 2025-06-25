"""Test suite for the cross-docked dataset with RDKit-calculated metrics."""

import unittest
import torch

from topoflow.data.graphs.datasets import CrossDocked
from topoflow.data.graphs.datamodule import MolecularDataModule


class TestCrossDockedDataset(unittest.TestCase):
    """Test suite for the cross-docked dataset with RDKit-calculated metrics."""

    def setUp(self):
        """Set up the dataset for testing."""
        self.train_dataset = CrossDocked(root="data/CrossDocked", split="train")
        self.val_dataset = CrossDocked(root="data/CrossDocked", split="val")
        self.test_dataset = CrossDocked(root="data/CrossDocked", split="test")
        self.datamodule = MolecularDataModule(
            self.train_dataset,
            val_dataset=self.val_dataset,
            test_dataset=self.test_dataset,
            batch_size=32,
            num_workers=0,
        )
        self.datamodule.setup()

    def test_dataset(self):
        """Test that the dataset is set up correctly."""
        assert len(self.train_dataset) > 0, "Train dataset should not be empty"
        assert len(self.val_dataset) > 0, "Validation dataset should not be empty"
        assert len(self.test_dataset) > 0, "Test dataset should not be empty"
        data = self.train_dataset[0]
        assert "x_lig" in data, "Data should contain 'x' (features)"
        assert "x_pocket" in data, "Data should contain 'x_pocket' (features)"
        assert "pos_lig" in data, "Data should contain 'pos_lig' (positions)"
        assert "pos_pocket" in data, "Data should contain 'pos_pocket' (positions)"

        train_loader = self.datamodule.train_dataloader()
        batch = next(iter(train_loader))
        assert "x_lig" in batch, "Batch should contain 'x_lig' (ligand features)"
        assert "x_pocket" in batch, "Batch should contain 'x_pocket' (pocket features)"
        assert "pos_lig" in batch, "Batch should contain 'pos_lig' (ligand positions)"
        assert (
            "pos_pocket" in batch
        ), "Batch should contain 'pos_pocket' (pocket positions)"
        assert (
            "batch_lig" in batch
        ), "Batch should contain 'batch_lig' (ligand batch indices)"
        assert (
            "batch_pocket" in batch
        ), "Batch should contain 'batch_pocket' (pocket batch indices)"

        assert (
            torch.max(batch.batch_lig) == 31
        ), "Batch indices should be within the expected range"
        assert (
            torch.max(batch.batch_pocket) == 31
        ), "Batch indices should be within the expected range"
