"""Test suite for the QM9 dataset with RDKit-calculated metrics."""

import torch
import unittest

from topoflow.data import MolecularDataModule
from topoflow.data import QM9WithRDKitMetrics


class TestQM9DataModule(unittest.TestCase):
    """Test suite for the QM9 dataset with RDKit-calculated metrics."""

    def setUp(self):
        """Set up the data module for testing."""
        root = "data/QM9_RDKit"
        self.dataset = QM9WithRDKitMetrics(root=root, normalize=True)
        self.data_module = MolecularDataModule(
            self.dataset, batch_size=32, num_workers=0
        )
        self.data_module.setup()

    def test_dataset(self):
        """Test that the data module is set up correctly."""
        self.assertIsNotNone(self.data_module.train_dataset)
        self.assertIsNotNone(self.data_module.val_dataset)
        self.assertIsNotNone(self.data_module.test_dataset)

        self.assertGreater(len(self.data_module.train_dataset), 0)
        self.assertGreater(len(self.data_module.val_dataset), 0)
        self.assertGreater(len(self.data_module.test_dataset), 0)

        self.assertEqual(self.dataset.num_features, 8)
        self.assertEqual(self.dataset.x.shape[1], 8)
        assert torch.all(self.dataset.y >= 0), "Target values should be non-negative"
        assert torch.all(
            self.dataset.y <= 1
        ), "Target values should be in [0, 1] for normalized dataset"
