"""Test suite for the metrics in the topoflow.utils.metrics module."""

import unittest
import torch
from topoflow.utils.metrics import qed_scorer, logp_scorer, sas_scorer

from tests.conftest import get_toy_molecule


class TestMetrics(unittest.TestCase):
    """Test suite for the metrics in the topoflow.utils.metrics module."""

    def setUp(self):
        """Set up test cases with toy molecules."""
        self.water_molecule = get_toy_molecule("water")
        self.benzene_molecule = get_toy_molecule("benzene")
        self.invalid_molecule = get_toy_molecule("invalid")

    def test_qed_scorer(self):
        """Test the QED scorer on a water molecule."""
        qed_score = qed_scorer(self.water_molecule["rdkit_mol"])
        self.assertIsNotNone(qed_score)
        self.assertTrue(isinstance(qed_score, torch.Tensor))
        self.assertGreaterEqual(qed_score.item(), 0.0)

    def test_logp_scorer(self):
        """Test the LogP scorer on a benzene molecule."""
        logp_score = logp_scorer(self.benzene_molecule["rdkit_mol"])
        self.assertIsNotNone(logp_score)
        self.assertTrue(isinstance(logp_score, torch.Tensor))
        self.assertGreaterEqual(logp_score.item(), -6.0)
        self.assertLessEqual(logp_score.item(), 6.0)

    def test_sas_scorer(self):
        """Test the Synthetic Accessibility Score (SAS) scorer on a water molecule."""
        sas_score = sas_scorer(self.water_molecule["rdkit_mol"])
        self.assertIsNotNone(sas_score)
        self.assertTrue(isinstance(sas_score, torch.Tensor))
        self.assertGreaterEqual(sas_score.item(), 1.0)
        self.assertLessEqual(sas_score.item(), 10.0)

    def test_invalid_molecule(self):
        """Test the metrics on an invalid molecule."""
        qed_score = qed_scorer(self.invalid_molecule["rdkit_mol"])
        self.assertIsNone(qed_score)
        logp_score = logp_scorer(self.invalid_molecule["rdkit_mol"])
        self.assertIsNone(logp_score)
        sas_score = sas_scorer(self.invalid_molecule["rdkit_mol"])
        self.assertIsNone(sas_score)
