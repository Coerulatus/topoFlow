"""Module for calculating various molecular metrics using RDKit."""

import sys
import os
from rdkit.Chem import QED, Descriptors
import torch

sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))
from SA_Score import sascorer


class Metric(torch.nn.Module):
    """Base class for molecular metrics."""

    def __init__(self, scorer_function=None):
        """Initialize the Metric module."""
        super(Metric, self).__init__()
        self.scorer_function = scorer_function

    def __call__(self, mol):
        """Calculate the metric for the given molecule."""
        if self.scorer_function is None:
            raise NotImplementedError("Scorer function must be defined.")
        try:
            score = self.scorer_function(mol)
            return torch.tensor([score], dtype=torch.float)
        except Exception as e:
            print(f"Error calculating metric: {e}")
            return None


class QEDMetric(Metric):
    """Quantitative Estimation of Drug-likeness (QED) metric."""

    def __init__(self):
        """Initialize the QED metric."""
        super(QEDMetric, self).__init__(scorer_function=QED.default)


def qed_scorer(mol):
    """Calculate the QED score for a molecule."""
    return QEDMetric()(mol)


class LogPMetric(Metric):
    """LogP metric, which calculates the octanol-water partition coefficient."""

    def __init__(self):
        """Initialize the LogP metric."""
        super(LogPMetric, self).__init__(scorer_function=Descriptors.MolLogP)


def logp_scorer(mol):
    """Calculate the LogP score for a molecule."""
    return LogPMetric()(mol)


class SASMetric(Metric):
    """Synthetic Accessibility Score (SAS) metric."""

    def __init__(self):
        """Initialize the Synthetic Accessibility Score (SAS) metric."""
        super(SASMetric, self).__init__(scorer_function=sascorer.calculateScore)


def sas_scorer(mol):
    """Calculate the Synthetic Accessibility Score (SAS) for a molecule."""
    return SASMetric()(mol)
