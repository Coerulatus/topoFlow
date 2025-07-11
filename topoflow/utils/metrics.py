"""Module for calculating various molecular metrics using RDKit."""

import sys
import os
from rdkit.Chem import QED, Descriptors
import torch

sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))
from SA_Score import sascorer
from topoflow.utils.molecules import data_to_rdmol


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
            if isinstance(mol, list):
                if len(mol) == 1:
                    mol = mol[0]
                else:
                    raise ValueError(
                        "Expected a single molecule, got a list of length > 1."
                    )
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


def wasserstein_1d(a, b):
    """Calculate the 1D Wasserstein distance between two 1D distributions."""
    # they might not have same length, pad with zeros if necessary
    a_sorted = torch.sort(a)[0]
    b_sorted = torch.sort(b)[0]

    # Uniform weights
    n, m = len(a_sorted), len(b_sorted)
    a_cdf = torch.arange(1, n + 1, dtype=torch.float32) / n
    b_cdf = torch.arange(1, m + 1, dtype=torch.float32) / m

    # Interpolate to common support (merge the sorted values)
    support = torch.sort(torch.cat([a_sorted, b_sorted]))[0]
    a_interp = torch.searchsorted(a_sorted, support, right=True) / n
    b_interp = torch.searchsorted(b_sorted, support, right=True) / m

    return torch.sum(
        torch.abs(a_interp - b_interp)
        * torch.diff(torch.cat([torch.tensor([0.0]), support]))
    )


def get_distributions(dataset, metrics, atom_decoder=None):
    """Calculate distributions for specified metrics from a dataset."""
    mols = [
        data_to_rdmol(data, atom_decoder=atom_decoder, sanitize=True)
        for data in dataset
    ]
    distribution = {
        "qed": [qed_scorer(mol) for mol in mols],
        "logp": [logp_scorer(mol) for mol in mols],
        "sas": [sas_scorer(mol) for mol in mols],
    }

    return distribution
