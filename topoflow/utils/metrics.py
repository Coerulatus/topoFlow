"""Module for calculating various molecular metrics using RDKit."""

import sys
import os
from rdkit.Chem import QED, Descriptors, Lipinski, Crippen
import numpy as np
from rdkit import Chem
import torch

sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))
from SA_Score import sascorer
from topoflow.utils.molecules import data_to_rdmol, rdmol_to_smiles


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


class LipinskiMetric(Metric):
    """Lipinski's Rule of Five metric."""

    def __init__(self):
        """Initialize the Lipinski metric."""
        super(LipinskiMetric, self).__init__(scorer_function=self.calculate_lipinski)


def lipinski_scorer(mol):
    """Calculate the Lipinski score for a molecule."""
    return LipinskiMetric()(mol)


@staticmethod
def calculate_lipinski(rdmol):
    """Calculate the Lipinski score for a molecule."""
    rule_1 = Descriptors.ExactMolWt(rdmol) < 500
    rule_2 = Lipinski.NumHDonors(rdmol) <= 5
    rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
    rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
    rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
    return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def compute_connectivity(valid, connectivity_thresh=0.5):  # TODO:choose threshold
    """Calculate connectivity of molecules.

    Consider molecule connected if its largest fragment contains at least x% of all atoms, where x is determined by self.connectivity_thresh (defaults to 1).
    """
    if len(valid) < 1:
        return [], 0.0

    connected_smiles = []
    for mol in valid:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= connectivity_thresh:
                smiles = rdmol_to_smiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)
        except Exception as e:
            print(f"Error processing molecule: {e}")
            continue

    return len(connected_smiles) / len(valid), connected_smiles


def compute_uniqueness(connected):
    """Calculate uniqueness of molecules in the dataset.

    Args:
        connected: list of SMILES strings representing connected molecules.

    Returns:
        uniqueness ratio and a list of unique SMILES strings.
    """
    if len(connected) < 1:
        return 0.0, []

    return len(set(connected)) / len(connected), list(set(connected))


def compute_novelty(connected, known_smiles):
    """Calculate novelty of molecules in the dataset.

    Args:
        connected: list of SMILES strings representing connected molecules.
        known_smiles: set of SMILES strings representing known molecules.

    Returns:
        novelty ratio and a list of novel SMILES strings.
    """
    if len(connected) < 1 or known_smiles is None:
        return 0.0, []

    novel = [smiles for smiles in connected if smiles not in known_smiles]
    return len(novel) / len(connected), novel


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
