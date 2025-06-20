"""Module for calculating various molecular metrics using RDKit."""

import sys
import os
from rdkit.Chem import QED, Descriptors
import torch

try:
    rdkit_contrib_path = os.path.join(
        os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"
    )
    if rdkit_contrib_path not in sys.path:
        sys.path.append(rdkit_contrib_path)
    from SA_Score import sascorer
except KeyError:
    print(
        "WARNING: CONDA_PREFIX environment variable not set. Attempting common RDKit Contrib path fallback for SA_Score."
    )
    # Common fallback path for RDKit Contrib if CONDA_PREFIX isn't set.
    # You may need to inspect your RDKit installation to find the correct path.
    # E.g., on Linux/macOS: /path/to/your/conda/env/lib/pythonX.Y/site-packages/rdkit/Chem/rdSAScore.py
    # or /opt/conda/envs/your_env_name/share/RDKit/Contrib
    fallback_path_guess = os.path.join(sys.prefix, "share", "RDKit", "Contrib")
    if fallback_path_guess not in sys.path:
        sys.path.append(fallback_path_guess)
    try:
        from SA_Score import sascorer
    except ImportError:
        print(
            "ERROR: Could not import SA_Score. Make sure RDKit is installed and SA_Score.py is accessible."
        )
        sascorer = None  # Set to None to handle gracefully if import fails
except ImportError:
    print(
        "ERROR: Could not import SA_Score. Make sure RDKit is installed and SA_Score.py is accessible."
    )
    sascorer = None  # Set to None to handle gracefully if import fails


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
