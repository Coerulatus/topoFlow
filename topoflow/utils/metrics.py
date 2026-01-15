"""Module for calculating various molecular metrics using RDKit."""

import sys
import os
from rdkit.Chem import QED, Descriptors, Lipinski, Crippen, AllChem
import subprocess
import tempfile
import re
import numpy as np
from rdkit import Chem
import torch
from tqdm import tqdm

from topoflow.utils.sa_score import sascorer
from topoflow.utils.molecules import data_to_rdmol, rdmol_to_smiles
from topoflow.utils.visualization import save_pocket_to_pdb, save_ligand_to_sdf
from topoflow.utils.gnina_path import GNINA_EXECUTABLE


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

            # Handle tuple/list returns (e.g. Gnina returning multiple scores)
            if isinstance(score, (tuple, list, np.ndarray)):
                return torch.tensor([score], dtype=torch.float)

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

    @staticmethod
    def calculate_lipinski(rdmol):
        """Calculate the Lipinski score for a molecule."""
        rule_1 = Descriptors.ExactMolWt(rdmol) < 500
        rule_2 = Lipinski.NumHDonors(rdmol) <= 5
        rule_3 = Lipinski.NumHAcceptors(rdmol) <= 10
        rule_4 = (logp := Crippen.MolLogP(rdmol) >= -2) & (logp <= 5)
        rule_5 = Chem.rdMolDescriptors.CalcNumRotatableBonds(rdmol) <= 10
        return np.sum([int(a) for a in [rule_1, rule_2, rule_3, rule_4, rule_5]])


def lipinski_scorer(mol):
    """Calculate the Lipinski score for a molecule."""
    return LipinskiMetric()(mol)


class GninaDocker:
    """Handles GNINA docking."""

    def _calculate_box(self, data):
        """Calculates dynamic bounding box centered on the pocket."""
        # Get pocket coordinates
        if hasattr(data, "x_pocket") and data.x_pocket is not None:
            pos = data.x_pocket
            if isinstance(pos, torch.Tensor):
                pos = pos.cpu().numpy()
        else:
            raise ValueError("Data object missing 'x_pocket' coordinates.")

        # Calculate center
        center = np.mean(pos, axis=0)

        # Calculate size: max distance from center * 2 + padding (e.g., 6A)
        # This ensures the box covers the whole pocket plus some room for the ligand
        max_dist = np.max(np.linalg.norm(pos - center, axis=1))
        size = (max_dist * 2) + 8.0

        return center, size

    def _run_gnina(self, data):
        """Runs GNINA and returns [vina_score, cnn_score]."""
        with tempfile.TemporaryDirectory() as temp_dir:
            receptor_path = os.path.join(temp_dir, "receptor.pdb")
            ligand_path = os.path.join(temp_dir, "ligand.sdf")
            output_path = os.path.join(temp_dir, "out.sdf")

            # 1. Save Files
            try:
                save_pocket_to_pdb(data, receptor_path, verbose=False)
                save_ligand_to_sdf(data, ligand_path, verbose=False)
            except Exception as e:
                print(f"Error saving GNINA inputs: {e}")
                return None

            # 2. Calculate Box
            try:
                center, size = self._calculate_box(data)
            except ValueError as e:
                print(f"Skipping GNINA: {e}")
                return None

            # 3. Build Command
            cmd = [
                GNINA_EXECUTABLE,
                "--receptor",
                receptor_path,
                "--ligand",
                ligand_path,
                "--out",
                output_path,
                "--center_x",
                f"{center[0]:.3f}",
                "--center_y",
                f"{center[1]:.3f}",
                "--center_z",
                f"{center[2]:.3f}",
                "--size_x",
                f"{size:.3f}",
                "--size_y",
                f"{size:.3f}",
                "--size_z",
                f"{size:.3f}",
                "--cpu",
                "4",
                "--seed",
                "0",
                "--num_modes",
                "1",  # Top pose only
                "--exhaustiveness",
                "8",
            ]

            # 4. Run subprocess
            try:
                subprocess.run(cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print("GNINA failed or not found.")
                print("Error Output:\n", e.stderr)
                return None

            # 5. Parse Output
            vina = None
            cnn = None

            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read()

                    # Extract Vina (minimizedAffinity)
                    v_match = re.search(r"> <minimizedAffinity>\s+([-\d\.]+)", content)
                    if v_match:
                        vina = float(v_match.group(1))

                    # Extract CNN (CNNaffinity)
                    c_match = re.search(r"> <CNNaffinity>\s+([\d\.]+)", content)
                    if c_match:
                        cnn = float(c_match.group(1))

            # Return both scores as a list [Vina, CNN]
            if vina is not None and cnn is not None:
                return [vina, cnn]
            return None


# Initialize instance
docker = GninaDocker()


def gnina_scorer(data):
    """Returns [Vina_Score, CNN_Affinity] as a list."""
    return docker._run_gnina(data)


def compute_connectivity(valid, connectivity_thresh=0.5):
    """Calculate connectivity of molecules."""
    if len(valid) < 1:
        return 0.0, [], []

    connected_smiles = []
    connected_mols = []
    for mol in valid:
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= connectivity_thresh:
                smiles = rdmol_to_smiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)
                    connected_mols.append(largest_mol)
        except Exception as e:
            print(f"Error processing molecule: {e}")
            continue

    return len(connected_smiles) / len(valid), connected_mols, connected_smiles


def compute_uniqueness(connected):
    """Calculate uniqueness of molecules in the dataset."""
    if len(connected) < 1:
        return 0.0, []

    return len(set(connected)) / len(connected), list(set(connected))


def compute_novelty(connected, known_smiles):
    """Calculate novelty of molecules in the dataset."""
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


scorer_dict = {
    "qed": qed_scorer,
    "logp": logp_scorer,
    "sas": sas_scorer,
    "lipinski": lipinski_scorer,
    "gnina": gnina_scorer,  # Returns [Vina, CNN]
}


def get_distributions(dataset, metrics, atom_decoder=None):
    """Calculate distributions.

    Handles both 'molecular' metrics (QED/LogP) and 'structural' metrics (Gnina).
    """
    distribution = {}

    # Pre-convert RDKit mols for standard metrics to avoid doing it repeatedly
    # (Only needed if we have standard metrics in the list)
    standard_metrics = ["qed", "logp", "sas", "lipinski"]
    needs_rdmols = any(m in standard_metrics for m in metrics)

    rdmols = []
    if needs_rdmols:
        rdmols = [
            data_to_rdmol(data, atom_decoder=atom_decoder, sanitize=True)
            for data in dataset
        ]

    if not rdmols and needs_rdmols:
        # Warning: if your dataset has invalid mols, this might be empty
        pass

    for metric in metrics:
        scorer = scorer_dict.get(metric)
        if scorer is None:
            print(f"Metric {metric} not recognized.")
            continue

        scores = []

        # CASE A: Structural Metrics (Gnina) -> Pass raw Data object
        if metric == "gnina":
            for data in tqdm(dataset, desc=f"Calculating {metric} scores."):
                s = scorer(data)  # Returns [vina, cnn]
                if s is not None:
                    scores.append(s)

        # CASE B: Standard Metrics -> Pass RDKit Mol
        else:
            for mol in rdmols:
                if mol is not None:  # check for conversion failures
                    s = scorer(mol)
                    if s is not None:
                        scores.append(s)

        # Convert to tensor
        if scores:
            # If scores is a list of lists/tuples (Gnina), this creates (N, 2)
            # If scores is a list of scalars (QED), this creates (N,)
            distribution[metric] = torch.tensor(scores, dtype=torch.float)
        else:
            distribution[metric] = torch.tensor([])

    return distribution
