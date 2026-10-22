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
            print(f"Error calculating metric {self.scorer_function.__name__}: {e}")
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


def mol_size_scorer(mol):
    """Calculate the number of atoms in a molecule."""
    if mol is None:
        return None
    return torch.tensor([mol.GetNumAtoms()], dtype=torch.float)


class GninaDocker:
    """Handles GNINA docking."""

    def _get_pdb_coords(self, pdb_path):
        """Extract heavy-atom 3D coordinates from a PDB file."""
        coords = []
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM  ", "HETATM")):
                    atom_name = line[12:16].strip()
                    element = line[76:78].strip() if len(line) >= 78 else ""
                    if element == "H" or (not element and atom_name.startswith("H")):
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        coords.append([x, y, z])
                    except ValueError:
                        continue
        if not coords:
            raise ValueError(f"No heavy atoms found in {pdb_path}")
        return np.array(coords, dtype=np.float32)

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

    def _run_gnina(self, data, norm_params=None, atom_decoder=None):
        """Runs GNINA and returns [vina_score, cnn_score]."""
        import sys

        env_extra = os.environ.copy()
        conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
        if conda_prefix:
            lib_path = os.path.join(conda_prefix, "lib")
            # Update LD_LIBRARY_PATH so the binary finds libcudnn.so.9
            current_ld = env_extra.get("LD_LIBRARY_PATH", "")
            env_extra["LD_LIBRARY_PATH"] = f"{lib_path}:{current_ld}"

        has_original_receptor = (
            hasattr(data, "pdb_path")
            and data.pdb_path is not None
            and os.path.exists(data.pdb_path)
        )
        has_original_ligand = (
            hasattr(data, "ligand_sdf_path")
            and data.ligand_sdf_path is not None
            and os.path.exists(data.ligand_sdf_path)
        )
        use_original_files = has_original_receptor and has_original_ligand

        pocket_pos = None
        pocket_center = None
        if has_original_receptor:
            receptor_path = data.pdb_path
            try:
                pocket_pos = self._get_pdb_coords(receptor_path)
                pocket_center = np.mean(pocket_pos, axis=0)
            except Exception as e:
                print(f"Warning: Could not parse receptor PDB {receptor_path}: {e}")
                has_original_receptor = False
                use_original_files = False

        if not use_original_files:
            # --- UNNORMALIZATION LOGIC (Fallback) ---
            data_to_save = data.clone()
            if norm_params is not None:
                try:
                    device = data_to_save.x_lig.device
                    std = torch.tensor(
                        norm_params["std"], device=device, dtype=torch.float32
                    )
                    mean = torch.tensor(
                        norm_params["mean"], device=device, dtype=torch.float32
                    )

                    data_to_save.x_lig = data_to_save.x_lig * std + mean
                    data_to_save.x_pocket = data_to_save.x_pocket * std + mean
                except Exception as e:
                    print(f"Warning: Unnormalization failed: {e}")

            if has_original_receptor and pocket_center is not None:
                lig_np = data_to_save.x_lig.detach().cpu().numpy()
                lig_center = np.mean(lig_np, axis=0)
                if (
                    np.linalg.norm(lig_center - pocket_center) > 10.0
                    and np.linalg.norm(lig_center) < 10.0
                    and np.linalg.norm(pocket_center) > 10.0
                ):
                    offset = torch.tensor(
                        pocket_center,
                        device=data_to_save.x_lig.device,
                        dtype=data_to_save.x_lig.dtype,
                    )
                    data_to_save.x_lig = data_to_save.x_lig + offset

        with tempfile.TemporaryDirectory() as temp_dir:
            if not use_original_files:
                if not has_original_receptor:
                    receptor_path = os.path.join(temp_dir, "receptor.pdb")
                ligand_path = os.path.join(temp_dir, "ligand.sdf")
                # 1. Save Files
                try:
                    if not has_original_receptor:
                        save_pocket_to_pdb(
                            data_to_save,
                            receptor_path,
                            atom_decoder=atom_decoder,
                            verbose=False,
                        )
                    save_ligand_to_sdf(
                        data_to_save,
                        ligand_path,
                        atom_decoder=atom_decoder,
                        verbose=False,
                    )
                except Exception as e:
                    print(f"Error saving GNINA inputs: {e}")
                    return [0, 0]
            else:
                ligand_path = data.ligand_sdf_path

            output_path = os.path.join(temp_dir, "out.sdf")

            # 2. Calculate Box (and recenter (0,0,0)-centered ligand to pocket_center if needed)
            try:
                if use_original_files or has_original_receptor:
                    suppl = Chem.SDMolSupplier(
                        ligand_path, removeHs=False, sanitize=False
                    )
                    mol = suppl[0] if len(suppl) > 0 else None
                    if mol is None:
                        raise ValueError(f"Could not read ligand from {ligand_path}")

                    conf = mol.GetConformer()
                    pos = np.array(conf.GetPositions(), dtype=np.float64)
                    lig_center = np.mean(pos, axis=0)

                    # If ligand is centered at origin (0,0,0) while receptor pocket is in world coords,
                    # shift the ligand into the pocket's coordinate frame.
                    if (
                        pocket_center is not None
                        and np.linalg.norm(lig_center - pocket_center) > 10.0
                        and np.linalg.norm(lig_center) < 10.0
                        and np.linalg.norm(pocket_center) > 10.0
                    ):
                        pos = pos + pocket_center
                        for atom_idx in range(mol.GetNumAtoms()):
                            conf.SetAtomPosition(
                                atom_idx,
                                (
                                    float(pos[atom_idx, 0]),
                                    float(pos[atom_idx, 1]),
                                    float(pos[atom_idx, 2]),
                                ),
                            )
                        shifted_ligand_path = os.path.join(
                            temp_dir, "ligand_recentered.sdf"
                        )
                        writer = Chem.SDWriter(shifted_ligand_path)
                        writer.SetKekulize(False)
                        writer.write(mol)
                        writer.close()
                        ligand_path = shifted_ligand_path
                        lig_center = np.mean(pos, axis=0)

                    max_dist = np.max(np.linalg.norm(pos - lig_center, axis=1))
                    # If ligand is still far outside the pocket, anchor the box on pocket_center
                    if (
                        pocket_center is not None
                        and np.linalg.norm(lig_center - pocket_center) > 15.0
                    ):
                        center = pocket_center
                        pocket_max_dist = np.max(
                            np.linalg.norm(pocket_pos - pocket_center, axis=1)
                        )
                        size = max((max_dist * 2) + 12.0, (pocket_max_dist * 2) + 4.0)
                    else:
                        center = lig_center
                        size = (max_dist * 2) + 12.0
                else:
                    center, size = self._calculate_box(data_to_save)
            except Exception as e:
                print(f"Skipping GNINA: {e}")
                return [0, 0]

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
                "1",
                "--exhaustiveness",
                "8",
            ]

            # 4. Run subprocess
            try:
                subprocess.run(
                    cmd, capture_output=True, text=True, check=True, env=env_extra
                )
            except subprocess.CalledProcessError as e:
                print(f"GNINA failed for {ligand_path}:")
                print("Error Output:\n", e.stderr)
                return [0.0, 0.0]

            # 5. Parse Output
            vina = None
            cnn = None

            if os.path.exists(output_path):
                with open(output_path, "r") as f:
                    content = f.read()

                    # Extract Vina (minimizedAffinity)
                    v_match = re.search(r"> <minimizedAffinity>\s+([-\d\.]+)", content)
                    if v_match:
                        try:
                            vina = float(v_match.group(1))
                        except ValueError:
                            vina = None

                    # Extract CNN (CNNaffinity)
                    c_match = re.search(r"> <CNNaffinity>\s+([-\d\.]+)", content)
                    if c_match:
                        try:
                            cnn = float(c_match.group(1))
                        except ValueError:
                            cnn = None

            if vina is None or cnn is None:
                print(
                    f"Warning: GNINA completed but failed to parse scores for {ligand_path}. Content length: {len(content) if os.path.exists(output_path) else 0}"
                )

            # Return both scores as a list [Vina, CNN]
            # Ensure we return valid floats or 0.0, never None in the list
            return [vina if vina is not None else 0.0, cnn if cnn is not None else 0.0]


# Initialize instance
docker = GninaDocker()


def gnina_scorer(data, norm_params=None, atom_decoder=None):
    """Returns [Vina_Score, CNN_Affinity] as a list."""
    return docker._run_gnina(data, norm_params=norm_params, atom_decoder=atom_decoder)


def compute_connectivity(valid, connectivity_thresh=0.5):
    """Calculate connectivity of molecules."""
    if len(valid) < 1:
        return 0.0, [], [], []

    connected_smiles = []
    connected_mols = []
    keep_idxs = []
    for i, mol in enumerate(valid):
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            if largest_mol.GetNumAtoms() / mol.GetNumAtoms() >= connectivity_thresh:
                smiles = rdmol_to_smiles(largest_mol)
                if smiles is not None:
                    connected_smiles.append(smiles)
                    connected_mols.append(largest_mol)
                    keep_idxs.append(i)
        except Exception as e:
            print(f"Error processing molecule: {e}")
            continue

    return (
        len(connected_smiles) / len(valid),
        connected_mols,
        connected_smiles,
        keep_idxs,
    )


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
    a_sorted = torch.sort(a)[0]
    b_sorted = torch.sort(b)[0]

    n, m = len(a_sorted), len(b_sorted)

    # Combine to find all points where the CDFs change
    support = torch.sort(torch.cat([a_sorted, b_sorted]))[0]

    # We evaluate the CDF step functions at the left edge of each interval.
    # Therefore, we drop the last point of the support.
    eval_points = support[:-1]

    # Calculate empirical CDFs at the evaluation points
    # (Note: .float() is added to ensure safe float division)
    a_interp = torch.searchsorted(a_sorted, eval_points, right=True).float() / n
    b_interp = torch.searchsorted(b_sorted, eval_points, right=True).float() / m

    # Calculate the widths of the intervals
    deltas = torch.diff(support)

    # Integrate |CDF_a - CDF_b| over the support
    return torch.sum(torch.abs(a_interp - b_interp) * deltas)


scorer_dict = {
    "qed": qed_scorer,
    "logp": logp_scorer,
    "sas": sas_scorer,
    "lipinski": lipinski_scorer,
    "gnina": gnina_scorer,  # Returns [Vina, CNN]
    "mol_size": mol_size_scorer,
}


def get_distributions(dataset, metrics, atom_decoder=None, norm_params=None):
    """Calculate distributions.

    Handles both 'molecular' metrics (QED/LogP) and 'structural' metrics (Gnina).
    """
    distribution = {}

    # Pre-convert RDKit mols for standard metrics to avoid doing it repeatedly
    # (Only needed if we have standard metrics in the list)
    standard_metrics = ["qed", "logp", "sas", "lipinski", "mol_size"]
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
                # Pass norm_params and atom_decoder to ensure unnormalized docking
                s = scorer(
                    data, norm_params=norm_params, atom_decoder=atom_decoder
                )  # Returns [vina, cnn]
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
            if metric == "gnina":
                distribution["vina"] = torch.tensor(
                    [s[0] for s in scores], dtype=torch.float
                )
                distribution["cnn_affinity"] = torch.tensor(
                    [s[1] for s in scores], dtype=torch.float
                )
            else:
                distribution[metric] = torch.tensor(scores, dtype=torch.float)
        else:
            distribution[metric] = torch.tensor([])

    return distribution
