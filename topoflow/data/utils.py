"""Utility functions for processing PDB and SDF files into graph representations."""

from abc import ABC, abstractmethod
from pathlib import Path
from time import time
import shutil
import random
import os
import requests

import numpy as np
from tqdm import tqdm
import gdown

from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.Data.PDBData import protein_letters_3to1
from rdkit import Chem
from scipy.ndimage import gaussian_filter

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

import topoflow.utils.constants as constants
from topoflow.utils.constants import covalent_radii, dataset_params
from topoflow.utils.molecules import make_mol_openbabel, rdmol_to_smiles


class MockDataset(torch.utils.data.Dataset):
    """A mock dataset to be used with lightning."""

    def __init__(self, length):
        """Initialize a mock dataset with a specified length."""
        self.data = torch.ones(length)

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Get item by index."""
        return self.data[idx]


class MockDatamodule(pl.LightningDataModule):
    """A mock datamodule to be used with lightning trainer."""

    def __init__(self, train_length, val_length=0, test_length=0, batch_size: int = 64):
        """Initialize the mock datamodule."""
        super().__init__()
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.batch_size = batch_size

    def train_dataloader(self):
        """Returns a DataLoader for the training dataset."""
        return DataLoader(
            MockDataset(self.train_length), batch_size=self.batch_size, shuffle=True
        )

    def val_dataloader(self):
        """Returns a DataLoader for the validation dataset."""
        return DataLoader(
            MockDataset(self.val_length), batch_size=self.batch_size, shuffle=False
        )

    def test_dataloader(self):
        """Returns a DataLoader for the test dataset."""
        return DataLoader(
            MockDataset(self.test_length), batch_size=self.batch_size, shuffle=False
        )


class Sampleable(ABC):
    """Abstract class for sampleable distributions."""

    @property
    @abstractmethod
    def dim(self):
        """Dimensionality of the distribution."""
        pass

    @abstractmethod
    def sample(self, num_samples):
        """Sample from the distribution.

        Args:
            num_samples (int): Number of samples to draw from the distribution.

        Returns:
            torch.Tensor: Samples from the distribution.
        """
        pass


def three_to_one(resname):
    """Converts three-letter amino acid codes to one-letter codes."""
    return protein_letters_3to1.get(resname, None)


def process_ligand_and_pocket(
    pdbfile, sdffile, atom_dict, dist_cutoff, ca_only, amino_acid_dict
):
    """Processes a single ligand-pocket pair.

    Args:
        pdbfile (Path): Path to the protein PDB file.
        sdffile (Path): Path to the ligand SDF file.
        atom_dict (dict): Dictionary mapping atom symbols to integer IDs.
        dist_cutoff (float): Distance cutoff for identifying interacting pocket residues.
        ca_only (bool): If True, only consider C-alpha atoms for pocket representation.
        amino_acid_dict (dict): Dictionary mapping three-letter amino acid codes to integer IDs.

    Returns:
        tuple: A tuple containing (ligand_data, pocket_data) dictionaries.
               Returns (None, None) if processing fails for any reason.
    """
    try:
        pdb_struct = PDBParser(QUIET=True).get_structure("", pdbfile)
    except Exception as e:
        print(f"Error parsing PDB file {pdbfile}: {e}")
        return None, None

    try:
        ligand = Chem.SDMolSupplier(str(sdffile))[0]
    except Exception as e:
        print(f"Cannot read SDF mol ({sdffile}): {e}")
        return None, None

    if ligand is None:
        print(f"Ligand is None for {sdffile}")
        return None, None

    # remove H atoms if not in atom_dict, other atom types that aren't allowed
    # should stay so that the entire ligand can be removed from the dataset
    lig_atoms = [
        a.GetSymbol()
        for a in ligand.GetAtoms()
        if (a.GetSymbol().capitalize() in atom_dict or a.GetAtomicNum() != 1)
    ]
    lig_coords = np.array(
        [
            list(ligand.GetConformer(0).GetAtomPosition(idx))
            for idx in range(ligand.GetNumAtoms())
        ]
    )

    try:
        lig_one_hot = np.stack(
            [
                np.eye(1, len(atom_dict), atom_dict[a.capitalize()]).squeeze()
                for a in lig_atoms
            ]
        )
    except KeyError as e:
        print(f"KeyError: {e} not in atom dict ({sdffile})")
        return None, None

    # Find interacting pocket residues based on distance cutoff
    pocket_residues = []
    for residue in pdb_struct[0].get_residues():
        res_coords = np.array([a.get_coord() for a in residue.get_atoms()])
        # Ensure res_coords is not empty before checking min()
        if (
            res_coords.size > 0
            and lig_coords.size > 0
            and is_aa(residue.get_resname(), standard=True)
            and (
                ((res_coords[:, None, :] - lig_coords[None, :, :]) ** 2).sum(-1) ** 0.5
            ).min()
            < dist_cutoff
        ):
            pocket_residues.append(residue)

    pocket_ids = [f"{res.parent.id}:{res.id[1]}" for res in pocket_residues]
    ligand_data = {
        "lig_coords": lig_coords,
        "lig_one_hot": lig_one_hot,
    }

    pocket_data = {}  # Initialize pocket_data

    if ca_only:
        try:
            pocket_one_hot = []
            full_coords = []
            for res in pocket_residues:
                for atom in res.get_atoms():
                    if atom.name == "CA":
                        # Ensure the amino_acid_dict has the three_to_one mapping
                        aa_one_letter = three_to_one(res.get_resname())
                        if aa_one_letter in amino_acid_dict:
                            pocket_one_hot.append(
                                np.eye(
                                    1,
                                    len(amino_acid_dict),
                                    amino_acid_dict[aa_one_letter],
                                ).squeeze()
                            )
                            full_coords.append(atom.coord)
                        else:
                            print(
                                f"Warning: Amino acid {res.get_resname()} not in amino_acid_dict."
                            )
                            # Optionally, append a placeholder for unknown AA types
                            # pocket_one_hot.append(np.eye(1, len(amino_acid_dict), len(amino_acid_dict) - 1).squeeze()) # if 'others' is last entry
            if pocket_one_hot:  # Check if list is not empty before stacking
                pocket_one_hot = np.stack(pocket_one_hot)
                full_coords = np.stack(full_coords)
            else:
                return None, None  # No CA atoms found or recognized
        except KeyError as e:
            print(f"KeyError: {e} not in amino acid dict ({pdbfile}, {sdffile})")
            return None, None

        pocket_data = {
            "pocket_coords": full_coords,
            "pocket_one_hot": pocket_one_hot,
            "pocket_ids": pocket_ids,
        }
    else:
        full_atoms_elements = []
        full_coords = []
        for res in pocket_residues:
            for atom in res.get_atoms():
                full_atoms_elements.append(atom.element)
                full_coords.append(atom.coord)

        if not full_atoms_elements:  # No atoms found for pocket
            return None, None

        full_atoms_elements = np.array(full_atoms_elements)
        full_coords = np.stack(full_coords)

        try:
            pocket_one_hot = []
            for a_elem in full_atoms_elements:
                # Prioritize atom_dict for atom types in full atom mode
                if a_elem.capitalize() in atom_dict:
                    atom_idx = atom_dict[a_elem.capitalize()]
                    pocket_one_hot.append(np.eye(1, len(atom_dict), atom_idx).squeeze())
                else:
                    # Fallback for unknown atoms (e.g., if 'others' is in atom_dict)
                    if "others" in atom_dict:
                        pocket_one_hot.append(
                            np.eye(1, len(atom_dict), atom_dict["others"]).squeeze()
                        )
                    else:
                        print(
                            f"Warning: Atom {a_elem} not in atom_dict and no 'others' fallback."
                        )
                        # This sample might need to be skipped or handled differently
                        return None, None
            if pocket_one_hot:
                pocket_one_hot = np.stack(pocket_one_hot)
            else:
                return None, None  # No valid pocket atoms recognized
        except KeyError as e:
            print(f"KeyError: {e} not in atom dict ({pdbfile})")
            return None, None

        pocket_data = {
            "pocket_coords": full_coords,
            "pocket_one_hot": pocket_one_hot,
            "pocket_ids": pocket_ids,
        }

    # Ensure pocket data is not empty
    if "pocket_coords" not in pocket_data or pocket_data["pocket_coords"].size == 0:
        return None, None

    return ligand_data, pocket_data


def compute_smiles(positions, one_hot, mask, dataset_info):
    """Computes SMILES strings for molecules from their positions and atom types."""
    print("Computing SMILES ...")

    atom_types = np.argmax(one_hot, axis=-1)

    sections = np.where(np.diff(mask))[0] + 1
    positions = [torch.from_numpy(x) for x in np.split(positions, sections)]
    atom_types = [torch.from_numpy(x) for x in np.split(atom_types, sections)]

    mols_smiles = []

    pbar = tqdm(enumerate(zip(positions, atom_types)), total=len(np.unique(mask)))
    for i, (pos, atom_type) in pbar:
        # Assuming build_molecule uses dataset_info correctly
        mol = make_mol_openbabel(pos, atom_type, dataset_info["atom_decoder"])
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except ValueError:
            continue

        mol_smiles = rdmol_to_smiles(mol)
        if mol_smiles is not None:
            mols_smiles.append(mol_smiles)
        pbar.set_description(f"{len(mols_smiles)}/{i + 1} successful")

    return mols_smiles


def get_n_nodes(lig_mask, pocket_mask, smooth_sigma=None):
    """Computes the joint distribution of ligand's and pocket's number of nodes."""
    idx_lig, n_nodes_lig = np.unique(lig_mask, return_counts=True)
    idx_pocket, n_nodes_pocket = np.unique(pocket_mask, return_counts=True)

    # It's possible for unique masks to have different lengths if some samples were skipped
    # We should only assert equality if we are sure the indices should match.
    # For robust handling, we should make sure the indices align or process based on common IDs.
    # For now, keeping the original assertion but be mindful if you get errors.
    assert np.all(
        idx_lig == idx_pocket
    ), "Mask indices do not align, samples might be missing."

    # Pad histograms if sizes don't match, to prevent index out of bounds
    max_n_lig = np.max(n_nodes_lig) if n_nodes_lig.size > 0 else 0
    max_n_pocket = np.max(n_nodes_pocket) if n_nodes_pocket.size > 0 else 0

    joint_histogram = np.zeros((max_n_lig + 1, max_n_pocket + 1))

    for nlig, npocket in zip(n_nodes_lig, n_nodes_pocket):
        joint_histogram[nlig, npocket] += 1

    print(
        f"Original histogram: {np.count_nonzero(joint_histogram)}/"
        f"{joint_histogram.shape[0] * joint_histogram.shape[1]} bins filled"
    )

    # Smooth the histogram
    if smooth_sigma is not None:
        filtered_histogram = gaussian_filter(
            joint_histogram,
            sigma=smooth_sigma,
            order=0,
            mode="constant",
            cval=0.0,
            truncate=4.0,
        )

        print(
            f"Smoothed histogram: {np.count_nonzero(filtered_histogram)}/"
            f"{filtered_histogram.shape[0] * filtered_histogram.shape[1]} bins filled"
        )

        joint_histogram = filtered_histogram

    return joint_histogram


def get_bond_length_arrays(atom_mapping):
    """Converts bond length dictionaries from constants into arrays.

    Requires `constants.bonds1`, `constants.bonds2`, `constants.bonds3`.
    """
    bond_arrays = []
    for i in range(3):
        bond_dict = getattr(constants, f"bonds{i + 1}")
        bond_array = np.zeros((len(atom_mapping), len(atom_mapping)))
        for a1 in atom_mapping.keys():
            for a2 in atom_mapping.keys():
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    bond_len = bond_dict[a1][a2]
                else:
                    bond_len = 0
                bond_array[atom_mapping[a1], atom_mapping[a2]] = bond_len

        assert np.all(bond_array == bond_array.T)
        bond_arrays.append(bond_array)

    return bond_arrays


def get_lennard_jones_rm(atom_mapping):
    """Calculates R_m for the Lennard-Jones potential based on covalent radii and bond lengths.

    Requires `constants.covalent_radii` and `constants.bondsX`.
    """
    # Bond radii for the Lennard-Jones potential
    LJ_rm = np.zeros((len(atom_mapping), len(atom_mapping)))

    for a1 in atom_mapping.keys():
        for a2 in atom_mapping.keys():
            all_bond_lengths = []
            for btype in ["bonds1", "bonds2", "bonds3"]:
                bond_dict = getattr(constants, btype)
                if a1 in bond_dict and a2 in bond_dict[a1]:
                    all_bond_lengths.append(bond_dict[a1][a2])

            if len(all_bond_lengths) > 0:
                # take the shortest possible bond length because slightly larger
                # values aren't penalized as much
                bond_len = min(all_bond_lengths)
            else:
                if a1 == "others" or a2 == "others":
                    bond_len = 0  # Default or placeholder if not found
                else:
                    # Replace missing values with sum of average covalent radii
                    # Assuming 'covalent_radii' is accessible from 'constants'
                    bond_len = constants.covalent_radii.get(
                        a1, 0
                    ) + constants.covalent_radii.get(a2, 0)

            LJ_rm[atom_mapping[a1], atom_mapping[a2]] = bond_len

    assert np.all(LJ_rm == LJ_rm.T)
    return LJ_rm


def get_type_histograms(lig_one_hot, pocket_one_hot, atom_encoder, aa_encoder):
    """Generates histograms of ligand atom types and pocket amino acid types."""
    atom_decoder = list(atom_encoder.keys())
    atom_counts = {k: 0 for k in atom_encoder.keys()}
    # Ensure indices are within bounds
    for a_idx in lig_one_hot.argmax(1):
        if a_idx < len(atom_decoder):
            atom_counts[atom_decoder[a_idx]] += 1

    aa_decoder = list(aa_encoder.keys())
    aa_counts = {k: 0 for k in aa_encoder.keys()}
    for r_idx in pocket_one_hot.argmax(1):
        if r_idx < len(aa_decoder):
            aa_counts[aa_decoder[r_idx]] += 1

    return atom_counts, aa_counts


def saveall(
    filename,
    pdb_and_mol_ids,
    lig_coords,
    lig_one_hot,
    lig_mask,
    pocket_coords,
    pocket_one_hot,
    pocket_mask,
):
    """Saves processed data to a .npz file."""
    np.savez(
        filename,
        names=pdb_and_mol_ids,
        lig_coords=lig_coords,
        lig_one_hot=lig_one_hot,
        lig_mask=lig_mask,
        pocket_coords=pocket_coords,
        pocket_one_hot=pocket_one_hot,
        pocket_mask=pocket_mask,
    )
    return True


def download_google_drive_file(google_drive_url, destination_path, cookies_file=None):
    """Downloads a file from Google Drive using gdown library.

    Can take a full shareable URL or just the file ID.

    Args:
        google_drive_url (str): The Google Drive URL (shareable link or file ID).
        destination_path (Path): The Path object for the destination file.
        cookies_file (str, optional): Path to a cookies.txt file for authentication.
                                      Defaults to None.

    Returns:
        bool: True if download is successful, False otherwise.
    """
    if destination_path.exists():
        print(f"File {destination_path} already exists. Skipping download.")
        return True
    print(
        f"Attempting to download from {google_drive_url} to {destination_path} using gdown..."
    )

    try:
        destination_path.parent.mkdir(parents=True, exist_ok=True)

        gdown.download(
            google_drive_url, output=str(destination_path), quiet=False, fuzzy=True
        )

        print(f"Download complete: {destination_path}")
        return True
    except Exception as e:
        print(f"An unexpected error occurred during download with gdown: {e}")
        return False


def download_crossdocked(basedir: Path, outdir: Path = None, random_seed: int = 42):
    """Downloads the CrossDocked dataset.

    Args:
        basedir (Path): Base directory where the raw dataset is located
                        (e.g., contains 'crossdocked_pocket10/' and 'split_by_name.pt').
        outdir (Path, optional): Output directory for processed data.
                                 If None, a default path under `basedir` is used.
                                 Defaults to None.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
    """
    print(f"Starting CrossDocked dataset processing from: {basedir}")
    print(
        f"Output directory for processed data: {outdir if outdir else 'default under basedir'}"
    )

    random.seed(random_seed)  # Set random seed for reproducibility

    # datadir = basedir / 'crossdocked_pocket10/'

    # Read data split
    split_path = Path(basedir, "split_by_name.pt")
    crossdocked_path = Path(basedir, "crossdocked_pocket10")
    if not split_path.exists():
        print(f"Required split file '{split_path}' not found locally.")
        print("Attempting to download it...")

        if not download_google_drive_file(
            "https://drive.google.com/file/d/1mycOKpphVBQjxEbpn1AwdpQs8tNVbxKY/view?usp=drive_link",
            split_path,
        ):
            # If download fails, then raise the FileNotFoundError
            raise FileNotFoundError(
                f"Required split file '{split_path}' not found and could not be downloaded. "
                "Please ensure the raw CrossDocked dataset "
                "(including 'split_by_name.pt' and 'crossdocked_pocket10/') "
                "is placed in your 'basedir' or check the download link."
            )
    if not crossdocked_path.exists():
        crossdocked_tar_path = Path(basedir, "crossdocked_pocket10.tar.gz")
        if download_google_drive_file(
            "https://drive.google.com/file/d/10KGuj15mxOJ2FBsduun2Lggzx0yPreEU/view?usp=drive_link",
            crossdocked_tar_path,
        ):
            import tarfile

            with tarfile.open(crossdocked_tar_path, "r") as tar:
                tar.extractall(path=basedir)
            # delete the tar file after extraction
            # os.remove(crossdocked_tar_path)
        else:
            raise FileNotFoundError(
                f"Required CrossDocked dataset archive '{crossdocked_tar_path}' not found and could not be downloaded. "
                "Please ensure the raw CrossDocked dataset is placed in your 'basedir' or check the download link."
            )
