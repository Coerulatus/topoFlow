"""QM9 dataset with RDKit-calculated molecular properties (QED, SAS, LogP)."""

import os
from collections import defaultdict
from pathlib import Path
import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from rdkit import Chem
from scipy.ndimage import gaussian_filter
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random
import shutil
from time import time

from topoflow.utils.metrics import qed_scorer, logp_scorer, sas_scorer
from topoflow.data.utils import (
    download_crossdocked,
    process_ligand_and_pocket,
    saveall,
    get_n_nodes_hist,
)
from topoflow.utils.molecules import data_to_rdmol, rdmol_to_smiles
import topoflow.utils.constants as constants


class QM9WithRDKitMetrics(InMemoryDataset):
    """A custom PyTorch Geometric dataset that extends the QM9 dataset to include RDKit-calculated properties."""

    def __init__(
        self, root, transform=None, pre_transform=None, normalize=True, fraction=1.0
    ):
        """A custom PyTorch Geometric dataset that extends QM9.

        This class adds RDKit-calculated molecular properties (QED, SAS, LogP) as new attributes to each Data object.
        """
        self.original_qm9_root = os.path.join(root, "QM9_original")
        self.normalize = normalize
        qed_min, qed_max = 0.0, 1.0
        logp_min, logp_max = -6.0, 6.0
        sas_min, sas_max = 1.0, 10.0
        self.norm_params = {
            "qed": (qed_min, qed_max),
            "logp": (logp_min, logp_max),
            "sas": (sas_min, sas_max),
        }
        self.fraction = fraction
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """QM9 dataset does not have raw files in the traditional sense."""
        return []

    @property
    def processed_file_names(self):
        """Filename for the processed dataset with RDKit metrics."""
        return "qm9_rdkit_metrics.pt"

    def download(self):
        """Download the original QM9 dataset if it doesn't exist."""
        QM9(self.original_qm9_root)

    def process(self):
        """Process the original QM9 dataset and calculate RDKit metrics."""
        original_dataset = QM9(self.original_qm9_root)
        if self.fraction < 1.0:
            # Sample a fraction of the dataset
            num_samples = int(len(original_dataset) * self.fraction)
            original_dataset = original_dataset[:num_samples]

        data_list = []
        # Iterate through each molecule in the original QM9 dataset
        for i, data in enumerate(
            tqdm(original_dataset, desc="Calculating RDKit metrics")
        ):
            # Get the SMILES string (QM9 stores it in `data.smiles` usually)
            smiles = (
                data.smiles[0] if isinstance(data.smiles, torch.Tensor) else data.smiles
            )

            # Convert SMILES to RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                print(
                    f"Warning: Could not parse SMILES for molecule {i}: {smiles}. Skipping."
                )
                continue

            # Calculate RDKit Descriptors
            # 1. Quantitative Estimation of Drug-likeness (QED)
            qed_score = torch.tensor([qed_scorer(mol)], dtype=torch.float)

            # 2. LogP (Partition Coefficient, related to solubility/lipophilicity)
            logp_score = torch.tensor([logp_scorer(mol)], dtype=torch.float)

            # 3. Synthetic Accessibility Score (SAS)
            # SAS score typically ranges from 1 (easy to synthesize) to 10 (hard to synthesize)
            sas_score = torch.tensor([sas_scorer(mol)], dtype=torch.float)

            if self.normalize:
                qed_min, qed_max = self.norm_params["qed"]
                logp_min, logp_max = self.norm_params["logp"]
                sas_min, sas_max = self.norm_params["sas"]
                if qed_score < qed_min or qed_score > qed_max:
                    raise ValueError(
                        f"QED score {qed_score.item()} out of bounds [{qed_min}, {qed_max}]"
                    )
                if logp_score < logp_min or logp_score > logp_max:
                    raise ValueError(
                        f"LogP score {logp_score.item()} out of bounds [{logp_min}, {logp_max}]"
                    )
                if sas_score < sas_min or sas_score > sas_max:
                    raise ValueError(
                        f"SAS score {sas_score.item()} out of bounds [{sas_min}, {sas_max}]"
                    )
                qed_score = (qed_score - qed_min) / (qed_max - qed_min)
                logp_score = (logp_score - logp_min) / (logp_max - logp_min)
                sas_score = (sas_score - sas_min) / (sas_max - sas_min)

            new_y = torch.cat([qed_score, logp_score, sas_score], dim=0)
            data.y = new_y

            # We only care about the atom types and positions for the graph structure
            data.x = torch.cat([data.x[:, :5], data.pos], dim=1)
            del data.pos  # Remove position if not needed

            data_list.append(data)

        # Save the processed data
        if self.pre_filter is not None:
            data_list = self.pre_filter(data_list)
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class CrossDocked(InMemoryDataset):
    """Cross-docked dataset with RDKit-calculated molecular properties."""

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        pre_transform=None,
        normalize: bool = True,
        outdir: str = None,
        ca_only: bool = False,
        distance_cutoff: float = 8.0,
        normalize_coords: bool = True,
        center_pocket: bool = True,
        random_seed: int = 42,
    ):
        """Initialize the CrossDocked dataset.

        Args:
            root (str): Root directory where the dataset is stored.
            split (str): The dataset split to use ('train', 'val', 'test').
            transform (callable, optional): A function/transform that takes in a Data object and
                returns a transformed version. The data object will be transformed before every access.
            pre_transform (callable, optional): A function/transform that takes in a Data object and
                returns a transformed version. The data object will be transformed before being saved to disk.
            normalize (bool): Whether to normalize the molecular properties (QED, SAS, LogP).
            outdir (str, optional): Directory to save processed data. If None, uses the default processed directory.
            ca_only (bool): If True, only process C-alpha atoms in the protein pockets.
            distance_cutoff (float): Distance cutoff for processing ligand-pocket interactions.
            normalize_coords (bool): If True, normalize the coordinates of the ligands and pockets.
            center_pocket (bool): If True, center the pocket coordinates around the origin.
            random_seed (int): Random seed for reproducibility.
        """
        self.split = split
        self.normalize = normalize
        self.outdir = outdir
        self.ca_only = ca_only
        self.distance_cutoff = distance_cutoff
        self.normalize_coords = normalize_coords
        self.center_pocket = center_pocket
        self.random_seed = random_seed

        super().__init__(root, transform, pre_transform)
        self.load()

    @property
    def raw_file_names(self):
        """Return the raw file names for the CrossDocked dataset."""
        return ["crossdocked_pocket10", "split_by_name.pt"]

    @property
    def processed_file_names(self):
        """Return the filename for the processed dataset."""
        return f"{self.split}_crossdocked_pyg.pt"

    def download(self):
        """Download the raw CrossDocked dataset files."""
        if not os.path.exists(str(self.raw_dir)):
            os.makedirs(str(self.raw_dir))

        download_crossdocked(basedir=str(self.raw_dir), outdir=str(self.processed_dir))
        print("Raw data download and initial NPZ processing complete.")

    def process(self):
        """Process the raw data into PyTorch Geometric Data objects.

        Processes the raw data (NPZ files created by download_crossdocked)
        into a single PyTorch Geometric Data object with slices for the
        specific split requested during initialization.
        """
        dataset_info = constants.dataset_params["crossdock_full"]
        amino_acid_dict = dataset_info["aa_encoder"]
        atom_dict = dataset_info["atom_encoder"]
        atom_decoder = dataset_info["atom_decoder"]
        if self.processed_dir is None:
            suffix = (
                "_crossdock" if "H" in atom_dict else "_crossdock_noH"
            )  # This checks if H is in atom_dict keys
            suffix += "_full_temp"
            processed_dir = Path(self.raw_dir, f"processed{suffix}")
        else:
            processed_dir = self.processed_dir

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir, exist_ok=True)
        print(f"Processed data will be saved to: {processed_dir}")

        datadir = Path(self.raw_dir, "crossdocked_pocket10/")

        split_path = Path(self.raw_dir, "split_by_name.pt")
        print(f"Loading data split from: {split_path}")
        data_split = torch.load(split_path, weights_only=False)
        print(f"Data split loaded successfully. Keys: {list(data_split.keys())}")

        # TODO: Remove this when loading whole dataset
        # data_split["train"] = data_split["train"][:1]

        # There is no validation set, copy 300 training examples (the validation set
        # is not very important in this application)
        data_split["val"] = random.sample(data_split["train"], 256)

        n_train_before = len(data_split["train"])
        n_val_before = len(data_split["val"])
        n_test_before = len(data_split["test"])

        failed_save = []
        n_samples_after = {}

        for split in data_split.keys():
            lig_coords_list = []
            lig_one_hot_list = []
            lig_mask_list = []
            pocket_coords_list = []
            pocket_one_hot_list = []
            pocket_mask_list = []
            pdb_and_mol_ids_list = []
            count = 0

            pdb_sdf_split_dir = os.path.join(processed_dir, split)
            os.makedirs(pdb_sdf_split_dir, exist_ok=True)
            target_file = os.path.join(processed_dir, f"{split}.npz")
            if os.path.exists(target_file):
                print(
                    f"Target file {target_file} already exists. Skipping processing for {split}."
                )
                continue

            print(f"\nProcessing '{split}' split...")

            tic = time()
            num_failed = 0
            pbar = tqdm(data_split[split])
            pbar.set_description(f"#failed: {num_failed}")

            for pocket_fn_rel, ligand_fn_rel in pbar:
                sdffile = datadir / ligand_fn_rel
                pdbfile = datadir / pocket_fn_rel

                if not pdbfile.exists():
                    num_failed += 1
                    failed_save.append((pocket_fn_rel, ligand_fn_rel, "PDB not found"))
                    pbar.set_description(f"#failed: {num_failed}")
                    continue
                if not sdffile.exists():
                    num_failed += 1
                    failed_save.append((pocket_fn_rel, ligand_fn_rel, "SDF not found"))
                    pbar.set_description(f"#failed: {num_failed}")
                    continue

                try:
                    ligand_data, pocket_data = process_ligand_and_pocket(
                        pdbfile,
                        sdffile,
                        atom_dict=atom_dict,
                        dist_cutoff=self.distance_cutoff,
                        ca_only=False,
                        amino_acid_dict=amino_acid_dict,
                    )

                    if ligand_data is None or pocket_data is None:
                        num_failed += 1
                        failed_save.append(
                            (pocket_fn_rel, ligand_fn_rel, "Processing returned None")
                        )
                        pbar.set_description(f"#failed: {num_failed}")
                        continue

                except (
                    KeyError,
                    AssertionError,
                    FileNotFoundError,
                    IndexError,
                    ValueError,
                ) as e:
                    print(
                        f"Error processing {pocket_fn_rel}, {ligand_fn_rel}: {type(e).__name__} {e}"
                    )
                    num_failed += 1
                    failed_save.append((pocket_fn_rel, ligand_fn_rel, str(e)))
                    pbar.set_description(f"#failed: {num_failed}")
                    continue
                except Exception as e:  # Catch any other unexpected errors
                    print(
                        f"Unhandled error processing {pocket_fn_rel}, {ligand_fn_rel}: {type(e).__name__} {e}"
                    )
                    num_failed += 1
                    failed_save.append((pocket_fn_rel, ligand_fn_rel, str(e)))
                    pbar.set_description(f"#failed: {num_failed}")
                    continue

                if len(ligand_data["lig_coords"]) == len(
                    ligand_data["lig_one_hot"]
                ) and len(pocket_data["pocket_coords"]) == len(
                    pocket_data["pocket_one_hot"]
                ):
                    if self.center_pocket:
                        # Center the pocket coordinates around the origin
                        pocket_coords = pocket_data["pocket_coords"]
                        mean_pocket = np.mean(pocket_coords, axis=0)
                        pocket_data["pocket_coords"] -= mean_pocket
                        ligand_data["lig_coords"] -= mean_pocket
                    pdb_and_mol_ids_list.append(f"{pocket_fn_rel}_{ligand_fn_rel}")
                    lig_coords_list.append(ligand_data["lig_coords"])
                    lig_one_hot_list.append(ligand_data["lig_one_hot"])
                    lig_mask_list.append(
                        count * np.ones(len(ligand_data["lig_coords"]))
                    )
                    pocket_coords_list.append(pocket_data["pocket_coords"])
                    pocket_one_hot_list.append(pocket_data["pocket_one_hot"])
                    pocket_mask_list.append(
                        count * np.ones(len(pocket_data["pocket_coords"]))
                    )
                    count += 1
                else:
                    print(
                        f"Skipping {pocket_fn_rel}, {ligand_fn_rel} due to mismatched lengths: "
                        f"{len(ligand_data['lig_coords'])} vs {len(ligand_data['lig_one_hot'])} "
                        f"and {len(pocket_data['pocket_coords'])} vs {len(pocket_data['pocket_one_hot'])}"
                    )
                    num_failed += 1
                    failed_save.append(
                        (pocket_fn_rel, ligand_fn_rel, "Mismatched lengths")
                    )
                    pbar.set_description(f"#failed: {num_failed}")
                    continue

                if split in {"val", "test"}:
                    new_rec_name = Path(pdbfile).stem.replace("_", "-")
                    pdb_file_out = Path(pdb_sdf_split_dir, f"{new_rec_name}.pdb")
                    shutil.copy(pdbfile, pdb_file_out)

                    new_lig_name = (
                        new_rec_name + "_" + Path(sdffile).stem.replace("_", "-")
                    )
                    sdf_file_out = Path(pdb_sdf_split_dir, f"{new_lig_name}.sdf")
                    shutil.copy(sdffile, sdf_file_out)

                    with open(Path(pdb_sdf_split_dir, f"{new_lig_name}.txt"), "w") as f:
                        f.write(" ".join(pocket_data["pocket_ids"]))

            if lig_coords_list:
                lig_coords = np.concatenate(lig_coords_list, axis=0)
                lig_one_hot = np.concatenate(lig_one_hot_list, axis=0)
                lig_mask = np.concatenate(lig_mask_list, axis=0)
                pocket_coords = np.concatenate(pocket_coords_list, axis=0)
                pocket_one_hot = np.concatenate(pocket_one_hot_list, axis=0)
                pocket_mask = np.concatenate(pocket_mask_list, axis=0)
                saveall(
                    target_file,
                    pdb_and_mol_ids_list,
                    lig_coords,
                    lig_one_hot,
                    lig_mask,
                    pocket_coords,
                    pocket_one_hot,
                    pocket_mask,
                )
            else:
                print(
                    f"No valid samples processed for {split} split. Skipping .npz save."
                )

            n_samples_after[split] = len(pdb_and_mol_ids_list)
            print(f"Processing {split} set took {(time() - tic) / 60.0:.2f} minutes")
            print(f"Failed samples in {split} split: {num_failed}")

        if failed_save:
            print("\n--- Summary of Failed Samples ---")
            for entry in failed_save:
                print(f"Failed: PDB/SDF: {entry[0]}, {entry[1]} - Reason: {entry[2]}")
        else:
            print("\nAll samples processed successfully (no failures recorded).")

        data_list = []

        split_name = self.split

        split_npz_path = os.path.join(self.processed_dir, f"{split_name}.npz")

        if not os.path.exists(split_npz_path):
            raise FileNotFoundError(
                f"Processed NPZ file for split '{split_name}' not found at {split_npz_path}. "
                "Please ensure `download()` method (which calls `download_crossdocked`) "
                "completed successfully and generated this file."
            )

        print(f"Loading data from {split_npz_path} for split '{split_name}'...")
        loaded_data = np.load(split_npz_path, allow_pickle=True)

        names = loaded_data["names"]
        lig_coords_all = loaded_data["lig_coords"]
        lig_one_hot_all = loaded_data["lig_one_hot"]
        lig_mask_all = loaded_data["lig_mask"]
        pocket_coords_all = loaded_data["pocket_coords"]
        pocket_one_hot_all = loaded_data["pocket_one_hot"]
        pocket_mask_all = loaded_data["pocket_mask"]

        unique_sample_ids = np.unique(lig_mask_all)

        # Normalization of coordinates if required
        if self.normalize_coords and split_name == "train":
            print(
                "Normalizing coordinates for ligands and pockets in training split..."
            )
            mean_lig = np.mean(lig_coords_all, axis=0)
            std_lig = np.std(lig_coords_all, axis=0)
            mean_pocket = np.mean(pocket_coords_all, axis=0)
            std_pocket = np.std(pocket_coords_all, axis=0)
            # Save normalization parameters for later use
            np.savez(
                os.path.join(self.processed_dir, "normalization_params.npz"),
                mean_lig=mean_lig,
                std_lig=std_lig,
                mean_pocket=mean_pocket,
                std_pocket=std_pocket,
            )
            # lig_coords_all = (lig_coords_all - mean_lig) / std_lig
            lig_coords_all = (lig_coords_all - mean_pocket) / std_pocket
            pocket_coords_all = (pocket_coords_all - mean_pocket) / std_pocket
        elif self.normalize_coords:
            print("Normalizing coordinates for validation/test splits...")
            norm_params_path = os.path.join(
                self.processed_dir, "normalization_params.npz"
            )
            if not os.path.exists(norm_params_path):
                raise FileNotFoundError(
                    f"Normalization parameters not found at {norm_params_path}. "
                    "Ensure the training split was processed with normalization enabled."
                )
            norm_params = np.load(norm_params_path)
            mean_lig = norm_params["mean_lig"]
            std_lig = norm_params["std_lig"]
            mean_pocket = norm_params["mean_pocket"]
            std_pocket = norm_params["std_pocket"]
            # lig_coords_all = (lig_coords_all - mean_lig) / std_lig
            lig_coords_all = (lig_coords_all - mean_pocket) / std_pocket
            pocket_coords_all = (pocket_coords_all - mean_pocket) / std_pocket

        # We need the histogram with the distribution of number of nodes (only for training)
        if split_name == "train":
            n_nodes_hist = get_n_nodes_hist(
                lig_mask_all, pocket_mask_all, smooth_sigma=1.0
            )
            n_nodes_hist_path = os.path.join(
                self.processed_dir, "size_distribution.npy"
            )
            np.save(n_nodes_hist_path, n_nodes_hist)

        if len(names) != len(unique_sample_ids):
            print(
                f"Warning: Mismatch between number of names ({len(names)}) and unique sample IDs ({len(unique_sample_ids)}) in {split_name}.npz. This might indicate corrupt data."
            )

        print(
            f"Reconstructing {len(unique_sample_ids)} samples from {split_name}.npz..."
        )
        lig_index_map = defaultdict(list)
        pocket_index_map = defaultdict(list)

        for i, sid in enumerate(lig_mask_all):
            lig_index_map[sid].append(i)
        for i, sid in enumerate(pocket_mask_all):
            pocket_index_map[sid].append(i)

        # Loop through samples
        for sample_idx in tqdm(unique_sample_ids):
            lig_indices = lig_index_map.get(sample_idx, [])
            pocket_indices = pocket_index_map.get(sample_idx, [])

            if len(lig_indices) == 0 or len(pocket_indices) == 0:
                print(
                    f"Warning: No ligand or pocket for sample {sample_idx}. Skipping."
                )
                continue

            # Integer indexing — much faster than boolean masks
            lig_coords = torch.from_numpy(lig_coords_all[lig_indices]).float()
            lig_one_hot = torch.from_numpy(lig_one_hot_all[lig_indices]).float()
            batch_lig = torch.zeros(len(lig_indices), dtype=torch.long)

            pocket_coords = torch.from_numpy(pocket_coords_all[pocket_indices]).float()
            pocket_one_hot = torch.from_numpy(
                pocket_one_hot_all[pocket_indices]
            ).float()
            batch_pocket = torch.zeros(len(pocket_indices), dtype=torch.long)

            data = Data(
                h_lig=lig_one_hot,
                x_lig=lig_coords,
                batch_lig=batch_lig,
                h_pocket=pocket_one_hot,
                x_pocket=pocket_coords,
                batch_pocket=batch_pocket,
            )
            data_list.append(data)
        print(
            f"Finished reconstructing samples for '{split_name}'. Total samples for this split: {len(data_list)}"
        )

        if not data_list:
            print(
                f"No valid data samples were processed for split '{split_name}'. Initializing as empty dataset."
            )
            self.data = Data()
            self.slices = {}
            return

        print(
            f"Collating {len(data_list)} individual Data objects for split '{split_name}'..."
        )
        data, slices = self.collate(data_list)

        print(f"Saving collated data to {self.processed_paths[0]}...")
        torch.save((data, slices), self.processed_paths[0])
        print("Processed dataset saved successfully.")

    def load(self):
        """Load the processed dataset."""
        if not os.path.exists(self.processed_paths[0]):
            raise FileNotFoundError(
                f"Processed dataset not found at {self.processed_paths[0]}. Please run process() first."
            )
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        print(f"Loaded dataset from {self.processed_paths[0]}.")

        try:
            self.distribution_hist = np.load(
                os.path.join(self.processed_dir, "size_distribution.npy"),
                allow_pickle=True,
            )
        except FileNotFoundError:
            raise FileNotFoundError(
                "Size distribution not found. Process the dataset with train split once to generate it."
            )

        norm_params_path = os.path.join(self.processed_dir, "normalization_params.npz")
        if not os.path.exists(norm_params_path):
            raise FileNotFoundError(
                f"Normalization parameters not found at {norm_params_path}. "
                "Ensure the training split was processed with normalization enabled."
            )
        self.norm_params = np.load(norm_params_path)

        smiles_path = self.processed_paths[0][:-3] + "_smiles.txt"
        if not os.path.exists(smiles_path):
            print("No SMILES file found, generating from RDKit molecules...")
            smiles_list = []
            for data in tqdm(self, desc="Converting data to RDKit molecules"):
                mol = data_to_rdmol(
                    data, constants.dataset_params["crossdock_full"]["atom_decoder"]
                )
                if len(mol) == 0:
                    continue
                mol = mol[0]
                smiles_list.append(rdmol_to_smiles(mol))
            with open(smiles_path, "w") as f:
                for smiles in smiles_list:
                    if smiles is not None:
                        f.write(smiles + "\n")
            print(f"Generated SMILES file at {smiles_path}")
        else:
            self.smiles_list = []
            with open(smiles_path, "r") as f:
                self.smiles_list = [line.strip() for line in f.readlines()]
            print(f"Loaded SMILES file from {smiles_path}")

        print(f"Dataset loaded successfully with {len(self)} samples.")
