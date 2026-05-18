"""Calculate and save reference metric distributions for a specific dataset split (e.g., test or val).

This script establishes a baseline by calculating chemical property distributions (QED, LogP,
SA Score, Lipinski, GNINA Affinity, Molecule Size) for the original molecules in the
CrossDocked dataset. It maps the PyTorch Geometric dataset entries back to their original
PDB/SDF files to ensure high-fidelity scoring. The resulting distributions are saved as
a .npy file, which is used by evaluation scripts to calculate Wasserstein-1 distances.

Parser Arguments:
    --data_dir (str): Dataset root directory.
    --data_name (str): Dataset configuration name (e.g., "crossdock_full").
    --split (str): The dataset split to process ("test" or "val").
    --output_path (str): Optional. Custom path to save the .npy result.

Files Produced:
    - A .npy file (e.g., test_distribution.npy) containing a dictionary of metric distributions.
"""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import topoflow.utils.constants as constants
from topoflow.utils.metrics import get_distributions
from topoflow.data.datasets import CrossDocked


def main():
    """Main function to calculate and save test set distributions for all metrics."""
    parser = argparse.ArgumentParser(
        description="Calculate and save test set distributions for all metrics."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="crossdock_full",
        help="Dataset name in constants.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split to process (test/val)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save .npy result. Defaults to data_dir/processed/split_distribution.npy",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(
            args.data_dir, "processed", f"{args.split}_distribution.npy"
        )

    print(f"Loading {args.split} dataset from {args.data_dir}...")
    dataset = CrossDocked(root=args.data_dir, split=args.split)

    # --- NEW: Link to original files for accurate baseline ---
    print("Linking dataset to original PDB and SDF files...")
    test_npz_path = os.path.join(args.data_dir, "processed", f"{args.split}.npz")
    if not os.path.exists(test_npz_path):
        # Try without extension or alternate locations
        test_npz_path = os.path.join(args.data_dir, "processed", f"{args.split}.npz")

    source_root = os.path.join(
        args.data_dir, "raw", "processed_crossdock_noH_full_temp", args.split
    )

    linked_data_list = []
    if os.path.exists(test_npz_path) and os.path.exists(source_root):
        test_names = np.load(test_npz_path, allow_pickle=True)["names"]
        for i in tqdm(range(len(dataset)), desc="Mapping files"):
            data = dataset[i]
            combined_name = test_names[i]

            # PDB Path construction
            if ".pdb" in combined_name:
                pdb_filename = combined_name.split(".pdb")[0] + ".pdb"
            else:
                pdb_filename = combined_name.split("_")[0] + ".pdb"
            pdb_filename = os.path.basename(pdb_filename)

            pdb_path = os.path.join(source_root, pdb_filename)
            if not os.path.exists(pdb_path):
                pdb_path = os.path.join(source_root, pdb_filename.replace("_", "-"))

            # Ligand SDF Path construction
            ligand_path = os.path.join(source_root, combined_name + ".sdf")
            if not os.path.exists(ligand_path):
                ligand_path = os.path.join(
                    source_root, combined_name.replace("_", "-") + ".sdf"
                )

            if os.path.exists(pdb_path) and os.path.exists(ligand_path):
                data.pdb_path = os.path.abspath(pdb_path)
                data.ligand_sdf_path = os.path.abspath(ligand_path)

            linked_data_list.append(data)
        scoring_dataset = linked_data_list
    else:
        print(
            f"Warning: Could not find original files in {source_root}. Falling back to PyG data."
        )
        scoring_dataset = dataset

    # Load Normalization Params
    norm_params = None
    norm_path = os.path.join(args.data_dir, "processed", "normalization_params.npz")
    if os.path.exists(norm_path):
        print(f"Loading normalization parameters from {norm_path}")
        norm_data = np.load(norm_path, allow_pickle=True)
        norm_params = {"mean": norm_data["mean"], "std": norm_data["std"]}

    atom_decoder = constants.dataset_params[args.data_name]["atom_decoder"]

    # We include all relevant metrics
    metrics = ["qed", "logp", "sas", "lipinski", "gnina", "mol_size"]

    print(f"Calculating distributions for metrics: {metrics}")
    print("Using original PDB/SDF files where available for GNINA...")

    distribution = get_distributions(
        scoring_dataset,
        metrics=metrics,
        atom_decoder=atom_decoder,
        norm_params=norm_params,
    )

    # Also add mol_id or name if possible for reference
    # distribution["names"] = dataset.data.names # If available

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, distribution)
    print(f"\nDistribution saved successfully to {args.output_path}")


if __name__ == "__main__":
    main()
