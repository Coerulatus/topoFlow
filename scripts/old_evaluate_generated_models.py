"""Evaluate generated molecules against test distributions using multiple splits for robust statistics.

This script evaluates TopoFlow models by comparing generated molecules to the reference
test set distribution. It maps generated SDFs back to their target pockets, converts
them to PyG format, and calculates metrics including QED, LogP, SA Score, Lipinski,
and GNINA scores. It specifically calculates Wasserstein (WL) distances between the
generated and reference distributions. Results are aggregated across multiple splits.

Parser Arguments:
    --data_dir (str): Path to the CrossDocked data.
    --output_base_dir (str): Base directory containing model subfolders with generated SDFs.

Files Produced:
    - pyg_splits.pt: PyG representation of the split data.
    - split_results.json: Raw results for each evaluation split.
    - evaluation_results.csv: Final aggregated stats (mean and std) for the model.
"""

import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
import scipy.stats
from torch_geometric.data import Data, Batch
import argparse

from topoflow.data.datasets import CrossDocked
from topoflow.utils.metrics import scorer_dict, get_distributions
import topoflow.utils.constants as constants
from topoflow.utils.molecules import (
    mol_to_data,
    load_sdf_molecules_rdkit,
    data_to_rdmol,
)


def aggregate_simple_stats(values):
    """Calculates the mean and standard deviation for a list of scalar values."""
    if not values:
        return np.nan, np.nan
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


def get_test_distribution(data_dir, atom_decoder):
    """Loads or generates the test distribution for the CrossDocked dataset, calculating missing metrics if necessary."""
    dist_path = os.path.join(data_dir, "processed", "test_distribution.npy")
    test_dist = {}
    if os.path.exists(dist_path):
        print(f"Loading test distribution from {dist_path}")
        test_dist = np.load(dist_path, allow_pickle=True).item()

    all_metrics = ["qed", "logp", "sas", "lipinski", "gnina", "mol_size"]

    # Check which metrics are actually missing
    missing_metrics = []
    if "qed" not in test_dist:
        missing_metrics.append("qed")
    if "logp" not in test_dist:
        missing_metrics.append("logp")
    if "sas" not in test_dist:
        missing_metrics.append("sas")
    if "lipinski" not in test_dist:
        missing_metrics.append("lipinski")
    if "vina" not in test_dist or "cnn_affinity" not in test_dist:
        missing_metrics.append("gnina")
    if "mol_size" not in test_dist:
        missing_metrics.append("mol_size")

    if not missing_metrics:
        return test_dist

    print(f"Generating missing metrics for test distribution: {missing_metrics}")
    test_dataset = CrossDocked(root=data_dir, split="test")

    # Load Normalization Params
    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    norm_params = None
    if os.path.exists(norm_path):
        norm_data = np.load(norm_path, allow_pickle=True)
        norm_params = {"mean": norm_data["mean"], "std": norm_data["std"]}

    for m in missing_metrics:
        print(f"Calculating metric: {m}")
        m_dist = get_distributions(
            test_dataset,
            metrics=[m],
            atom_decoder=atom_decoder,
            norm_params=norm_params,
        )
        test_dist.update(m_dist)
        # Save after each metric is calculated
        np.save(dist_path, test_dist)
        print(f"Saved updated test distribution (added {m}) to {dist_path}")

    return test_dist


def main():
    """Main function to evaluate generated molecules against the test distribution of the CrossDocked dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/CrossDocked/")
    parser.add_argument(
        "--output_base_dir", type=str, default="./outputs/test_molecules/"
    )
    args = parser.parse_args()

    data_name = "crossdock_full"
    atom_encoder = constants.dataset_params[data_name]["atom_encoder"]
    atom_decoder = constants.dataset_params[data_name]["atom_decoder"]

    # 1. Get test distribution
    test_dist = get_test_distribution(args.data_dir, atom_decoder)

    # 2. Load test dataset to get pockets
    test_dataset = CrossDocked(root=args.data_dir, split="test")
    test_npz = np.load(
        os.path.join(args.data_dir, "processed", "test.npz"), allow_pickle=True
    )
    test_names = test_npz["names"]

    # Map pocket name stem to dataset index
    # Example name: 'GSTP1_HUMAN_2_210_0/14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10.pdb_...'
    pocket_stem_to_idx = {}
    for i, name in enumerate(test_names):
        # Extract pocket part: GSTP1_HUMAN_2_210_0/14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10.pdb
        pocket_part = name.split(".pdb")[0] + ".pdb"
        # Stem: 14gs_A_rec_20gs_cbd_lig_tt_min_0_pocket10
        pocket_stem = (
            os.path.basename(pocket_part).replace(".pdb", "").replace("_", "-")
        )
        pocket_stem_to_idx[pocket_stem] = i

    model_folders = [
        d
        for d in os.listdir(args.output_base_dir)
        if os.path.isdir(os.path.join(args.output_base_dir, d))
    ]

    for model_folder in model_folders:
        folder_path = os.path.join(args.output_base_dir, model_folder)
        mapping_path = os.path.join(folder_path, "mapping.txt")
        sdfs_dir = os.path.join(folder_path, "sdfs")
        scored_path = os.path.join(folder_path, "pyg_scored_flat.pt")

        if os.path.exists(scored_path):
            print(f"Skipping {model_folder}, pyg_scored_flat.pt already exists.")
            continue

        if not os.path.exists(mapping_path) or not os.path.exists(sdfs_dir):
            print(f"Skipping {model_folder}, mapping.txt or sdfs/ not found.")
            continue

        print(f"Evaluating model: {model_folder}")

        # Read mapping
        pocket_to_sdf = {}
        with open(mapping_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    sdf_name = parts[0]
                    pdb_name = parts[1]
                    pocket_stem = pdb_name.replace(".pdb", "")
                    pocket_to_sdf[pocket_stem] = sdf_name

        # Collect molecules
        protein_molecules = {}  # pocket_stem -> list of PyG Data objects

        for pocket_stem, sdf_name in tqdm(
            pocket_to_sdf.items(), desc=f"Loading molecules for {model_folder}"
        ):
            sdf_path = os.path.join(sdfs_dir, sdf_name)
            if not os.path.exists(sdf_path):
                continue

            if pocket_stem not in pocket_stem_to_idx:
                # Try fallback: some mapping might have different dash/underscore
                found = False
                for k in pocket_stem_to_idx.keys():
                    if k.replace("-", "_") == pocket_stem.replace("-", "_"):
                        pocket_idx = pocket_stem_to_idx[k]
                        found = True
                        break
                if not found:
                    continue
            else:
                pocket_idx = pocket_stem_to_idx[pocket_stem]

            pocket_data = test_dataset[pocket_idx]

            mols = load_sdf_molecules_rdkit(sdf_path)
            pyg_mols = []
            for mol in mols:
                try:
                    x_lig, h_lig, _ = mol_to_data(mol, atom_encoder)
                    # Create combined Data object
                    data = Data(
                        x_lig=x_lig,
                        h_lig=h_lig,
                        batch_lig=torch.zeros(x_lig.size(0), dtype=torch.long),
                        x_pocket=pocket_data.x_pocket,
                        h_pocket=pocket_data.h_pocket,
                        batch_pocket=torch.zeros(
                            pocket_data.x_pocket.size(0), dtype=torch.long
                        ),
                    )
                    pyg_mols.append(data)
                except Exception as e:
                    print(f"Error processing molecule in {sdf_name}: {e}")

            protein_molecules[pocket_stem] = pyg_mols

        # 3. Create 10 splits
        splits = [[] for _ in range(10)]
        for pocket_stem, mols in protein_molecules.items():
            for i in range(min(len(mols), 10)):
                splits[i].append(mols[i])

        # Save PyG splits
        torch.save(splits, os.path.join(folder_path, "pyg_splits.pt"))

        # 4. Evaluate each split
        split_results_path = os.path.join(folder_path, "split_results.json")
        if os.path.exists(split_results_path):
            import json

            with open(split_results_path, "r") as f:
                split_results = json.load(f)
            print(
                f"Loaded {len(split_results)} previously calculated splits from {split_results_path}"
            )
        else:
            split_results = []

        # Load Normalization Params
        norm_path = os.path.join(args.data_dir, "processed", "normalization_params.npz")
        norm_params = None
        if os.path.exists(norm_path):
            norm_data = np.load(norm_path, allow_pickle=True)
            norm_params = {"mean": norm_data["mean"], "std": norm_data["std"]}

        for i, split in enumerate(splits):
            if not split:
                continue

            if i < len(split_results):
                print(f"Skipping split {i+1}/10 (already calculated)")
                continue

            print(f"Evaluating split {i+1}/10 for {model_folder}...")
            # We can use get_distributions by passing a list of Data objects
            # but we need to convert them to RDKit mols for standard metrics
            split_rdmols = []
            for data in split:
                mol = data_to_rdmol(data, atom_decoder=atom_decoder, sanitize=True)
                if mol:
                    split_rdmols.append(mol)

            metrics = ["qed", "logp", "sas", "lipinski", "gnina"]
            results = {}

            # Standard metrics
            for m in ["qed", "logp", "sas", "lipinski"]:
                scores = []
                scorer = scorer_dict[m]
                for mol in split_rdmols:
                    s = scorer(mol)
                    if s is not None:
                        scores.append(s.item())
                results[m] = scores

            # Gnina metrics
            vina_scores = []
            cnn_scores = []
            gnina_scorer_func = scorer_dict["gnina"]
            for data in tqdm(split, desc=f"Calculating Gnina scores for split {i+1}"):
                s = gnina_scorer_func(
                    data, norm_params=norm_params, atom_decoder=atom_decoder
                )
                if s:
                    vina_scores.append(s[0])
                    cnn_scores.append(s[1])
            results["vina"] = vina_scores
            results["cnn_affinity"] = cnn_scores

            # Calculate WL distances
            wl_dists = {}
            for m in ["qed", "logp", "sas", "lipinski", "vina", "cnn_affinity"]:
                if results[m] and m in test_dist and len(test_dist[m]) > 0:
                    gen_vals = np.array(results[m])
                    test_vals = test_dist[m].numpy()
                    wl = scipy.stats.wasserstein_distance(gen_vals, test_vals)
                    wl_dists[f"wl_{m}"] = wl
                else:
                    wl_dists[f"wl_{m}"] = np.nan

            # Add means
            for m in ["qed", "logp", "sas", "lipinski", "vina", "cnn_affinity"]:
                results[f"{m}_mean"] = np.mean(results[m]) if results[m] else np.nan

            results.update(wl_dists)
            split_results.append(results)

            # Save split results after each split
            import json

            with open(split_results_path, "w") as f:
                json.dump(split_results, f)
            print(f"Saved split {i+1} results to {split_results_path}")

        # Aggregate across splits
        final_stats = {}
        all_metrics = ["qed", "logp", "sas", "lipinski", "vina", "cnn_affinity"]
        for m in all_metrics:
            means = [
                r[f"{m}_mean"] for r in split_results if not np.isnan(r[f"{m}_mean"])
            ]
            final_stats[f"{m}_mean"], final_stats[f"{m}_std"] = aggregate_simple_stats(
                means
            )

            wls = [r[f"wl_{m}"] for r in split_results if not np.isnan(r[f"wl_{m}"])]
            final_stats[f"wl_{m}_mean"], final_stats[f"wl_{m}_std"] = (
                aggregate_simple_stats(wls)
            )

        # Save to CSV
        df = pd.DataFrame([final_stats])
        df.to_csv(os.path.join(folder_path, "evaluation_results.csv"), index=False)
        print(f"Saved results for {model_folder} to evaluation_results.csv")


if __name__ == "__main__":
    main()
