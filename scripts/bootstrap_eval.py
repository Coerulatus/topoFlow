"""Script to evaluate molecules using bootstrap methodology, calculating validity, connectivity, uniqueness, and various chemical metrics."""

import os
import torch
import math
import numpy as np
import argparse
import pandas as pd
import scipy.stats
import concurrent.futures
import multiprocessing as mp
from functools import partial
from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data

from rdkit import Chem
from topoflow.utils.metrics import scorer_dict, compute_connectivity, compute_uniqueness
import topoflow.utils.constants as constants
from topoflow.utils.molecules import (
    data_to_rdmol,
    mol_to_data,
    load_sdf_molecules_rdkit,
)
from topoflow.data.datasets import CrossDocked


def round_normal(value, decimals):
    """Rounds a value to a specified number of decimal places using standard rounding."""
    return f"{value:.{decimals}f}"


def round_excess(value, decimals):
    """Rounds a value to a specified number of decimal places, rounding up to avoid underreporting variability."""
    factor = 10**decimals
    rounded_val = math.ceil(value * factor) / factor
    return f"{rounded_val:.{decimals}f}"


def aggregate_simple_stats(values):
    """Calculates the mean and standard deviation for a list of scalar values."""
    if not values:
        return np.nan, np.nan
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


# --- NEW: Isolated scoring function for parallel workers ---
def process_single_molecule(data, atom_decoder, norm_params=None):
    """Worker function to score a single molecule."""
    # Skip if already scored
    if hasattr(data, "is_scored") and data.is_scored:
        return data, False

    data.qed = float("nan")
    data.logp = float("nan")
    data.sas = float("nan")
    data.lipinski = float("nan")
    data.vina = float("nan")
    data.cnn_affinity = float("nan")
    data.is_valid = False

    mol = data_to_rdmol(data, atom_decoder=atom_decoder, sanitize=True)

    if mol is not None:
        data.is_valid = True
        for m in ["qed", "logp", "sas", "lipinski"]:
            scorer = scorer_dict[m]
            try:
                score = scorer(mol)
                if score is not None:
                    val = score.item() if hasattr(score, "item") else float(score)
                    setattr(data, m, val)
            except Exception:
                pass

    try:
        from topoflow.utils.metrics import docker

        # Pass both norm_params and atom_decoder to ensure correct coordinate space and elements
        gnina_score = docker._run_gnina(
            data, norm_params=norm_params, atom_decoder=atom_decoder
        )

        if gnina_score and len(gnina_score) == 2:
            data.vina = float(gnina_score[0])
            data.cnn_affinity = float(gnina_score[1])
    except Exception as e:
        print(
            f"Error during GNINA scoring for {getattr(data, 'mol_id', 'unknown')}: {e}"
        )

    data.is_scored = True
    return data, True


def main():
    """Main function to perform bootstrap evaluation of generated molecules."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/CrossDocked/")
    parser.add_argument(
        "--output_base_dir", type=str, default="./outputs/test_molecules_temp/"
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of .pt files even if they exist.",
    )
    parser.add_argument(
        "--n_groups",
        type=int,
        default=5,
        help="Number of groups to create for bootstrapping.",
    )
    parser.add_argument(
        "--test_distribution_path",
        type=str,
        default="./data/CrossDocked/processed/test_distribution.npy",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="Number of parallel workers for scoring (tune based on GPU VRAM).",
    )
    args = parser.parse_args()

    # Load Normalization Params
    norm_params = None
    norm_path = os.path.join(args.data_dir, "processed", "normalization_params.npz")
    if os.path.exists(norm_path):
        print(f"Loading normalization parameters from {norm_path}")
        norm_data = np.load(norm_path, allow_pickle=True)
        norm_params = {"mean": norm_data["mean"], "std": norm_data["std"]}
    else:
        print(
            f"Warning: Normalization file not found at {norm_path}. Scores may be inaccurate."
        )

    data_name = "crossdock_full"
    atom_encoder = constants.dataset_params[data_name]["atom_encoder"]
    atom_decoder = constants.dataset_params[data_name]["atom_decoder"]

    # Load test distribution
    if os.path.exists(args.test_distribution_path):
        test_distribution = np.load(
            args.test_distribution_path, allow_pickle=True
        ).item()
        # Reference size for grouping
        ref_size = len(next(iter(test_distribution.values())))
        print(
            f"Loaded test distribution from {args.test_distribution_path}. Group size: {ref_size}"
        )
    else:
        print(
            f"WARNING: Test distribution not found at {args.test_distribution_path}. W1 distance will not be calculated."
        )
        test_distribution = None
        ref_size = 0

    model_folders = [
        d
        for d in os.listdir(args.output_base_dir)
        if os.path.isdir(os.path.join(args.output_base_dir, d))
    ]

    print(f"Loading test dataset from {args.data_dir} to resolve pocket mappings...")
    test_dataset = CrossDocked(root=args.data_dir, split="test")
    test_npz_path = os.path.join(args.data_dir, "processed", "test.npz")
    if os.path.exists(test_npz_path):
        test_npz = np.load(test_npz_path, allow_pickle=True)
        test_names = test_npz["names"]
    else:
        test_names = []

    # AGGRESSIVE NORMALIZATION: Remove all _ and - and make lowercase for bulletproof matching
    pocket_stem_to_idx = {}
    for i, name in enumerate(test_names):
        pocket_part = name.split(".pdb")[0]
        clean_name = (
            os.path.basename(pocket_part).replace("_", "").replace("-", "").lower()
        )
        pocket_stem_to_idx[clean_name] = i

    for model_folder in model_folders:
        folder_path = os.path.join(args.output_base_dir, model_folder)
        splits_path = os.path.join(folder_path, "pyg_splits.pt")
        out_path = os.path.join(folder_path, "pyg_scored_flat.pt")
        result_path = os.path.join(folder_path, f"result_{model_folder}.txt")

        flat_mols = []

        # 1. Check if we already have scored molecules (potentially partially scored)
        if os.path.exists(out_path) and not args.force_rebuild:
            print(
                f"\nLoading existing scored/partially-scored molecules for {model_folder}..."
            )
            flat_mols = torch.load(out_path, weights_only=False)
            print(f"Loaded {len(flat_mols)} molecules from existing state.")
        else:
            # Check if we need to build/rebuild the splits
            if args.force_rebuild or not os.path.exists(splits_path):
                print(f"\n[!] Building PyG data for {model_folder}...")
                sdfs_dir = os.path.join(folder_path, "sdfs")

                if not os.path.exists(sdfs_dir):
                    print(f"Skipping {model_folder}, sdfs/ directory not found.")
                    continue

                # Read SDFs directly from the directory, ignoring mapping.txt
                sdf_files = [f for f in os.listdir(sdfs_dir) if f.endswith(".sdf")]

                protein_molecules = {}
                # Create a directory for individual SDFs to avoid reconstruction issues in scorer
                indiv_sdf_dir = os.path.join(folder_path, "individual_sdfs")
                os.makedirs(indiv_sdf_dir, exist_ok=True)

                # Based on check, the actual files are in data/CrossDocked/raw/processed_crossdock_noH_full_temp/test/
                protein_root = os.path.join(
                    args.data_dir, "raw", "processed_crossdock_noH_full_temp", "test"
                )

                # Fallback check
                if not os.path.exists(protein_root):
                    # Try another likely candidate
                    protein_root = os.path.join(args.data_dir, "processed", "test")

                for sdf_name in tqdm(
                    sdf_files, desc=f"Converting SDFs to PyG for {model_folder}"
                ):
                    sdf_path = os.path.join(sdfs_dir, sdf_name)

                    # Normalize the SDF name to match the test set dictionary
                    raw_stem = sdf_name.replace(".sdf", "")
                    clean_stem = raw_stem.replace("_", "").replace("-", "").lower()

                    if clean_stem not in pocket_stem_to_idx:
                        continue

                    pocket_idx = pocket_stem_to_idx[clean_stem]
                    pocket_data = test_dataset[pocket_idx]

                    # Get the original PDB path from test_names
                    combined_name = test_names[pocket_idx]

                    # The name in test.npz is usually pocket_ligand (e.g. pocket.pdb_ligand.sdf)
                    # We want the first part which is the PDB filename
                    if ".pdb" in combined_name:
                        pdb_filename = combined_name.split(".pdb")[0] + ".pdb"
                    else:
                        # Fallback if no .pdb extension found (unlikely)
                        pdb_filename = combined_name.split("_")[0] + ".pdb"

                    # Ensure it's just the filename if it had paths
                    pdb_filename = os.path.basename(pdb_filename)

                    actual_pdb_path = os.path.join(protein_root, pdb_filename)
                    if not os.path.exists(actual_pdb_path):
                        # Try mapping underscores to hyphens which is common in CrossDocked files
                        pdb_alt = pdb_filename.replace("_", "-")
                        actual_pdb_path = os.path.join(protein_root, pdb_alt)

                    if not os.path.exists(actual_pdb_path):
                        print(
                            f"Warning: Could not find protein PDB for {raw_stem} at {actual_pdb_path}"
                        )
                        # Final search: try to find it in protein_root via glob or raw_stem
                        import glob

                        search_res = glob.glob(
                            os.path.join(protein_root, f"*{raw_stem}*pocket10.pdb")
                        )
                        if search_res:
                            actual_pdb_path = search_res[0]
                        else:
                            continue

                    mols = load_sdf_molecules_rdkit(sdf_path)
                    pyg_mols = []

                    for idx, mol in enumerate(mols):
                        if mol is None:
                            continue
                        try:
                            # Save individual molecule to its own SDF file
                            mol_id = f"{raw_stem}_{idx}"
                            indiv_sdf_path = os.path.abspath(
                                os.path.join(indiv_sdf_dir, f"{mol_id}.sdf")
                            )

                            if not os.path.exists(indiv_sdf_path):
                                writer = Chem.SDWriter(indiv_sdf_path)
                                writer.write(mol)
                                writer.close()

                            x_lig, h_lig, _ = mol_to_data(mol, atom_encoder)
                            data = Data(
                                x_lig=x_lig,
                                h_lig=h_lig,
                                batch_lig=torch.zeros(x_lig.size(0), dtype=torch.long),
                                x_pocket=pocket_data.x_pocket,
                                h_pocket=pocket_data.h_pocket,
                                batch_pocket=torch.zeros(
                                    pocket_data.x_pocket.size(0), dtype=torch.long
                                ),
                                mol_id=mol_id,
                                pdb_path=os.path.abspath(actual_pdb_path),
                                ligand_sdf_path=indiv_sdf_path,
                            )
                            pyg_mols.append(data)
                        except Exception as e:
                            print(
                                f"\nFailed to convert molecule {idx} in {sdf_name}: {e}"
                            )

                    protein_molecules[raw_stem] = pyg_mols

                # Create a flat list of all successfully converted molecules
                flat_mols = []
                for pocket_stem, mols in protein_molecules.items():
                    flat_mols.extend(mols)

                # Save PyG flat list
                torch.save(flat_mols, splits_path)
                print(
                    f"Successfully converted and saved {len(flat_mols)} PyG molecules to {splits_path}"
                )

            else:
                print(f"\nLoading existing splits for model: {model_folder}")
                data_loaded = torch.load(splits_path, weights_only=False)
                if len(data_loaded) > 0 and isinstance(data_loaded[0], list):
                    print("Old split format detected. Flattening...")
                    flat_mols = [
                        mol for split in data_loaded for mol in split if mol is not None
                    ]
                else:
                    flat_mols = data_loaded
                print(f"Loaded a total of {len(flat_mols)} molecules to begin scoring.")

        if len(flat_mols) == 0:
            print(f"No molecules to score for {model_folder}. Skipping scoring phase.")
            continue

        # --- NEW: PARALLEL SCORING PHASE ---
        unscored_indices = [
            i
            for i, d in enumerate(flat_mols)
            if not (hasattr(d, "is_scored") and d.is_scored)
        ]

        needs_save = False

        if unscored_indices:
            print(
                f"Parallelizing {len(unscored_indices)} scoring tasks across {args.num_workers} workers..."
            )

            # Setup worker function with fixed parameters
            worker_func = partial(
                process_single_molecule,
                atom_decoder=atom_decoder,
                norm_params=norm_params,
            )

            # Using 'spawn' context is crucial to prevent PyTorch/CUDA hangs in subprocesses
            ctx = mp.get_context("spawn")

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_workers, mp_context=ctx
            ) as executor:
                # Submit all unscored tasks to the executor mapped to their original index
                future_to_idx = {
                    executor.submit(worker_func, flat_mols[idx]): idx
                    for idx in unscored_indices
                }

                completed = 0
                for future in tqdm(
                    concurrent.futures.as_completed(future_to_idx),
                    total=len(unscored_indices),
                    desc=f"Scoring {model_folder}",
                ):
                    idx = future_to_idx[future]
                    try:
                        scored_data, was_scored = future.result()
                        # Update the array in place with the processed copy
                        flat_mols[idx] = scored_data

                        if was_scored:
                            needs_save = True

                        completed += 1

                        # Checkpoint save every 100 molecules
                        if completed % 100 == 0:
                            torch.save(flat_mols, out_path)

                    except Exception as e:
                        print(f"\nTask failed for molecule at index {idx}: {e}")

        # Final save if any new molecules were processed
        if needs_save:
            torch.save(flat_mols, out_path)
            print(
                f"Successfully finished and saved {len(flat_mols)} scored molecules to {out_path}\n"
            )
        else:
            print(f"All molecules were already scored. Proceeding to evaluation.\n")

        # 2. Evaluation Phase (Grouping)
        if len(flat_mols) == 0:
            continue

        print(
            f"Starting grouped evaluation for {model_folder} with {args.n_groups} groups..."
        )

        all_group_metrics = {
            "val/validity": [],
            "val/connectivity": [],
            "val/uniqueness": [],
        }

        metrics_to_eval = [
            "qed",
            "logp",
            "sas",
            "lipinski",
            "vina",
            "cnn_affinity",
            "mol_size",
        ]
        for m in metrics_to_eval:
            all_group_metrics[f"val/{m}_mean"] = []
            if test_distribution is not None:
                all_group_metrics[f"val/{m}_w1_distance"] = []

        if ref_size == 0:
            ref_size = len(flat_mols) // args.n_groups
            if ref_size == 0:
                ref_size = len(flat_mols)

        for g in range(args.n_groups):
            # Sample a group
            if len(flat_mols) >= ref_size:
                indices = np.random.choice(len(flat_mols), ref_size, replace=False)
            else:
                indices = np.random.choice(len(flat_mols), ref_size, replace=True)

            group = [flat_mols[i] for i in indices]

            # Validity
            valid_mols_pyg = [d for d in group if getattr(d, "is_valid", False)]
            all_group_metrics["val/validity"].append(len(valid_mols_pyg) / len(group))

            if len(valid_mols_pyg) > 0:
                # Convert valid to RDKit for connectivity
                valid_rdmols = [
                    data_to_rdmol(d, atom_decoder=atom_decoder, sanitize=True)
                    for d in valid_mols_pyg
                ]
                valid_rdmols = [m for m in valid_rdmols if m is not None]

                conn_score, _, conn_smiles, _ = compute_connectivity(valid_rdmols)
                all_group_metrics["val/connectivity"].append(conn_score)

                uniq_score, _ = compute_uniqueness(conn_smiles)
                all_group_metrics["val/uniqueness"].append(uniq_score)

                for m in metrics_to_eval:
                    vals = []
                    for d in valid_mols_pyg:
                        val = getattr(d, m, float("nan"))
                        # Calculate mol_size on the fly if missing
                        if (val is None or np.isnan(val)) and m == "mol_size":
                            if hasattr(d, "x_lig"):
                                val = float(d.x_lig.size(0))
                            elif hasattr(d, "h_lig"):
                                val = float(d.h_lig.size(0))

                        if val is not None and not np.isnan(val):
                            vals.append(val)

                    if vals:
                        all_group_metrics[f"val/{m}_mean"].append(np.mean(vals))
                        if test_distribution is not None and m in test_distribution:
                            test_vals = np.array(test_distribution[m]).flatten()
                            w1_dist = scipy.stats.wasserstein_distance(vals, test_vals)
                            all_group_metrics[f"val/{m}_w1_distance"].append(w1_dist)
            else:
                all_group_metrics["val/connectivity"].append(0.0)
                all_group_metrics["val/uniqueness"].append(0.0)
                for m in metrics_to_eval:
                    all_group_metrics[f"val/{m}_mean"].append(np.nan)
                    if test_distribution is not None:
                        all_group_metrics[f"val/{m}_w1_distance"].append(np.nan)

        # Aggregate Results
        final_results = []
        for metric_name, values in sorted(all_group_metrics.items()):
            mean, std = aggregate_simple_stats(values)
            if not np.isnan(mean):
                mean_str = round_normal(mean, 2)
                std_str = round_excess(std, 2)
                final_results.append([metric_name, f"{mean_str} ± {std_str}"])

        # Add generation time if available
        timing_path = os.path.join(folder_path, "generation_time.txt")
        if os.path.exists(timing_path):
            try:
                with open(timing_path, "r") as f:
                    for line in f:
                        if "seconds_per_valid_mol:" in line:
                            s_per_valid = float(line.split(":")[1].strip())
                            final_results.append(
                                [
                                    "val/generation_time (valid s/mol)",
                                    f"{s_per_valid:.4f}",
                                ]
                            )
                        elif "seconds_per_attempted_mol:" in line:
                            s_per_attempted = float(line.split(":")[1].strip())
                            final_results.append(
                                [
                                    "val/generation_time (attempted s/mol)",
                                    f"{s_per_attempted:.4f}",
                                ]
                            )
            except Exception:
                pass

        df = pd.DataFrame(final_results, columns=["Metric", "Value (mean ± std)"])

        # Save to result file
        with open(result_path, "w") as f:
            f.write("# Validation Results\n\n")
            f.write(df.to_markdown(index=False))

        print(f"Results for {model_folder} saved to {result_path}")


if __name__ == "__main__":
    main()
