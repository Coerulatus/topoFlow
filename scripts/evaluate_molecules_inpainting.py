"""Evaluate generated molecules in specific folders against a target pocket, including scaffold checking and GNINA docking.

This script is designed for evaluating molecules generated for a specific target protein.
It calculates standard chemical metrics (QED, LogP, SAS, Lipinski) and performs GNINA
docking simulations to estimate binding affinity (Vina score and CNN affinity).
Additionally, it checks for the presence of a specific chemical scaffold and reports
the fraction of valid molecules that satisfy these structural constraints.

Parser Arguments:
    --base_dir (str): Directory containing subfolders of generated SDF files.
    --metrics (list): List of metrics to calculate (default: qed, logp, sas, lipinski, gnina).
    --scaffold_smiles (str): SMILES string of the scaffold to check for.
    --pocket_pdb_path (str): Path to the target protein's PDB file.
    --pocket_data_path (str): Path to save/load processed pocket PyG data.

Files Produced:
    - Markdown files (e.g., metrics_{folder_name}.md) containing summarized evaluation results.
"""

import argparse
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from rdkit import Chem

from topoflow.utils.metrics import (
    qed_scorer,
    logp_scorer,
    sas_scorer,
    lipinski_scorer,
    scorer_dict,
)
from topoflow.utils.molecules import (
    load_sdf_molecules_rdkit,
    pdb_to_data_object,
    mol_to_data,
)
from topoflow.utils.scaffold import check_scaffold_presence
import topoflow.utils.constants as constants

# --- GNINA/Pocket Configuration ---
ATOM_ENCODER = constants.dataset_params["crossdock_full"]["atom_encoder"]


def calculate_std_for_fraction(p, n):
    """Calculates the standard error (std dev) for a binomial fraction."""
    if n > 0 and 0 < p < 1:
        return np.sqrt(p * (1 - p) / n)
    return 0.0


def evaluate_folder(
    folder, results_path, scaffold_mol, metrics, pocket_data, atom_encoder
):
    """Evaluates all .sdf files in a given folder and saves a markdown report."""
    print(f"\n{'-'*50}\nEvaluating folder: {folder}")

    all_mols = []
    mols_idxs = []

    # --- Data Loading ---
    for file in os.listdir(folder):
        if file.endswith(".sdf"):
            file_path = os.path.join(folder, file)
            all_mols.extend(load_sdf_molecules_rdkit(file_path))
            try:
                idx = int(file.split("_")[-1].replace(".sdf", ""))
                mols_idxs.append(idx)
            except ValueError:
                mols_idxs.append(-1)

    total_mols = len(all_mols)
    if total_mols == 0:
        print(f"Skipping: No .sdf files found in {folder}")
        return

    # --- 1. Molecule Validation and Filtering ---
    valid_mols = []
    scaffold_present_counts = 0
    scaffold_present_counts_exact = 0
    exact_idxs = []
    no_match_idxs = []

    for idx, mol in enumerate(all_mols):
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
                valid_mols.append(mol)

                if check_scaffold_presence(mol, scaffold_mol):
                    scaffold_present_counts += 1
                else:
                    no_match_idxs.append(mols_idxs[idx])

                if check_scaffold_presence(mol, scaffold_mol, relaxed=False):
                    scaffold_present_counts_exact += 1
                    exact_idxs.append(mols_idxs[idx])

            except Exception:
                continue

    num_valid_mols = len(valid_mols)
    validity_fraction = num_valid_mols / total_mols if total_mols > 0 else 0

    scaffold_fraction = scaffold_present_counts / total_mols if total_mols > 0 else 0.0
    scaffold_fraction_exact = (
        scaffold_present_counts_exact / total_mols if total_mols > 0 else 0.0
    )

    validity_std = calculate_std_for_fraction(validity_fraction, total_mols)
    scaffold_std = calculate_std_for_fraction(scaffold_fraction, total_mols)
    scaffold_exact_std = calculate_std_for_fraction(scaffold_fraction_exact, total_mols)

    # --- 2. Metric Calculation ---
    results = {}

    for metric in metrics:
        if metric not in scorer_dict:
            print(f"Warning: Unsupported metric '{metric}'. Skipping.")
            continue

        scorer_func = scorer_dict[metric]

        # --- GNINA Specific Handling ---
        if metric == "gnina":
            vina_scores = []
            cnn_scores = []

            print("  -> Running GNINA docking evaluations...")
            for idxmol, mol in enumerate(tqdm(valid_mols, desc="GNINA Docking")):
                try:
                    # 1. Convert RDKit Mol back to tensors
                    coords, one_hot, batch = mol_to_data(mol, atom_encoder)

                    # 2. Construct a combined PyTorch Geometric Data object
                    combined_data = Data(x_lig=coords, h_lig=one_hot, batch_lig=batch)

                    # 3. Inherit all pocket properties from the pre-loaded pocket
                    if pocket_data is not None:
                        for key, value in pocket_data:
                            if "pocket" in key:
                                setattr(combined_data, key, value)

                    # 4. Score it!
                    score = scorer_func(combined_data)

                    # Ignored failed runs returning [0, 0]
                    if score != [0, 0]:
                        vina_scores.append(float(score[0]))
                        cnn_scores.append(float(score[1]))

                except Exception as e:
                    # Skip if gnina fails for this specific molecule
                    continue

            results["vina"] = vina_scores
            results["cnn_affinity"] = cnn_scores
        # --- Standard Metrics (QED, LogP, etc.) ---
        else:
            collected_scores = []
            for mol in valid_mols:
                try:
                    s = scorer_func(mol)
                    if s is not None:
                        collected_scores.append(
                            s.item() if hasattr(s, "item") else float(s)
                        )
                except Exception:
                    continue
            results[metric] = collected_scores

    # --- 3. Results Compilation ---
    res = pd.DataFrame(columns=["Metric", "Value (mean ± std)"])

    # Add Fractions
    validity_value = f"{validity_fraction:.4f} ± {validity_std:.4f}"
    validity_raw_counts = f"({num_valid_mols}/{total_mols})"
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "Metric": ["Validity Fraction"],
                    "Value (mean ± std)": [f"{validity_value} {validity_raw_counts}"],
                }
            ),
        ],
        ignore_index=True,
    )

    scaffold_value = f"{scaffold_fraction:.4f} ± {scaffold_std:.4f}"
    scaffold_raw_counts = f"({scaffold_present_counts}/{total_mols})"
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "Metric": ["Scaffold Presence"],
                    "Value (mean ± std)": [f"{scaffold_value} {scaffold_raw_counts}"],
                }
            ),
        ],
        ignore_index=True,
    )

    scaffold_exact_value = f"{scaffold_fraction_exact:.4f} ± {scaffold_exact_std:.4f}"
    scaffold_exact_raw_counts = f"({scaffold_present_counts_exact}/{total_mols})"
    res = pd.concat(
        [
            res,
            pd.DataFrame(
                {
                    "Metric": ["Scaffold Presence (Exact)"],
                    "Value (mean ± std)": [
                        f"{scaffold_exact_value} {scaffold_exact_raw_counts}"
                    ],
                }
            ),
        ],
        ignore_index=True,
    )

    # Add Computed Metrics
    for metric, scores in results.items():
        if scores:
            avg_score = np.mean(scores)
            std_score = np.std(scores)
            res = pd.concat(
                [
                    res,
                    pd.DataFrame(
                        {
                            "Metric": [metric],
                            "Value (mean ± std)": [
                                f"{avg_score:.4f} ± {std_score:.4f}"
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )
        else:
            res = pd.concat(
                [
                    res,
                    pd.DataFrame(
                        {
                            "Metric": [metric],
                            "Value (mean ± std)": [
                                "N/A (No valid mols/Failed computation)"
                            ],
                        }
                    ),
                ],
                ignore_index=True,
            )

    # Save
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    res.to_markdown(results_path, index=False)

    print(f"Metrics saved to: {results_path}")
    print(f"Total molecules loaded: {total_mols}")
    print(f"Valid molecules processed: {num_valid_mols}")
    print(f"Molecules containing scaffold: {scaffold_present_counts}")
    print(f"Molecules containing scaffold (Exact): {scaffold_present_counts_exact}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate generated molecules.")
    parser.add_argument(
        "--base_dir",
        type=str,
        default="./outputs/generated_ligands_wwp2_hect/",
        help="Base directory containing folders of .sdf files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["qed", "logp", "sas", "lipinski", "gnina"],
        help="Metrics to calculate.",
    )
    parser.add_argument(
        "--scaffold_smiles",
        type=str,
        default="c1ccc2c(c1)[nH]cc2",
        help="SMILES of the scaffold to check for presence.",
    )
    parser.add_argument(
        "--pocket_pdb_path",
        type=str,
        default="./data/target_protein/wwp2_hect.pdb",
        help="Path to the pocket PDB file.",
    )
    parser.add_argument(
        "--pocket_data_path",
        type=str,
        default="./data/target_protein/wwp2_hect_data.pt",
        help="Path to save/load the processed pocket data.",
    )

    args = parser.parse_args()

    scaffold_mol = Chem.MolFromSmiles(args.scaffold_smiles)
    if scaffold_mol is None:
        raise ValueError(
            f"Failed to create RDKit molecule from scaffold SMILES: {args.scaffold_smiles}"
        )

    if not os.path.exists(args.base_dir):
        raise FileNotFoundError(
            f"The specified base folder does not exist: {args.base_dir}"
        )

    # --- Load Pocket Data Once for GNINA ---
    pocket_data = None
    if "gnina" in args.metrics:
        print("Loading pocket data for GNINA docking...")
        if not os.path.exists(args.pocket_pdb_path):
            print(
                f"Warning: Pocket file not found at {args.pocket_pdb_path}. GNINA will likely fail."
            )
        else:
            pocket_data = pdb_to_data_object(
                pdb_file=args.pocket_pdb_path,
                atom_encoder=ATOM_ENCODER,
                save_path=args.pocket_data_path,
                normalization_path=None,
                center_pocket=False,
                force=True,  # Set to True if you need to re-parse the PDB
            )

    for item in os.listdir(args.base_dir):
        folder_path = os.path.join(args.base_dir, item)

        if os.path.isdir(folder_path):
            results_file = os.path.join(args.base_dir, f"metrics_{item}.md")
            evaluate_folder(
                folder=folder_path,
                results_path=results_file,
                scaffold_mol=scaffold_mol,
                metrics=args.metrics,
                pocket_data=pocket_data,
                atom_encoder=ATOM_ENCODER,
            )

    print(f"\n{'-'*50}\nAll evaluations completed successfully!")
