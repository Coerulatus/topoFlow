"""Script to evaluate generated molecules against various metrics and check for scaffold presence."""

import os
import numpy as np
import pandas as pd
from rdkit import Chem
from topoflow.utils.metrics import (
    qed_scorer,
    logp_scorer,
    sas_scorer,
    lipinski_scorer,
)
from topoflow.utils.molecules import load_sdf_molecules_rdkit
from topoflow.utils.scaffold import check_scaffold_presence  # Your custom function

folder = "./outputs/generated_ligands_wwp2_c2/"
results_path = "./results/molecule_evaluation/generated_ligands_wwp2_c2_metrics.md"
metrics = ["qed", "logp", "sas"]

# --- Define Scaffold ---
# IMPORTANT: Update this SMILES string if your base scaffold is different!
SCAFFOLD_SMILES = "c1ccc2c(c1)[nH]cc2"  # Example: Indole
scaffold_mol = Chem.MolFromSmiles(SCAFFOLD_SMILES)
if scaffold_mol is None:
    raise ValueError(
        f"Failed to create RDKit molecule from scaffold SMILES: {SCAFFOLD_SMILES}"
    )


# --- Data Loading ---
if not os.path.exists(folder):
    raise FileNotFoundError(f"The specified folder does not exist: {folder}")

all_mols = []
mols_idxs = []
for file in os.listdir(folder):
    if file.endswith(".sdf"):
        file_path = os.path.join(folder, file)
        all_mols.extend(load_sdf_molecules_rdkit(file_path))
        mols_idxs.append(int(file.split("_")[-1].replace(".sdf", "")))

total_mols = len(all_mols)

# --- 1. Molecule Validation and Filtering ---
valid_mols = []
# New list to store results of the scaffold check
scaffold_present_counts = 0
scaffold_present_counts_exact = 0
exact_idxs = []
no_match_idxs = []

for idx, mol in enumerate(all_mols):
    if mol is not None:
        try:
            # Sanitize the molecule. If this fails, it's considered invalid.
            Chem.SanitizeMol(mol)
            valid_mols.append(mol)

            # --- Check Scaffold Presence ---
            if check_scaffold_presence(mol, scaffold_mol):
                scaffold_present_counts += 1
            else:
                no_match_idxs.append(mols_idxs[idx])
            if check_scaffold_presence(mol, scaffold_mol, relaxed=False):
                scaffold_present_counts_exact += 1
                exact_idxs.append(mols_idxs[idx])

        except Exception:
            # Skip invalid mols
            continue

num_valid_mols = len(valid_mols)
validity_fraction = num_valid_mols / total_mols if total_mols > 0 else 0

# Calculate the fraction of valid molecules containing the scaffold
scaffold_fraction = scaffold_present_counts / total_mols if total_mols > 0 else 0.0
scaffold_fraction_exact = (
    scaffold_present_counts_exact / total_mols if total_mols > 0 else 0.0
)

# The rest of the metrics are calculated only on the valid_mols list
mols = valid_mols

# --- Metric Calculation and Conversion ---
results = {}
scorer_map = {
    "qed": qed_scorer,
    "logp": logp_scorer,
    "sas": sas_scorer,
    "lipinski": lipinski_scorer,
}

for metric in metrics:
    if metric not in scorer_map:
        raise ValueError(f"Unsupported metric: {metric}")

    scorer_func = scorer_map[metric]

    # Calculate scores on the filtered, valid molecules
    raw_scores = [scorer_func(mol) for mol in mols]

    # Ensure scores are collected as standard Python numbers by using .item()
    collected_scores = []
    for s in raw_scores:
        if s is not None:
            # Ensure proper type conversion for PyTorch/Numpy tensors to float
            try:
                # Use .item() for Tensors, otherwise use float()
                collected_scores.append(s.item() if hasattr(s, "item") else float(s))
            except Exception:
                continue  # Skip scores that fail conversion

    results[metric] = collected_scores


# --- Results Compilation ---
res = pd.DataFrame(columns=["Metric", "Value (mean ± std)"])

# Add Validity Fraction
validity_mean_std = f"{validity_fraction:.4f} ({num_valid_mols}/{total_mols})"
validity_row = pd.DataFrame(
    {"Metric": ["Validity Fraction"], "Value (mean ± std)": [validity_mean_std]}
)
res = pd.concat([res, validity_row], ignore_index=True)

# --- Add Scaffold Presence Fraction ---
scaffold_mean_std = f"{scaffold_fraction:.4f} ({scaffold_present_counts}/{total_mols})"
scaffold_row = pd.DataFrame(
    {"Metric": ["Scaffold Presence"], "Value (mean ± std)": [scaffold_mean_std]}
)
res = pd.concat([res, scaffold_row], ignore_index=True)

scaffold_exact_mean_std = (
    f"{scaffold_fraction_exact:.4f} ({scaffold_present_counts_exact}/{total_mols})"
)
scaffold_exact_row = pd.DataFrame(
    {
        "Metric": ["Scaffold Presence (Exact)"],
        "Value (mean ± std)": [scaffold_exact_mean_std],
    }
)
res = pd.concat([res, scaffold_exact_row], ignore_index=True)


for metric, scores in results.items():
    if scores:
        avg_score = np.mean(scores)
        std_score = np.std(scores)

        new_row = pd.DataFrame(
            {
                "Metric": [metric],
                "Value (mean ± std)": [f"{avg_score:.4f} ± {std_score:.4f}"],
            }
        )
        res = pd.concat([res, new_row], ignore_index=True)
    else:
        new_row = pd.DataFrame(
            {"Metric": [metric], "Value (mean ± std)": ["N/A (No valid mols)"]}
        )
        res = pd.concat([res, new_row], ignore_index=True)

# Ensure output directory exists before writing
if not os.path.exists(os.path.dirname(results_path)):
    os.makedirs(os.path.dirname(results_path))

res.to_markdown(results_path, index=False)

print(f"\nMetrics saved to: {results_path}")
print(f"Total molecules loaded: {total_mols}")
print(f"Valid molecules processed: {num_valid_mols}")
print(f"Molecules containing scaffold: {scaffold_present_counts}")
print(f"Molecules containing scaffold (Exact): {scaffold_present_counts_exact}")
print(f"Exact scaffold match indices in valid molecules: {sorted(exact_idxs)}")
print(f"No scaffold match indices in valid molecules: {sorted(no_match_idxs)}")
