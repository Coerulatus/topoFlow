"""Calculate the novelty and uniqueness of generated molecules relative to the training set.

This script assesses the generative quality of models by comparing the SMILES strings of
generated molecules against a reference list of SMILES from the training set. It calculates
"Novelty" (percentage of generated molecules not present in training) and "Uniqueness"
(percentage of unique SMILES among all generated molecules). The process is parallelized
using RDKit for SMILES generation.

Parser Arguments:
    --train_smiles (str): Path to the file containing training set SMILES.
    --test_dir (str): Directory containing generated molecules (SDFs).
    --output (str): Path to the output CSV file.
    --workers (int): Number of parallel workers for SMILES conversion.

Files Produced:
    - A CSV file (e.g., novelty_results.csv) containing novelty and uniqueness metrics for each model.
    - Printed markdown table in the console.
"""

import os
import argparse
import pandas as pd
from rdkit import Chem
from tqdm import tqdm
import torch
from concurrent.futures import ProcessPoolExecutor
from topoflow.utils.molecules import load_sdf_molecules_rdkit
from topoflow.utils.metrics import compute_novelty


def mol_to_smiles_task(file_path):
    """Worker task to read an SDF and return canonical SMILES."""
    try:
        mols = load_sdf_molecules_rdkit(file_path)
        smiles_list = []
        for mol in mols:
            if mol is not None:
                try:
                    # Canonical non-isomeric SMILES for fair comparison
                    s = Chem.MolToSmiles(mol, isomericSmiles=False)
                    if s:
                        smiles_list.append(s)
                except:
                    continue
        return smiles_list
    except Exception as e:
        return []


def main():
    """Main function to calculate novelty and uniqueness of generated molecules."""
    parser = argparse.ArgumentParser(
        description="Efficiently calculate novelty of generated molecules."
    )
    parser.add_argument(
        "--train_smiles",
        type=str,
        default="./data/CrossDocked/processed/train_crossdocked_pyg_smiles.txt",
        help="Path to training SMILES.",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="outputs/test_molecules/",
        help="Directory with model subfolders.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/novelty_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers.",
    )
    args = parser.parse_args()

    # 1. Load training SMILES into a set for O(1) lookup
    print(f"Loading training SMILES from {args.train_smiles}...")
    with open(args.train_smiles, "r") as f:
        # Using set for maximal efficiency during novelty check
        train_smiles = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(train_smiles)} unique training SMILES from ~100k records.")

    # 2. Identify models
    model_dirs = sorted(
        [
            d
            for d in os.listdir(args.test_dir)
            if os.path.isdir(os.path.join(args.test_dir, d))
        ]
    )
    model_dirs = [
        d for d in model_dirs if os.path.exists(os.path.join(args.test_dir, d, "sdfs"))
    ]

    if not model_dirs:
        print("No folders with 'sdfs' found.")
        return

    results = []

    for model in model_dirs:
        sdfs_dir = os.path.join(args.test_dir, model, "sdfs")
        sdf_files = [
            os.path.join(sdfs_dir, f)
            for f in os.listdir(sdfs_dir)
            if f.endswith(".sdf")
        ]

        print(f"\nEvaluating {model} ({len(sdf_files)} SDF files)...")

        # 3. Parallel SMILES generation (the bottleneck)
        generated_smiles = []
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            # Using imap to keep track of progress
            for chunk in tqdm(
                executor.map(mol_to_smiles_task, sdf_files),
                total=len(sdf_files),
                desc="SMILES",
            ):
                generated_smiles.extend(chunk)

        if not generated_smiles:
            continue

        # 4. Compute metrics
        novelty, novel_list = compute_novelty(generated_smiles, train_smiles)
        unique_gen = set(generated_smiles)
        uniqueness = len(unique_gen) / len(generated_smiles)

        print(
            f"  Total: {len(generated_smiles)} | Unique: {len(unique_gen)} ({uniqueness:.2%}) | Novel: {len(novel_list)} ({novelty:.2%})"
        )

        results.append(
            {
                "Model": model,
                "Novelty": novelty,
                "Uniqueness": uniqueness,
                "Total": len(generated_smiles),
                "Unique": len(unique_gen),
                "Novel": len(novel_list),
            }
        )

    # 5. Export results
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        print("\n" + df.to_markdown(index=False))


if __name__ == "__main__":
    main()
