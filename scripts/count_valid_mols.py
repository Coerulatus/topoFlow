"""Validate and count generated molecules in SDF files based on structural integrity and atom types.

This script provides a strict validation of generated molecules stored in SDF format.
It checks each molecule for: 1) Presence of "hallucinated" or unauthorized atom types
not found in the training data, and 2) Structural/geometric validity via RDKit's
SanitizeMol (e.g., checking for valid valence). It outputs a summary report with the
total count, strict validity rate, and reasons for rejection.

Parser Arguments:
    (None, the script uses hardcoded target directories in the main block).

Files Produced:
    - None (outputs summary statistics directly to the console).
"""

import os
from rdkit import Chem
from tqdm import tqdm

# Import the atom encoder to check for valid symbols
import topoflow.utils.constants as constants


def count_and_validate_mols(sdf_dir, allowed_atoms):
    """Count and validate molecules in a given SDF directory."""
    if not os.path.exists(sdf_dir):
        print(f"Directory not found: {sdf_dir}")
        return

    sdf_files = [f for f in os.listdir(sdf_dir) if f.endswith(".sdf")]
    print(f"Found {len(sdf_files)} SDF files in {sdf_dir}")

    total_mols = 0
    valid_mols = 0
    dummy_mols = 0  # Track how many molecules have hallucinated atoms
    sanitization_fails = 0  # Track structural failures

    for sdf_file in tqdm(
        sdf_files, desc=f"Validating {os.path.basename(os.path.dirname(sdf_dir))}"
    ):
        sdf_path = os.path.join(sdf_dir, sdf_file)
        # sanitize=False in supplier so we can manually check and count
        suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)

        for mol in suppl:
            if mol is None:
                continue

            total_mols += 1

            # --- STRICT CHECK: Filter out hallucinated/dummy atoms ---
            has_dummy_atom = False
            for atom in mol.GetAtoms():
                if atom.GetSymbol() not in allowed_atoms:
                    has_dummy_atom = True
                    break

            if has_dummy_atom:
                dummy_mols += 1
                continue  # Skip sanitization, this is invalid for our pipeline
            # ---------------------------------------------------------

            try:
                # Attempt to sanitize to check geometric validity
                Chem.SanitizeMol(mol)
                valid_mols += 1
            except Exception:
                # Sanitization failed (e.g., bad valence)
                sanitization_fails += 1

    print("\n" + "=" * 40)
    print(f"Results for: {sdf_dir}")
    print(f"Total Generated:       {total_mols}")
    print(f"Strict Valid:          {valid_mols}")
    print("-" * 40)
    print(f"Rejected (Dummy Atoms): {dummy_mols}")
    print(f"Rejected (Bad Valence): {sanitization_fails}")
    if total_mols > 0:
        print(f"Strict Validity Rate:   {valid_mols/total_mols:.2%}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    # 1. Fetch the exact allowed atoms from your project constants
    data_name = "crossdock_full"
    atom_encoder = constants.dataset_params[data_name]["atom_encoder"]
    allowed_atoms = set(atom_encoder.keys())

    # 2. Run the strict validation on all directories
    target_dirs = [
        "./outputs/test_molecules/ast_test/sdfs",
        "./outputs/test_molecules/ast_gat_test/sdfs",
        "./outputs/test_molecules/gat_test/sdfs",
        "./outputs/test_molecules/transformer_test/sdfs",
        "./outputs/test_molecules/targetdiff_test/sdfs",
        "./outputs/test_molecules/DiffSBDD_test/sdfs",
    ]

    for target_dir in target_dirs:
        count_and_validate_mols(target_dir, allowed_atoms)
