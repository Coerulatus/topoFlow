"""Script to sample molecules for different number of flow steps using TopoFlow."""

import argparse
import os
import torch
import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Batch

from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.utils.molecules import pdb_to_data_object, data_to_rdmol
import topoflow.utils.constants as constants


class MockDataLoader:
    """A mock dataloader to satisfy the Path object's need for a dimension attribute."""

    def __init__(self, dim):
        """Initialize the mock dataloader with a specified dimension."""
        self.total_dim = dim

    @property
    def dim(self):
        """Returns the total dimension of the data, which is required by the Path object."""
        return self.total_dim


def sample_for_pockets(
    model,
    pdb_dir,
    output_path,
    atom_encoder,
    atom_decoder,
    norm_path,
    mol_per_pocket=10,
    batch_size=16,
    n_steps=80,
    device="cuda",
    num_limit=None,
    max_attempts=50,
):
    """Main function to sample molecules for a directory of protein PDB files using TopoFlow."""
    # Establish the strict set of allowed atoms based on your dataset
    allowed_atoms = set(atom_encoder.keys())

    # ======================================================================
    # Processing Loop
    # ======================================================================
    os.makedirs(output_path, exist_ok=True)
    sdf_dir = os.path.join(output_path, "sdfs")
    os.makedirs(sdf_dir, exist_ok=True)

    mapping_path = os.path.join(output_path, "mapping.txt")
    existing_mappings = set()
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            for line in f:
                existing_mappings.add(line.strip())

    pdb_files = sorted([f for f in os.listdir(pdb_dir) if f.endswith(".pdb")])
    if num_limit:
        pdb_files = pdb_files[:num_limit]

    print(f"\n[Steps={n_steps}] Found {len(pdb_files)} PDB files. Starting sampling...")

    total_gen_time = 0
    total_mols_valid = 0
    total_mols_attempted = 0

    for pdb_file in tqdm(pdb_files, desc=f"Sampling Pockets (steps={n_steps})"):
        pdb_path = os.path.join(pdb_dir, pdb_file)
        protein_name = pdb_file.replace(".pdb", "")
        sdf_path = os.path.join(sdf_dir, f"{protein_name}.sdf")

        valid_mols = []

        try:
            # 1. Load and check existing SDF file (Resume Logic)
            if os.path.exists(sdf_path):
                suppl = Chem.SDMolSupplier(sdf_path, sanitize=False)
                for mol in suppl:
                    if mol is not None:
                        # STRICT CHECK: Filter out dummy atoms from old saves
                        has_dummy_atom = False
                        for atom in mol.GetAtoms():
                            if atom.GetSymbol() not in allowed_atoms:
                                has_dummy_atom = True
                                break
                        if has_dummy_atom:
                            continue

                        try:
                            Chem.SanitizeMol(mol)
                            valid_mols.append(mol)
                            if len(valid_mols) == mol_per_pocket:
                                break
                        except Exception:
                            pass  # Skip invalid molecules found in old files

            n_needed = mol_per_pocket - len(valid_mols)
            if n_needed <= 0:
                continue

            # 2. Convert PDB to TopoFlow Pocket Data
            pocket_data = pdb_to_data_object(
                pdb_path,
                atom_encoder=atom_encoder,
                normalization_path=norm_path,
                center_pocket=True,
                verbose=False,
            )

            # 3. Generate New Samples until Quota is Met
            attempts = 0
            start_time = time.time()
            mols_at_start = len(valid_mols)

            while len(valid_mols) < mol_per_pocket and attempts < max_attempts:
                attempts += 1

                # Use the original batching logic that worked natively with the model
                batch_pocket = Batch.from_data_list([pocket_data] * batch_size).to(
                    device
                )

                with torch.no_grad():
                    samples = model.sample(
                        n_samples=batch_size, n_steps=n_steps, pocket=batch_pocket
                    )

                total_mols_attempted += batch_size

                # Sanitize=True automatically filters out geometrically invalid graphs
                mols = data_to_rdmol(
                    samples, atom_decoder=atom_decoder, sanitize=True, verbose=False
                )

                for mol in mols:
                    if mol is not None:
                        # STRICT CHECK: Filter out newly generated dummy atoms
                        has_dummy_atom = False
                        for atom in mol.GetAtoms():
                            if atom.GetSymbol() not in allowed_atoms:
                                has_dummy_atom = True
                                break
                        if has_dummy_atom:
                            continue

                        valid_mols.append(mol)
                        if len(valid_mols) == mol_per_pocket:
                            break

            end_time = time.time()
            total_gen_time += end_time - start_time
            total_mols_valid += len(valid_mols) - mols_at_start

            if attempts == max_attempts and len(valid_mols) < mol_per_pocket:
                print(
                    f"\nWarning: Reached max attempts ({max_attempts}) for {pdb_file}. Generated {len(valid_mols)}/{mol_per_pocket} valid molecules."
                )

            # 4. Save Results
            if len(valid_mols) > 0:
                writer = Chem.SDWriter(sdf_path)
                for i, mol in enumerate(valid_mols):
                    mol.SetProp("_Name", f"{protein_name}_sample_{i}")
                    writer.write(mol)
                writer.close()

                mapping_line = f"{protein_name}.sdf\t{pdb_file}"
                if mapping_line not in existing_mappings:
                    existing_mappings.add(mapping_line)
                    with open(mapping_path, "a") as mapping_file:
                        mapping_file.write(mapping_line + "\n")
            else:
                if os.path.exists(sdf_path):
                    os.remove(sdf_path)
                print(
                    f"\nWarning: Failed to generate ANY valid molecules for {pdb_file}"
                )

        except Exception as e:
            print(f"\nError processing {pdb_file}: {e}")

    # Save timing results
    if total_mols_attempted > 0:
        avg_time_per_attempt = total_gen_time / total_mols_attempted
        avg_time_per_valid = (
            total_gen_time / total_mols_valid if total_mols_valid > 0 else float("inf")
        )

        timing_path = os.path.join(output_path, "generation_time.txt")
        with open(timing_path, "w") as f:
            f.write(f"total_time_seconds: {total_gen_time}\n")
            f.write(f"total_mols_valid: {total_mols_valid}\n")
            f.write(f"total_mols_attempted: {total_mols_attempted}\n")
            f.write(f"seconds_per_valid_mol: {avg_time_per_valid}\n")
            f.write(f"seconds_per_attempted_mol: {avg_time_per_attempt}\n")
        print(f"\n[Steps={n_steps}] Generation time saved to {timing_path}")
        print(f"Average time per valid molecule: {avg_time_per_valid:.4f}s")
        print(f"Average time per attempted molecule: {avg_time_per_attempt:.4f}s")


def main():
    """Main function to batch sample molecules for a directory of PDB pockets across different n_steps using TopoFlow."""
    parser = argparse.ArgumentParser(
        description="Batch sample molecules for a directory of PDB pockets across different n_steps."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to TopoFlow checkpoint.",
    )
    parser.add_argument(
        "--pdb_dir",
        type=str,
        default="./data/CrossDocked/processed/test",
        help="Directory containing pocket PDB files.",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="./outputs/test_n_steps/",
        help="Base output directory.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Project data directory (for distribution/norm).",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="crossdock_full",
        choices=["crossdock_full"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--mol_per_pocket",
        type=int,
        default=4,
        help="Number of VALID molecules required per pocket.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for sampling."
    )
    parser.add_argument(
        "--n_steps_list",
        type=int,
        nargs="+",
        default=[10, 20, 40, 60, 80, 100, 120],
        help="List of flow integration steps to test.",
    )
    parser.add_argument(
        "--num_limit",
        type=int,
        default=None,
        help="Limit number of pockets to process.",
    )
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {args.device}")

    # ======================================================================
    # Path & Model Setup
    # ======================================================================
    dist_path = os.path.join(args.data_dir, "processed/size_distribution.npy")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"Size distribution not found at {dist_path}")
    distribution_hist = np.load(dist_path, allow_pickle=True)

    mock_p_data = MockDataLoader(dim=14)
    path = GaussianConditionalProbabilityPath(
        p_data=mock_p_data,
        alpha=CosineAlpha(),
        beta=CosineBeta(),
        target_dim=14,
        distribution_hist=distribution_hist,
    )

    norm_path = os.path.join(args.data_dir, "processed/normalization_params.npz")
    model = FlowModel.load_from_checkpoint(
        args.checkpoint_path,
        path=path,
        norm_path=norm_path,
    )

    if "ast_gat" in os.path.basename(args.checkpoint_path):
        model.vector_field_model.use_gat = True

    model.to(args.device)
    model.eval()

    atom_encoder = constants.dataset_params[args.data_name]["atom_encoder"]
    atom_decoder = constants.dataset_params[args.data_name]["atom_decoder"]

    # ======================================================================
    # Iteration over n_steps
    # ======================================================================
    for n_steps in args.n_steps_list:
        output_path = os.path.join(args.output_base_dir, f"steps_{n_steps}")
        sample_for_pockets(
            model=model,
            pdb_dir=args.pdb_dir,
            output_path=output_path,
            atom_encoder=atom_encoder,
            atom_decoder=atom_decoder,
            norm_path=norm_path,
            mol_per_pocket=args.mol_per_pocket,
            batch_size=args.batch_size,
            n_steps=n_steps,
            device=args.device,
            num_limit=args.num_limit,
        )

    print(f"\nAll steps completed! Results saved in {args.output_base_dir}")


if __name__ == "__main__":
    main()
