"""Generate ligand molecules starting from a base scaffold, conditioned on a target protein pocket using inpainting.

This script implements scaffold-based generation (inpainting). Starting from a base
scaffold SMILES and a target protein pocket, the model generates new atoms and
bonds to complete the molecule. It performs a grid search over key hyperparameters
like t_max_inpainting (noise level for the scaffold) and force_bonds (fixed
scaffold bonds).

Parser Arguments:
    --pdb_path (str): Path to the target protein PDB file.
    --base_ligand_smile (str): SMILES string of the starting scaffold.
    --n_samples (int): Number of molecules to generate per hyperparameter combination.
    --t_max_values (list): List of t_max values for grid search.
    --force_bonds_values (list): List of True/False strings for fixing scaffold bonds.
    --base_output_dir (str): Base directory for generated molecules and logs.

Files Produced:
    - {output_folder}/generated_mol_inpainting_{idx}.sdf: The generated molecule.
    - {output_folder}/generated_mol_inpainting_{idx}.png: 2D visualization of the molecule.
    - generation_summary.csv: A summary of success rates and attempts for each parameter combination.
"""

import argparse
import os
import torch
import numpy as np
import time
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data, Batch

from topoflow.flow.flow_models import FlowModel
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.utils.molecules import pdb_to_data_object, data_to_rdmol, mol_to_data
import topoflow.utils.constants as constants
from topoflow.data.datasets import CrossDocked
from torch.utils.data import DataLoader


class MockDatamodule:
    """A simple mock datamodule to provide dataset lengths and batch size."""

    def __init__(self, train_length, val_length, test_length, batch_size):
        """Initialize the mock datamodule with dataset lengths and batch size."""
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.batch_size = batch_size


def extract_bonds(mol):
    """Extract bond information from an RDKit molecule."""
    bonds = []
    for bond in mol.GetBonds():
        bonds.append(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondTypeAsDouble())
        )
    return bonds


def run_inpainting(args):
    """Main execution function for custom pocket inpainting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load atom encoder
    dataset_info = constants.dataset_params["crossdock_full"]
    atom_encoder = dataset_info["atom_encoder"]
    atom_decoder = dataset_info["atom_decoder"]

    # --- 1. Global Setup ---
    print("Initializing environment and loading models...")
    print(f"Ligand smiles: {args.base_ligand_smile}")

    rdkit_ligand = None
    if args.base_ligand_smile and args.base_ligand_smile.strip():
        rdkit_ligand = Chem.MolFromSmiles(args.base_ligand_smile)
        if rdkit_ligand is None:
            raise ValueError(f"Could not parse SMILES: {args.base_ligand_smile}")
        AllChem.EmbedMolecule(rdkit_ligand, AllChem.ETKDG())
        extracted_bonds_list = extract_bonds(rdkit_ligand)
        coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)
        base_ligand = Data(x_lig=coords, h_lig=one_hot, batch_lig=batch)
    else:
        print(
            "No base ligand SMILES provided. Generating from scratch within the pocket."
        )
        # Create an empty Data object with the correct feature dimension
        num_atom_types = max(atom_encoder.values()) + 1
        base_ligand = Data(
            x_lig=torch.zeros((0, 3), dtype=torch.float32),
            h_lig=torch.zeros((0, num_atom_types), dtype=torch.float32),
        )
        extracted_bonds_list = []
        FORCE_BONDS_VALUES = [False]

    # Load pocket and ALWAYS center it
    pocket = pdb_to_data_object(
        pdb_file=args.pdb_path,
        atom_encoder=atom_encoder,
        save_path=args.data_path,
        normalization_path=args.normalization_path,
        force=True,
    )

    # Center base ligand if provided, using the pocket's centroid
    if rdkit_ligand is not None and hasattr(pocket, "original_centroid"):
        centroid = pocket.original_centroid.to(base_ligand.x_lig.device)
        # Centering BEFORE scaling (to match pdb_to_data_object logic)
        base_ligand.x_lig = base_ligand.x_lig - centroid

        # Scale if normalization was applied to pocket
        if args.normalization_path is not None:
            norm_params = np.load(args.normalization_path, allow_pickle=True)
            std = torch.tensor(norm_params["std"], dtype=torch.float32).to(
                base_ligand.x_lig.device
            )
            base_ligand.x_lig = base_ligand.x_lig / std

    data_dir = "./data/CrossDocked/"
    checkpoint_path = "/data/petretto/home/gmsv2427/Documents/topoFlow/checkpoints/to_evaluate/03_11_1199e_ast_gat_scaled_vel.ckpt"
    val_dataset = CrossDocked(root=data_dir, split="val")
    datamodule = MockDatamodule(
        train_length=1,
        val_length=len(val_dataset),
        test_length=1,
        batch_size=args.batch_size,
    )

    path = GaussianConditionalProbabilityPath(
        p_data=datamodule,
        alpha=CosineAlpha(),
        beta=CosineBeta(),
        target_dim=14,
        distribution_hist=val_dataset.distribution_hist,
    )

    model = FlowModel.load_from_checkpoint(
        checkpoint_path,
        path=path,
        norm_path=os.path.join(data_dir, "processed/normalization_params.npz"),
    )
    model.to(device)
    model.eval()

    # --- 2. Run Inpainting ---
    print(f"Starting inpainting for {args.n_samples} samples...")
    start_time = time.time()

    all_generated_mols = []

    # Batch processing
    for i in tqdm(range(0, args.n_samples, args.batch_size)):
        current_batch_size = min(args.batch_size, args.n_samples - i)

        with torch.no_grad():
            # Perform inpainting using the model's native method
            # FlowModel.inpainting internally calls inpainting_with_trajectory
            # which we updated to handle recentering.
            gen_data = model.inpainting(
                ligand=base_ligand.to(device),
                pocket=pocket.to(device),
                n_atoms_to_add=args.n_atoms_to_add,
                batch_size=current_batch_size,
                n_steps=args.n_steps,
                recenter=True,  # This will use pocket.original_centroid
            )

        # Convert back to RDKit molecules
        # Sanitize=True will filter out invalid molecules
        mols = data_to_rdmol(
            gen_data, atom_decoder=atom_decoder, sanitize=True, verbose=False
        )
        all_generated_mols.extend([m for m in mols if m is not None])

    end_time = time.time()
    print(f"Generation completed in {end_time - start_time:.2f} seconds.")
    print(f"Generated {len(all_generated_mols)} valid molecules.")

    # --- 3. Save Results ---
    if all_generated_mols:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        writer = Chem.SDWriter(str(args.output_path))
        for i, mol in enumerate(all_generated_mols):
            mol.SetProp("_Name", f"inpainted_sample_{i}")
            writer.write(mol)
        writer.close()
        print(f"Saved generated molecules to {args.output_path}")
    else:
        print("No valid molecules were generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Target Protein Inpainting with TopoFlow."
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./data/target_protein/wwp2_hect.pdb",
        help="Path to the protein pocket PDB file.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/target_protein/wwp2_hect_data.pt",
        help="Path to save/load processed pocket data.",
    )
    parser.add_argument(
        "--base_ligand_smile",
        type=str,
        default="C1=CC=C(C=C1)C(=O)O",  # Benzoic acid as example
        help="SMILES of the base ligand (scaffold) to start from.",
    )
    parser.add_argument(
        "--n_atoms_to_add",
        type=int,
        default=10,
        help="Number of atoms to add to the base ligand.",
    )
    parser.add_argument(
        "--n_samples", type=int, default=10, help="Number of molecules to generate."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for generation."
    )
    parser.add_argument(
        "--n_steps", type=int, default=80, help="Number of integration steps."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/target_protein_inpainting/generated_mols.sdf",
        help="Path to save generated molecules.",
    )
    parser.add_argument(
        "--normalization_path",
        type=str,
        default="./data/CrossDocked/processed/normalization_params.npz",
        help="Path to normalization params.",
    )

    args = parser.parse_args()
    run_inpainting(args)
