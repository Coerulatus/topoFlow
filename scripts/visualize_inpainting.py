"""Visualize the step-by-step process of scaffold-based molecular generation (inpainting) using PyMOL.

This script generates high-quality visualizations and PyMOL sessions showing
how the TopoFlow model grows a molecule from a starting scaffold. It captures
the trajectory from the noisy initial state to the final refined molecule.
The output includes multiple images and session files highlighting the scaffold,
the newly added atoms, and the target protein pocket.

Parser Arguments:
    --base_ligand_smile (str): Starting scaffold SMILES string.
    --pdb_path (str): Path to the target protein PDB file.
    --n_atoms_to_add (int): Number of atoms to be generated around the scaffold.
    --t_max (float): Time step for inpainting initialization.
    --rx, --ry, --rz (float): Rotation angles for orienting the scaffold in the pocket.
    --output_dir (str): Directory to save images and PyMOL sessions.

Files Produced:
    - 1_scaffold_only.png/.pse: Initial scaffold placement.
    - 2_noisy_start.png/.pse: Initial noisy atom distribution.
    - 3_final_molecule.png/.pse: Final generated molecule in the pocket.
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
from topoflow.utils.visualization import save_trajectory_as_session


class MockDatamodule:
    """A simple mock datamodule to provide dataset lengths and batch size."""

    def __init__(self, train_length, val_length, test_length, batch_size):
        """Initialize the mock datamodule with dataset lengths and batch size."""
        self.train_length = train_length
        self.val_length = val_length
        self.test_length = test_length
        self.batch_size = batch_size


def run_visualize_inpainting(args):
    """Main execution function for visual pocket inpainting."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load atom encoder
    dataset_info = constants.dataset_params["crossdock_full"]
    atom_encoder = dataset_info["atom_encoder"]
    atom_decoder = dataset_info["atom_decoder"]

    # --- 1. Global Setup ---
    print("Initializing environment and loading models...")

    # Load pocket and ALWAYS center it
    pocket = pdb_to_data_object(
        pdb_file=args.pdb_path,
        atom_encoder=atom_encoder,
        normalization_path="./data/CrossDocked/processed/normalization_params.npz",
        force=True,
    ).to(device)

    # Prepare base ligand (scaffold)
    rdkit_ligand = Chem.MolFromSmiles(args.base_ligand_smile)
    if rdkit_ligand is None:
        raise ValueError(f"Could not parse SMILES: {args.base_ligand_smile}")
    AllChem.EmbedMolecule(rdkit_ligand, AllChem.ETKDG())

    coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)
    base_ligand = Data(x_lig=coords, h_lig=one_hot, batch_lig=batch).to(device)

    # Center base ligand using pocket's centroid (which is stored in Angstroms)
    if hasattr(pocket, "original_centroid"):
        centroid = pocket.original_centroid.to(device)
        base_ligand.x_lig = base_ligand.x_lig - centroid

        # Scale to match the normalized pocket
        norm_params = np.load(
            "./data/CrossDocked/processed/normalization_params.npz", allow_pickle=True
        )
        std = torch.tensor(norm_params["std"], dtype=torch.float32).to(device)
        base_ligand.x_lig = base_ligand.x_lig / std

    data_dir = "./data/CrossDocked/"
    checkpoint_path = "/home/marco/Documents/phd/topoFlow/checkpoints/runs_cluster/to_evaluate/03_11_1199e_ast_gat_scaled_vel.ckpt"
    val_dataset = CrossDocked(root=data_dir, split="val")
    datamodule = MockDatamodule(
        train_length=1,
        val_length=len(val_dataset),
        test_length=1,
        batch_size=1,
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

    # --- 2. Run Inpainting Trajectory ---
    print("Starting inpainting trajectory generation...")
    with torch.no_grad():
        trajectory = model.inpainting_with_trajectory(
            ligand=base_ligand,
            pocket=pocket,
            n_atoms_to_add=args.n_atoms_to_add,
            batch_size=1,
            n_steps=args.n_steps,
            recenter=True,  # Moves it back home!
        )

    # --- 3. Save Visualization ---
    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving trajectory visualization to {args.output_path}...")
    save_trajectory_as_session(
        trajectory, args.output_path, atom_decoder=atom_decoder, verbose=True
    )
    print("Visualization saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Inpainting with TopoFlow.")
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./data/target_protein/wwp2_hect.pdb",
        help="Path to the protein pocket PDB file.",
    )
    parser.add_argument(
        "--base_ligand_smile",
        type=str,
        default="C1=CC=C(C=C1)C(=O)O",
        help="SMILES of the scaffold.",
    )
    parser.add_argument(
        "--n_atoms_to_add",
        type=int,
        default=10,
        help="Number of atoms to add.",
    )
    parser.add_argument("--n_steps", type=int, default=80, help="Integration steps.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="./outputs/inpainting_visualization/trajectory",
        help="Path to save PyMOL session/images.",
    )

    args = parser.parse_args()
    run_visualize_inpainting(args)
