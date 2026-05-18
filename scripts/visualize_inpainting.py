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
import lightning as pl
from rdkit import Chem
from rdkit.Chem import AllChem

from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
import topoflow.utils.constants as constants
from topoflow.utils.molecules import (
    pdb_to_data_object,
    mol_to_data,
)
from topoflow.utils.visualization import visualize_data
from torch_geometric.data import Data


def set_seed(seed):
    """Sets the random seed for reproducibility across torch, numpy, and lightning."""
    pl.seed_everything(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rotation_matrix(rx, ry, rz):
    """Returns a 3x3 rotation matrix for the given angles in degrees.

    Args:
        rx (float): Rotation around X axis in degrees.
        ry (float): Rotation around Y axis in degrees.
        rz (float): Rotation around Z axis in degrees.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    rx = np.radians(rx)
    ry = np.radians(ry)
    rz = np.radians(rz)

    Rx = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]]
    )

    Ry = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]]
    )

    Rz = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]]
    )

    return Rz @ Ry @ Rx


def main():
    """Main function to visualize the inpainting process of scaffold-based molecular generation."""
    parser = argparse.ArgumentParser(description="Visualize the inpainting process.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--n_atoms_to_add", type=int, default=10, help="Number of atoms to add."
    )
    parser.add_argument(
        "--n_steps", type=int, default=100, help="Number of flow steps."
    )
    parser.add_argument(
        "--t_max", type=float, default=0.9, help="t_max for inpainting."
    )
    parser.add_argument(
        "--force_scaffold", type=bool, default=True, help="Force scaffold."
    )
    parser.add_argument(
        "--rx", type=float, default=0.0, help="Rotation around X axis in degrees."
    )
    parser.add_argument(
        "--ry", type=float, default=0.0, help="Rotation around Y axis in degrees."
    )
    parser.add_argument(
        "--rz", type=float, default=0.0, help="Rotation around Z axis in degrees."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/inpainting_visualization",
        help="Output directory.",
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./data/target_protein/wwp2_hect.pdb",
        help="Path to protein PDB.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/marco/Documents/phd/topoFlow/checkpoints/runs_cluster/to_evaluate/03_11_1199e_ast_gat_scaled_vel.ckpt",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--base_ligand_smile",
        type=str,
        default="c1ccc2c(c1)[nH]cc2",
        help="Base ligand SMILES (Indole).",
    )

    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]

    # 1. Load data and model
    print("Loading model and data...")
    data_dir = "./data/CrossDocked/"
    val_dataset = CrossDocked(root=data_dir, split="val")
    alpha = CosineAlpha()
    beta = CosineBeta()
    p_data = DataLoaderSampleable(
        val_dataset, batch_size=1, shuffle=True, num_workers=0
    )

    distribution_hist = np.load(
        "./data/CrossDocked/processed/size_distribution_full.npy", allow_pickle=True
    )
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=alpha,
        beta=beta,
        target_dim=14,
        distribution_hist=distribution_hist,
        device=device,
    )

    model = FlowModel.load_from_checkpoint(
        args.checkpoint_path,
        path=path,
        norm_path="./data/CrossDocked/processed/normalization_params.npz",
    ).to(device)
    model.eval()

    # 2. Prepare Scaffold and Pocket
    print("Preparing scaffold and pocket...")
    rdkit_ligand = Chem.MolFromSmiles(args.base_ligand_smile)
    AllChem.EmbedMolecule(rdkit_ligand, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(rdkit_ligand)  # Optimize geometry to avoid deformations

    # Extract explicit bonds from the scaffold
    from topoflow.utils.molecules import extract_bonds

    explicit_bonds = extract_bonds(rdkit_ligand)

    # Get raw coordinates
    coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)

    # Apply rotation if specified
    if args.rx != 0 or args.ry != 0 or args.rz != 0:
        print(f"Applying rotation: rx={args.rx}, ry={args.ry}, rz={args.rz}")
        R = torch.from_numpy(get_rotation_matrix(args.rx, args.ry, args.rz)).float()
        # Rotate around the centroid of the scaffold
        centroid = coords.mean(dim=0)
        coords = (coords - centroid) @ R.T + centroid

    # Normalize scaffold coordinates
    norm_params = np.load(
        "./data/CrossDocked/processed/normalization_params.npz", allow_pickle=True
    )
    mean = torch.tensor(norm_params["mean"], dtype=torch.float32)
    std = torch.tensor(norm_params["std"], dtype=torch.float32)
    coords_norm = (coords - mean) / std
    base_ligand = Data(x_lig=coords_norm, h_lig=one_hot, batch_lig=batch).to(device)

    pocket = pdb_to_data_object(
        pdb_file=args.pdb_path,
        atom_encoder=atom_encoder,
        normalization_path="./data/CrossDocked/processed/normalization_params.npz",
        center_pocket=False,
        force=True,
    ).to(device)

    # 3. Perform inpainting and get trajectory
    print(f"Performing inpainting (adding {args.n_atoms_to_add} atoms)...")
    trajectory = model.inpainting_with_trajectory(
        ligand=base_ligand,
        pocket=pocket,
        n_atoms_to_add=args.n_atoms_to_add,
        n_steps=args.n_steps,
        t_max_inpainting=args.t_max,
        force_scaffold=args.force_scaffold,
        batch_size=1,
    )

    # 4. Visualization
    print("Generating PyMOL images...")
    camera = [
        0.363415956,
        -0.772950709,
        0.520053267,
        -0.188191637,
        -0.607617855,
        -0.771607637,
        0.912414014,
        0.182542145,
        -0.366285205,
        0.000004224,
        0.000004929,
        -43.471538544,
        0.621637106,
        -0.829541802,
        -0.770337045,
        20.621967316,
        66.323265076,
        -20.000000000,
    ]

    # Image 1: Scaffold only in the pocket at t_max (clean-ish if force_scaffold=True)
    t_eval = torch.linspace(0, 1, args.n_steps)
    t_max_idx = (t_eval - args.t_max).abs().argmin().item()
    t_max_frame = trajectory[t_max_idx]
    scaffold_mask = t_max_frame.scaffold_mask

    print(f"Saving Image 1: Scaffold only (at t={t_eval[t_max_idx]:.2f})...")
    visualize_data(
        t_max_frame,
        image_path=os.path.join(args.output_dir, "1_scaffold_only.png"),
        session_path=os.path.join(args.output_dir, "1_scaffold_only.pse"),
        scaffold_mask=scaffold_mask,
        ligand_mask=scaffold_mask,  # Only show scaffold atoms
        pymol_camera=camera,
        width=2400,
        height=2400,
        add_ligand_bonds=True,
        explicit_bonds=explicit_bonds,
    )

    # Image 2: Noisy starting atoms (scaffold = blue, others = yellow) at t=0
    first_frame = trajectory[0]
    print("Saving Image 2: Noisy starting state (at t=0.00)...")
    visualize_data(
        first_frame,
        image_path=os.path.join(args.output_dir, "2_noisy_start.png"),
        session_path=os.path.join(args.output_dir, "2_noisy_start.pse"),
        scaffold_mask=scaffold_mask,
        pymol_camera=camera,
        width=2400,
        height=2400,
        add_ligand_bonds=False,
        explicit_bonds=explicit_bonds,
    )

    # Image 3: Final molecule (scaffold = blue, new = yellow)
    final_frame = trajectory[-1]
    print("Saving Image 3: Final generated molecule (keeping largest fragment)...")

    # 1. Infer connectivity to find fragments
    from topoflow.utils.molecules import make_mol_openbabel

    rdmol_final = make_mol_openbabel(
        final_frame.x_lig.cpu().numpy(),
        final_frame.h_lig.cpu(),  # Pass as tensor
        atom_decoder=constants.dataset_params["crossdock_full"]["atom_decoder"],
        explicit_bonds=explicit_bonds,
    )

    # 2. Get largest fragment indices
    if rdmol_final is not None:
        frags = Chem.GetMolFrags(rdmol_final, asMols=False)
        largest_frag_indices = max(frags, key=len)
        mask_largest = torch.zeros(final_frame.x_lig.shape[0], dtype=torch.bool)
        mask_largest[torch.tensor(largest_frag_indices)] = True
    else:
        mask_largest = None

    visualize_data(
        final_frame,
        image_path=os.path.join(args.output_dir, "3_final_molecule.png"),
        session_path=os.path.join(args.output_dir, "3_final_molecule.pse"),
        scaffold_mask=scaffold_mask,
        ligand_mask=mask_largest,  # Only show the largest fragment
        pymol_camera=camera,
        width=2400,
        height=2400,
        add_ligand_bonds=True,  # Show bonds for final molecule
        explicit_bonds=explicit_bonds,
    )

    print(f"Inpainting visualization completed. Images saved to {args.output_dir}")


if __name__ == "__main__":
    main()
