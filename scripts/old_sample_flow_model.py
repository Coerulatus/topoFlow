"""Sample a specified number of molecules for every protein pocket in a dataset split using a trained TopoFlow model.

This script is the primary tool for bulk molecular generation. It iterates through
all pockets in a given dataset split (test or val) and uses a trained continuous
normalizing flow model to generate new ligands. For each pocket, it produces multiple
samples, saves them as a multi-molecule SDF file, and also exports the corresponding
PyTorch Geometric data objects for further analysis.

Parser Arguments:
    --checkpoint_path (str): Path to the trained model checkpoint (.ckpt).
    --output_dir (str): Root directory for saving generated outputs.
    --molecules_per_pocket (int): Number of samples to generate for each target.
    --n_steps (int): Number of integration steps for the ODE solver.
    --split (str): Dataset split to use ("val" or "test").

Files Produced:
    - Subdirectories for each pocket containing:
        - {pocket_id}_generated.sdf: Generated molecules in SDF format.
        - {pocket_id}_generated.pt: PyG data objects for the generated molecules.
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Batch, DataLoader as PyGDataLoader

from topoflow.flow.flow_models import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
import topoflow.utils.constants as constants
from topoflow.utils.molecules import data_to_rdmol, data_to_list


def sample_molecules(
    checkpoint_path,
    data_dir,
    data_name,
    batch_size,
    molecules_per_pocket,
    output_dir,
    split="test",
    device="auto",
    n_steps=20,
):
    """Samples molecules from a trained model for pockets in a dataset split."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ======================================================================
    # Data Loading
    # ======================================================================
    print(f"Setting up {split} dataset...")
    dataset = CrossDocked(root=data_dir, split=split)

    # Load names for identifying pockets
    split_npz_path = os.path.join(data_dir, "processed", f"{split}.npz")
    if not os.path.exists(split_npz_path):
        # Fallback to names in the dataset if possible, or just use indices
        print(f"Warning: {split_npz_path} not found. Using indices for naming.")
        names = [f"pocket_{i}" for i in range(len(dataset))]
    else:
        names = np.load(split_npz_path, allow_pickle=True)["names"]

    # ======================================================================
    # Path Initialization (needed for model loading)
    # ======================================================================
    alpha = CosineAlpha()
    beta = CosineBeta()
    p_data = DataLoaderSampleable(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    distribution_hist_path = os.path.join(
        data_dir, "processed", "size_distribution.npy"
    )
    if not os.path.exists(distribution_hist_path):
        raise FileNotFoundError(
            f"Size distribution not found at {distribution_hist_path}"
        )
    distribution_hist = np.load(distribution_hist_path, allow_pickle=True)

    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=alpha,
        beta=beta,
        target_dim=14,
        distribution_hist=distribution_hist,
    )

    # ======================================================================
    # Model Loading
    # ======================================================================
    print(f"Loading model from checkpoint: {checkpoint_path}")
    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    model = FlowModel.load_from_checkpoint(
        checkpoint_path,
        path=path,
        norm_path=norm_path,
    )

    # Special handling for AST-GAT model variant
    if "ast_gat" in os.path.basename(checkpoint_path):
        print("Detected ast_gat model, setting use_gat=True")
        model.vector_field_model.use_gat = True

    model.to(device)
    model.eval()
    model.max_batched_samples = batch_size
    atom_decoder = constants.dataset_params[data_name]["atom_decoder"]

    # ======================================================================
    # Sampling Loop
    # ======================================================================
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"Sampling {molecules_per_pocket} molecules per pocket for {len(dataset)} pockets..."
    )

    loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)

    for idx, pocket in enumerate(tqdm(loader, desc="Sampling Pockets")):
        # Extract and sanitize pocket name
        full_name = names[idx]
        # Format usually: pocket_path_ligand_path. We want a clean filename.
        pocket_id = full_name.split(".pdb")[0].replace("/", "_")

        pocket_output_dir = os.path.join(output_dir, pocket_id)
        os.makedirs(pocket_output_dir, exist_ok=True)

        n_left = molecules_per_pocket
        mols_all = []
        pyg_samples_all = []

        batch_idx = 0
        while n_left > 0:
            n_batch = min(n_left, model.max_batched_samples)
            # Replicate pocket for the batch
            repeated_pocket = Batch.from_data_list([pocket] * n_batch).to(device)

            # Sample using the model
            with torch.no_grad():
                samples = model.sample(
                    n_samples=n_batch, n_steps=n_steps, pocket=repeated_pocket
                )

            # Convert to RDKit molecules for saving
            mols = data_to_rdmol(samples, atom_decoder=atom_decoder, sanitize=True)
            mols_all.extend(mols)

            # Convert to individual PyG objects
            pyg_samples = data_to_list(samples)
            pyg_samples_all.extend(pyg_samples)

            n_left -= n_batch
            batch_idx += 1

        # Save as SDF
        sdf_path = os.path.join(pocket_output_dir, f"{pocket_id}_generated.sdf")
        writer = Chem.SDWriter(sdf_path)
        for i, m in enumerate(mols_all):
            if m is not None:
                m.SetProp("_Name", f"{pocket_id}_sample_{i}")
                writer.write(m)
        writer.close()

        # Save PyG objects for further analysis
        pt_path = os.path.join(pocket_output_dir, f"{pocket_id}_generated.pt")
        torch.save(pyg_samples_all, pt_path)

    print(f"\nSampling complete. Results saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample molecules for protein pockets using TopoFlow."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the generated molecules.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Directory containing the dataset.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="crossdock_full",
        help="Dataset name (for atom decoder).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to use.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for sampling."
    )
    parser.add_argument(
        "--molecules_per_pocket",
        type=int,
        default=10,
        help="Number of molecules to generate per pocket.",
    )
    parser.add_argument(
        "--n_steps", type=int, default=20, help="Number of integration steps."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (cuda, cpu, or auto)."
    )

    args = parser.parse_args()

    sample_molecules(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        data_name=args.data_name,
        batch_size=args.batch_size,
        molecules_per_pocket=args.molecules_per_pocket,
        output_dir=args.output_dir,
        split=args.split,
        device=args.device,
        n_steps=args.n_steps,
    )
