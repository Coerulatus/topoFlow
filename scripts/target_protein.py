"""Script to generate ligand molecules conditioned on a target protein pocket using a pre-trained flow model."""

import os
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
import torch.serialization
import torch_geometric.data

from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.utils.molecules import pdb_to_data_object, make_mol_openbabel
from topoflow.utils.visualization import visualize_trajectory


def get_largest_fragment_info(mol: Chem.Mol):
    """Finds the largest connected fragment in an RDKit molecule.

    Returns:
        tuple: (largest_fragment_mol, largest_fragment_size)
    """
    if mol is None:
        return None, 0

    # Get fragments as atom indices, then convert to RDKit molecules
    try:
        frags = Chem.GetMolFrags(mol, asMols=True)

        largest_fragment = None
        max_size = 0

        for fragment in frags:
            size = fragment.GetNumAtoms()
            if size > max_size:
                max_size = size
                largest_fragment = fragment

        return largest_fragment, max_size
    except Exception as e:
        print(f"Error in getting largest fragment: {e}")
        return None, 0


# --- Configuration ---
atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]
atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]
pdb_path = "./data/target_protein/wwp2_c2.pdb"
data_path = "./data/target_protein/wwp2_c2_data.pt"
output_folder = (
    "./outputs/generated_ligands_wwp2_c2/"  # Folder to save passing molecules
)
FRAGMENT_THRESHOLD_PERCENT = 0.50  # % of atoms must be in the largest fragment
N_SAMPLES_TO_GENERATE = 256  # Total number of molecules to generate
N_STEPS = 100  # Steps for the ODE simulator
BATCH_SIZE = 4  # Sample in batches

# --- Environment Setup (No changes needed) ---
pocket = pdb_to_data_object(
    pdb_file=pdb_path,
    atom_encoder=atom_encoder,
    save_path=data_path,
    center_pocket=True,
    force=True,
)

data_dir = "./data/CrossDocked/"
checkpoint_path = "checkpoints/runs_cluster/10_21_499e_ast_gat_scaled_vel.ckpt"
val_dataset = CrossDocked(root=data_dir, split="val")
datamodule = MockDatamodule(
    train_length=1, val_length=len(val_dataset), test_length=1, batch_size=BATCH_SIZE
)
alpha = CosineAlpha()
beta = CosineBeta()
p_data = DataLoaderSampleable(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8
)
distribution_hist = np.load(
    "./data/CrossDocked/processed/size_distribution.npy", allow_pickle=True
)

path = GaussianConditionalProbabilityPath(
    p_data=p_data,
    alpha=alpha,
    beta=beta,
    target_dim=14,
    distribution_hist=distribution_hist,
)

print(f"Loading model from checkpoint: {checkpoint_path}")
model = FlowModel.load_from_checkpoint(
    checkpoint_path,
    path=path,
    norm_path="./data/CrossDocked/processed/normalization_params.npz",
)
model.eval()
model.max_batched_samples = BATCH_SIZE
# --- End Environment Setup ---

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

print(f"Starting generation of {N_SAMPLES_TO_GENERATE} molecules...")

num_generated = 0
num_saved = 0
total_batches = (N_SAMPLES_TO_GENERATE + BATCH_SIZE - 1) // BATCH_SIZE

for i in tqdm(range(total_batches), desc="Generating Batches"):

    current_batch_size = min(BATCH_SIZE, N_SAMPLES_TO_GENERATE - num_generated)
    if current_batch_size <= 0:
        break

    # 1. Sample a batch of data objects
    res = model.sample(n_samples=current_batch_size, n_steps=N_STEPS, pocket=pocket)

    # Check if sampling failed (e.g., returned an empty result)
    if res is None or res.x_lig is None or res.x_lig.numel() == 0:
        tqdm.write("Warning: Sampling returned empty data. Skipping batch.")
        num_generated += current_batch_size
        continue

    # 2. Convert batch to a list of RDKit molecules
    # We use the batch_lig index to separate individual molecules from the batched tensor
    unique_batch_ids = torch.unique(res.batch_lig)
    batch_lig = res.batch_lig[res.batch_lig != -1]
    for mol_idx, batch_id in enumerate(unique_batch_ids):
        mask = batch_lig == batch_id

        lig_coords = res.x_lig[mask, :3].detach().cpu()
        lig_features = res.h_lig[mask, :].detach().cpu()

        # Total number of atoms in the sampled ligand
        total_atoms = lig_coords.shape[0]

        # Use your custom openbabel function to build the RDKit molecule
        rdmol = make_mol_openbabel(lig_coords, lig_features, atom_decoder)

        if rdmol is None:
            tqdm.write(
                f"Warning: RDKit molecule creation failed for sample {num_generated + 1}. Skipping."
            )
            num_generated += 1
            continue

        # 3. Find the largest connected fragment
        largest_frag, frag_size = get_largest_fragment_info(rdmol)

        # 4. Check the percentage threshold
        if total_atoms > 0:
            frag_percent = frag_size / total_atoms
        else:
            frag_percent = 0.0

        if frag_percent >= FRAGMENT_THRESHOLD_PERCENT:
            # 5. Save the molecule to the folder
            base_name = f"generated_mol_{num_saved}_frag_{frag_percent:.2f}"
            sdf_file_path = os.path.join(output_folder, f"{base_name}.sdf")
            png_file_path = os.path.join(
                output_folder, f"{base_name}.png"
            )  # <-- NEW PATH

            # Save the largest fragment only (this ensures we save the connected part)
            if largest_frag:
                writer = Chem.SDWriter(sdf_file_path)
                writer.write(largest_frag)
                writer.close()
                num_saved += 1
                tqdm.write(
                    f"Saved {base_name}.sdf. Fragment size: {frag_size}/{total_atoms} ({frag_percent:.2f})"
                )

            if largest_frag:
                try:
                    # Compute 2D coordinates for plotting
                    AllChem.Compute2DCoords(largest_frag)

                    # Draw and save the 2D image
                    Draw.MolToFile(largest_frag, png_file_path, size=(300, 300))
                    tqdm.write(
                        f"Saved {base_name}.sdf and {base_name}.png. Fragment size: {frag_size}/{total_atoms} ({frag_percent:.2f})"
                    )
                except Exception as e:
                    tqdm.write(
                        f"Warning: Failed to compute 2D coords or save PNG for molecule {num_saved}: {e}"
                    )
        num_generated += 1

        if num_generated >= N_SAMPLES_TO_GENERATE:
            break

print("\n--- Generation Summary ---")
print(f"Total samples generated: {num_generated}")
print(
    f"Total valid molecules saved (Fragment > {FRAGMENT_THRESHOLD_PERCENT:.0%}): {num_saved}"
)
print(f"Output directory: {output_folder}")
