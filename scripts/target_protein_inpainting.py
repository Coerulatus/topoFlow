"""Script to generate ligand molecules starting from a base scaffold, conditioned on a target protein pocket using a pre-trained flow model."""

import os
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
from torch_geometric.data import Data

from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.utils.molecules import (
    pdb_to_data_object,
    make_mol_openbabel,
    mol_to_data,
    get_largest_fragment_info,
    extract_bonds,
)
from topoflow.utils.visualization import visualize_trajectory

if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 4  # Increased BATCH_SIZE for efficiency, you can adjust this
    N_SAMPLES_TO_GENERATE = 256
    N_STEPS = 80  # Number of steps in the flow sampling process
    FRAGMENT_THRESHOLD_PERCENT = 0.0  # % of atoms must be in the largest fragment
    T_MAX_INPAINTING = 1

    # New parameter for retry limit (optional but recommended to prevent infinite loops)
    MAX_ATTEMPTS = N_SAMPLES_TO_GENERATE * 5

    atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]
    atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]
    pdb_path = "./data/target_protein/wwp2_c2.pdb"
    data_path = "./data/target_protein/wwp2_c2_data_normalized.pt"
    normalization_path = "./data/CrossDocked/processed/normalization_params_full.npz"
    output_folder = "./outputs/generated_ligands_wwp2_c2/t_max_100_force_bonds"  # Folder to save passing molecules

    # --- Environment Setup (No changes needed, keeping original imports/setup) ---
    base_ligand_smile = "c1ccc2c(c1)[nH]cc2"
    rdkit_ligand = Chem.MolFromSmiles(base_ligand_smile)
    AllChem.Compute2DCoords(rdkit_ligand)
    bonds = extract_bonds(rdkit_ligand)
    coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)
    if normalization_path is not None:
        norm_params = np.load(normalization_path, allow_pickle=True)
        mean = norm_params["mean"]
        std = norm_params["std"]
        coords = (coords - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(
            std, dtype=torch.float32
        )
    base_ligand = Data(x_lig=coords, h_lig=one_hot, batch_lig=batch)

    pocket = pdb_to_data_object(
        pdb_file=pdb_path,
        atom_encoder=atom_encoder,
        save_path=data_path,
        normalization_path=normalization_path,
        center_pocket=False,
        force=True,
    )

    data_dir = "./data/CrossDocked/"
    checkpoint_path = "checkpoints/runs_cluster/10_21_499e_ast_gat_scaled_vel.ckpt"
    val_dataset = CrossDocked(root=data_dir, split="val")
    datamodule = MockDatamodule(
        train_length=1,
        val_length=len(val_dataset),
        test_length=1,
        batch_size=BATCH_SIZE,
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

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    print(f"Starting generation of {N_SAMPLES_TO_GENERATE} molecules...")

    num_saved = 0
    attempt_count = 0

    # Use tqdm to track the progress of SAVED molecules
    pbar = tqdm(total=N_SAMPLES_TO_GENERATE, desc="Molecules Saved")

    while num_saved < N_SAMPLES_TO_GENERATE and attempt_count < MAX_ATTEMPTS:
        attempt_count += 1

        # Determine the number of molecules to generate in this batch
        # We cap it at the BATCH_SIZE, even if we need fewer than BATCH_SIZE left
        current_batch_size = BATCH_SIZE

        # 1. Sample a batch of data objects
        res = model.inpainting(
            ligand=base_ligand,
            pocket=pocket,
            n_atoms_to_add=10,
            n_steps=N_STEPS,
            t_max_inpainting=T_MAX_INPAINTING,
        )

        # Check for immediate sampling failure (e.g., returned an empty result)
        if res is None or res.x_lig is None or res.x_lig.numel() == 0:
            tqdm.write("Warning: Sampling returned empty data. Retrying batch.")
            continue  # Retry the loop without incrementing num_saved

        # 2. Convert batch to a list of RDKit molecules and process them
        unique_batch_ids = torch.unique(res.batch_lig)
        batch_lig = res.batch_lig[res.batch_lig != -1]

        # Only iterate over the number of unique samples generated in the batch
        for batch_id in unique_batch_ids:
            if num_saved >= N_SAMPLES_TO_GENERATE:
                break  # Stop processing if target is reached

            if batch_id.item() == -1:
                continue

            mask = batch_lig == batch_id

            # Extract single molecule data (this part assumes your data structure)
            lig_coords = res.x_lig[mask].detach().cpu()
            lig_features = res.h_lig[mask].detach().cpu()

            # Total number of atoms in the sampled ligand
            total_atoms = lig_coords.shape[0]

            # Use your custom openbabel function to build the RDKit molecule
            rdmol = make_mol_openbabel(
                lig_coords, lig_features, atom_decoder, explicit_bonds=bonds
            )

            if rdmol is None:
                tqdm.write(
                    f"Warning: RDKit molecule creation failed for a sample in attempt {attempt_count}. Skipping."
                )
                continue  # Move to the next sample in the batch

            # 3. Find the largest connected fragment
            largest_frag, frag_size = get_largest_fragment_info(rdmol)

            # 4. Check the percentage threshold
            frag_percent = frag_size / total_atoms if total_atoms > 0 else 0.0

            if frag_percent >= FRAGMENT_THRESHOLD_PERCENT:
                # 5. Molecule passed all checks, proceed to save and count
                base_name = f"generated_mol_inpainting_{num_saved}"
                sdf_file_path = os.path.join(output_folder, f"{base_name}.sdf")
                png_file_path = os.path.join(output_folder, f"{base_name}.png")

                # Save the largest fragment only
                if largest_frag:
                    try:
                        writer = Chem.SDWriter(sdf_file_path)
                        writer.write(largest_frag)
                        writer.close()

                        # Compute 2D coordinates for plotting
                        AllChem.Compute2DCoords(largest_frag)

                        # Draw and save the 2D image
                        Draw.MolToFile(largest_frag, png_file_path, size=(300, 300))

                        # --- SUCCESS: Increment count and update progress bar ---
                        num_saved += 1
                        pbar.update(1)
                        tqdm.write(
                            f"Saved {base_name}.sdf (Attempt {attempt_count}). Fragment size: {frag_size}/{total_atoms} ({frag_percent:.2f})"
                        )
                    except Exception as e:
                        tqdm.write(
                            f"Warning: Failed to save molecule or draw PNG: {e}. Skipping count."
                        )
            else:
                tqdm.write(
                    f"Fragment percent {frag_percent:.2f} below threshold (Attempt {attempt_count}). Skipping saving."
                )

    pbar.close()

    if num_saved < N_SAMPLES_TO_GENERATE:
        print(
            f"\nGeneration finished. Only {num_saved}/{N_SAMPLES_TO_GENERATE} molecules were successfully saved after {attempt_count} attempts."
        )
    else:
        print(
            f"\nSuccessfully generated and saved {num_saved} molecules in {attempt_count} attempts."
        )
