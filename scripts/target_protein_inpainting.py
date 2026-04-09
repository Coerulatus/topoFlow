"""Script to generate ligand molecules starting from a base scaffold, conditioned on a target protein pocket using a pre-trained flow model."""

import os
import csv
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
    make_mol_openbabel_with_bonds,
    mol_to_data,
    get_largest_fragment_info,
    extract_bonds,
)

if __name__ == "__main__":
    # --- Configuration ---
    BATCH_SIZE = 4
    N_SAMPLES_TO_GENERATE = 256
    N_STEPS = 80
    FRAGMENT_THRESHOLD_PERCENT = 0.0
    MAX_ATTEMPTS = N_SAMPLES_TO_GENERATE * 5

    # Grid search parameters
    T_MAX_VALUES = [0.9, 1.0]
    FORCE_BONDS_VALUES = [True, False]

    atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]
    atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]
    pdb_path = "./data/target_protein/wwp2_hect.pdb"
    data_path = "./data/target_protein/wwp2_hect_data_normalized.pt"
    normalization_path = "/data/petretto/home/gmsv2427/Documents/topoFlow/data/CrossDocked/processed/normalization_params_full.npz"
    base_output_dir = "./outputs/generated_ligands_wwp2_hect"

    # --- 1. Global Setup ---
    print("Initializing environment and loading models...")
    base_ligand_smile = "c1ccc2c(c1)[nH]cc2"

    # Conditional setup: only process the base ligand if a SMILES string is provided
    if base_ligand_smile:
        print(f"Base ligand provided: {base_ligand_smile}")
        rdkit_ligand = Chem.MolFromSmiles(base_ligand_smile)
        AllChem.Compute2DCoords(rdkit_ligand)

        extracted_bonds = extract_bonds(rdkit_ligand)

        coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)
        if normalization_path is not None:
            norm_params = np.load(normalization_path, allow_pickle=True)
            mean = norm_params["mean"]
            std = norm_params["std"]
            coords = (coords - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(
                std, dtype=torch.float32
            )
        base_ligand = Data(x_lig=coords, h_lig=one_hot, batch_lig=batch)
    else:
        print("No base ligand provided. Will use standard sampling.")
        extracted_bonds = None
        base_ligand = None

    pocket = pdb_to_data_object(
        pdb_file=pdb_path,
        atom_encoder=atom_encoder,
        save_path=data_path,
        normalization_path=normalization_path,
        center_pocket=False,
        force=True,
    )

    data_dir = "./data/CrossDocked/"
    checkpoint_path = "/data/petretto/home/gmsv2427/Documents/topoFlow/checkpoints/topoflow/run_02_11_1200e_ast_gat_scaled_vel/topoflow-epoch=1199.ckpt"
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

    # Ensure base output directory exists
    os.makedirs(base_output_dir, exist_ok=True)

    # List to hold our summary metrics for the CSV
    summary_results = []

    # --- 2. Parameter Grid Search Loop ---
    for t_max_val in T_MAX_VALUES:
        for force_val in FORCE_BONDS_VALUES:

            t_max_str = str(t_max_val).replace(".", "_")
            output_folder = os.path.join(
                base_output_dir, f"t_max_{t_max_str}_force_bonds_{force_val}"
            )
            os.makedirs(output_folder, exist_ok=True)

            print(f"\n{'='*50}")
            print(
                f"Starting Run: t_max_inpainting={t_max_val}, force_bonds={force_val}"
            )
            print(f"Saving to: {output_folder}")
            print(f"{'='*50}")

            num_saved = 0
            attempt_count = 0
            total_attempted_molecules = 0  # Tracks total individual molecules generated

            pbar = tqdm(
                total=N_SAMPLES_TO_GENERATE,
                desc=f"Molecules Saved (t={t_max_val}, force={force_val})",
            )

            while num_saved < N_SAMPLES_TO_GENERATE and attempt_count < MAX_ATTEMPTS:
                attempt_count += 1

                if not base_ligand_smile:
                    res = model.sample(
                        n_samples=BATCH_SIZE, n_steps=N_STEPS, pocket=pocket
                    )
                else:
                    res = model.inpainting(
                        ligand=base_ligand,
                        pocket=pocket,
                        n_atoms_to_add=10,
                        n_steps=N_STEPS,
                        t_max_inpainting=t_max_val,
                    )

                if res is None or res.x_lig is None or res.x_lig.numel() == 0:
                    tqdm.write("Warning: Sampling returned empty data. Retrying batch.")
                    continue

                unique_batch_ids = torch.unique(res.batch_lig)
                valid_batch_ids = [bid for bid in unique_batch_ids if bid != -1]

                # Add the actual number of molecules produced in this pass
                total_attempted_molecules += len(valid_batch_ids)

                batch_lig = res.batch_lig[res.batch_lig != -1]

                for batch_id in valid_batch_ids:
                    if num_saved >= N_SAMPLES_TO_GENERATE:
                        break

                    mask = batch_lig == batch_id
                    lig_coords = res.x_lig[mask].detach().cpu()
                    lig_features = res.h_lig[mask].detach().cpu()
                    total_atoms = lig_coords.shape[0]

                    bonds_to_use = (
                        extracted_bonds if (force_val and base_ligand_smile) else None
                    )

                    rdmol = make_mol_openbabel_with_bonds(
                        lig_coords,
                        lig_features,
                        atom_decoder,
                        explicit_bonds=bonds_to_use,
                    )

                    if rdmol is None:
                        continue

                    largest_frag, frag_size = get_largest_fragment_info(rdmol)

                    # --- NEW: Strict RDKit Validity Check ---
                    if largest_frag is None:
                        continue

                    try:
                        # Sanitize checks valences, aromaticity, and basic chemical logic
                        Chem.SanitizeMol(largest_frag)
                    except Exception:
                        # Discard this invalid molecule and move to the next in the batch
                        continue
                    # ----------------------------------------

                    frag_percent = frag_size / total_atoms if total_atoms > 0 else 0.0

                    if frag_percent >= FRAGMENT_THRESHOLD_PERCENT:
                        prefix = (
                            "generated_mol_sample"
                            if not base_ligand_smile
                            else "generated_mol_inpainting"
                        )
                        base_name = f"{prefix}_{num_saved}"
                        sdf_file_path = os.path.join(output_folder, f"{base_name}.sdf")
                        png_file_path = os.path.join(output_folder, f"{base_name}.png")

                        try:
                            writer = Chem.SDWriter(sdf_file_path)
                            writer.write(largest_frag)
                            writer.close()
                            AllChem.Compute2DCoords(largest_frag)
                            Draw.MolToFile(largest_frag, png_file_path, size=(300, 300))

                            num_saved += 1
                            pbar.update(1)
                        except Exception as e:
                            pass

            pbar.close()

            # Calculate success rate using the true number of individual molecules attempted
            success_rate = (
                (num_saved / total_attempted_molecules * 100)
                if total_attempted_molecules > 0
                else 0.0
            )

            summary_results.append(
                {
                    "t_max_inpainting": t_max_val,
                    "force_bonds": force_val,
                    "molecules_saved": num_saved,
                    "model_calls": attempt_count,
                    "total_molecules_generated": total_attempted_molecules,
                    "success_rate_percent": round(success_rate, 2),
                }
            )

            if num_saved < N_SAMPLES_TO_GENERATE:
                print(
                    f"-> Run finished early. Saved {num_saved}/{N_SAMPLES_TO_GENERATE} molecules out of {total_attempted_molecules} total generated candidates."
                )
            else:
                print(
                    f"-> Run successful. Saved {num_saved} molecules out of {total_attempted_molecules} total generated candidates."
                )

    # --- 3. Save Summary CSV ---
    csv_file_path = os.path.join(base_output_dir, "generation_summary.csv")
    print(f"\nWriting summary logs to: {csv_file_path}")

    with open(csv_file_path, mode="w", newline="") as csv_file:
        # Added new columns for better clarity
        fieldnames = [
            "t_max_inpainting",
            "force_bonds",
            "molecules_saved",
            "model_calls",
            "total_molecules_generated",
            "success_rate_percent",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in summary_results:
            writer.writerow(row)

    print("All combinations completed and logged successfully!")
