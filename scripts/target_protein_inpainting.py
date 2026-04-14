"""Script to generate ligand molecules starting from a base scaffold, conditioned on a target protein pocket using a pre-trained flow model."""

import argparse
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
from topoflow.flow.utils import DistributionNodes
from topoflow.utils.molecules import (
    pdb_to_data_object,
    make_mol_openbabel,
    mol_to_data,
    get_largest_fragment_info,
    extract_bonds,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ligands using inpainting conditioned on a protein pocket."
    )
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument(
        "--n_samples", type=int, default=1024, help="Number of samples to generate."
    )
    parser.add_argument("--n_steps", type=int, default=80, help="Number of flow steps.")
    parser.add_argument(
        "--fragment_threshold",
        type=float,
        default=0.0,
        help="Fragment size threshold percent.",
    )
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=None,
        help="Maximum attempts. Defaults to 5 * n_samples.",
    )
    parser.add_argument(
        "--t_max_values",
        nargs="+",
        type=float,
        default=[0.9, 1.0],
        help="Grid search t_max values.",
    )
    parser.add_argument(
        "--force_bonds_values",
        nargs="+",
        type=str,
        default=["True", "False"],
        help="Grid search force_bonds values (True/False).",
    )
    parser.add_argument(
        "--pdb_path",
        type=str,
        default="./data/target_protein/wwp2_c2.pdb",
        help="Path to protein PDB.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/target_protein/wwp2_c2_data_normalized.pt",
        help="Path to save processed protein data.",
    )
    parser.add_argument(
        "--normalization_path",
        type=str,
        default="./data/CrossDocked/processed/normalization_params.npz",
        help="Path to normalization params.",
    )
    parser.add_argument(
        "--base_output_dir",
        type=str,
        default="./outputs/generated_ligands_wwp2_c2",
        help="Base output directory.",
    )
    parser.add_argument(
        "--base_ligand_smile",
        type=str,
        default="c1ccc2c(c1)[nH]cc2",
        help="Base ligand SMILES.",
    )

    args = parser.parse_args()

    # Process grid search parameters
    T_MAX_VALUES = args.t_max_values
    FORCE_BONDS_VALUES = [
        v.lower() in ("true", "yes", "t", "1") for v in args.force_bonds_values
    ]

    if args.max_attempts is None:
        args.max_attempts = args.n_samples * 5

    atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]
    atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]

    # --- 1. Global Setup ---
    print("Initializing environment and loading models...")
    print(f"Ligand smiles: {args.base_ligand_smile}")
    rdkit_ligand = Chem.MolFromSmiles(args.base_ligand_smile)
    AllChem.EmbedMolecule(rdkit_ligand, AllChem.ETKDG())

    extracted_bonds_list = extract_bonds(rdkit_ligand)

    coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)
    if args.normalization_path is not None:
        norm_params = np.load(args.normalization_path, allow_pickle=True)
        mean = norm_params["mean"]
        std = norm_params["std"]
        coords = (coords - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(
            std, dtype=torch.float32
        )
    base_ligand = Data(x_lig=coords, h_lig=one_hot, batch_lig=batch)

    pocket = pdb_to_data_object(
        pdb_file=args.pdb_path,
        atom_encoder=atom_encoder,
        save_path=args.data_path,
        normalization_path=args.normalization_path,
        center_pocket=False,
        force=True,
    )

    data_dir = "./data/CrossDocked/"
    checkpoint_path = "/data/petretto/home/gmsv2427/Documents/topoFlow/checkpoints/to_evaluate/03_11_1199e_ast_gat_scaled_vel.ckpt"
    val_dataset = CrossDocked(root=data_dir, split="val")
    datamodule = MockDatamodule(
        train_length=1,
        val_length=len(val_dataset),
        test_length=1,
        batch_size=args.batch_size,
    )
    alpha = CosineAlpha()
    beta = CosineBeta()
    p_data = DataLoaderSampleable(
        val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )
    distribution_hist_path = "/data/petretto/home/gmsv2427/Documents/topoFlow/data/CrossDocked/processed/size_distribution_full.npy"
    distribution_hist = np.load(distribution_hist_path, allow_pickle=True)
    dist_nodes = DistributionNodes(distribution_hist)

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
    model.max_batched_samples = args.batch_size

    # Ensure base output directory exists
    os.makedirs(args.base_output_dir, exist_ok=True)

    # List to hold our summary metrics for the CSV
    summary_results = []

    # Get number of atoms in pocket and base ligand for sampling
    n_pocket_atoms = pocket.x_pocket.shape[0]
    n_base_ligand_atoms = base_ligand.x_lig.shape[0]

    # --- 2. Parameter Grid Search Loop ---
    for t_max_val in T_MAX_VALUES:
        for force_val in FORCE_BONDS_VALUES:

            t_max_str = str(t_max_val).replace(".", "_")
            output_folder = os.path.join(
                args.base_output_dir, f"t_max_{t_max_str}_force_bonds_{force_val}"
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
            pbar = tqdm(
                total=args.n_samples,
                desc=f"Molecules Saved (t={t_max_val}, force={force_val})",
            )

            while num_saved < args.n_samples and attempt_count < args.max_attempts:
                # Determine how many to sample in this batch
                current_batch_size = min(
                    args.batch_size, args.max_attempts - attempt_count
                )
                if current_batch_size <= 0:
                    break

                attempt_count += current_batch_size

                # Sample target ligand size from distribution given pocket size for the whole batch
                target_ligand_sizes = dist_nodes.sample_conditional(
                    n2=torch.tensor([n_pocket_atoms] * current_batch_size)
                )
                n_atoms_to_add = torch.clamp(
                    target_ligand_sizes - n_base_ligand_atoms, min=1
                )  # Force at least 1 atom to be added to grow the molecule

                res = model.inpainting(
                    ligand=base_ligand,
                    pocket=pocket,
                    n_atoms_to_add=n_atoms_to_add,
                    n_steps=args.n_steps,
                    t_max_inpainting=t_max_val,
                    batch_size=current_batch_size,
                )

                if res is None or res.x_lig is None or res.x_lig.numel() == 0:
                    tqdm.write("Warning: Sampling returned empty data. Retrying batch.")
                    continue

                unique_batch_ids = torch.unique(res.batch_lig)
                # Filter out -1 if present
                unique_batch_ids = unique_batch_ids[unique_batch_ids != -1]

                for batch_id in unique_batch_ids:
                    if num_saved >= args.n_samples:
                        break

                    mask = res.batch_lig == batch_id
                    lig_coords = res.x_lig[mask].detach().cpu()
                    lig_features = res.h_lig[mask].detach().cpu()
                    total_atoms = lig_coords.shape[0]

                    bonds_to_use = extracted_bonds_list if force_val else None

                    rdmol = make_mol_openbabel(
                        lig_coords,
                        lig_features,
                        atom_decoder,
                        explicit_bonds=bonds_to_use,
                    )

                    if rdmol is None:
                        continue

                    largest_frag, frag_size = get_largest_fragment_info(rdmol)
                    frag_percent = frag_size / total_atoms if total_atoms > 0 else 0.0

                    if frag_percent >= args.fragment_threshold:
                        base_name = f"generated_mol_inpainting_{num_saved}"
                        sdf_file_path = os.path.join(output_folder, f"{base_name}.sdf")
                        png_file_path = os.path.join(output_folder, f"{base_name}.png")

                        if largest_frag:
                            try:
                                writer = Chem.SDWriter(sdf_file_path)
                                writer.write(largest_frag)
                                writer.close()
                                AllChem.Compute2DCoords(largest_frag)
                                Draw.MolToFile(
                                    largest_frag, png_file_path, size=(300, 300)
                                )

                                num_saved += 1
                                pbar.update(1)
                            except Exception as e:
                                pass

            pbar.close()

            # Calculate success rate and log the metrics for this run
            success_rate = (
                (num_saved / attempt_count * 100) if attempt_count > 0 else 0.0
            )
            summary_results.append(
                {
                    "t_max_inpainting": t_max_val,
                    "force_bonds": force_val,
                    "molecules_saved": num_saved,
                    "total_attempts": attempt_count,
                    "success_rate_percent": round(success_rate, 2),
                }
            )

            if num_saved < args.n_samples:
                print(
                    f"-> Run finished early. Saved {num_saved}/{args.n_samples} molecules in {attempt_count} attempts."
                )
            else:
                print(
                    f"-> Run successful. Saved {num_saved} molecules in {attempt_count} attempts."
                )

    # --- 3. Save Summary CSV ---
    csv_file_path = os.path.join(args.base_output_dir, "generation_summary.csv")
    print(f"\nWriting summary logs to: {csv_file_path}")

    with open(csv_file_path, mode="w", newline="") as csv_file:
        fieldnames = [
            "t_max_inpainting",
            "force_bonds",
            "molecules_saved",
            "total_attempts",
            "success_rate_percent",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for row in summary_results:
            writer.writerow(row)

    print("All combinations completed and logged successfully!")
