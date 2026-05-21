"""Analyze entropy during the molecular sampling process.

This script loads a trained flow model and a protein pocket from the CrossDocked validation set.
It performs molecular sampling while tracking the entropy of the estimated vector field at various time steps.
The entropy is calculated for different types of interactions (Global Ligand, Global Pocket,
Ligand Self-loop, Local Radius/Cross) and plotted against the cardinality of the interactions
to visualize the model's uncertainty over time.

Parser Arguments:
    --checkpoint_path (str): Path to the model checkpoint.
    --data_dir (str): Directory containing the training/validation data.
    --output_dir (str): Directory to save the entropy plots.
    --n_atoms (int): Optional. Number of atoms for the sampled molecule. Defaults to the original molecule's size.

Files Produced:
    - PNG plots (e.g., entropy_t_0.1.png) showing Entropy vs Cardinality for specific time steps in the output_dir.
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import Batch

from topoflow.flow.flow_models import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
import topoflow.utils.constants as constants


def main():
    """Main function to analyze entropy during the molecular sampling process."""
    parser = argparse.ArgumentParser(description="Analyze entropy during sampling.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/home/marco/Documents/phd/topoFlow/checkpoints/runs_cluster/to_evaluate/03_11_1199e_ast_gat_scaled_vel.ckpt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Directory containing the training data.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs/entropy_analysis",
        help="Directory to save the plots.",
    )
    parser.add_argument(
        "--n_atoms",
        type=int,
        default=None,
        help="Number of atoms for the sampled molecule.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Load Dataset
    print("Loading validation dataset...")
    val_dataset = CrossDocked(root=args.data_dir, split="val")

    # 2. Setup Path
    alpha = CosineAlpha()
    beta = CosineBeta()
    p_data = DataLoaderSampleable(val_dataset, batch_size=1, shuffle=False)

    distribution_hist_path = os.path.join(
        args.data_dir, "processed", "size_distribution.npy"
    )
    distribution_hist = np.load(distribution_hist_path, allow_pickle=True)

    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=alpha,
        beta=beta,
        target_dim=14,
        distribution_hist=distribution_hist,
    )

    # 3. Load Model
    print(f"Loading model from {args.checkpoint_path}...")
    norm_path = os.path.join(args.data_dir, "processed", "normalization_params.npz")
    model = FlowModel.load_from_checkpoint(
        args.checkpoint_path, path=path, norm_path=norm_path, map_location=device
    )
    model.to(device)
    model.eval()

    # 4. Pick a pocket from validation set
    # Pick a random index or a fixed one for reproducibility
    pocket_idx = 0
    pocket_data = val_dataset[pocket_idx]
    print(f"Picked pocket index {pocket_idx}")

    n_atoms = args.n_atoms if args.n_atoms is not None else pocket_data.x_lig.shape[0]
    print(f"Sampling molecule with {n_atoms} atoms.")

    # 5. Sample with entropy
    entropy_times = [float(t) / 10.0 for t in range(11)]  # 0.0, 0.1, ..., 1.0
    print(f"Sampling with entropy at times: {entropy_times}")

    # We need to wrap pocket_data in a Batch or object that has the expected attributes
    # The sample method expects a 'pocket' object with x_pocket, h_pocket, batch_pocket
    class PocketWrapper:
        def __init__(self, data):
            self.x_pocket = data.x_pocket
            self.h_pocket = data.h_pocket
            self.batch_pocket = torch.zeros(
                data.x_pocket.shape[0], dtype=torch.long, device=data.x_pocket.device
            )

    wrapped_pocket = PocketWrapper(pocket_data)

    # Run sampling
    data_gen, entropies = model.sample_with_entropy(
        n_samples=1,
        n_steps=50,
        pocket=wrapped_pocket,
        entropy_times=entropy_times,
        n_atoms=n_atoms,
    )

    # 6. Plotting
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Plotting results to {args.output_dir}...")

    # Type Mapping:
    # 0: Global Ligand
    # 1: Global Pocket
    # 2: Ligand Self-loops
    # 3: Local Radius / Cross-Interface
    type_names = {
        0: "Global Ligand",
        1: "Global Pocket",
        2: "Ligand Self-loop",
        3: "Local Radius/Cross",
    }

    for t, diag_full in entropies.items():
        # diag_full is {"per_cardinality": diag, "summary": summary}
        diag = diag_full["per_cardinality"]
        # diag is {cardinality: [{'entropy': float, 'type': int}, ...]}
        data_list = []
        for card, items in diag.items():
            for item in items:
                data_list.append(
                    {
                        "Cardinality": card,
                        "Entropy": item["entropy"],
                        "Type": type_names.get(
                            item["type"], f"Unknown({item['type']})"
                        ),
                    }
                )

        df = pd.DataFrame(data_list)
        # select only type 3
        df = df[df["Type"] == "Local Radius/Cross"]

        if df.empty:
            print(f"No Local Radius/Cross data for t={t:.1f}")
            continue

        plt.figure(figsize=(10, 6))

        # Plot all values with scatter
        sns.scatterplot(
            data=df, x="Cardinality", y="Entropy", hue="Type", alpha=0.3, legend=False
        )

        # Plot line through the mean
        sns.lineplot(
            data=df,
            x="Cardinality",
            y="Entropy",
            hue="Type",
            marker="o",
            estimator="mean",
            errorbar=None,
        )

        plt.title(f"Entropy vs Cardinality at t={t:.1f} (Local Radius/Cross)")
        plt.grid(True, alpha=0.3)
        plt.ylim(-0.05, 1.05)  # Entropy is normalized [0, 1]

        save_path = os.path.join(args.output_dir, f"entropy_t_{t:.1f}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"Saved plot for t={t:.1f} to {save_path}")

    print("Analysis complete.")


if __name__ == "__main__":
    main()
