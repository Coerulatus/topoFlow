"""Calculate and save test set distributions for all metrics."""

import os
import argparse
import numpy as np
import torch
from tqdm import tqdm
import topoflow.utils.constants as constants
from topoflow.utils.metrics import get_distributions
from topoflow.data.datasets import CrossDocked


def main():
    """Main function to calculate and save test set distributions for all metrics."""
    parser = argparse.ArgumentParser(
        description="Calculate and save test set distributions for all metrics."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="crossdock_full",
        help="Dataset name in constants.",
    )
    parser.add_argument(
        "--split", type=str, default="test", help="Split to process (test/val)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save .npy result. Defaults to data_dir/processed/split_distribution.npy",
    )
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(
            args.data_dir, "processed", f"{args.split}_distribution.npy"
        )

    print(f"Loading {args.split} dataset from {args.data_dir}...")
    dataset = CrossDocked(root=args.data_dir, split=args.split)

    atom_decoder = constants.dataset_params[args.data_name]["atom_decoder"]

    # We include all relevant metrics
    metrics = ["qed", "logp", "sas", "lipinski", "gnina", "mol_size"]

    print(f"Calculating distributions for metrics: {metrics}")
    print(
        "This may take a while if GNINA is included and needs to run on the whole set..."
    )

    distribution = get_distributions(
        dataset, metrics=metrics, atom_decoder=atom_decoder
    )

    # Also add mol_id or name if possible for reference
    # distribution["names"] = dataset.data.names # If available

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    np.save(args.output_path, distribution)
    print(f"\nDistribution saved successfully to {args.output_path}")


if __name__ == "__main__":
    main()
