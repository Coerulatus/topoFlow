"""Utility script for counting and splitting .smol dataset files."""

import argparse
import pickle
import random
from pathlib import Path


def count_and_optionally_split(data_dir, split_ratio=0.05, force_split=False):
    """Count smol files and split validation from train if missing."""
    data_dir = Path(data_dir)

    partitions = ["train", "val", "test"]
    counts = {}
    data_cache = {}

    # 1. Count the molecules
    print(f"--- Counting molecules in {data_dir} ---")
    for split in partitions:
        path = data_dir / f"{split}.smol"
        if not path.exists():
            print(f"{split:5s} : 0 pairs (File not found)")
            counts[split] = 0
            continue

        bytes_data = path.read_bytes()
        try:
            data_list = pickle.loads(bytes_data)
            counts[split] = len(data_list)
            data_cache[split] = data_list
            print(f"{split:5s} : {counts[split]} pairs")
        except Exception as e:
            print(f"{split:5s} : Error reading file - {e}")
            counts[split] = 0

    # 2. Handle missing validation set
    if counts.get("val", 0) == 0 or force_split:
        print("\n--- Validation set is empty or missing ---")
        if counts.get("train", 0) > 0:
            train_data = data_cache.get("train", [])
            val_size = int(len(train_data) * split_ratio)

            if val_size > 0:
                print(
                    f"Splitting {split_ratio*100}% of training data ({val_size} pairs) into validation set..."
                )

                # Shuffle and split
                random.shuffle(train_data)
                new_val_data = train_data[:val_size]
                new_train_data = train_data[val_size:]

                # Save new splits
                train_path = data_dir / "train.smol"
                val_path = data_dir / "val.smol"

                train_path.write_bytes(pickle.dumps(new_train_data))
                val_path.write_bytes(pickle.dumps(new_val_data))

                print("Successfully created new train.smol and val.smol!")
                print(f"New train: {len(new_train_data)} pairs")
                print(f"New val  : {len(new_val_data)} pairs")
            else:
                print("Not enough training data to create a validation split.")
        else:
            print("Training set is also empty. Cannot create a validation split.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Count smol dataset pairs and handle missing validation."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/CrossDocked/processed_flowr",
        help="Path to the directory containing .smol files",
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.05,
        help="Fraction of training data to move to validation if validation is empty",
    )
    parser.add_argument(
        "--force_split",
        action="store_true",
        help="Force recreating the validation split from train even if val.smol exists",
    )

    args = parser.parse_args()
    count_and_optionally_split(args.data_dir, args.split_ratio, args.force_split)
