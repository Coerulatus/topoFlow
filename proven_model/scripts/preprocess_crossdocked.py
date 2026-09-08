import torch
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from tqdm import tqdm

from flowr.util.pocket import PocketComplexBatch
from flowr.data.preprocess_pdbs import process_complex
import argparse


def main(args):
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    splits = torch.load(data_dir / "split_by_name.pt")
    crossdocked_dir = data_dir / "crossdocked_pocket10"

    for split, items in splits.items():
        if args.debug and split == "train":
            items = items[:100]
        elif args.debug and split == "test":
            items = items[:20]

        systems = []
        for pocket_pdb, ligand_sdf in tqdm(items, desc=f"Processing {split}"):
            pdb_path = crossdocked_dir / pocket_pdb
            sdf_path = crossdocked_dir / ligand_sdf

            if not pdb_path.exists() or not sdf_path.exists():
                print(f"Skipping {pocket_pdb} or {ligand_sdf} as it doesn't exist.")
                continue

            try:
                _complex = process_complex(
                    struct_path=str(pdb_path),
                    sdf_path=str(sdf_path),
                    txt_path=None,
                    remove_hs=True,
                    kekulize=False,
                    add_bonds_to_protein=True,
                    add_hs_to_protein=False,
                    pocket_cutoff=6.0,
                    cut_pocket=False,
                    compute_interactions=False,
                    split=split,
                )
                systems.append(_complex)
            except Exception as e:
                print(f"Error processing {pocket_pdb}: {e}")

        # save
        batch = PocketComplexBatch(systems)
        save_file = save_dir / f"{split}.smol"
        bytes_data = batch.to_bytes()
        save_file.write_bytes(bytes_data)
        print(f"Saved {len(systems)} systems to {save_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
