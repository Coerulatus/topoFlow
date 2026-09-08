import os
import torch
import numpy as np
from topoflow.data.datasets import CrossDocked
import topoflow.utils.constants as constants

def debug_coordinates(data_dir="./data/CrossDocked/", sample_idx=0):
    # 1. Load Normalization Params
    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    if not os.path.exists(norm_path):
        print(f"Normalization params not found at {norm_path}")
        return
    
    norm_data = np.load(norm_path, allow_pickle=True)
    mean = torch.tensor(norm_data["mean"]).float()
    std = torch.tensor(norm_data["std"]).float()
    print(f"Normalization - Mean: {mean}, Std: {std.mean().item():.4f}")

    # 2. Load Val Dataset
    dataset = CrossDocked(root=data_dir, split="val")
    data = dataset[sample_idx]
    
    x_lig_norm = data.x_lig
    x_poc_norm = data.x_pocket
    
    # 3. Unnormalize
    # Formula: x_real = (x_norm * std) + mean
    # Note: CrossDocked centers per-sample first, then applies global norm.
    # The 'mean' here is the global mean of the *already centered* pocket atoms from training.
    x_lig_real = (x_lig_norm * std) + mean
    x_poc_real = (x_poc_norm * std) + mean

    # 4. Calculate Distance
    dist = torch.cdist(x_lig_real, x_poc_real)
    min_dist = dist.min().item()
    print(f"\nSample {sample_idx}:")
    print(f"Minimum L-P distance in real space: {min_dist:.4f} Angstroms")
    
    # 5. Save to XYZ
    def save_xyz(coords, filename, atom_types=None):
        num_atoms = coords.size(0)
        with open(filename, "w") as f:
            f.write(f"{num_atoms}\n")
            f.write("Generated for inspection\n")
            for j in range(num_atoms):
                # Just use 'C' and 'O' for visualization if types are missing
                symbol = "C" if atom_types is None else "O" 
                f.write(f"{symbol} {coords[j,0]:.6f} {coords[j,1]:.6f} {coords[j,2]:.6f}\n")
        print(f"Saved to {filename}")

    save_xyz(x_lig_real, "debug_ligand.xyz")
    save_xyz(x_poc_real, "debug_pocket.xyz")

if __name__ == "__main__":
    debug_coordinates()
