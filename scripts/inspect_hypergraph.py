import os
import torch
import numpy as np
from torch_geometric.utils import scatter
from topoflow.data.datasets import CrossDocked
from topoflow.flow.vector_field_models import AllSetTransformerVectorField
import topoflow.utils.constants as constants

def inspect_hypergraph(data_dir="./data/CrossDocked/", num_samples=3):
    # 1. Load Normalization Params
    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    if not os.path.exists(norm_path):
        print(f"Normalization params not found at {norm_path}")
        return
    
    norm_data = np.load(norm_path, allow_pickle=True)
    norm_std = norm_data["std"]
    print(f"Loaded norm_std: {norm_std.mean():.4f} (mean)")

    # 2. Load Val Dataset
    dataset = CrossDocked(root=data_dir, split="val")
    print(f"Loaded validation set with {len(dataset)} samples.\n")

    # 3. Initialize Model (ast_gat)
    model = AllSetTransformerVectorField(
        h_dim=128,
        num_heads=16,
        num_layers=2,
        use_gat=True,
        norm_std=norm_std
    )
    model.eval()

    for i in range(num_samples):
        data = dataset[i]
        x_lig_gt = data.x_lig
        x_pocket = data.x_pocket
        batch_lig = torch.zeros(x_lig_gt.size(0), dtype=torch.long)
        batch_pocket = torch.zeros(x_pocket.size(0), dtype=torch.long)
        
        print(f"--- Sample {i} ({x_lig_gt.size(0)} ligand atoms, {x_pocket.size(0)} pocket atoms) ---")
        
        # Debug distances for t=1
        dist_lp = torch.cdist(x_lig_gt, x_pocket)
        min_lp = dist_lp.min().item()
        print(f"  [GT Debug] Min L-P distance (normalized): {min_lp:.4f}")
        print(f"  [GT Debug] Ligand mean pos: {x_lig_gt.mean(0)}")
        print(f"  [GT Debug] Pocket mean pos: {x_pocket.mean(0)}")
        print(f"  [GT Debug] Model cutoff (normalized): {model.cutoff_cross:.4f}")

        for label, x_lig in [("t=1 (GT)", x_lig_gt), ("t=0 (Gaussian)", torch.randn_like(x_lig_gt))]:
            # Get Hypergraph structure
            edge_index_v2e, _, gat_edge_index, he_types, batch_he = model.create_incidence_with_types(
                x_lig, x_pocket, batch_lig, batch_pocket, num_graphs=1
            )

            # 1. Hyperedge Stats
            num_he = he_types.size(0)
            ones = torch.ones(edge_index_v2e.size(1), device=edge_index_v2e.device)
            cardinalities = scatter(ones, edge_index_v2e[0], dim=0, reduce="sum")
            
            # Group hyperedges by type (0: lig_centroid, 1: poc_centroid, 2: local_ligand)
            print(f"  [{label}]")
            print(f"    Hyperedges total: {num_he}")
            for t in range(3):
                mask = he_types == t
                count = mask.sum().item()
                if count > 0:
                    cards_t = cardinalities[mask]
                    print(f"      Type {t} count: {count}, mean cardinality: {cards_t.mean().item():.2f}")

            # 2. GAT Edge Stats
            # Re-run create_edge_index to get separated counts
            edge_ll, edge_lp, _ = model.create_edge_index(
                x_lig, x_pocket, batch_lig, batch_pocket
            )
            # Re-calculate pp by hand since it's filtered inside the model usually
            # But we can just use the counts from the logic:
            # edge_index is [ll, pp, lp, pl]
            
            num_ll = edge_ll.size(1)
            num_lp = edge_lp.size(1)
            # Total GAT edges
            num_gat = gat_edge_index.size(1)
            # pp is the rest (excluding pl which is same as lp)
            num_pp = num_gat - num_ll - 2 * num_lp
            
            print(f"    GAT Edges total: {num_gat}")
            print(f"      LL edges: {num_ll}")
            print(f"      LP edges: {num_lp}")
            print(f"      PP edges: {num_pp}")
            print("")

if __name__ == "__main__":
    inspect_hypergraph()
