import torch
from topoflow.flow.vector_field_models.allsettransformer import (
    AllSetTransformerVectorField,
)


def check_hypergraph():
    """
    Check the incidence matrix and GAT graph construction for the AllSetTransformerVectorField.
    Uses a batch size of 2 to ensure no cross-batch edges are created.
    """

    # 1. Prepare Mock Data (Batch size = 2)
    x_lig = torch.zeros((4, 14))
    # Batch 0 Ligand (2 atoms, close together)
    x_lig[0, :3] = torch.tensor([0.0, 0.0, 0.0])
    x_lig[1, :3] = torch.tensor([1.0, 0.0, 0.0])
    # Batch 1 Ligand (2 atoms, close together)
    x_lig[2, :3] = torch.tensor([0.0, 0.0, 0.0])
    x_lig[3, :3] = torch.tensor([1.0, 0.0, 0.0])

    x_pocket = torch.zeros((6, 14))
    # Batch 0 Pocket (3 atoms)
    x_pocket[0, :3] = torch.tensor([0.5, 0.5, 0.0])
    x_pocket[1, :3] = torch.tensor([2.0, 0.0, 0.0])
    x_pocket[2, :3] = torch.tensor([0.0, -0.5, 0.0])
    # Batch 1 Pocket (3 atoms)
    x_pocket[3, :3] = torch.tensor([0.5, 0.5, 0.0])
    x_pocket[4, :3] = torch.tensor([2.0, 0.0, 0.0])
    x_pocket[5, :3] = torch.tensor([0.0, -0.5, 0.0])

    batch_lig = torch.tensor([0, 0, 1, 1])
    batch_pocket = torch.tensor([0, 0, 0, 1, 1, 1])

    # 2. Instantiate Model and Set Cutoffs
    model = AllSetTransformerVectorField(h_dim=32, num_layers=1)
    model.cutoff_lig = 1.1
    model.cutoff_cross = 0.8
    model.use_gat = True

    print("=== Hypergraph & GAT Graph Construction Check ===")
    print(f"Cutoffs: Lig-Lig={model.cutoff_lig}A, Lig-Pocket={model.cutoff_cross}A\n")

    # 3. Generate Incidence and GAT edges
    print(
        "Edge LP:", model.create_edge_index(x_lig, x_pocket, batch_lig, batch_pocket)[1]
    )
    edge_index_v2e, edge_index_e2v, gat_edge_index, edge_types, final_batch_he = (
        model.create_incidence_with_types(
            x_lig, x_pocket, batch_lig, batch_pocket, num_graphs=2
        )
    )

    N_lig = x_lig.shape[0]
    N_poc = x_pocket.shape[0]
    N_total = N_lig + N_poc

    # --- Check GAT Graph Edges ---
    print("Checking GAT Graph edges for cross-batch leaks...")
    src, dst = gat_edge_index
    # Identify batch for each node in the global graph
    # Global nodes: [0..3] are ligands, [4..9] are pockets
    global_batch = torch.cat([batch_lig, batch_pocket], dim=0)

    src_batch = global_batch[src]
    dst_batch = global_batch[dst]

    cross_batch_gat = (src_batch != dst_batch).sum().item()
    print(f"Number of cross-batch GAT edges: {cross_batch_gat}")
    if cross_batch_gat > 0:
        bad_indices = torch.where(src_batch != dst_batch)[0]
        for idx in bad_indices:
            s, d = src[idx].item(), dst[idx].item()
            print(
                f"Bad Edge: {s} (Batch {global_batch[s].item()}) -> {d} (Batch {global_batch[d].item()})"
            )
    assert cross_batch_gat == 0, f"Found {cross_batch_gat} cross-batch GAT edges!"

    # Check if ligand-pocket cross edges exist and are correct (not scrambled)
    lp_edges = 0
    for s, d in zip(src.tolist(), dst.tolist()):
        if s < N_lig and d >= N_lig:
            lp_edges += 1
            assert global_batch[s] == global_batch[d], "Cross-batch LP edge!"
        if s >= N_lig and d < N_lig:
            lp_edges += 1
            assert global_batch[s] == global_batch[d], "Cross-batch PL edge!"

    print(f"Found {lp_edges} valid ligand-pocket cross edges.")
    assert lp_edges > 0, "No ligand-pocket cross edges found, check your cutoffs!"

    # --- Check Hyperedges ---
    print("\nChecking Hyperedges for cross-batch leaks...")
    num_hyperedges = edge_index_v2e[0].max().item() + 1
    dense = torch.zeros((N_total, num_hyperedges))
    dense[edge_index_v2e[1], edge_index_v2e[0]] = 1.0

    cross_batch_he = 0
    type_map = {0: "Global Ligand", 1: "Global Pocket", 2: "Local Interface"}

    for he_idx in range(num_hyperedges):
        nodes = torch.where(dense[:, he_idx] > 0)[0]
        if len(nodes) == 0:
            continue

        # Check if all nodes in this hyperedge belong to the same batch
        he_batches = global_batch[nodes]
        if len(torch.unique(he_batches)) > 1:
            cross_batch_he += 1
            print(
                f"WARNING: Hyperedge {he_idx} spans multiple batches: {he_batches.tolist()}"
            )

    print(f"Number of cross-batch Hyperedges: {cross_batch_he}")
    assert cross_batch_he == 0, f"Found {cross_batch_he} cross-batch hyperedges!"

    print("\nAll checks passed successfully!")


if __name__ == "__main__":
    check_hypergraph()
