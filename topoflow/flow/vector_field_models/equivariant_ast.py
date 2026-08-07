"""Equivariant AllSet Transformer modules for hypergraph neural networks."""

import torch
import torch.nn as nn
from torch_scatter import scatter


class EquivariantAllSetTransformerLayer(nn.Module):
    """Equivariant V2E and E2V block."""

    def __init__(self, h_dim, num_heads=4):
        """Initialize the layer."""
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = num_heads

        # V2E: Nodes to Hyperedges
        self.v2e_attn = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, 1)
        )
        self.v2e_msg = nn.Sequential(
            nn.Linear(h_dim + 1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

        # E2V: Hyperedges to Nodes
        self.e2v_attn = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.SiLU(), nn.Linear(h_dim, 1)
        )
        self.e2v_msg = nn.Sequential(
            nn.Linear(h_dim + 1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

        # E2V: Coordinate update
        self.e2v_coord = nn.Sequential(
            nn.Linear(h_dim * 2 + 1, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, 1, bias=False),
        )

        self.node_update = nn.Sequential(
            nn.Linear(h_dim * 2, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

    def forward(self, x, h, edge_index_v2e, edge_index_e2v, node_mask=None):
        """Forward pass of the equivariant AllSet layer.

        x: [N, 3]
        h: [N, h_dim]
        edge_index_v2e: [2, E] (row 0: hyperedge, row 1: node)
        node_mask: [N, 1] boolean or float mask indicating which nodes can move (e.g. ligand)
        """
        num_nodes = h.size(0)
        num_hyperedges = (
            edge_index_v2e[0].max().item() + 1 if edge_index_v2e.numel() > 0 else 0
        )

        if num_hyperedges == 0:
            return x, h

        he_idx, n_idx = edge_index_v2e[0], edge_index_v2e[1]

        # --- V2E (Nodes to Hyperedges) ---
        # 1. Compute attention for hyperedge coordinate centroid
        alpha_v2e = self.v2e_attn(h[n_idx])  # [E, 1]
        # Softmax over each hyperedge
        alpha_max = scatter(
            alpha_v2e, he_idx, dim=0, dim_size=num_hyperedges, reduce="max"
        )
        alpha_exp = torch.exp(alpha_v2e - alpha_max[he_idx])
        alpha_sum = scatter(
            alpha_exp, he_idx, dim=0, dim_size=num_hyperedges, reduce="sum"
        )
        alpha_norm = alpha_exp / (alpha_sum[he_idx] + 1e-8)

        # 2. Equivariant hyperedge center
        x_e = scatter(
            alpha_norm * x[n_idx],
            he_idx,
            dim=0,
            dim_size=num_hyperedges,
            reduce="sum",
        )

        # 3. Invariant distances
        dist_sq_v2e = torch.sum((x[n_idx] - x_e[he_idx]) ** 2, dim=-1, keepdim=True)
        dist_v2e = torch.sqrt(dist_sq_v2e + 1e-8)

        # 4. Invariant messages to hyperedge
        msg_v2e_input = torch.cat([h[n_idx], dist_v2e], dim=-1)
        m_v2e = self.v2e_msg(msg_v2e_input)
        h_e = scatter(
            alpha_norm * m_v2e,
            he_idx,
            dim=0,
            dim_size=num_hyperedges,
            reduce="sum",
        )

        # --- E2V (Hyperedges to Nodes) ---
        n_idx2, he_idx2 = edge_index_e2v[0], edge_index_e2v[1]

        # 1. Attention for E2V
        e2v_attn_input = torch.cat([h[n_idx2], h_e[he_idx2]], dim=-1)
        beta_e2v = self.e2v_attn(e2v_attn_input)
        beta_max = scatter(beta_e2v, n_idx2, dim=0, dim_size=num_nodes, reduce="max")
        beta_exp = torch.exp(beta_e2v - beta_max[n_idx2])
        beta_sum = scatter(beta_exp, n_idx2, dim=0, dim_size=num_nodes, reduce="sum")
        beta_norm = beta_exp / (beta_sum[n_idx2] + 1e-8)

        # 2. Invariant distances for E2V
        dist_sq_e2v = torch.sum((x[n_idx2] - x_e[he_idx2]) ** 2, dim=-1, keepdim=True)
        dist_e2v = torch.sqrt(dist_sq_e2v + 1e-8)

        # 3. Coordinate update (Equivariant)
        coord_update_input = torch.cat([h[n_idx2], h_e[he_idx2], dist_e2v], dim=-1)
        phi_x = self.e2v_coord(coord_update_input)  # [E, 1]
        delta_x_msg = (x[n_idx2] - x_e[he_idx2]) * phi_x * beta_norm
        delta_x = scatter(delta_x_msg, n_idx2, dim=0, dim_size=num_nodes, reduce="sum")

        if node_mask is not None:
            delta_x = delta_x * node_mask

        x_out = x + delta_x

        # 4. Feature update
        msg_e2v_input = torch.cat([h_e[he_idx2], dist_e2v], dim=-1)
        m_e2v = self.e2v_msg(msg_e2v_input)
        h_update_msg = scatter(
            beta_norm * m_e2v, n_idx2, dim=0, dim_size=num_nodes, reduce="sum"
        )

        h_out = h + self.node_update(torch.cat([h, h_update_msg], dim=-1))

        return x_out, h_out


class EquivariantAllSetTransformer(nn.Module):
    """Equivariant AllSet Transformer network."""

    def __init__(self, h_dim, num_layers=2):
        """Initialize the network."""
        super().__init__()
        self.layers = nn.ModuleList(
            [EquivariantAllSetTransformerLayer(h_dim) for _ in range(num_layers)]
        )

    def forward(self, x, h, edge_index_v2e, edge_index_e2v, node_mask=None):
        """Forward pass for the network."""
        for layer in self.layers:
            x, h = layer(x, h, edge_index_v2e, edge_index_e2v, node_mask=node_mask)
        return x, h
