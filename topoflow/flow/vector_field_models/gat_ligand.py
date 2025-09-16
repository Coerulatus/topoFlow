"""GAT-based vector field model for ligand-only prediction."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import radius_graph, radius
from torch_geometric.utils import to_undirected


class SimpleTimeEmbedding(nn.Module):
    """Simple time embedding."""

    def __init__(self, h_dim):
        """Simple time embedding."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

    def forward(self, t):
        """Forward pass."""
        return self.net(t.view(-1, 1))


class GATVectorField(nn.Module):
    """Ligand-only prediction with GATv2 model."""

    def __init__(
        self,
        h_dim,
        spatial_dim=3,
        atom_dim=11,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
        add_time_to="ligand",  # "ligand", "both", or "none"
    ):
        """Ligand-only prediction with self-attention inside each ligand and cross-attention to pocket atoms."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.x_dim = spatial_dim + atom_dim
        self.h_dim = h_dim
        self.add_time_to = add_time_to

        self.cutoff_lig = 10.0
        self.cutoff_poc = 3.0
        self.cutoff_cross = 3.0

        self.lig_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.poc_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.time_embed = SimpleTimeEmbedding(h_dim)

        self.layers = nn.ModuleList(
            [
                GATv2Conv(h_dim, h_dim // num_heads, heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

        # output ligand velocity only
        self.out_vel = nn.Sequential(
            nn.LayerNorm(h_dim), nn.Linear(h_dim, self.x_dim, bias=False)
        )

    def create_edge_index(self, x_lig, x_pocket, batch_lig, batch_pocket):
        """Create edge index for GATv2Conv."""
        lig_pos = x_lig[:, :3]
        poc_pos = x_pocket[:, :3]

        N1, N2 = lig_pos.size(0), poc_pos.size(0)

        edge_ll = radius_graph(
            lig_pos, r=self.cutoff_lig, batch=batch_lig, loop=False
        )  # [2, E_ll]
        # Intra-pocket edges
        edge_pp = radius_graph(
            poc_pos, r=self.cutoff_poc, batch=batch_pocket, loop=False
        )  # [2, E_pp]

        # Cross edges using bipartite radius:
        # radius(x=poc, y=lig) returns [row_y, col_x] = [lig_idx, poc_idx]
        edge_lp = radius(
            poc_pos,
            lig_pos,
            r=self.cutoff_cross,
            batch_x=batch_pocket,
            batch_y=batch_lig,
        )  # [2, E_lp] with row in [0..N1-1], col in [0..N2-1]

        # Offset pocket node ids by N1 to form one homogeneous graph:
        # lig->lig stays as-is; poc->poc gets +N1 on both ends
        edge_pp_global = edge_pp + N1
        # lig->poc: src=lig, dst=poc+N1
        edge_lp_global = torch.stack([edge_lp[0], edge_lp[1] + N1], dim=0)

        # Final edge index (directed); deduplicate if desired
        edge_index = torch.cat([edge_ll, edge_pp_global, edge_lp_global], dim=1)

        # Make undirected
        edge_index = to_undirected(edge_index, num_nodes=N1 + N2)

        return edge_index

    def forward(self, x_lig, t, x_pocket, batch_lig, batch_pocket):
        """Forward pass."""
        device = x_lig.device
        B = int(torch.max(batch_lig) + 1)

        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        # Embed tokens
        lig_h = self.lig_embed(x_lig)
        poc_h = self.poc_embed(x_pocket)

        # Time embeddings
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        if t.ndim == 0:
            t = t.repeat(B)
        t_emb = self.time_embed(t)  # [B, h]
        if self.add_time_to in ("ligand", "both"):
            lig_h = lig_h + t_emb[batch_lig[is_lig], :]
        if self.add_time_to in ("both",):
            poc_h = poc_h + t_emb[batch_pocket[is_poc], :]

        # Create edge index for GATv2Conv
        edge_index = self.create_edge_index(
            x_lig, x_pocket, batch_lig[is_lig], batch_pocket[is_poc]
        )

        # Creates features for GATv2Conv
        x = torch.cat([lig_h, poc_h], dim=0)

        # Encoder
        for layer in self.layers:
            x = layer(x, edge_index)

        N1 = lig_h.size(0)
        x_lig = x[:N1, :]

        # Predict ligand velocities only
        vel_lig = self.out_vel(x_lig)  # [N_lig, spatial_dim]
        return vel_lig
