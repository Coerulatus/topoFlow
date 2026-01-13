"""Hybrid Transformer Vector Field using pairwise distances."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTimeEmbedding(nn.Module):
    """Simple MLP-based time embedding."""

    def __init__(self, h_dim):
        """Initialize the SimpleTimeEmbedding module."""
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

    def forward(self, t):
        """Forward pass for time embedding."""
        t_reshaped = t.view(-1, 1)
        return self.time_embed(t_reshaped)


class TransformerDistanceVectorField(nn.Module):
    """Hybrid Transformer Vector Field: Equivariant (Space) + Invariant (Types)."""

    def __init__(
        self,
        h_dim,
        spatial_dim=3,
        atom_dim=11,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
        ffn_mult=4,
        add_time_to="ligand",
    ):
        """Initialize the TransformerDistanceVectorField model."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.x_dim = spatial_dim + atom_dim
        self.h_dim = h_dim
        self.add_time_to = add_time_to

        # Note: We embed the atom features (11 dim), but NOT the coords.
        # Coords are used only for geometry/distances.
        self.lig_embed = nn.Sequential(
            nn.Linear(atom_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.poc_embed = nn.Sequential(
            nn.Linear(atom_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.time_embed = SimpleTimeEmbedding(h_dim)

        self.dist_coeff_head = nn.Linear(h_dim, 2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=ffn_mult * h_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.direction_weight_net = nn.Sequential(
            nn.Linear(h_dim * 2 + 1, h_dim), nn.SiLU(), nn.Linear(h_dim, 1, bias=False)
        )

        self.feature_vel_net = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, atom_dim, bias=True),
        )

        self.spatial_vel_net = nn.Sequential(
            nn.LayerNorm(self.x_dim + h_dim),
            nn.Linear(self.x_dim + h_dim, h_dim),
            nn.SiLU(),
            nn.Linear(h_dim, spatial_dim, bias=True),
        )

    def forward(self, x_lig, t, x_pocket, batch_lig, batch_pocket):
        """Forward pass of the TransformerDistanceVectorField."""
        device = x_lig.device

        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1
        batch_all = batch_lig.clone()
        batch_all[is_poc] = batch_pocket[is_poc]

        # x_lig is [N, 14] -> Split into [N, 3] (coords) and [N, 11] (types)
        lig_coords = x_lig[:, : self.spatial_dim]
        lig_feat = x_lig[:, self.spatial_dim :]

        poc_coords = x_pocket[:, : self.spatial_dim]
        poc_feat = x_pocket[:, self.spatial_dim :]

        all_coords = torch.cat([lig_coords, poc_coords], dim=0)

        lig_h = self.lig_embed(lig_feat)
        poc_h = self.poc_embed(poc_feat)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        if t.ndim == 0:
            B = int(batch_all.max().item() + 1)
            t = t.repeat(B)
        t_emb = self.time_embed(t)

        if self.add_time_to in ("ligand", "both"):
            lig_h = lig_h + t_emb[batch_lig[is_lig], :]
        if self.add_time_to in ("both",):
            poc_h = poc_h + t_emb[batch_pocket[is_poc], :]

        all_h = torch.cat([lig_h, poc_h], dim=0)

        # diff_vecs: [N_total, N_total, 3]
        diff_vecs = all_coords[:, None, :] - all_coords[None, :, :]
        dist_sq = torch.sum(diff_vecs**2, dim=-1)
        raw_dist = torch.sqrt(dist_sq + 1e-6)
        unit_vecs = diff_vecs / (raw_dist.unsqueeze(-1) + 1e-6)

        node_coeffs = self.dist_coeff_head(all_h)
        node_slope = F.softplus(node_coeffs[:, 0])
        node_bias = node_coeffs[:, 1]
        pairwise_slope = (node_slope.unsqueeze(0) + node_slope.unsqueeze(1)) / 2.0
        pairwise_bias = (node_bias.unsqueeze(0) + node_bias.unsqueeze(1)) / 2.0
        learnable_dist_bias = -1.0 * (pairwise_slope * raw_dist + pairwise_bias)

        batch_mask = batch_all[:, None] != batch_all[None, :]

        final_attn_mask = learnable_dist_bias.clone()
        batch_mask = batch_all[:, None] != batch_all[None, :]
        final_attn_mask.masked_fill_(batch_mask, -torch.inf)
        final_attn_mask.diagonal().fill_(0.0)
        x_encoded = self.transformer(all_h.unsqueeze(0), mask=final_attn_mask).squeeze(
            0
        )

        num_lig = len(lig_h)
        num_total = len(all_h)

        # # Create pairs: Ligand atoms (i) attending to All atoms (j)
        # h_i = x_encoded[:num_lig].unsqueeze(1).repeat(1, num_total, 1)  # [Lig, All, H]
        # h_j = x_encoded.unsqueeze(0).repeat(num_lig, 1, 1)  # [Lig, All, H]
        # d_ij = raw_dist[:num_lig, :].unsqueeze(-1)  # [Lig, All, 1]

        # # Predict Weights
        # pair_input = torch.cat([h_i, h_j, d_ij], dim=-1)

        # # Output is [Lig, All, 1]
        # # We apply Tanh here (optional but recommended) to cap the max influence
        # # This prevents "exploding" gradients if weights get too large.
        # weights = self.direction_weight_net(pair_input)
        # weights = torch.tanh(weights)

        # # Mask interactions from other batches
        # mask_slice = batch_mask[:num_lig, :]
        # weights = weights.masked_fill(mask_slice.unsqueeze(-1), 0.0)

        # # Normalize weights by number of neighbors (optional, effectively "mean" pooling)
        # # This makes the velocity scale independent of the number of atoms.
        # N_neighbors = (~mask_slice).sum(dim=1, keepdim=True).unsqueeze(-1)
        # weights = weights / (N_neighbors + 1e-6)

        # # Sum of Normalized Directions
        # # vel = Sum( weight * unit_vector )
        # relevant_unit_vecs = unit_vecs[:num_lig, :, :]  # [Lig, All, 3]
        # vel_spatial = torch.sum(relevant_unit_vecs * weights, dim=1)  # [Lig, 3]

        # 7. Predict Feature Velocity (Standard MLP)
        vel_features = self.feature_vel_net(x_encoded[:num_lig])

        # 8. Combine
        vel_spatial = self.spatial_vel_net(
            torch.cat([x_lig, x_encoded[:num_lig]], dim=-1)
        )
        vel_total = torch.cat([vel_spatial, vel_features], dim=-1)

        return vel_total
