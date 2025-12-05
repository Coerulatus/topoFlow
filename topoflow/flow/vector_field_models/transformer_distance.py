"""Transformer Distance Vector Field Model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTimeEmbedding(nn.Module):
    """Simple MLP-based time embedding."""

    def __init__(self, h_dim):
        """Initializes the SimpleTimeEmbedding module."""
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

    def forward(self, t):
        """Forward pass of the time embedding."""
        t_reshaped = t.view(-1, 1)
        return self.time_embed(t_reshaped)


class TransformerDistanceVectorField(nn.Module):
    """Transformer Distance Vector Field Model."""

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
        """Initializes the Transformer Distance Vector Field model."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.x_dim = spatial_dim + atom_dim
        self.h_dim = h_dim
        self.add_time_to = add_time_to

        # Embeddings
        self.lig_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.poc_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.time_embed = SimpleTimeEmbedding(h_dim)

        # Maps hidden state to 2 values: Slope (scale) and Bias (shift)
        self.dist_coeff_head = nn.Linear(h_dim, 2)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=ffn_mult * h_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Usually stabilizes geometric transformers
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.out_vel = nn.Sequential(
            nn.LayerNorm(h_dim), nn.Linear(h_dim, self.x_dim, bias=False)
        )

    def forward(self, x_lig, t, x_pocket, batch_lig, batch_pocket):
        """Forward pass of the Transformer Distance Vector Field model."""
        device = x_lig.device

        # 1. Prepare Batching Logic
        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        # Create a unified batch index
        batch_all = batch_lig.clone()
        batch_all[is_poc] = batch_pocket[is_poc]

        # 2. Initial Embeddings
        lig_h = self.lig_embed(x_lig)
        poc_h = self.poc_embed(x_pocket)

        # 3. Time Embeddings
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        # Ensure t is broadcast correctly if scalar
        if t.ndim == 0:
            # Calculate actual batch size based on unique batch indices
            B = int(batch_all.max().item() + 1)
            t = t.repeat(B)

        t_emb = self.time_embed(t)  # [B, h]

        # Add time conditionally
        if self.add_time_to in ("ligand", "both"):
            lig_h = lig_h + t_emb[batch_lig[is_lig], :]
        if self.add_time_to in ("both",):
            poc_h = poc_h + t_emb[batch_pocket[is_poc], :]

        all_h = torch.cat([lig_h, poc_h], dim=0)

        # 4. Spatial Calculations
        coords_lig = x_lig[:, : self.spatial_dim]
        coords_poc = x_pocket[:, : self.spatial_dim]
        coords = torch.cat([coords_lig, coords_poc], dim=0)

        # Calculate raw Euclidean distance [N, N]
        raw_dist = torch.cdist(coords, coords, p=2)

        # Calculate per-node slope and bias from features
        # coeffs shape: [N, 2] -> index 0 is slope, index 1 is bias
        node_coeffs = self.dist_coeff_head(all_h)

        # Enforce positive slope
        node_slope = F.softplus(node_coeffs[:, 0])
        node_bias = node_coeffs[:, 1]

        # Broadcast to pairwise [N, N]
        # We average the slope of atom i and atom j
        pairwise_slope = (node_slope.unsqueeze(0) + node_slope.unsqueeze(1)) / 2.0
        pairwise_bias = (node_bias.unsqueeze(0) + node_bias.unsqueeze(1)) / 2.0

        # Transform the physical distance
        # Logic: Attention = Softmax(QK + Bias).
        # Typically, large distance = large negative bias (to reduce attention).
        learnable_dist_bias = -1.0 * (pairwise_slope * raw_dist + pairwise_bias)

        # 5. Create Attention Mask
        batch_mask = batch_all[:, None] != batch_all[None, :]

        # Initialize mask with 0 (attend) and -inf (ignore)
        final_attn_mask = torch.zeros_like(raw_dist)
        final_attn_mask.masked_fill_(batch_mask, float("-inf"))

        # Add the learnable distance bias to the valid regions
        # This modulates the attention scores directly
        final_attn_mask = final_attn_mask + learnable_dist_bias
        final_attn_mask.diagonal().fill_(0.0)
        # 6. Transformer Pass
        x_encoded = self.transformer(all_h.unsqueeze(0), mask=final_attn_mask).squeeze(
            0
        )

        # 7. Output
        lig_h_out = x_encoded[: len(lig_h), :]
        vel_lig = self.out_vel(lig_h_out)

        return vel_lig
