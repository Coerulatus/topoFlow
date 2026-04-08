"""Transformer-based Vector Field Model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleTimeEmbedding(nn.Module):
    """Simple MLP-based time embedding."""

    def __init__(self, h_dim):
        """Initialize the simple time embedding."""
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

    def forward(self, t):
        """Embed time `t` into a higher-dimensional space.

        Args:
            t: Tensor of shape `[batch_size, 1]` representing time.

        Returns:
            Embedded time tensor of shape `[batch_size, h_dim]`.
        """
        t_reshaped = t.view(-1, 1)
        return self.time_embed(t_reshaped)


class TransformerVectorField(nn.Module):
    """Transformer-based vector field model for Topological Flow, with learned type embedding."""

    def __init__(
        self,
        h_dim,
        spatial_dim=3,
        atom_dim=11,
        type_embed_dim=8,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        combine="concat",
    ):
        """Initialize the Transformer vector field model."""
        super().__init__()
        self.x_dim = spatial_dim + atom_dim
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.h_dim = h_dim
        self.time_embed_dim = h_dim

        # Type embedding (0 = pocket, 1 = ligand)
        self.type_embed = nn.Embedding(2, h_dim)  # 2 types: ligand, pocket

        # Time embedding MLP
        self.time_embed = SimpleTimeEmbedding(h_dim)

        # New: Embedding for pocket velocities.
        # This will embed the 3D velocity vector into the h_dim space.
        self.pocket_vel_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

        # Input embedding MLP
        self.x_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.combine = combine
        # Updated: The feature projection now needs to handle an extra embedding.
        if combine == "concat":
            self.feature_projection = nn.Linear(4 * h_dim, h_dim)
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=4 * h_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final output projection
        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, spatial_dim, bias=False),
        )

        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(inplace=False),
            nn.Linear(h_dim, atom_dim),
        )

    def forward(self, x, t, u_t_ref_pocket, batch_lig, batch_pocket):
        """Forward pass of the vector field model with pocket velocity conditioning."""
        device = x.device

        batch_x = torch.cat([batch_lig, batch_pocket], dim=0)

        # Build type labels: 1 for ligand, 0 for pocket
        type_labels = torch.cat(
            [
                torch.ones(len(batch_lig), dtype=torch.long),
                torch.zeros(len(batch_pocket), dtype=torch.long),
            ]
        ).to(device)

        # Get learned type embedding
        type_embed = self.type_embed(type_labels)  # [num_x, h_dim]

        # Embed x
        x_embed = self.x_embed(x)  # [num_x, h_dim]

        # Time embedding
        num_samples = torch.max(batch_pocket) + 1
        if len(t.shape) == 0:
            t = t.repeat(num_samples, 1)  # Ensure t is a column vector
        time_embed = self.time_embed(t)  # [batch_size, h_dim]

        t_per_x = time_embed[batch_x]  # [num_x, h_dim]

        # Create a zero tensor for the ligand velocities
        u_t_ref_lig = torch.zeros([len(batch_lig), self.x_dim], device=device)
        # Combine the ligand zeros with the pocket velocities
        u_t_ref = torch.cat([u_t_ref_lig, u_t_ref_pocket], dim=0)
        # Embed the reference velocities
        pocket_vel_embed = self.pocket_vel_embed(u_t_ref)

        # Combine all embeddings.
        # Now there are four embeddings to combine: x, type, time, and pocket velocity.
        if self.combine == "add":
            x = x_embed + type_embed + t_per_x + pocket_vel_embed
        elif self.combine == "concat":
            x = torch.cat([x_embed, type_embed, t_per_x, pocket_vel_embed], dim=-1)
            x = self.feature_projection(x)

        # Create attention mask: prevent attention across graphs
        attn_mask = (batch_x[:, None] != batch_x[None, :]).float().to(device)
        attn_mask = attn_mask.masked_fill(attn_mask > 0, float("-inf"))

        # Transformer forward
        x_encoded = self.transformer(x.unsqueeze(0), mask=attn_mask).squeeze(0)

        # Output projections
        coords_out = self.out_coord_linear(x_encoded)  # [num_x, spatial_dim]
        atom_type_out = self.out_atom_type_linear(x_encoded)  # [num_x, atom_dim]

        return torch.cat([coords_out, atom_type_out], dim=-1)
