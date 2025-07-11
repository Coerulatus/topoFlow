"""Transformer-based Vector Field Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerVectorField(nn.Module):
    """Transformer-based vector field model for Topological Flow, with learned type embedding."""

    def __init__(
        self, x_dim, h_dim, type_embed_dim=8, num_heads=4, num_layers=2, dropout=0.1
    ):
        """Initialize the Transformer vector field model."""
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.type_embed_dim = type_embed_dim

        # Type embedding (0 = pocket, 1 = ligand)
        self.type_embed = nn.Embedding(2, type_embed_dim)  # 2 types: ligand, pocket

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

        # Input embedding MLP
        self.x_embed = nn.Sequential(
            nn.Linear(x_dim + type_embed_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

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
        self.out_proj = nn.Linear(h_dim, x_dim)

    def forward(self, x, t, batch_lig, batch_pocket):
        """Forward pass of the vector field model.

        Args:
            x: [num_x, x_dim] — node features (ligand + pocket)
            t: scalar or [batch_size] — time step
            batch_lig: [num_lig] — ligand graph index
            batch_pocket: [num_pocket] — pocket graph index
        Returns:
            [num_x, x_dim] — vector field
        """
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
        type_embed = self.type_embed(type_labels)  # [num_x, type_embed_dim]

        # Concatenate to input
        x = torch.cat([x, type_embed], dim=1)  # [num_x, x_dim + type_embed_dim]

        # Embed x
        x = self.x_embed(x)  # [num_x, h_dim]

        # Expand t to per-node
        if len(t.shape) == 0:
            t = t.unsqueeze(0).repeat(batch_x.shape[0], 1)
        t_embed = self.time_embed(t[:, None]).squeeze()
        t_per_x = t_embed[batch_x]

        x = x + t_per_x  # Time conditioning

        # Create attention mask: prevent attention across graphs
        attn_mask = (batch_x[:, None] != batch_x[None, :]).to(device)
        attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))

        # Transformer forward
        x_encoded = self.transformer(x.unsqueeze(0), mask=attn_mask).squeeze(0)

        return self.out_proj(x_encoded)
