"""Transformer-based Vector Field Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerVectorField(nn.Module):
    """Transformer-based vector field model for Topological Flow."""

    def __init__(self, x_dim, h_dim, num_heads=4, num_layers=2, dropout=0.1):
        """Initialize the Transformer-based vector field model."""
        super().__init__()
        self.x_dim = x_dim
        self.h_dim = h_dim

        # Time embedding MLP
        self.time_embed = nn.Sequential(
            nn.Linear(1, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        # X embedding
        self.x_embed = nn.Sequential(
            nn.Linear(x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

        # Positional encoding is optional; not included here
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=4 * h_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final projection to vector field
        self.out_proj = nn.Linear(h_dim, x_dim)

    def forward(self, x, t, batch_lig, batch_pocket):
        """Forward pass of the Transformer vector field model.

        Args:
            x: Tensor of shape [num_x, h_dim]
            t: Tensor of shape [batch_size] — one t per graph/batch
            batch_lig: Tensor of shape [num_lig] — ligands associated with each sample
            batch_pocket: Tensor of shape [num_pocket] — pockets associated with each sample

        Returns:
            Tensor of shape [num_x, h_dim] — the vector field
        """
        device = x.device

        batch_x = torch.cat([batch_lig, batch_pocket], dim=0)

        # Time embedding: [batch_size, 1] -> [batch_size, h_dim]
        if len(t.shape) == 0:
            t = t.unsqueeze(0)
            t = t.repeat(batch_x.shape[0], 1)  # [num_x, 1]
        t_embed = self.time_embed(t[:, None]).squeeze()  # [batch_size, h_dim]
        t_per_x = t_embed[batch_x]  # [num_x, h_dim]

        # X embedding: [num_x, x_dim] -> [num_x, h_dim]
        x = self.x_embed(x).squeeze()  # [num_x, h_dim]

        x = x + t_per_x  # Add time-conditioning

        # Create attention mask: prevent attention across different graphs
        attn_mask = (batch_x[:, None] != batch_x[None, :]).to(device)  # [num_x, num_x]
        attn_mask = attn_mask.masked_fill(
            attn_mask, float("-inf")
        )  # Transformer uses -inf

        # Transformer forward
        x_encoded = self.transformer(x.unsqueeze(0), mask=attn_mask).squeeze(
            0
        )  # [num_x, h_dim]

        return self.out_proj(x_encoded)  # [num_x, x_dim]
