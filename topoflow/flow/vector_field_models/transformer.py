"""Transformer-based Vector Field Model."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeFourierEncoding(nn.Module):
    """Encoder for continuous timesteps in `[0, 1]`."""

    def __init__(self, h_dim, max_len=100):
        """Initialize the Time Fourier Encoding."""
        super().__init__()
        self.h_dim = h_dim
        self.max_len = max_len

    def forward(self, t):
        """Encode a tensor of timesteps.

        Args:
            t: 1-D tensor with values in `[0, 1]`.

        Returns:
            Tensor of shape `(B, h_dim)` with sine/cosine features.
        """
        t = t.view(-1)  # Ensure t is a 1-D tensor
        t_scaled = t * self.max_len
        half_dim = self.h_dim // 2
        emb = math.log(self.max_len) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb
        )
        emb = torch.outer(t_scaled.float(), emb)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

        if self.h_dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant")

        assert emb.shape == (
            t.shape[0],
            self.h_dim,
        ), f"Expected shape ({t.shape[0], self.h_dim}), got {emb.shape}"
        return emb

    def out_dim(self):
        """Return the output dimension of the time encoding."""
        return self.h_dim


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

    def forward_old(self, x, t, batch_lig, batch_pocket):
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
        if t.ndim == 0:
            t = t.view(1, 1)
        elif t.ndim == 1:
            t = t[:, None]  # [batch_size, 1]
        t_embed = self.time_embed(t)  # [batch_size, h_dim]

        # Now assign each node the right t embedding via batch_x
        t_per_x = t_embed[batch_x]  # [num_x, h_dim]

        # Add time conditioning
        x = torch.cat([x, t_per_x], dim=-1)

        # Create attention mask: prevent attention across graphs
        attn_mask = (batch_x[:, None] != batch_x[None, :]).to(device)
        attn_mask = attn_mask.masked_fill(attn_mask, float("-inf"))

        # Transformer forward
        x_encoded = self.transformer(x.unsqueeze(0), mask=attn_mask).squeeze(0)

        return self.out_proj(x_encoded)
