"""Transformer-based vector field model for Topological Flow, using xformers."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention, fmha


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


class AttentionLayer(nn.Module):
    """Multi-headed self-attention layer using xformers.

    This layer is a direct replacement for the attention part of a standard transformer.
    """

    def __init__(self, h_dim, num_heads, dropout=0.1):
        """Initialize the Attention Layer."""
        super().__init__()
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.head_dim = h_dim // num_heads

        self.to_q = nn.Linear(h_dim, h_dim)
        self.to_k = nn.Linear(h_dim, h_dim)
        self.to_v = nn.Linear(h_dim, h_dim)

        self.to_out = nn.Linear(h_dim, h_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        """Forward pass of the attention layer.

        Args:
            x (Tensor): Input tensor of shape (sequence_length, h_dim).
            attn_bias (AttentionBias): The attention mask object.
        """
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Reshape for multi-head attention (1 ,sequence_length, num_heads, head_dim)
        # First dimension is batch size, which is 1 here since we handle batching via attn_bias
        q = q.view(1, -1, self.num_heads, self.head_dim)
        k = k.view(1, -1, self.num_heads, self.head_dim)
        v = v.view(1, -1, self.num_heads, self.head_dim)

        # Use memory_efficient_attention from xformers with the bias object
        out = memory_efficient_attention(q, k, v, attn_bias=attn_bias)

        # Reshape output
        out = out.view(-1, self.h_dim)
        out = self.to_out(out)

        return self.dropout(out)


class TransformerEncoderLayerWithXformers(nn.Module):
    """A single Transformer Encoder layer using xformers for attention."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """Initialize the Transformer Encoder Layer."""
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.self_attn = AttentionLayer(h_dim=d_model, num_heads=nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.SiLU()

    def forward(self, src, attn_bias):
        """Pass the attention bias to the attention layer."""
        src2 = self.self_attn(src, attn_bias)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout1(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    """Custom Transformer Encoder that handles xformers AttentionBias."""

    def __init__(self, encoder_layer, num_layers, norm=None):
        """Initialize the Custom Transformer Encoder."""
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.norm = norm

    def forward(self, src, attn_bias):
        """Pass the attention bias to each layer."""
        output = src
        for mod in self.layers:
            output = mod(output, attn_bias)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerVectorField(nn.Module):
    """Transformer-based vector field model for Topological Flow, with learned type embedding."""

    def __init__(
        self,
        h_dim,
        spatial_dim=3,
        atom_dim=11,
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

        self.type_embed = nn.Embedding(2, h_dim)
        self.time_embed = SimpleTimeEmbedding(h_dim)
        self.pocket_vel_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.x_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.combine = combine
        if combine == "concat":
            self.feature_projection = nn.Linear(4 * h_dim, h_dim)

        encoder_layer = TransformerEncoderLayerWithXformers(
            d_model=h_dim,
            nhead=num_heads,
            dim_feedforward=4 * h_dim,
            dropout=dropout,
        )
        self.transformer = CustomTransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.out_coord_linear = nn.Sequential(
            nn.LayerNorm(h_dim),
            nn.Linear(h_dim, spatial_dim, bias=False),
        )

        self.out_atom_type_linear = nn.Sequential(
            nn.Linear(h_dim, h_dim),
            nn.SiLU(inplace=False),
            nn.Linear(h_dim, atom_dim),
        )

    def forward(self, x, t, u_t_ref_pocket, batch_x, batch_lig, batch_pocket):
        """Forward pass of the vector field model with pocket velocity conditioning."""
        device = x.device

        # Create a boolean mask for ligand atoms
        is_ligand = batch_lig != -1
        # Create a boolean mask for pocket atoms
        is_pocket = batch_pocket != -1

        # Create a full type_labels tensor for all atoms (0 for pocket, 1 for ligand)
        type_labels_all = torch.zeros(len(x), dtype=torch.long, device=device)
        type_labels_all[is_ligand] = 1

        # Get the type embedding for each atom
        type_embed = self.type_embed(type_labels_all)

        # Initialize a zero tensor for all atoms to store the velocity embeddings.
        u_t_ref_all = torch.zeros(len(x), self.x_dim, device=device)
        u_t_ref_all[is_pocket] = u_t_ref_pocket
        vel_embed = self.pocket_vel_embed(u_t_ref_all)

        # Embed x
        x_embed = self.x_embed(x)

        # Time embedding
        num_molecules = torch.max(batch_x) + 1
        if len(t.shape) == 0:
            t = t.repeat(num_molecules, 1)
        time_embed = self.time_embed(t)
        t_per_x = time_embed[batch_x]

        # Combine all embeddings.
        if self.combine == "add":
            x_combined = x_embed + type_embed + t_per_x + vel_embed
        elif self.combine == "concat":
            x_combined = torch.cat([x_embed, type_embed, t_per_x, vel_embed], dim=-1)
            x_combined = self.feature_projection(x_combined)

        # Create the BlockDiagonalMask from the batch tensor
        batch_sizes = batch_x.bincount()
        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(batch_sizes.tolist())

        # Transformer forward
        x_encoded = self.transformer(x_combined, attn_bias=attn_bias)

        # Output projections
        coords_out = self.out_coord_linear(x_encoded)
        atom_type_out = self.out_atom_type_linear(x_encoded)

        return torch.cat([coords_out, atom_type_out], dim=-1)
