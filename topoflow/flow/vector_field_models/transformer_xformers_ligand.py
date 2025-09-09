"""Transformer-based vector field model with xformers attention for ligand-only prediction with cross-attention to pocket atoms."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from xformers.ops import memory_efficient_attention, fmha


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


class SelfAttention(nn.Module):
    """Self-attention layer."""

    def __init__(self, h_dim, n_heads, dropout=0.0):
        """Self-attention layer."""
        super().__init__()
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.q = nn.Linear(h_dim, h_dim)
        self.k = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, h_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, attn_bias):
        """Forward pass for self-attention layer."""
        # x: [S, h]
        q = self.q(x).view(1, -1, self.n_heads, self.head_dim)
        k = self.k(x).view(1, -1, self.n_heads, self.head_dim)
        v = self.v(x).view(1, -1, self.n_heads, self.head_dim)
        y = memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(
            -1, self.h_dim
        )
        return self.drop(self.out(y))


class CrossAttention(nn.Module):
    """Cross-attention layer."""

    def __init__(self, h_dim, n_heads, dropout=0.0):
        """Cross-attention layer."""
        super().__init__()
        self.h_dim = h_dim
        self.n_heads = n_heads
        self.head_dim = h_dim // n_heads
        self.q = nn.Linear(h_dim, h_dim)
        self.k = nn.Linear(h_dim, h_dim)
        self.v = nn.Linear(h_dim, h_dim)
        self.out = nn.Linear(h_dim, h_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_x, kv_x, attn_bias):
        """Forward pass for cross-attention layer."""
        # q_x: [S_q, h], kv_x: [S_kv, h]
        q = self.q(q_x).view(1, -1, self.n_heads, self.head_dim)
        k = self.k(kv_x).view(1, -1, self.n_heads, self.head_dim)
        v = self.v(kv_x).view(1, -1, self.n_heads, self.head_dim)
        y = memory_efficient_attention(q, k, v, attn_bias=attn_bias).view(
            -1, self.h_dim
        )
        return self.drop(self.out(y))


class LigandCrossEncoderLayer(nn.Module):
    """Encoder layer with ligand self-attention and ligand-to-pocket cross-attention."""

    def __init__(self, h_dim, n_heads, ffn_mult=4, dropout=0.0):
        """Encoder layer with ligand self-attention and ligand-to-pocket cross-attention."""
        super().__init__()
        self.self_attn = SelfAttention(h_dim, n_heads, dropout)
        self.cross_attn = CrossAttention(h_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(h_dim)
        self.norm2 = nn.LayerNorm(h_dim)
        self.norm3 = nn.LayerNorm(h_dim)

        self.ff = nn.Sequential(
            nn.Linear(h_dim, ffn_mult * h_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * h_dim, h_dim),
            nn.Dropout(dropout),
        )

    def forward(self, lig_x, poc_x, lig_self_bias, lig_to_poc_bias):
        """Forward pass."""
        # Ligand self-attention
        lig_x = self.norm1(lig_x + self.self_attn(lig_x, lig_self_bias))
        # Ligand-to-pocket cross attention
        lig_x = self.norm2(lig_x + self.cross_attn(lig_x, poc_x, lig_to_poc_bias))
        # FFN
        lig_x = self.norm3(lig_x + self.ff(lig_x))
        return lig_x


class TransformerVectorField(nn.Module):
    """Ligand-only prediction with self-attention inside each ligand and cross-attention to pocket atoms."""

    def __init__(
        self,
        h_dim,
        spatial_dim=3,
        atom_dim=11,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
        ffn_mult=4,
        add_time_to="ligand",  # "ligand", "both", or "none"
    ):
        """Ligand-only prediction with self-attention inside each ligand and cross-attention to pocket atoms."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.x_dim = spatial_dim + atom_dim
        self.h_dim = h_dim
        self.add_time_to = add_time_to

        self.lig_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.poc_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.time_embed = SimpleTimeEmbedding(h_dim)

        self.layers = nn.ModuleList(
            [
                LigandCrossEncoderLayer(
                    h_dim, num_heads, ffn_mult=ffn_mult, dropout=dropout
                )
                for _ in range(num_layers)
            ]
        )

        # output ligand velocity only
        self.out_vel = nn.Sequential(
            nn.LayerNorm(h_dim), nn.Linear(h_dim, self.x_dim, bias=False)
        )

    @staticmethod
    def _sort_by_batch(x, batch):
        """Sort x and batch by batch, returning also the indices to unsort."""
        # returns x_sorted, batch_sorted, idx, inv_idx
        idx = torch.argsort(batch)
        x_sorted = x.index_select(0, idx)
        batch_sorted = batch.index_select(0, idx)
        inv_idx = torch.empty_like(idx)
        inv_idx[idx] = torch.arange(idx.numel(), device=idx.device)
        return x_sorted, batch_sorted, idx, inv_idx

    @staticmethod
    def _seqlens_from_batch(batch, n_groups=None):
        """Compute sequence lengths from batch vector."""
        if n_groups is None:
            n_groups = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        # batch is already non-negative here
        return torch.bincount(batch, minlength=n_groups).tolist()

    def forward(self, x_lig, t, x_pocket, batch_lig, batch_pocket):
        """Forward pass."""
        device = x_lig.device
        B = int(torch.max(torch.cat([batch_lig, batch_pocket])) + 1)

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

        # Sort by batch to make contiguous groups for block-diagonal masks
        # lig_h, b_lig, lig_idx, lig_inv = self._sort_by_batch(lig_h, batch_lig)
        # poc_h, b_poc, poc_idx, _ = self._sort_by_batch(poc_h, is_poc)

        # Build per-batch sequence lengths
        lig_seqlens = self._seqlens_from_batch(batch_lig[is_lig], n_groups=B)
        poc_seqlens = self._seqlens_from_batch(batch_pocket[is_poc], n_groups=B)

        # Biases: ligand self, ligand-to-pocket cross
        lig_self_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(lig_seqlens)
        lig_to_poc_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(
            lig_seqlens, poc_seqlens
        )

        # Encoder
        for layer in self.layers:
            lig_h = layer(lig_h, poc_h, lig_self_bias, lig_to_poc_bias)

        # Unsort ligand back to original order
        # lig_h = lig_h.index_select(0, lig_inv)

        # Predict ligand velocities only
        vel_lig = self.out_vel(lig_h)  # [N_lig, spatial_dim]
        return vel_lig
