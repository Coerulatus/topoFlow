"""Allset transformer class."""

import torch
import math
import numpy as np
from typing import Literal
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import softmax
from torch_geometric.utils import scatter as pyg_scatter
from torch_geometric.utils import to_undirected, to_dense_adj
from torch_geometric.nn.pool import radius, radius_graph


def dedup_columns(row: torch.LongTensor, col: torch.LongTensor, num_rows: int):
    """Remove duplicate columns from a binary sparse COO matrix.

    Args:
        row: 1-D LongTensors of length E (COO coordinates)
        col: 1-D LongTensors of length E (COO coordinates)
        num_rows: total number of rows in the matrix.

    Returns:
        BoolTensor, length E: True for coordinates to keep (first column of each duplicate set).
        LongTensor: Sorted list of unique column indices that remain.
    """
    device = row.device
    E = row.size(0)
    C = int(col.max()) + 1

    colptr = torch.zeros(C + 1, device=device, dtype=torch.long)
    colptr.index_add_(0, col + 1, torch.ones(E, device=device, dtype=torch.long))
    colptr = colptr.cumsum(0)
    nnz_per_col = colptr[1:] - colptr[:-1]
    max_nnz = int(nnz_per_col.max())

    pad = torch.full((C * max_nnz,), -1, device=device, dtype=row.dtype)
    slot = (col * max_nnz) + torch.arange(E, device=device) % max_nnz
    pad.index_copy_(0, slot, row)
    rows_padded = pad.view(C, max_nnz)

    rows_sorted, _ = rows_padded.sort(dim=1)

    base = num_rows + 1
    sig = torch.zeros(C, device=device, dtype=torch.int64)
    for k in range(max_nnz):
        sig = sig * base + rows_sorted[:, k].clamp(min=0)

    sort_idx = sig.argsort()
    sig_sorted = sig[sort_idx]
    change = torch.ones_like(sig_sorted, dtype=torch.bool)
    change[1:] = sig_sorted[1:] != sig_sorted[:-1]
    first_pos = sort_idx[change]

    keep_col_mask = torch.zeros(C, device=device, dtype=torch.bool)
    keep_col_mask[first_pos] = True

    keep_mask = keep_col_mask[col]
    _, new_col = torch.unique(col[keep_mask], return_inverse=True)
    return row[keep_mask], new_col


def add_ligands_to_hyperedges(row_radius, col_radius, n_lig):
    """Add ligand atoms to hyperedges that should include them.

    When finding pocket atoms close to ligand atoms radius only returns the pocket atoms, so for each set of neighboring pocket atoms we need to add the corresponding ligand atom to the hyperedge.
    """
    row_radius = row_radius + n_lig

    hyperedge_ids = torch.unique(col_radius)
    row_radius = torch.cat([row_radius, hyperedge_ids], dim=0)
    col_radius = torch.cat([col_radius, hyperedge_ids], dim=0)

    return row_radius, col_radius


class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding for temporal conditioning."""

    def __init__(self, dim: int, max_period: int = 10000, time_scale: float = 1000.0):
        """Initialize the sinusoidal positional embedding."""
        super().__init__()
        self.dim = dim
        self.time_scale = time_scale
        half_dim = self.dim // 2
        frequencies = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        args = (timesteps[:, None].float() * self.time_scale) * self.frequencies[
            None, :
        ]
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class AdvancedTimeEmbedding(nn.Module):
    """Advanced temporal conditioning embedding with positional frequencies."""

    def __init__(self, h_dim: int):
        """Initialize the advanced time embedding."""
        super().__init__()
        self.sin_embed = SinusoidalPositionalEmbedding(h_dim)
        self.net = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the advanced time embedding for the given timesteps."""
        t_emb = self.net(self.sin_embed(t))
        if len(t_emb.shape) == 2:
            return t_emb
        else:
            return t_emb.mean(1)


class MessagePassing(torch.nn.Module):
    """Message Passing Base Class."""

    def __init__(
        self,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        att: bool = False,
        initialization: Literal[
            "uniform", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        """Define message passing."""
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialization_gain

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        match self.initialization:
            case "uniform":
                if self.weight is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight.data.uniform_(-stdv, stdv)
                if self.att:
                    stdv = 1.0 / math.sqrt(self.att_weight.size(1))
                    self.att_weight.data.uniform_(-stdv, stdv)
            case "xavier_uniform":
                if self.weight is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_uniform_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case "xavier_normal":
                if self.weight is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_normal_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case _:
                raise ValueError(
                    f"Initialization {self.initialization} not recognized."
                )

    def message(self, x_source, x_target=None):
        """Construct message from source cells to target cells."""
        return x_source

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages."""
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def aggregate(self, x_message):
        """Aggregate messages on each target cell."""
        return pyg_scatter(x_message, self.target_index_i, dim=0, reduce=self.aggr_func)

    def forward(self, x_source, edge_index, x_target=None):
        r"""Forward pass."""
        self.target_index_i = edge_index[0]
        self.source_index_j = edge_index[1]

        x_message = self.message(x_source=x_source, x_target=x_target)
        x_message = x_message.index_select(-2, self.source_index_j)

        if self.att:
            attention_values = self.attention(x_source=x_source, x_target=x_target)
            x_message = attention_values.view(-1, 1) * x_message

        return self.aggregate(x_message)


class AllSetTransformerLayer(nn.Module):
    """AllSet Transformer Layer Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads: int = 4,
        number_queries: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 1,
        mlp_activation=nn.ReLU,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
        **kwargs,
    ) -> None:
        """Implementation of the AllSetTransformer Layer."""
        super().__init__()

        if heads <= 0:
            raise ValueError(f"heads ({heads}) must be positive")

        if mlp_num_layers <= 0:
            raise ValueError(f"mlp_num_layers ({mlp_num_layers}) must be positive")

        if (hidden_channels % heads) != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            )

        if hidden_channels < heads:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be greater than or equal to heads ({heads})"
            )

        self.dropout = dropout
        self.skip_proj = (
            nn.Linear(in_channels, hidden_channels)
            if in_channels != hidden_channels
            else nn.Identity()
        )
        self.vertex2edge = AllSetTransformerBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
            number_queries=number_queries,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
        )

        self.edge2vertex = AllSetTransformerBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
            number_queries=number_queries,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
        )

    def reset_parameters(self) -> None:
        """Reset parameters."""
        self.vertex2edge.reset_parameters()
        self.edge2vertex.reset_parameters()

    def forward(
        self,
        x_0,
        edge_index_v2e,
        edge_index_e2v,
        t_emb=None,
        batch_all=None,
        batch_he=None,
        return_stats=False,
    ):
        """Forward computation."""
        x_0_skip = self.skip_proj(x_0)
        if return_stats:
            x_1, stats = self.vertex2edge(
                x_0,
                edge_index_v2e,
                t_emb=t_emb,
                batch_target=batch_he,
                return_stats=True,
            )
        else:
            x_1 = self.vertex2edge(
                x_0, edge_index_v2e, t_emb=t_emb, batch_target=batch_he
            )

        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_0_new = F.relu(
            self.edge2vertex(x_1, edge_index_e2v, t_emb=t_emb, batch_target=batch_all)
        )
        x_0_new = F.dropout(x_0_new, p=self.dropout, training=self.training)

        x_0 = x_0_skip + x_0_new

        if return_stats:
            return x_0, x_1, stats
        return x_0, x_1


class AllSetTransformerBlock(nn.Module):
    r"""AllSetTransformer Block Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads: int = 4,
        number_queries: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 1,
        mlp_activation=None,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
    ) -> None:
        """A module for AllSet Transformer block in a bipartite graph."""
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.number_queries = number_queries
        self.dropout = dropout

        self.multihead_att = MultiHeadAttention(
            in_channels=self.in_channels,
            hidden_channels=hidden_channels // heads,
            heads=self.heads,
            number_queries=number_queries,
            initialization=initialization,
        )

        self.mlp = MLP(
            in_channels=self.hidden_channels,
            hidden_channels=[self.hidden_channels] * mlp_num_layers,
            norm_layer=mlp_norm,
            activation_layer=mlp_activation,
            dropout=mlp_dropout,
        )

        self.ln0 = nn.LayerNorm(self.hidden_channels)
        self.ln1 = nn.LayerNorm(self.hidden_channels)
        self.time_proj = nn.Linear(self.hidden_channels, self.hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.multihead_att.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        self.time_proj.reset_parameters()
        if hasattr(self.mlp, "reset_parameters") and callable(
            self.mlp.reset_parameters
        ):
            self.mlp.reset_parameters()

    def forward(
        self, x_source, edge_index, t_emb=None, batch_target=None, return_stats=False
    ):
        """Forward computation."""
        self.target_index_i = edge_index[0]
        self.source_index_j = edge_index[1]

        if return_stats:
            x_message_on_target, stats = self.multihead_att(
                x_source, edge_index, return_stats=True
            )
        else:
            x_message_on_target = self.multihead_att(x_source, edge_index)

        q_weight = self.multihead_att.Q_weight.permute(1, 0, 2).unsqueeze(0)
        x_message_on_target = x_message_on_target.unsqueeze(1) + q_weight

        x_message_on_target = x_message_on_target.reshape(
            -1, self.number_queries, self.hidden_channels
        )
        if t_emb is not None and batch_target is not None:
            x_message_on_target = x_message_on_target + self.time_proj(t_emb)[
                batch_target
            ].unsqueeze(1)

        x_message_on_target = self.ln0(x_message_on_target)

        x_message_on_target = self.ln1(
            x_message_on_target + F.relu(self.mlp(x_message_on_target))
        )

        if return_stats:
            return x_message_on_target.mean(dim=1), stats
        return x_message_on_target.mean(dim=1)


class MultiHeadAttention(MessagePassing):
    """Computes the multi-head attention mechanism (QK^T)V of transformer-based architectures."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        aggr_norm: bool = False,
        update_func=None,
        heads: int = 4,
        number_queries: int = 1,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        """Implementation of the multi-head attention mechanism."""
        super().__init__(
            att=True,
            initialization=initialization,
            initialization_gain=initialization_gain,
        )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.heads = heads
        self.number_queries = number_queries

        self.K_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.in_channels, self.hidden_channels)
        )
        self.Q_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.number_queries, self.hidden_channels)
        )
        self.V_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.in_channels, self.hidden_channels)
        )

    def reset_parameters(self):
        """Reset learnable parameters."""
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.V_weight, gain=self.initialization_gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.V_weight, gain=self.initialization_gain)
        else:
            raise RuntimeError("Initialization method not recognized.")

    def attention(self, x_source, edge_index):
        """Compute attention weights for messages."""
        # Explicit einsum replaces matmul to prevent dim=0 broadcasting errors
        x_K = torch.einsum("nc,hcd->hnd", x_source, self.K_weight)
        alpha = torch.einsum("hsd,hqd->hsq", x_K, self.Q_weight)
        alpha = alpha / math.sqrt(self.hidden_channels)
        alpha = F.leaky_relu(alpha, 0.2)

        alpha = alpha.permute(1, 0, 2)
        alpha_msg = alpha[self.source_index_j]

        attn_weights = softmax(alpha_msg, index=self.target_index_i, dim=0)

        if self.number_queries == 1:
            attn_weights = attn_weights.squeeze(-1)

        return attn_weights

    def attention_with_entropy(self, x_source, edge_index):
        """Compute attention weights and their normalized entropy for messages."""
        x_K = torch.einsum("nc,hcd->hnd", x_source, self.K_weight)
        alpha = torch.einsum("hsd,hqd->hsq", x_K, self.Q_weight)
        alpha = alpha / math.sqrt(self.hidden_channels)
        alpha = F.leaky_relu(alpha, 0.2)

        alpha = alpha.permute(1, 0, 2)
        alpha_msg = alpha[self.source_index_j]

        attn_weights = softmax(alpha_msg, index=self.target_index_i, dim=0)

        eps = 1e-12
        p = attn_weights.clamp(min=eps)
        ent_per_msg = -p * torch.log(p)

        entropy = pyg_scatter(ent_per_msg, self.target_index_i, dim=0, reduce="sum")

        ones = torch.ones_like(self.source_index_j, dtype=torch.float)
        card_msg = pyg_scatter(ones, self.target_index_i, dim=0, reduce="sum")
        cardinality = card_msg.view(-1, 1, 1)

        log_k = torch.log(cardinality.clamp(min=1.0001))
        mask = (cardinality > 1.1).float()
        norm_entropy = (entropy * mask) / (log_k + (1 - mask))
        norm_entropy = torch.clamp(norm_entropy, 0.0, 1.0)

        if self.number_queries == 1:
            attn_weights = attn_weights.squeeze(-1)
            norm_entropy = norm_entropy.squeeze(-1)

        return attn_weights, norm_entropy, card_msg

    def forward(self, x_source, edge_index, return_stats=False):
        """Forward pass."""
        self.target_index_i = edge_index[0]
        self.source_index_j = edge_index[1]

        if return_stats:
            attn_weights, entropy, cardinality = self.attention_with_entropy(
                x_source, edge_index
            )
        else:
            attn_weights = self.attention(x_source, edge_index)

        # Explicit einsum here as well
        x_message = torch.einsum("nc,hcd->nhd", x_source, self.V_weight)
        x_message = x_message[self.source_index_j] * attn_weights.unsqueeze(-1)
        out = pyg_scatter(x_message, self.target_index_i, dim=0, reduce="sum")

        if return_stats:
            return out, {"entropy": entropy, "cardinality": cardinality}
        return out


class AllSetTransformer(torch.nn.Module):
    """AllSet Neural Network Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=1,
        heads=4,
        dropout=0.2,
        mlp_num_layers=2,
        mlp_dropout=0.2,
        **kwargs,
    ):
        """A module that combines multiple AllSet layers to form a neural network."""
        super().__init__()

        self.layers = torch.nn.ModuleList(
            AllSetTransformerLayer(
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                heads=heads,
                mlp_num_layers=mlp_num_layers,
                mlp_dropout=mlp_dropout,
                **kwargs,
            )
            for i in range(n_layers)
        )

    def forward(
        self,
        x_0,
        edge_index_v2e,
        edge_index_e2v,
        t_emb=None,
        batch_all=None,
        batch_he=None,
        return_stats=False,
    ):
        """Forward computation."""
        for i, layer in enumerate(self.layers):
            if return_stats and i == len(self.layers) - 1:
                x_0, x_1, stats = layer(
                    x_0,
                    edge_index_v2e,
                    edge_index_e2v,
                    t_emb=t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                    return_stats=True,
                )
            else:
                x_0, x_1 = layer(
                    x_0,
                    edge_index_v2e,
                    edge_index_e2v,
                    t_emb=t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                )

        if return_stats:
            return x_0, x_1, stats
        return x_0, x_1


class GeometricMessagePassing(torch.nn.Module):
    """Geometric Message Passing Base Class."""

    def __init__(
        self,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        att: bool = False,
        initialization: Literal[
            "uniform", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        """Define geometric message passing."""
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialization_gain

    def message(self, x_source, x_target=None, edge_attr=None):
        """Construct message from source cells to target cells."""
        return x_source

    def attention(self, x_source, x_target=None, edge_attr=None):
        """Compute attention weights for messages."""
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def aggregate(self, x_message):
        """Aggregate messages on each target cell."""
        return pyg_scatter(x_message, self.target_index_i, dim=0, reduce=self.aggr_func)

    def forward(self, x_source, edge_index, edge_attr=None, x_target=None):
        r"""Forward pass."""
        self.target_index_i = edge_index[0]
        self.source_index_j = edge_index[1]

        x_message = self.message(
            x_source=x_source, x_target=x_target, edge_attr=edge_attr
        )
        x_message = x_message.index_select(-2, self.source_index_j)

        if edge_attr is not None and getattr(self, "edge_proj", None) is not None:
            e_emb = self.edge_proj(edge_attr)
            x_message = x_message + e_emb

        if self.att:
            attention_values = self.attention(
                x_source=x_source, x_target=x_target, edge_attr=edge_attr
            )
            x_message = attention_values.view(-1, 1) * x_message

        return self.aggregate(x_message)


class GeometricMultiHeadAttention(GeometricMessagePassing):
    """Computes the multi-head attention mechanism with geometric edge features."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        aggr_norm: bool = False,
        update_func=None,
        heads: int = 4,
        number_queries: int = 1,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
        edge_dim: int | None = None,
    ) -> None:
        """Implementation of the multi-head attention mechanism with geometric edge features."""
        super().__init__(
            att=True,
            initialization=initialization,
            initialization_gain=initialization_gain,
        )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.heads = heads
        self.number_queries = number_queries

        self.K_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.in_channels, self.hidden_channels)
        )
        self.Q_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.number_queries, self.hidden_channels)
        )
        self.V_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.in_channels, self.hidden_channels)
        )

        if edge_dim is not None:
            self.edge_proj_k = nn.Linear(edge_dim, self.hidden_channels, bias=False)
            self.edge_proj_v = nn.Linear(edge_dim, self.hidden_channels, bias=False)
        else:
            self.edge_proj_k = None
            self.edge_proj_v = None

    def reset_parameters(self):
        """Reset learnable parameters."""
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.V_weight, gain=self.initialization_gain)
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.V_weight, gain=self.initialization_gain)
        else:
            raise RuntimeError("Initialization method not recognized.")

    def attention(self, x_source, edge_index, edge_attr=None):
        """Compute attention weights for messages, incorporating edge attributes if provided."""
        x_K = torch.einsum("nc,hcd->hnd", x_source, self.K_weight)

        if edge_attr is not None and self.edge_proj_k is not None:
            e_k = self.edge_proj_k(edge_attr)
            x_K_edge = x_K[:, self.source_index_j, :]
            x_K_edge = x_K_edge + e_k.unsqueeze(0)
            alpha_msg = torch.einsum("hed,hqd->heq", x_K_edge, self.Q_weight)
        else:
            alpha = torch.einsum("hsd,hqd->hsq", x_K, self.Q_weight)
            alpha_msg = alpha.permute(1, 0, 2)[self.source_index_j]

        alpha_msg = alpha_msg / math.sqrt(self.hidden_channels // self.heads)
        alpha_msg = F.leaky_relu(alpha_msg, 0.2)

        attn_weights = softmax(alpha_msg, index=self.target_index_i, dim=0)

        if self.number_queries == 1:
            attn_weights = attn_weights.squeeze(-1)

        return attn_weights

    def attention_with_entropy(self, x_source, edge_index, edge_attr=None):
        """Compute attention weights and their normalized entropy for messages, incorporating edge attributes if provided."""
        x_K = torch.einsum("nc,hcd->hnd", x_source, self.K_weight)

        if edge_attr is not None and self.edge_proj_k is not None:
            e_k = self.edge_proj_k(edge_attr)
            x_K_edge = x_K[:, self.source_index_j, :]
            x_K_edge = x_K_edge + e_k.unsqueeze(0)
            alpha_msg = torch.einsum("hed,hqd->heq", x_K_edge, self.Q_weight)
        else:
            alpha = torch.einsum("hsd,hqd->hsq", x_K, self.Q_weight)
            alpha_msg = alpha.permute(1, 0, 2)[self.source_index_j]

        alpha_msg = alpha_msg / math.sqrt(self.hidden_channels // self.heads)
        alpha_msg = F.leaky_relu(alpha_msg, 0.2)

        attn_weights = softmax(alpha_msg, index=self.target_index_i, dim=0)

        eps = 1e-12
        p = attn_weights.clamp(min=eps)
        ent_per_msg = -p * torch.log(p)

        entropy = pyg_scatter(ent_per_msg, self.target_index_i, dim=0, reduce="sum")

        ones = torch.ones_like(self.source_index_j, dtype=torch.float)
        card_msg = pyg_scatter(ones, self.target_index_i, dim=0, reduce="sum")
        cardinality = card_msg.view(-1, 1, 1)

        log_k = torch.log(cardinality.clamp(min=1.0001))
        mask = (cardinality > 1.1).float()
        norm_entropy = (entropy * mask) / (log_k + (1 - mask))
        norm_entropy = torch.clamp(norm_entropy, 0.0, 1.0)

        if self.number_queries == 1:
            attn_weights = attn_weights.squeeze(-1)
            norm_entropy = norm_entropy.squeeze(-1)

        return attn_weights, norm_entropy, card_msg

    def forward(self, x_source, edge_index, edge_attr=None, return_stats=False):
        """Forward pass."""
        self.target_index_i = edge_index[0]
        self.source_index_j = edge_index[1]

        if return_stats:
            attn_weights, entropy, cardinality = self.attention_with_entropy(
                x_source, edge_index, edge_attr
            )
        else:
            attn_weights = self.attention(x_source, edge_index, edge_attr)

        x_message = torch.einsum("nc,hcd->hnd", x_source, self.V_weight)
        x_message = x_message.permute(1, 0, 2)[self.source_index_j]

        if edge_attr is not None and self.edge_proj_v is not None:
            x_message = x_message + self.edge_proj_v(edge_attr).unsqueeze(1)

        x_message = x_message * attn_weights.unsqueeze(-1)
        out = pyg_scatter(x_message, self.target_index_i, dim=0, reduce="sum")

        if return_stats:
            return out, {"entropy": entropy, "cardinality": cardinality}
        return out


class GeometricAllSetTransformerBlock(nn.Module):
    r"""Geometric AllSetTransformer Block Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads: int = 4,
        number_queries: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 1,
        mlp_activation=None,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        edge_dim=None,
    ) -> None:
        """A module for Geometric AllSet Transformer block."""
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.number_queries = number_queries
        self.dropout = dropout

        self.multihead_att = GeometricMultiHeadAttention(
            in_channels=self.in_channels,
            hidden_channels=hidden_channels // heads,
            heads=self.heads,
            number_queries=number_queries,
            initialization=initialization,
            edge_dim=edge_dim,
        )

        self.mlp = MLP(
            in_channels=self.hidden_channels,
            hidden_channels=[self.hidden_channels] * mlp_num_layers,
            norm_layer=mlp_norm,
            activation_layer=mlp_activation,
            dropout=mlp_dropout,
        )

        self.ln0 = nn.LayerNorm(self.hidden_channels)
        self.ln1 = nn.LayerNorm(self.hidden_channels)
        self.time_proj = nn.Linear(self.hidden_channels, self.hidden_channels)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.multihead_att.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        self.time_proj.reset_parameters()
        if hasattr(self.mlp, "reset_parameters") and callable(
            self.mlp.reset_parameters
        ):
            self.mlp.reset_parameters()

    def forward(
        self,
        x_source,
        edge_index,
        edge_attr=None,
        t_emb=None,
        batch_target=None,
        return_stats=False,
    ):
        """Forward computation."""
        self.target_index_i = edge_index[0]
        self.source_index_j = edge_index[1]

        if return_stats:
            x_message_on_target, stats = self.multihead_att(
                x_source, edge_index, edge_attr, return_stats=True
            )
        else:
            x_message_on_target = self.multihead_att(x_source, edge_index, edge_attr)

        q_weight = self.multihead_att.Q_weight.permute(1, 0, 2).unsqueeze(0)
        x_message_on_target = x_message_on_target.unsqueeze(1) + q_weight

        x_message_on_target = x_message_on_target.reshape(
            -1, self.number_queries, self.hidden_channels
        )
        if t_emb is not None and batch_target is not None:
            x_message_on_target = x_message_on_target + self.time_proj(t_emb)[
                batch_target
            ].unsqueeze(1)

        x_message_on_target = self.ln0(x_message_on_target)

        x_message_on_target = self.ln1(
            x_message_on_target + F.relu(self.mlp(x_message_on_target))
        )

        if return_stats:
            return x_message_on_target.mean(dim=1), stats
        return x_message_on_target.mean(dim=1)


class GeometricAllSetTransformerLayer(nn.Module):
    """Geometric AllSet Transformer Layer Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads: int = 4,
        number_queries: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 1,
        mlp_activation=nn.ReLU,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
        edge_dim=None,
        **kwargs,
    ) -> None:
        """Implementation of the Geometric AllSetTransformer Layer."""
        super().__init__()

        if heads <= 0:
            raise ValueError(f"heads ({heads}) must be positive")

        if mlp_num_layers <= 0:
            raise ValueError(f"mlp_num_layers ({mlp_num_layers}) must be positive")

        if (hidden_channels % heads) != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            )

        self.dropout = dropout

        self.vertex2edge = GeometricAllSetTransformerBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
            number_queries=number_queries,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
            edge_dim=edge_dim,
        )

        self.edge2vertex = GeometricAllSetTransformerBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
            number_queries=number_queries,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
            edge_dim=edge_dim,
        )

    def reset_parameters(self) -> None:
        """Reset parameters."""
        self.vertex2edge.reset_parameters()
        self.edge2vertex.reset_parameters()

    def forward(
        self,
        x_0,
        edge_index_v2e,
        edge_index_e2v,
        edge_attr_v2e=None,
        edge_attr_e2v=None,
        t_emb=None,
        batch_all=None,
        batch_he=None,
        return_stats=False,
    ):
        r"""Forward computation."""
        if return_stats:
            x_1, stats = self.vertex2edge(
                x_0,
                edge_index_v2e,
                edge_attr_v2e,
                t_emb=t_emb,
                batch_target=batch_he,
                return_stats=True,
            )
        else:
            x_1 = self.vertex2edge(
                x_0, edge_index_v2e, edge_attr_v2e, t_emb=t_emb, batch_target=batch_he
            )

        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_0 = F.relu(
            self.edge2vertex(
                x_1, edge_index_e2v, edge_attr_e2v, t_emb=t_emb, batch_target=batch_all
            )
        )
        x_0 = F.dropout(x_0, p=self.dropout, training=self.training)

        if return_stats:
            return x_0, x_1, stats
        return x_0, x_1


class GeometricAllSetTransformer(torch.nn.Module):
    """Geometric AllSet Neural Network Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=1,
        heads=4,
        dropout=0.2,
        mlp_num_layers=2,
        mlp_dropout=0.2,
        edge_dim=None,
        **kwargs,
    ):
        """A module that combines multiple Geometric AllSet layers."""
        super().__init__()

        self.layers = torch.nn.ModuleList(
            GeometricAllSetTransformerLayer(
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                heads=heads,
                mlp_num_layers=mlp_num_layers,
                mlp_dropout=mlp_dropout,
                edge_dim=edge_dim,
                **kwargs,
            )
            for i in range(n_layers)
        )

    def forward(
        self,
        x_0,
        edge_index_v2e,
        edge_index_e2v,
        edge_attr_v2e=None,
        edge_attr_e2v=None,
        t_emb=None,
        batch_all=None,
        batch_he=None,
        return_stats=False,
    ):
        """Forward computation."""
        for i, layer in enumerate(self.layers):
            if return_stats and i == len(self.layers) - 1:
                x_0, x_1, stats = layer(
                    x_0,
                    edge_index_v2e,
                    edge_index_e2v,
                    edge_attr_v2e,
                    edge_attr_e2v,
                    t_emb=t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                    return_stats=True,
                )
            else:
                x_0, x_1 = layer(
                    x_0,
                    edge_index_v2e,
                    edge_index_e2v,
                    edge_attr_v2e,
                    edge_attr_e2v,
                    t_emb=t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                )

        if return_stats:
            return x_0, x_1, stats
        return x_0, x_1


class MLP(nn.Sequential):
    """MLP Module."""

    def __init__(
        self,
        in_channels,
        hidden_channels,
        norm_layer=None,
        activation_layer=torch.nn.ReLU,
        dropout: float = 0.0,
        inplace: bool | None = None,
        bias: bool = False,
    ) -> None:
        """A module for a multi-layer perceptron."""
        params = {} if inplace is None else {"inplace": inplace}
        layers: list[nn.Module] = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(nn.Dropout(dropout, **params))

        super().__init__(*layers)


class GATModel(nn.Module):
    """GAT model with isolated residual paths."""

    def __init__(
        self,
        h_dim,
        num_heads,
        num_layers,
        dropout,
    ):
        """Initialize GAT model."""
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GATv2Conv(h_dim, h_dim // num_heads, heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.layers_norm = nn.ModuleList(
            [nn.LayerNorm(h_dim) for _ in range(num_layers)]
        )
        self.time_projs = nn.ModuleList(
            [nn.Linear(h_dim, h_dim) for _ in range(num_layers)]
        )

    def forward(self, x, edge_index, t_emb=None, batch_all=None):
        """Forward pass."""
        for layer, norm, time_proj in zip(
            self.layers, self.layers_norm, self.time_projs
        ):
            x_res = layer(x, edge_index)
            x_res = F.silu(x_res)
            if t_emb is not None and batch_all is not None:
                x_res = x_res + (time_proj(t_emb)[batch_all])
            x_res = norm(x_res)
            x = x + x_res
        return x


class AllSetTransformerVectorField(nn.Module):
    """Ligand-only prediction with AllSetTransformer model."""

    def __init__(
        self,
        h_dim,
        spatial_dim=3,
        atom_dim=11,
        num_heads=4,
        num_layers=4,
        dropout=0.0,
        add_time_to="layers",
        use_gat=True,
        use_geometric_ast=False,
        norm_std=None,
    ):
        """Ligand-only prediction with self-attention inside each ligand and cross-attention to pocket atoms."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.x_dim = spatial_dim + atom_dim
        self.h_dim = h_dim
        self.add_time_to = add_time_to
        self.use_geometric_ast = use_geometric_ast

        # Cutoffs in Angstroms
        self.cutoff_lig = 5.0
        self.cutoff_poc = 5.0
        self.cutoff_cross = 5.0

        if norm_std is not None:
            if isinstance(norm_std, (torch.Tensor, np.ndarray, list, tuple)):
                std_val = float(np.mean(norm_std))
            else:
                std_val = float(norm_std)
            # Scale down cutoffs to match normalized space
            self.cutoff_lig /= std_val
            self.cutoff_poc /= std_val
            self.cutoff_cross /= std_val

        self.lig_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.poc_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.time_embed = AdvancedTimeEmbedding(h_dim)

        if self.use_geometric_ast:
            self.model = GeometricAllSetTransformer(
                in_channels=h_dim,
                hidden_channels=h_dim,
                edge_dim=4,
                n_layers=num_layers,
                heads=num_heads,
                dropout=dropout,
                mlp_num_layers=2,
                mlp_dropout=dropout,
            )
        else:
            self.model = AllSetTransformer(
                in_channels=h_dim,
                hidden_channels=h_dim,
                n_layers=num_layers,
                heads=num_heads,
                dropout=dropout,
                mlp_num_layers=2,
                mlp_dropout=dropout,
            )

        self.use_gat = use_gat
        if use_gat:
            self.gat_model = GATModel(
                h_dim=h_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                dropout=dropout,
            )

        self.out_vel = nn.Sequential(
            nn.LayerNorm(h_dim), nn.Linear(h_dim, self.x_dim, bias=False)
        )

    def format_diagnostics(self, stats, edge_types):
        """Organizes entropy by cardinality and returns summary statistics."""
        entropy = stats["entropy"].mean(dim=-1).cpu()
        cards = stats["cardinality"].cpu().int()
        types = edge_types.cpu()

        diag_dict = {}
        for i in range(len(cards)):
            c = cards[i].item()
            if c not in diag_dict:
                diag_dict[c] = []
            diag_dict[c].append({"entropy": entropy[i].item(), "type": types[i].item()})

        summary_stats = {
            "mean_entropy": entropy.mean().item(),
            "max_entropy": entropy.max().item(),
            "min_entropy": entropy.min().item(),
        }

        return {"per_cardinality": diag_dict, "summary": summary_stats}

    def create_edge_index(self, x_lig, x_pocket, filt_batch_lig, filt_batch_poc):
        """Create edge index for GATv2Conv directly using dense node inputs."""
        lig_pos = x_lig[:, :3]
        poc_pos = x_pocket[:, :3]

        N1 = lig_pos.size(0)

        edge_ll = radius_graph(
            lig_pos,
            r=self.cutoff_lig,
            batch=filt_batch_lig,
            loop=False,
            max_num_neighbors=512,
        )

        edge_lp = radius(
            poc_pos,
            lig_pos,
            r=self.cutoff_cross,
            batch_x=filt_batch_poc,
            batch_y=filt_batch_lig,
            max_num_neighbors=512,
        )

        if not self.use_gat:
            return edge_ll, edge_lp, None

        edge_pp = radius_graph(
            poc_pos,
            r=self.cutoff_poc,
            batch=filt_batch_poc,
            loop=False,
            max_num_neighbors=512,
        )

        edge_pp_global = (
            edge_pp + N1
            if edge_pp.numel() > 0
            else torch.empty((2, 0), dtype=torch.long, device=x_lig.device)
        )

        edge_lp_global = (
            torch.stack([edge_lp[0], edge_lp[1] + N1], dim=0)
            if edge_lp.numel() > 0
            else torch.empty((2, 0), dtype=torch.long, device=x_lig.device)
        )
        edge_pl_global = (
            torch.stack([edge_lp[1] + N1, edge_lp[0]], dim=0)
            if edge_lp.numel() > 0
            else torch.empty((2, 0), dtype=torch.long, device=x_lig.device)
        )

        edge_index = torch.cat(
            [edge_ll, edge_pp_global, edge_lp_global, edge_pl_global], dim=1
        )

        return edge_ll, edge_lp, edge_index

    def create_incidence_with_types(
        self, x_lig, x_pocket, batch_lig, batch_pocket, num_graphs
    ):
        """Create incidence matrices for hyperedges with type annotations and batch assignments."""
        device = x_lig.device
        n_lig = x_lig.shape[0]
        n_poc = x_pocket.shape[0]

        filt_batch_lig = batch_lig[batch_lig != -1].view(-1)
        filt_batch_poc = batch_pocket[batch_pocket != -1].view(-1)

        B_lig = num_graphs
        B_poc = num_graphs
        row_list, col_list, type_list, he_batches = [], [], [], []
        current_hyperedge_id = 0

        valid_lig_batches = torch.arange(B_lig, device=device)
        row_list.append(torch.arange(n_lig, device=device))
        col_list.append(filt_batch_lig + current_hyperedge_id)

        type_list.append(torch.full((B_lig,), 0, device=device))
        he_batches.append(valid_lig_batches)
        current_hyperedge_id += B_lig

        valid_poc_batches = torch.arange(B_poc, device=device)
        row_list.append(torch.arange(n_poc, device=device) + n_lig)
        col_list.append(filt_batch_poc + current_hyperedge_id)

        type_list.append(torch.full((B_poc,), 1, device=device))
        he_batches.append(valid_poc_batches)
        current_hyperedge_id += B_poc

        edge_ll, edge_lp, edge_index = self.create_edge_index(
            x_lig, x_pocket, filt_batch_lig, filt_batch_poc
        )

        lig_idx_to_he = torch.arange(n_lig, device=device) + current_hyperedge_id

        row_list.append(torch.arange(n_lig, device=device))
        col_list.append(lig_idx_to_he)

        if edge_lp is not None and edge_lp.numel() > 0:
            row_list.append(edge_lp[1] + n_lig)
            col_list.append(lig_idx_to_he[edge_lp[0]])

        if edge_ll is not None and edge_ll.numel() > 0:
            row_list.append(edge_ll[1])
            col_list.append(lig_idx_to_he[edge_ll[0]])

        type_list.append(torch.full((n_lig,), 2, device=device))
        he_batches.append(filt_batch_lig)
        current_hyperedge_id += n_lig

        final_rows = torch.cat(row_list, dim=0)
        final_cols = torch.cat(col_list, dim=0)
        final_types = torch.cat(type_list, dim=0)
        final_batch_he = torch.cat(he_batches, dim=0)

        edge_index_v2e = torch.stack([final_cols, final_rows], dim=0)
        edge_index_e2v = torch.stack([final_rows, final_cols], dim=0)

        return edge_index_v2e, edge_index_e2v, edge_index, final_types, final_batch_he

    def create_geometric_incidence(
        self, x_lig, x_pocket, batch_lig, batch_pocket, num_graphs
    ):
        """Create incidence matrices for hyperedges with geometric attributes, type annotations, and batch assignments."""
        device = x_lig.device
        n_lig = x_lig.shape[0]
        n_poc = x_pocket.shape[0]

        filt_batch_lig = batch_lig[batch_lig != -1].view(-1)
        filt_batch_poc = batch_pocket[batch_pocket != -1].view(-1)

        B_lig = num_graphs
        B_poc = num_graphs
        lig_pos = x_lig[:, :3]
        poc_pos = x_pocket[:, :3]

        row_list, col_list, type_list, he_batches = [], [], [], []
        current_hyperedge_id = 0

        valid_lig_batches = torch.arange(B_lig, device=device)
        row_list.append(torch.arange(n_lig, device=device))
        col_list.append(filt_batch_lig + current_hyperedge_id)

        type_list.append(torch.full((B_lig,), 0, device=device))
        he_batches.append(valid_lig_batches)
        current_hyperedge_id += B_lig

        valid_poc_batches = torch.arange(B_poc, device=device)
        row_list.append(torch.arange(n_poc, device=device) + n_lig)
        col_list.append(filt_batch_poc + current_hyperedge_id)

        type_list.append(torch.full((B_poc,), 1, device=device))
        he_batches.append(valid_poc_batches)
        current_hyperedge_id += B_poc

        edge_ll, edge_lp, gat_edge_index = self.create_edge_index(
            x_lig, x_pocket, filt_batch_lig, filt_batch_poc
        )

        lig_idx_to_he = torch.arange(n_lig, device=device) + current_hyperedge_id

        row_list.append(torch.arange(n_lig, device=device))
        col_list.append(lig_idx_to_he)

        if edge_lp is not None and edge_lp.numel() > 0:
            row_list.append(edge_lp[1] + n_lig)
            col_list.append(lig_idx_to_he[edge_lp[0]])

        if edge_ll is not None and edge_ll.numel() > 0:
            row_list.append(edge_ll[1])
            col_list.append(lig_idx_to_he[edge_ll[0]])

        type_list.append(torch.full((n_lig,), 2, device=device))
        he_batches.append(filt_batch_lig)
        current_hyperedge_id += n_lig

        final_rows = torch.cat(row_list, dim=0)
        final_cols = torch.cat(col_list, dim=0)
        final_types = torch.cat(type_list, dim=0)
        final_batch_he = torch.cat(he_batches, dim=0)

        center_pos = torch.zeros((current_hyperedge_id, 3), device=device)

        center_pos[:B_lig] = pyg_scatter(lig_pos, filt_batch_lig, dim=0, reduce="mean")
        center_pos[B_lig : B_lig + B_poc] = pyg_scatter(
            poc_pos, filt_batch_poc, dim=0, reduce="mean"
        )

        offset = B_lig + B_poc
        center_pos[offset : offset + n_lig] = lig_pos

        pos_all = torch.cat([lig_pos, poc_pos], dim=0)
        source_pos = pos_all[final_rows]
        target_pos = center_pos[final_cols]

        rel_vec = source_pos - target_pos
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)
        direction = rel_vec / (dist + 1e-6)
        edge_attr = torch.cat([dist, direction], dim=-1)

        edge_index_v2e = torch.stack([final_cols, final_rows], dim=0)
        edge_index_e2v = torch.stack([final_rows, final_cols], dim=0)

        return (
            edge_index_v2e,
            edge_index_e2v,
            edge_attr,
            gat_edge_index,
            final_types,
            final_batch_he,
        )

    def forward(
        self,
        x_lig,
        t,
        x_pocket,
        batch_lig,
        batch_pocket,
        return_diagnostics=False,
        num_graphs=None,
    ):
        """Forward pass."""
        device = x_lig.device

        # Extract valid batch indices
        filt_batch_lig = batch_lig[batch_lig != -1].view(-1)
        filt_batch_poc = batch_pocket[batch_pocket != -1].view(-1)

        # Handle purely single-graph edge cases gracefully
        # B = int(torch.max(filt_batch_lig) + 1) if filt_batch_lig.numel() > 0 else 1
        B = (
            num_graphs
            if num_graphs is not None
            else (
                int(torch.max(filt_batch_lig).item()) + 1
                if filt_batch_lig.numel() > 0
                else 1
            )
        )

        lig_h = self.lig_embed(x_lig)
        poc_h = self.poc_embed(x_pocket)

        if not torch.is_tensor(t):
            t = torch.tensor(t, device=device)
        if t.ndim == 0:
            t = t.repeat(B)
        t_emb = self.time_embed(t)

        batch_all = torch.cat([filt_batch_lig, filt_batch_poc], dim=0)

        if self.add_time_to == "input":
            lig_h = lig_h + t_emb[filt_batch_lig, :]
            poc_h = poc_h + t_emb[filt_batch_poc, :]
            layer_t_emb = None
        else:
            layer_t_emb = t_emb

        if self.use_geometric_ast:
            (
                edge_index_v2e,
                edge_index_e2v,
                edge_attr_v2e,
                gat_edge_index,
                edge_types,
                batch_he,
            ) = self.create_geometric_incidence(
                x_lig, x_pocket, batch_lig, batch_pocket, B
            )
            edge_attr_e2v = torch.cat(
                [edge_attr_v2e[:, :1], -edge_attr_v2e[:, 1:]], dim=-1
            )
        else:
            (
                edge_index_v2e,
                edge_index_e2v,
                gat_edge_index,
                edge_types,
                batch_he,
            ) = self.create_incidence_with_types(
                x_lig, x_pocket, batch_lig, batch_pocket, B
            )

        h = torch.cat([lig_h, poc_h], dim=0)

        if self.use_geometric_ast:
            if return_diagnostics:
                h_ast, _, stats = self.model(
                    h,
                    edge_index_v2e,
                    edge_index_e2v,
                    edge_attr_v2e,
                    edge_attr_e2v,
                    t_emb=layer_t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                    return_stats=True,
                )
            else:
                h_ast, _ = self.model(
                    h,
                    edge_index_v2e,
                    edge_index_e2v,
                    edge_attr_v2e,
                    edge_attr_e2v,
                    t_emb=layer_t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                )
        else:
            if return_diagnostics:
                h_ast, _, stats = self.model(
                    h,
                    edge_index_v2e,
                    edge_index_e2v,
                    t_emb=layer_t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                    return_stats=True,
                )
            else:
                h_ast, _ = self.model(
                    h,
                    edge_index_v2e,
                    edge_index_e2v,
                    t_emb=layer_t_emb,
                    batch_all=batch_all,
                    batch_he=batch_he,
                )

        if self.use_gat:
            h_gat = self.gat_model(
                h, gat_edge_index, t_emb=layer_t_emb, batch_all=batch_all
            )
            h = h_gat + h_ast
        else:
            h = h_ast

        N1 = lig_h.size(0)
        h_lig = h[:N1, :]

        vel_lig = self.out_vel(h_lig)

        if return_diagnostics:
            return vel_lig, self.format_diagnostics(stats, edge_types)
        return vel_lig
