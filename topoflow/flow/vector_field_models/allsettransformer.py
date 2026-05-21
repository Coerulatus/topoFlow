"""Allset transformer class."""

import torch
import math
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
    C = int(col.max()) + 1  # number of columns

    # ----- 1.  build colptr so we know how many entries each column owns
    colptr = torch.zeros(C + 1, device=device, dtype=torch.long)
    colptr.index_add_(0, col + 1, torch.ones(E, device=device, dtype=torch.long))
    colptr = colptr.cumsum(0)  # CSR pointer, size C+1
    nnz_per_col = colptr[1:] - colptr[:-1]
    max_nnz = int(nnz_per_col.max())  # longest column

    # ----- 2.  pack each column’s row indices into a fixed-length tensor
    pad = torch.full(
        (C * max_nnz,), -1, device=device, dtype=row.dtype
    )  # flat buffer, filled with –1
    # scatter rows into their column slice
    slot = (col * max_nnz) + torch.arange(E, device=device) % max_nnz
    pad.index_copy_(0, slot, row)
    rows_padded = pad.view(C, max_nnz)  # [C, max_nnz]

    # sort inside each column so equal patterns align
    rows_sorted, _ = rows_padded.sort(dim=1)  # –1 pads move to front

    # ----- 3.  hash each column’s pattern into a single int64 signature
    base = num_rows + 1  # safe mixed-radix base
    sig = torch.zeros(C, device=device, dtype=torch.int64)
    for k in range(max_nnz):  # small loop over slots
        sig = sig * base + rows_sorted[:, k].clamp(min=0)

    # ----- 4.  keep the first column of each signature class
    sort_idx = sig.argsort()  # O(C log C)
    sig_sorted = sig[sort_idx]
    # boundaries where signature changes
    change = torch.ones_like(sig_sorted, dtype=torch.bool)
    change[1:] = sig_sorted[1:] != sig_sorted[:-1]
    first_pos = sort_idx[change]  # col indices to KEEP

    keep_col_mask = torch.zeros(C, device=device, dtype=torch.bool)
    keep_col_mask[first_pos] = True

    # ----- 5.  translate to coordinate mask for original COO lists
    keep_mask = keep_col_mask[col]
    _, new_col = torch.unique(col[keep_mask], return_inverse=True)  # broadcast lookup
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


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    """Broadcasts `src` to the shape of `other`."""
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    return src.expand(other.size())


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)

    return out.scatter_add_(dim, index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Add all values from the `src` tensor into `out` at the indices."""
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: torch.Tensor | None = None,
    dim_size: int | None = None,
) -> torch.Tensor:
    """Compute the mean value of all values from the `src` tensor into `out`."""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


SCATTER_DICT = {"sum": scatter_sum, "mean": scatter_mean, "add": scatter_sum}


def scatter(scatter: str):
    """Return the scatter function."""
    if scatter not in SCATTER_DICT:
        raise ValueError(f"scatter must be string: {list(SCATTER_DICT.keys())}")

    return SCATTER_DICT[scatter]


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
        """Define message passing.

        This class defines message passing through a single neighborhood N,
        by de
        1. 🟥 Create messages going from source cells to target cells through N.
        2. 🟧 Aggregate messages coming from different sources cells onto each target cell.

        This class should not be instantiated directly, but rather inherited
        through subclasses that effectively define a message passing function.
        This class does not have trainable weights, but its subclasses should
        define these weights.

        Args:
            aggr_func (Literal["sum", "mean", "add"], default="sum"): Aggregation function to use.
            att (bool, default=False): Whether to use attention.
            initialization (Literal["uniform", "xavier_uniform", "xavier_normal"], default="xavier_uniform"): Initialization method for the weights of the layer.
            initialization_gain (float, default=1.414): Gain for the weight initialization.

        References:
           [1] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
            Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub.
            Topological deep learning: going beyond graph data (2023).
            https://arxiv.org/abs/2206.00606.

           [2] Papillon, Sanborn, Hajij, Miolane.
            Architectures of topological deep learning: a survey on topological neural networks (2023).
            https://arxiv.org/abs/2304.10031.
        """
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialization_gain

    def reset_parameters(self):
        r"""Reset learnable parameters.

        Notes:
            This function will be called by subclasses of MessagePassing that have trainable weights.
        """
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
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the message method in order to replace it with their own message mechanism.

        Args:
            x_source (Tensor, shape = (..., n_source_cells, in_channels)):
                Input features on source cells.
                Assumes that all source cells have the same rank r.
            x_target (Tensor, shape = (..., n_target_cells, in_channels)):
                Input features on target cells.
                Assumes that all target cells have the same rank s.
                Optional. If not provided, x_target is assumed to be x_source,
                i.e. source cells send messages to themselves.

        Returns:
            torch.Tensor, shape = (..., n_source_cells, in_channels): Messages on source cells.
        """
        return x_source

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages.

        This provides a default attention function to the message-passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the attention method in order to replace it with their own attention mechanism.

        The implementation follows [1]_.

        Args:
            x_source (torch.Tensor, shape = (n_source_cells, in_channels)): Input features on source cells. Assumes that all source cells have the same rank r.
            x_target (torch.Tensor, shape = (n_target_cells, in_channels)): Input features on target cells. Assumes that all target cells have the same rank s.

        Returns:
            torch.Tensor, shape = (n_messages, 1): one scalar per message between a source and a target cell.
        """
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
        """Aggregate messages on each target cell.

        A target cell receives messages from several source cells.
        This function aggregates these messages into a single output
        feature per target cell.

        🟧 This function corresponds to the within-neighborhood aggregation
        defined in [1]_ and [2]_.

        Args:
            x_message (torch.Tensor, shape = (..., n_messages, out_channels)): Features associated with each message. One message is sent from a source cell to a target cell.

        Returns:
            Tensor, shape = (...,  n_target_cells, out_channels): Output features on target cells. Each target cell aggregates messages from several source cells. Assumes that all target cells have the same rank s.
        """
        aggr = scatter(self.aggr_func)
        return aggr(x_message, self.target_index_i, 0)

    def forward(self, x_source, neighborhood, x_target=None):
        r"""Forward pass.

        This implements message passing for a given neighborhood:

        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        The message passing is decomposed into two steps:

        1. 🟥 Message: A message :math:`m_{y \rightarrow x}^{\left(r \rightarrow s\right)}`
        travels from a source cell :math:`y` of rank r to a target cell :math:`x` of rank s
        through a neighborhood of :math:`x`, denoted :math:`\mathcal{N} (x)`,
        via the message function :math:`M_\mathcal{N}`:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                = M_{\mathcal{N}}\left(\mathbf{h}_x^{(s)}, \mathbf{h}_y^{(r)}, \Theta \right),

        where:

        - :math:`\mathbf{h}_y^{(r)}` are input features on the source cells, called `x_source`,
        - :math:`\mathbf{h}_x^{(s)}` are input features on the target cells, called `x_target`,
        - :math:`\Theta` are optional parameters (weights) of the message passing function.

        Optionally, attention can be applied to the message, such that:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                \leftarrow att(\mathbf{h}_y^{(r)}, \mathbf{h}_x^{(s)}) . m_{y \rightarrow x}^{\left(r \rightarrow s\right)}

        2. 🟧 Aggregation: Messages are aggregated across source cells :math:`y` belonging to the
        neighborhood :math:`\mathcal{N}(x)`:

        .. math::
            m_x^{\left(r \rightarrow s\right)}
                = \text{AGG}_{y \in \mathcal{N}(x)} m_{y \rightarrow x}^{\left(r\rightarrow s\right)},

        resulting in the within-neighborhood aggregated message :math:`m_x^{\left(r \rightarrow s\right)}`.

        Details can be found in [1]_ and [2]_.

        Args:
            x_source (Tensor, shape = (..., n_source_cells, in_channels)): Input features on source cells. Assumes that all source cells have the same rank r.
            neighborhood (torch.sparse, shape = (n_target_cells, n_source_cells)): Neighborhood matrix.
            x_target (Tensor, shape = (..., n_target_cells, in_channels)): Input features on target cells. Assumes that all target cells have the same rank s. Optional. If not provided, x_target is assumed to be x_source,
                i.e. source cells send messages to themselves.

        Returns:
            torch.Tensor, shape = (..., n_target_cells, out_channels): Output features on target cells. Assumes that all target cells have the same rank s.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        neighborhood_values = neighborhood.values()

        x_message = self.message(x_source=x_source, x_target=x_target)
        x_message = x_message.index_select(-2, self.source_index_j)

        if self.att:
            attention_values = self.attention(x_source=x_source, x_target=x_target)
            neighborhood_values = torch.multiply(neighborhood_values, attention_values)

        x_message = neighborhood_values.view(-1, 1) * x_message
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
        """Implementation of the AllSetTransformer Layer proposed in [1]_.

        Args:
            in_channels (int): Dimension of the input features.
            hidden_channels (int): Dimension of the hidden features.
            heads (int, default=4): Number of attention heads.
            number_queries (int, default=1): Number of queries.
            dropout (float, optional): Dropout probability.
            mlp_num_layers (int, default=1): Number of layers in the MLP.
            mlp_activation (callable or None, optional): Activation function in the MLP.
            mlp_dropout (float, optional): Dropout probability in the MLP.
            mlp_norm (str or None, optional): Type of layer normalization in the MLP.
            **kwargs (optional): Additional arguments for the layer modules.

        References:
        .. [1] Chien, Pan, Peng and Milenkovic.
            You are AllSet: a multiset function framework for hypergraph neural networks.
            ICLR 2022.
            https://arxiv.org/abs/2106.13264
        """
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

    def forward(self, x_0, incidence_1, return_stats=False):
        r"""Forward computation.

        Vertex to edge:

        .. math::
            \begin{align*}
                &🟧 \quad m_{\rightarrow z}^{(\rightarrow 1)}
                    = AGG_{y \in \\mathcal{B}(z)} (h_y^{t, (0)}, h_z^{t,(1)}) \quad \text{with attention}\\
                &🟦 \quad h_z^{t+1,(1)}
                    = \text{LN}(m_{\rightarrow z}^{(\rightarrow 1)} + \text{MLP}(m_{\rightarrow z}^{(\rightarrow 1)} ))
            \end{align*}

        Edge to vertex:

        .. math::
            \begin{align*}
                &🟧 \quad m_{\rightarrow x}^{(\rightarrow 0)}
                    = AGG_{z \in \mathcal{C}(x)} (h_z^{t+1,(1)}, h_x^{t,(0)}) \quad \text{with attention}\\
                &🟦 \quad h_x^{t+1,(0)}
                    = \text{LN}(m_{\rightarrow x}^{(\rightarrow 0)} + \text{MLP}(m_{\rightarrow x}^{(\rightarrow 0)} ))
            \end{align*}

        Args:
            x_0 (torch.Tensor, shape = (n_nodes, channels)): Node input features.
            incidence_1 (torch.sparse, shape = (n_nodes, n_hyperedges)): Incidence matrix :math:`B_1` mapping hyperedges to nodes.
            return_stats (bool, default=False): Whether to return attention statistics.

        Returns:
            torch.Tensor: Output node features.
            torch.Tensor: Output hyperedge features.
            dict (optional): Attention statistics.
        """
        if x_0.shape[-2] != incidence_1.shape[-2]:
            raise ValueError(
                f"Shape of incidence matrix ({incidence_1.shape}) does not have the correct number of nodes ({x_0.shape[0]})."
            )

        if return_stats:
            x_1, stats = self.vertex2edge(
                x_0, incidence_1.transpose(1, 0), return_stats=True
            )
        else:
            x_1 = self.vertex2edge(x_0, incidence_1.transpose(1, 0))

        x_1 = F.relu(x_1)
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_0 = F.relu(self.edge2vertex(x_1, incidence_1))
        x_0 = F.dropout(x_0, p=self.dropout, training=self.training)

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
        """A module for AllSet Transformer block in a bipartite graph.

        Args:
            in_channels (int): Dimension of the input features.
            hidden_channels (int): Dimension of the hidden features.
            heads (int, default=4): Number of attention heads.
            number_queries (int, default=1): Number of queries.
            dropout (float, default=0.0): Dropout probability.
            mlp_num_layers (int, default=1): Number of layers in the MLP.
            mlp_activation (callable or None, optional): Activation function in the MLP.
            mlp_dropout (float, optional): Dropout probability in the MLP.
            mlp_norm (str or None, optional): Type of layer normalization in the MLP.
            initialization (Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"): Initialization method.
        """
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.number_queries = number_queries
        self.dropout = dropout

        # For neighbor nodes (source side, key)
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

        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.multihead_att.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        if hasattr(self.mlp, "reset_parameters") and callable(
            self.mlp.reset_parameters
        ):
            self.mlp.reset_parameters()

    def forward(self, x_source, neighborhood, return_stats=False):
        """Forward computation.

        Args:
            x_source (Tensor, shape = (…, n_source_cells, in_channels)):
                Input features.
            neighborhood (torch.sparse, shape = (n_target_cells, n_source_cells)):
                Neighborhood matrix.
            return_stats (bool, default=False):
                Whether to return attention statistics.

        Returns:
            x_message_on_target: Output sum over features on target cells.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()

        # Obtain MH from Eq(7) in AllSet paper [1]
        if return_stats:
            x_message_on_target, stats = self.multihead_att(
                x_source, neighborhood, return_stats=True
            )
        else:
            x_message_on_target = self.multihead_att(x_source, neighborhood)

        # Obtain Y from Eq(8) in AllSet paper [1]
        # Skip-connection (broadcased)
        x_message_on_target = x_message_on_target + self.multihead_att.Q_weight.permute(
            1, 0, 2
        )
        x_message_on_target = x_message_on_target.unsqueeze(2)
        # Permute: n,h,q,c -> n,q,h,c
        x_message_on_target = x_message_on_target.permute(0, 2, 1, 3)
        x_message_on_target = self.ln0(
            x_message_on_target.reshape(-1, self.number_queries, self.hidden_channels)
        )

        # Obtain LN(Y+FF(Y)) in Eq(8) in AllSet paper [1]
        x_message_on_target = self.ln1(
            x_message_on_target + F.relu(self.mlp(x_message_on_target))
        )

        if return_stats:
            return x_message_on_target.sum(dim=1), stats
        return x_message_on_target.sum(dim=1)


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
        """MH module from Eq(7) in AllSet paper [1]_.

        Args:
            in_channels (int): Dimension of input features.
            hidden_channels (int): Dimension of hidden features.
            aggr_norm (bool, default=False): Whether to normalize the aggregated message by the neighborhood size.
            update_func (str or None, optional): Update method to apply to message.
            heads (int, default=4): Number of attention heads.
            number_queries (int, default=1): Number of queries.
            initialization (Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"): Initialization method.
            initialization_gain (float, default=1.414): Gain factor for initialization.
        """
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
        # self.KV_weight = torch.nn.Parameter(
        #     torch.randn(self.heads, self.in_channels, 2 * self.hidden_channels)
        # )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.V_weight, gain=self.initialization_gain)
            # torch.nn.init.xavier_uniform_(self.KV_weight, gain=self.initialization_gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.V_weight, gain=self.initialization_gain)
            # torch.nn.init.xavier_normal_(self.KV_weight, gain=self.initialization_gain)

        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def attention(self, x_source, neighborhood):
        """Compute (QK^T) of transformer-based architectures.

        Args:
            x_source (torch.Tensor, shape = (n_source_cells, in_channels)): Input features on source cells.
                Assumes that all source cells have the same rank r.
            neighborhood (torch.sparse, shape = (n_target_cells, n_source_cells)): Neighborhood matrix.

        Returns:
            torch.Tensor, shape = (n_target_cells, heads, number_queries, n_source_cells): Attention weights: one scalar per message between a source and a target cell.
        """
        x_K = torch.matmul(x_source, self.K_weight).permute(1, 0, 2)
        alpha = (x_K * self.Q_weight.permute(1, 0, 2)).sum(-1)
        alpha = F.leaky_relu(alpha, 0.2)
        return softmax(alpha[self.source_index_j], index=self.target_index_i)

    def attention_with_entropy(self, x_source, neighborhood):
        """Compute attention and calculate entropy per hyperedge."""
        # x_source: [S, C], K_weight: [H, C, D] -> [H, S, D]
        x_K = torch.matmul(x_source, self.K_weight)
        # alpha shape: [H, S, Q]
        alpha = torch.einsum("hsd,hqd->hsq", x_K, self.Q_weight)
        # Scaling to avoid extremely large values before softmax
        alpha = alpha / math.sqrt(self.hidden_channels // self.heads)
        alpha = F.leaky_relu(alpha, 0.2)

        # [n_source, heads, number_queries]
        alpha = alpha.permute(1, 0, 2)
        # [n_messages, heads, number_queries]
        alpha_msg = alpha[self.source_index_j]

        # Get attention weights (p_i)
        attn_weights = softmax(alpha_msg, index=self.target_index_i, dim=0)

        # Calculate Entropy: -sum(p * log(p))
        eps = 1e-12
        p = attn_weights.clamp(min=eps)
        ent_per_msg = -p * torch.log(p)

        # Sum entropy components for each target (hyperedge)
        # [n_targets, heads, number_queries]
        entropy = pyg_scatter(ent_per_msg, self.target_index_i, dim=0, reduce="sum")

        # Calculate cardinality (k) for normalization
        ones = torch.ones_like(self.source_index_j, dtype=torch.float)
        card_msg = pyg_scatter(ones, self.target_index_i, dim=0, reduce="sum")
        cardinality = card_msg.view(-1, 1, 1)

        # Normalized Entropy: H / log(k)
        log_k = torch.log(cardinality.clamp(min=1.0001))
        mask = (cardinality > 1.1).float()
        norm_entropy = (entropy * mask) / (log_k + (1 - mask))
        norm_entropy = torch.clamp(norm_entropy, 0.0, 1.0)

        if self.number_queries == 1:
            attn_weights = attn_weights.squeeze(-1)
            norm_entropy = norm_entropy.squeeze(-1)

        return attn_weights, norm_entropy, card_msg

    def forward(self, x_source, neighborhood, return_stats=False):
        """Forward pass.

        Computes (QK^T)V attention mechanism of transformer-based architectures.
        Module MH from Eq (7) in AllSet paper [1]_.

        Args:
            x_source (Tensor, shape = (..., n_source_cells, in_channels)): Input features on source cells. Assumes that all source cells have the same rank r.
            neighborhood (torch.sparse, shape = (n_target_cells, n_source_cells)): Neighborhood matrix.
            return_stats (bool, default=False): Whether to return attention statistics (entropy and cardinality) along with the output.

        Returns:
            Tensor, shape = (..., n_target_cells, out_channels): Output features on target cells.
                Assumes that all target cells have the same rank s.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()

        attn_weights, entropy, cardinality = self.attention_with_entropy(
            x_source, neighborhood
        )

        x_message = torch.matmul(x_source, self.V_weight)
        x_message = x_message.permute(1, 0, 2)[
            self.source_index_j
        ] * attn_weights.unsqueeze(-1)
        out = pyg_scatter(x_message, self.target_index_i, dim=0, reduce="sum")

        if return_stats:
            return out, {"entropy": entropy, "cardinality": cardinality}
        return out


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
        """A module for a multi-layer perceptron (MLP).

        Args:
            in_channels (int): Dimension of the input features.
            hidden_channels (list of int): List of dimensions of the hidden features.
            norm_layer (callable or None, optional): Type of layer normalization.
            activation_layer (callable or None, optional): Type of activation function.
            dropout (float, optional): Dropout probability.
            inplace (bool, default=False): Whether to do the operation in-place.
            bias (bool, default=False): Whether to add bias.
        """
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
        """A module that combines multiple AllSet layers [1]_ to form a neural network.

        Args:
            in_channels (int): Dimension of the input features.
            hidden_channels (int): Dimension of the hidden features.
            n_layers (int, default: 2): Number of AllSet layers in the network.
            heads (int, default: 4): Number of attention heads.
            dropout (float, default: 0.2): Dropout probability.
            mlp_num_layers (int, default: 2): Number of layers in the MLP.
            mlp_dropout (float, default: 0.2): Dropout probability in the MLP.
            **kwargs: Additional arguments for the inner layers.

        References:
        .. [1] Chien, Pan, Peng and Milenkovic.
            You are AllSet: a multiset function framework for hypergraph neural networks.
            ICLR 2022.
            https://arxiv.org/abs/2106.13264.
        """
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

    def forward(self, x_0, incidence_1, return_stats=False):
        """Forward computation.

        Args:
            x_0 (torch.Tensor): Input features.
            incidence_1 (torch.Tensor): Incidence matrix. (n_nodes, n_hyperedges)
            return_stats (bool, default=False): Whether to return attention statistics.

        Returns:
            torch.Tensor: Output prediction.
            torch.Tensor: Output hyperedge features.
            dict (optional): Attention statistics from the last layer.
        """
        for i, layer in enumerate(self.layers):
            if return_stats and i == len(self.layers) - 1:
                x_0, x_1, stats = layer(x_0, incidence_1, return_stats=True)
            else:
                x_0, x_1 = layer(x_0, incidence_1)

        if return_stats:
            return x_0, x_1, stats
        return x_0, x_1


class GATModel(nn.Module):
    """GAT model."""

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

    def forward(self, x, edge_index):
        """Forward pass."""
        for layer, norm in zip(self.layers, self.layers_norm):
            x_new = layer(x, edge_index)
            x_new = norm(x_new)
            x = x + x_new  # Residual connection
            x = F.silu(x)  # Nonlinear activation after residual sum
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
        add_time_to="ligand",  # "ligand", "both", or "none"
        use_gat=True,
    ):
        """Ligand-only prediction with self-attention inside each ligand and cross-attention to pocket atoms."""
        super().__init__()
        self.spatial_dim = spatial_dim
        self.atom_dim = atom_dim
        self.x_dim = spatial_dim + atom_dim
        self.h_dim = h_dim
        self.add_time_to = add_time_to

        self.cutoff_lig = 10.0
        self.cutoff_poc = 1.0
        self.cutoff_cross = 1.5

        self.lig_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.poc_embed = nn.Sequential(
            nn.Linear(self.x_dim, h_dim), nn.SiLU(), nn.Linear(h_dim, h_dim)
        )
        self.time_embed = SimpleTimeEmbedding(h_dim)

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

    def create_edge_index(self, x_lig, x_pocket, batch_lig, batch_pocket):
        """Create edge index for GATv2Conv."""
        lig_pos = x_lig[:, :3]
        poc_pos = x_pocket[:, :3]

        N1, N2 = lig_pos.size(0), poc_pos.size(0)

        edge_ll = radius_graph(
            lig_pos,
            r=self.cutoff_lig,
            batch=batch_lig,
            loop=False,
            max_num_neighbors=1000,
        )  # [2, E_ll]
        # Intra-pocket edges
        edge_pp = radius_graph(
            poc_pos,
            r=self.cutoff_poc,
            batch=batch_pocket,
            loop=False,
            max_num_neighbors=1000,
        )  # [2, E_pp]

        # Cross edges using bipartite radius:
        # radius(x=poc, y=lig) returns [row_y, col_x] = [lig_idx, poc_idx]
        edge_lp = radius(
            poc_pos,
            lig_pos,
            r=self.cutoff_cross,
            batch_x=batch_pocket,
            batch_y=batch_lig,
            max_num_neighbors=1000,
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

        return edge_lp, edge_index

    def create_incidence(self, x_lig, x_pocket, batch_lig, batch_pocket):
        """Create hyperedges."""
        device = x_lig.device
        n_lig = x_lig.shape[0]
        n_poc = x_pocket.shape[0]
        n_nodes = n_lig + n_poc
        n_hyperedges = 0
        n_values = 0
        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1
        # Hyperedges with all ligand atoms in same batch
        valid = torch.unique(batch_lig)
        non_zeros = (batch_lig.unsqueeze(1) == valid.unsqueeze(0)).nonzero(
            as_tuple=False
        )
        row = non_zeros[:, 0]
        col = non_zeros[:, 1]
        n_hyperedges += valid.size(0)
        n_values += row.size(0)

        # Hyperedges with all pocket atoms in same batch
        valid = torch.unique(batch_pocket)
        non_zeros = (batch_pocket.unsqueeze(1) == valid.unsqueeze(0)).nonzero(
            as_tuple=False
        )
        row = torch.cat([row, non_zeros[:, 0] + n_lig], dim=0)
        col = torch.cat([col, non_zeros[:, 1] + n_hyperedges], dim=0)
        n_hyperedges += valid.size(0)
        n_values += non_zeros.size(0)

        # Create incidence matrix with lp hyperedges
        edge_lp, edge_index = self.create_edge_index(
            x_lig, x_pocket, batch_lig[is_lig], batch_pocket[is_poc]
        )
        rows_lp = edge_lp[0]
        cols_lp = edge_lp[1]

        # Ligand self-connections: each ligand belongs to its own hyperedge
        rows_lig = torch.arange(n_lig, device=device)
        cols_lig = torch.arange(n_lig, device=device)

        # Concatenate all
        rows = torch.cat([rows_lig, cols_lp + n_lig], dim=0)
        cols = torch.cat([cols_lig, rows_lp], dim=0)
        row = torch.cat([row, rows], dim=0)
        col = torch.cat([col, cols + n_hyperedges], dim=0)
        n_values += rows.size(0)
        n_hyperedges += n_lig

        # Create incidence matrix
        value = torch.ones(n_values, dtype=torch.long, device=device)
        incidence1 = torch.sparse_coo_tensor(
            torch.stack([row, col], dim=0),
            value,
            (n_nodes, n_hyperedges),
            device=device,
        )
        incidence1 = incidence1.coalesce()
        return incidence1, edge_index

    def forward(
        self, x_lig, t, x_pocket, batch_lig, batch_pocket, return_diagnostics=False
    ):
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
        # Create incidence matrix
        incidence1, edge_index, edge_types = self.create_incidence_with_types(
            x_lig, x_pocket, batch_lig, batch_pocket
        )
        # Creates features
        h = torch.cat([lig_h, poc_h], dim=0)

        # AllSetTransformer
        if return_diagnostics:
            h_ast, _, stats = self.model(h, incidence1, return_stats=True)
        else:
            h_ast, _ = self.model(h, incidence1)

        # GAT
        if self.use_gat:
            h_gat = self.gat_model(h, edge_index)
            h = h_gat + h_ast  # GAT + AST
        else:
            h = h_ast

        N1 = lig_h.size(0)
        h_lig = h[:N1, :]

        # Predict ligand velocities only
        vel_lig = self.out_vel(h_lig)  # [N_lig, x_dim]

        if return_diagnostics:
            return vel_lig, self.format_diagnostics(stats, edge_types)
        return vel_lig

    def format_diagnostics(self, stats, edge_types):
        """Organizes entropy by cardinality and returns summary statistics."""
        entropy = stats["entropy"].mean(dim=-1).cpu()  # Average across heads
        cards = stats["cardinality"].cpu().int()
        types = edge_types.cpu()

        diag_dict = {}
        for i in range(len(cards)):
            c = cards[i].item()
            if c not in diag_dict:
                diag_dict[c] = []
            diag_dict[c].append({"entropy": entropy[i].item(), "type": types[i].item()})

        # Summary statistics over all hyperedges
        summary_stats = {
            "mean_entropy": entropy.mean().item(),
            "max_entropy": entropy.max().item(),
            "min_entropy": entropy.min().item(),
        }

        return {"per_cardinality": diag_dict, "summary": summary_stats}

    def create_incidence_with_types(self, x_lig, x_pocket, batch_lig, batch_pocket):
        """Create hyperedges and track their semantic types."""
        device = x_lig.device
        n_lig = x_lig.shape[0]
        n_poc = x_pocket.shape[0]
        n_nodes = n_lig + n_poc

        row_list = []
        col_list = []
        type_list = []

        current_hyperedge_id = 0
        is_lig = batch_lig != -1
        is_poc = batch_pocket != -1

        # ----- 1. Global Ligand Hyperedges (Type 0) -----
        valid_lig_batches = torch.unique(batch_lig)
        non_zeros_lig = (
            batch_lig.unsqueeze(1) == valid_lig_batches.unsqueeze(0)
        ).nonzero(as_tuple=False)

        row_list.append(non_zeros_lig[:, 0])
        col_list.append(non_zeros_lig[:, 1] + current_hyperedge_id)

        num_global_lig = valid_lig_batches.size(0)
        type_list.append(torch.full((num_global_lig,), 0, device=device))
        current_hyperedge_id += num_global_lig

        # ----- 2. Global Pocket Hyperedges (Type 1) -----
        valid_poc_batches = torch.unique(batch_pocket)
        non_zeros_poc = (
            batch_pocket.unsqueeze(1) == valid_poc_batches.unsqueeze(0)
        ).nonzero(as_tuple=False)

        row_list.append(non_zeros_poc[:, 0] + n_lig)
        col_list.append(non_zeros_poc[:, 1] + current_hyperedge_id)

        num_global_poc = valid_poc_batches.size(0)
        type_list.append(torch.full((num_global_poc,), 1, device=device))
        current_hyperedge_id += num_global_poc

        # ----- 3. Ligand Self-loops (Type 2) -----
        # Each ligand atom belongs to its own unique hyperedge
        row_list.append(torch.arange(n_lig, device=device))
        col_list.append(torch.arange(n_lig, device=device) + current_hyperedge_id)

        type_list.append(torch.full((n_lig,), 2, device=device))
        current_hyperedge_id += n_lig

        # ----- 4. Local Radius / Cross-Interface (Type 3) -----
        edge_lp, edge_index = self.create_edge_index(
            x_lig, x_pocket, batch_lig[is_lig], batch_pocket[is_poc]
        )
        rows_lp = edge_lp[0]  # Ligand indices
        cols_lp = edge_lp[1]  # Pocket indices (local within pocket set)

        # In your original logic: row = cols_lp + n_lig, col = rows_lp
        # This means one hyperedge per ligand atom, containing its pocket neighbors.
        row_list.append(cols_lp + n_lig)
        col_list.append(rows_lp + current_hyperedge_id)

        # The number of hyperedges added here is equal to the number of ligand atoms (n_lig)
        # because rows_lp ranges from 0 to n_lig-1
        type_list.append(torch.full((n_lig,), 3, device=device))
        current_hyperedge_id += n_lig

        # ----- Final Assembly -----
        final_rows = torch.cat(row_list, dim=0)
        final_cols = torch.cat(col_list, dim=0)
        final_types = torch.cat(type_list, dim=0)  # [n_hyperedges]

        value = torch.ones(final_rows.size(0), dtype=torch.long, device=device)
        incidence1 = torch.sparse_coo_tensor(
            torch.stack([final_rows, final_cols], dim=0),
            value,
            (n_nodes, current_hyperedge_id),
            device=device,
        )

        return incidence1.coalesce(), edge_index, final_types
