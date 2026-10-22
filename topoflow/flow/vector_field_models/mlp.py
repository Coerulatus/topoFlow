"""MLP-based vector field model."""

import torch


def build_mlp(dims, activation=torch.nn.SiLU):
    """Builds a multi-layer perceptron (MLP).

    This function constructs a sequential MLP model with a customizable number of
    layers, dimensions, and activation functions.

    Args:
        dims: A list of integers specifying the dimensions for each layer. The
            first element is the input dimension, the last is the output
            dimension, and intermediate elements define hidden layer dimensions.
        activation: The activation function to use between hidden layers.
            Defaults to `torch.nn.SiLU`.

    Returns:
        A `torch.nn.Sequential` model representing the MLP.
    """
    layers = []
    for i in range(len(dims) - 1):
        layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(activation())

    return torch.nn.Sequential(*layers)


class MLPVectorField(torch.nn.Module):
    r"""An MLP network designed to learn a vector field $u_t^\theta(x)$.

    This model takes an input state `x`, time `t`, and an optional conditioning
    variable `y` to predict a vector field. It supports different methods for
    integrating additional features (time and conditioning variable).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        y_classes=None,
        hidden_dims=None,
        activation=torch.nn.SiLU,
        additional_features="add",
    ):
        """Initializes the MLPVectorField.

        Args:
            input_dim: The dimensionality of the input state `x`.
            output_dim: The dimensionality of the output vector field.
            y_classes: The number of possible classes for the conditioning variable `y`.
                Required if `y` is used. Defaults to None.
            hidden_dims: A list of integers specifying the dimensions for the
                hidden layers of the MLP. Defaults to [64, 64].
            activation: The activation function to use between MLP layers.
                Defaults to `torch.nn.SiLU`.
            additional_features: Method to incorporate time and conditioning
                features. Can be 'concat' (concatenate to input) or 'add'
                (add as learned embeddings). Defaults to 'add'.

        Raises:
            ValueError: If `additional_features` is not 'concat' or 'add'.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]

        self.input_dim = input_dim
        self.additional_features = additional_features

        if self.additional_features == "concat":
            mlp_input_dim = input_dim + 1
            if y_classes is not None:
                mlp_input_dim += 1
            self.mlp = build_mlp(
                [mlp_input_dim] + hidden_dims + [output_dim], activation
            )
        elif self.additional_features == "add":
            self.mlp = build_mlp([input_dim] + hidden_dims + [output_dim], activation)
            self.t_embedder = torch.nn.Sequential(
                torch.nn.Linear(1, input_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(input_dim, input_dim),
            )
            self.y_embedder = None
            if y_classes is not None:
                self.y_embedder = torch.nn.Embedding(y_classes, input_dim)
        else:
            raise ValueError("additional_features must be 'concat' or 'add'")
        self.layer_norm = torch.nn.LayerNorm(input_dim)

    def forward(
        self,
        x_lig=None,
        t=None,
        x_pocket=None,
        batch_lig=None,
        batch_pocket=None,
        x=None,
        num_graphs=None,
    ):
        """Forward pass of the vector field model.

        Args:
            x_lig: [num_lig, x_dim] — ligand node features
            t: scalar or [batch_size] — time step
            x_pocket: [num_pocket, x_dim] — pocket node features
            batch_lig: [num_lig] — ligand graph index
            batch_pocket: [num_pocket] — pocket graph index
            x: [num_x, x_dim] — node features (ligand + pocket), or None if x_lig/x_pocket provided
            num_graphs: optional number of graphs in batch
        Returns:
            [num_lig, x_dim] or [num_x, x_dim] — vector field
        """
        lig_only = False
        if x is None and x_lig is not None:
            lig_only = True
            n_lig = x_lig.shape[0]
            x = (
                torch.cat([x_lig, x_pocket], dim=0)
                if x_pocket is not None
                else x_lig
            )

        x = self.layer_norm(x)

        is_lig = batch_lig != -1 if batch_lig is not None else None
        is_poc = batch_pocket != -1 if batch_pocket is not None else None

        if t.ndim == 0:
            t = t.unsqueeze(0).unsqueeze(-1).repeat(x.shape[0], 1)
        elif t.ndim == 1:
            if t.shape[0] != x.shape[0] and is_lig is not None and is_poc is not None:
                batch_all = torch.cat([batch_lig[is_lig], batch_pocket[is_poc]], dim=0)
                t = t[batch_all].unsqueeze(-1)
            else:
                t = t.unsqueeze(-1)

        if self.additional_features == "add":
            t_emb = self.t_embedder(t)
            t_emb = self.layer_norm(t_emb)
            xt = x + t_emb
        else:
            xt = torch.cat([x, t], dim=-1)

        out = self.mlp(xt)
        return out[:n_lig] if lig_only else out
