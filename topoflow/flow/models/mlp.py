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
        additional_features="concat",
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
                (add as learned embeddings). Defaults to 'concat'.

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

    def forward(self, x, t, y=None):
        """Forward pass through the MLP vector field.

        Args:
            x: Input tensor representing the current state. [batch_size, input_dim]
            t: Time tensor. [batch_size, 1] (or [batch_size] which will be unsqueezed).
            y: Optional conditioning variable. [batch_size] (Long tensor for embedding).
                Defaults to None.

        Returns:
            The predicted vector field. [batch_size, output_dim]
        """
        x = self.layer_norm(x)

        if t.ndim == 0:
            t = t.unsqueeze(0).unsqueeze(-1).repeat(x.shape[0], 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)

        if self.additional_features == "add":
            t_emb = self.t_embedder(t)
            t_emb = self.layer_norm(t_emb)
            xt = x + t_emb
            if y is not None and self.y_embedder is not None:
                y_emb = self.y_embedder(y)
                y_emb = self.layer_norm(y_emb)
                xt = xt + y_emb
        else:
            xt = torch.cat([x, t], dim=-1)
            if y is not None:
                y_float = y.view(-1, 1).float()
                xt = torch.cat([xt, y_float], dim=-1)

        return self.mlp(xt)


class MLP(torch.nn.Module):
    """A general Multi-layer perceptron (MLP) model."""

    def __init__(
        self, input_dim, output_dim, hidden_dims=None, activation=torch.nn.SiLU
    ):
        """Initializes the MLP model.

        Args:
            input_dim: The dimensionality of the input.
            output_dim: The dimensionality of the output.
            hidden_dims: A list of integers specifying the dimensions for the
                hidden layers. Defaults to [64, 64].
            activation: The activation function to use between layers.
                Defaults to `torch.nn.SiLU`.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 64]
        self.mlp = build_mlp([input_dim] + hidden_dims + [output_dim], activation)
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        """Forward pass through the MLP.

        Args:
            x: Input tensor.

        Returns:
            Output tensor from the MLP.
        """
        return self.mlp(x)

    def get_loss(self, x, y):
        """Computes the cross-entropy loss between the model's predictions and target labels.

        Args:
            x: Input tensor to the MLP. [batch_size, *input_shape]
            y: Target labels (class indices). [batch_size]

        Returns:
            The computed cross-entropy loss.
        """
        if x.ndim > 2:
            x = x.view(x.shape[0], -1)
        y_hot = torch.nn.functional.one_hot(
            y, num_classes=self.mlp[-1].out_features
        ).float()
        return self.loss_fn(self.forward(x), y_hot)
