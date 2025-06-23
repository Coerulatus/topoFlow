"""FiLM-based neural network architecture for learning vector fields."""

import numpy as np
import torch
import torch.nn as nn


class FiLMLayer(nn.Module):
    """A Feature-wise Linear Modulation (FiLM) layer.

    This layer applies an affine transformation to its input, where the scaling (gamma)
    and shifting (beta) parameters are provided externally, allowing for
    conditional modulation of features.
    """

    def __init__(self, input_dim, output_dim):
        """Initializes the FiLMLayer.

        Args:
            input_dim: The dimensionality of the input features to the linear layer.
            output_dim: The dimensionality of the output features from the linear layer.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.film_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x, gamma, beta):
        """Applies the FiLM transformation.

        The operation performed is `gamma * self.film_layer(x) + beta`.

        Args:
            x: The input tensor to the linear layer.
            gamma: The scaling tensor for the affine transformation.
            beta: The shifting tensor for the affine transformation.

        Returns:
            The output tensor after applying the FiLM transformation.
        """
        return gamma * self.film_layer(x) + beta


class FiLMVectorField(nn.Module):
    """A neural network that learns a vector field using FiLM layers.

    This model can incorporate time and an optional conditioning variable
    (e.g., class labels) to modulate its internal layers, enabling the
    learning of conditional vector fields. It also supports residual connections.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        y_classes=None,
        hidden_dim=None,
        residual=False,
    ):
        """Initializes the FiLMVectorField.

        Args:
            input_dim: The dimensionality of the input to the vector field (e.g., state `x`).
            output_dim: The dimensionality of the output vector field.
            y_classes: The number of possible classes for the conditioning variable `y`.
                If None, no embedding for `y` is created. Defaults to None.
            hidden_dim: A list of integers specifying the dimensions for the hidden layers.
                Defaults to [128, 128].
            residual: If True, residual connections are used within the network.
                Requires `input_dim`, `output_dim`, and all `hidden_dim` to be equal.
                Defaults to False.

        Raises:
            ValueError: If `residual` is True but `input_dim`, `output_dim`, or
                any `hidden_dim` are not all equal.
        """
        super().__init__()
        if hidden_dim is None:
            hidden_dim = [128, 128]

        if residual and (
            input_dim != output_dim or any(h != output_dim for h in hidden_dim)
        ):
            raise ValueError(
                "Residual connections require input_dim to match output_dim, and all hidden_dim to match output_dim."
            )

        self.y_embed = (
            nn.Embedding(y_classes, hidden_dim[0]) if y_classes is not None else None
        )
        self.t_embed = nn.Sequential(
            nn.Linear(1, hidden_dim[0]),
            nn.SiLU(),
            nn.Linear(hidden_dim[0], hidden_dim[0]),
        )
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dim)):
            if i == 0:
                self.layers.append(FiLMLayer(input_dim, hidden_dim[i]))
            else:
                self.layers.append(FiLMLayer(hidden_dim[i - 1], hidden_dim[i]))
        self.out = nn.Linear(hidden_dim[-1], output_dim)

        self.gamma_beta = nn.ModuleList()
        dim_mult = 1 + (1 if y_classes is not None else 0)
        for i in range(len(hidden_dim)):
            self.gamma_beta.append(
                nn.Linear(hidden_dim[0] * dim_mult, hidden_dim[i] * 2)
            )

        self.residual = residual
        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim[i]) for i in range(len(hidden_dim))]
        )

    def forward(self, x, t, y):
        """Forward pass through the FiLM vector field.

        Args:
            x: The input tensor representing the state. Shape: [batch_size, input_dim].
            t: The time tensor. Shape: [batch_size, 1] or [batch_size].
            y: The optional conditioning variable tensor (e.g., class labels). Shape: [batch_size].

        Returns:
            The predicted vector field. Shape: [batch_size, output_dim].
        """
        B = x.shape[0]
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(B, -1)
        t_emb = self.t_embed(t)

        if self.y_embed is not None and y is not None:
            y_emb = self.y_embed(y)
            context = torch.cat([t_emb, y_emb], dim=-1)
        else:
            context = t_emb

        for i, layer in enumerate(self.layers):
            gamma_beta = self.gamma_beta[i](context)
            gamma, beta = torch.chunk(gamma_beta, 2, dim=-1)

            if self.residual:
                x_res = x
                x = layer(x, gamma, beta)
                x = x + x_res
            else:
                x = layer(x, gamma, beta)
            x = self.norms[i](x)

        x = self.out(x)
        return x
