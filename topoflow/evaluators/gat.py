"""Graph Attention Network (GAT) implementation using PyTorch Geometric and PyTorch Lightning."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from pytorch_lightning import LightningModule
from sklearn.metrics import mean_squared_error, mean_absolute_error


class GAT(torch.nn.Module):
    """Graph Attention Network (GAT) model."""

    def __init__(
        self,
        in_channels,
        edge_in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=4,
        dropout=0.6,
    ):
        """Initialize the GAT model.

        Args:
            in_channels (int): Number of input features per node.
            edge_in_channels (int): Number of input features per edge.
            hidden_channels (int): Number of hidden features per node.
            out_channels (int): Number of output features per node.
            num_layers (int): Number of GAT layers.
            heads (int): Number of attention heads in each GAT layer.
            dropout (float): Dropout rate for GAT layers.
        """
        super().__init__()

        self.num_layers = num_layers

        # Input layer
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GATv2Conv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                edge_dim=edge_in_channels,
                heads=heads,
                dropout=dropout,
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_channels * heads,
                    out_channels=hidden_channels,
                    edge_dim=edge_in_channels,
                    heads=heads,
                    dropout=dropout,
                )
            )

        # Output layer (heads=1 and concat=False to get final output shape [num_nodes, out_channels])
        self.convs.append(
            GATv2Conv(
                in_channels=hidden_channels * heads,
                out_channels=out_channels,
                edge_dim=edge_in_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )
        )

    def forward(self, x, edge_index, edge_attr):
        """Forward pass through multiple GAT layers."""
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            x = F.elu(x) if i < self.num_layers - 1 else x
        return x


class GATEvaluator(LightningModule):
    """Graph Attention Network Evaluator for PyTorch Lightning."""

    def __init__(
        self,
        in_channels,
        edge_in_channels,
        hidden_channels,
        out_channels,
        num_layers=2,
        heads=4,
        dropout=0.6,
        lr=1e-3,
        monitor_metric="mae",
        optimizer=None,
    ):
        """Initialize the GATEvaluator.

        Args:
            in_channels (int): Number of input features per node.
            edge_in_channels (int): Number of input features per edge.
            hidden_channels (int): Number of hidden features per node.
            out_channels (int): Number of output features per node.
            num_layers (int): Number of GAT layers.
            heads (int): Number of attention heads in each GAT layer.
            dropout (float): Dropout rate for GAT layers.
            lr (float): Learning rate for the optimizer.
            monitor_metric (str): Metric to monitor for validation and test steps.
            optimizer (callable): Optimizer class to use for training.
        """
        super().__init__()

        # Graph Attention Network
        self.gat = GAT(
            in_channels=in_channels,
            edge_in_channels=edge_in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )

        # Training hyperparameters
        self.lr = lr
        self.monitor_metric = monitor_metric
        if optimizer is not None:
            self.optimizer = optimizer(self.parameters(), lr=self.lr)
        self.save_hyperparameters(ignore=["optimizer"])

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass."""
        x = self.gat(x, edge_index, edge_attr)  # Node-level output
        x = global_mean_pool(x, batch)  # Graph-level output
        return x

    def _compute_loss_and_metric(self, batch, stage):
        """Common logic for training, validation, and test steps."""
        x, edge_index, edge_attr, batch_index, y = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
            batch.y,
        )

        y_hat = self(x, edge_index, edge_attr, batch_index)
        y = y.view_as(y_hat)

        loss = (
            F.mse_loss(y_hat, y)
            if self.monitor_metric == "mse"
            else F.l1_loss(y_hat, y)
        )

        metric = (
            mean_squared_error(y_hat, y)
            if self.monitor_metric == "mse"
            else mean_absolute_error(y_hat, y)
        )

        if stage == "val":
            stage = "val_monitor"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(
            f"{stage}/{self.monitor_metric}",
            metric,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def compute_score(self, batch):
        """Compute the model's prediction on the given input."""
        batch = batch[0]  # For compatibility with compute_score of the other metrics
        if batch is None:
            return torch.tensor(0.0)
        x, edge_index, edge_attr, batch = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        return self.forward(x, edge_index, edge_attr, batch=batch).cpu()

    def training_step(self, batch, batch_idx):
        """Training step for updating the model parameters."""
        return self._compute_loss_and_metric(batch, "train")

    def validation_step(self, batch, batch_idx):
        """Validation step for evaluating the model on validation data."""
        self._compute_loss_and_metric(batch, "val")

    def test_step(self, batch, batch_idx):
        """Test step for evaluating the model on test data."""
        self._compute_loss_and_metric(batch, "test")

    def configure_optimizers(self):
        """Configure the optimizer for training."""
        return self.optimizer
