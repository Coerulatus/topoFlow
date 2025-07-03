"""Graph Attention Network (GAT) implementation using PyTorch Geometric and PyTorch Lightning."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from lightning.pytorch import LightningModule
from torchmetrics.functional import mean_squared_error, mean_absolute_error
from rdkit import Chem
import topoflow.utils.constants as constants
from topoflow.utils.molecules import data_to_rdmol


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
        y_cols=None,
        norm_params=None,
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
            y_cols (list, optional): List of column names for labels.
            norm_params (dict, optional): Normalization parameters used for y.
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
        if optimizer is None:
            self.optimizer = torch.optim.Adam(
                self.parameters(),
                lr=lr,
                weight_decay=1e-5,
            )
        else:
            self.optimizer = optimizer(
                self.parameters(),
                lr=lr,
                weight_decay=1e-5,
            )
        self.y_cols = (
            y_cols if y_cols is not None else [f"y_{i}" for i in range(out_channels)]
        )
        self.norm_params = [[] for _ in self.y_cols]
        found_keys = 0
        for key in norm_params:
            for i, col in enumerate(self.y_cols):
                if key in col:
                    self.norm_params[i] = norm_params[key]
                    found_keys += 1
        assert found_keys == len(
            self.y_cols
        ), f"Expected {len(self.y_cols)} normalization parameters, but found {found_keys}. \
                Please ensure that y_cols contain the keys in norm_params."
        self.norm_params = torch.tensor(self.norm_params, dtype=torch.float32)
        # TODO: add param dataset_name
        self.atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]
        self.save_hyperparameters(ignore=["optimizer"])

    def forward(self, x, edge_index, edge_attr, batch):
        """Forward pass."""
        x = self.gat(x, edge_index, edge_attr)  # Node-level output
        x = global_mean_pool(x, batch)  # Graph-level output
        return x

    def _compute_loss_and_metric(self, batch, stage):
        """Common logic for training, validation, and test steps."""
        batch_size = batch.idx.shape[0]

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

        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        self.log(
            f"{stage}/{self.monitor_metric}",
            metric,
            prog_bar=True,
            on_epoch=True,
            sync_dist=True,
            batch_size=batch_size,
        )
        if stage == "val":

            def unnormalize(y, i):
                """Unnormalize the output using the provided normalization parameters."""
                return (
                    y * (self.norm_params[i, 1] - self.norm_params[i, 0])
                    + self.norm_params[i, 0]
                )

            metric_func = (
                mean_squared_error
                if self.monitor_metric == "mse"
                else mean_absolute_error
            )
            metrics = {
                y_col: metric_func(unnormalize(y_hat[:, i], i), unnormalize(y[:, i], i))
                for i, y_col in enumerate(self.y_cols)
            }
            self.log_dict(
                {f"{stage}/{y_col}": metric for y_col, metric in metrics.items()},
                batch_size=batch_size,
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

    def evaluate(self, batch, validate=True):
        """Evaluate the model on a batch of data and compute predicted molecular properties.

        Args:
            batch: A PyTorch Geometric Batch object.
            validate (bool): If True, sanitize and filter invalid molecules before evaluating validity.

        Returns:
            dict: A dictionary with keys 'validity', 'qed', 'logp', 'sas' from model predictions.
        """
        if validate:
            mols = data_to_rdmol(batch, self.atom_decoder)
            mols_valid = []
            mask = []
            for i, mol in enumerate(mols):
                try:
                    Chem.SanitizeMol(mol)
                    mols_valid.append(mol)
                    mask.append(i)
                except Exception:
                    continue

        if not mols_valid:
            return {
                "validity": torch.tensor(0.0),
                "qed": torch.tensor(0.0),
                "logp": torch.tensor(0.0),
                "sas": torch.tensor(0.0),
            }

        validity = torch.tensor(len(mols_valid) / len(mols), dtype=torch.float)

        x, edge_index, edge_attr, batch_index = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )
        y_hat = self(x, edge_index, edge_attr, batch_index).cpu()  # [num_graphs, 3]

        if validate:
            y_hat = y_hat[mask]

        metrics = {
            "validity": validity,
            "qed": y_hat[:, 0].mean(),
            "logp": y_hat[:, 1].mean(),
            "sas": y_hat[:, 2].mean(),
        }

        return metrics
