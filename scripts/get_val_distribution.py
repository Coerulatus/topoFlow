"""Script to train a Topological Flow model on a given dataset."""

import argparse
import os
import psutil
import numpy as np
import torch
import lightning.pytorch as pl
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
from topoflow.flow.schedulers import LinearAlpha, LinearBeta, CosineAlpha, CosineBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.vector_field_models import (
    TransformerVectorField,
    MLPVectorField,
    TransformerVectorFieldLigand,
    GATVectorField,
    AllSetTransformerVectorField,
    TransformerDistanceVectorField,
)
from topoflow.flow.flow_models_ligand import FlowModel
import topoflow.utils.constants as constants
from topoflow.flow.utils import InverseTimeFactor, InverseTimeFactorNoClamping
from topoflow.data.augmentations import RandomRotateTranslate
from topoflow.utils.metrics import get_distributions


def train_flow_model(
    data_dir,
    data_name,
):
    """Train a Topological Flow model on the specified dataset.

    Args:
        data_dir (str): Directory containing the training data.
        data_name (str): Name of the dataset to use.
        augment (bool): Whether to apply data augmentations.
        vector_field_model (str): Type of vector field model to use.
        hidden_dim (int): Dimension of the hidden layers in the model.
        evaluator (str): Evaluator to use for model evaluation. Either 'rdkit' or a path to a GATEvaluator checkpoint.
        checkpoint_dir (str): Directory to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        start_lr (float): Initial learning rate for the optimizer.
        max_lr (float): Maximum learning rate for the scheduler.
        min_lr (float): Minimum learning rate for the scheduler.
        max_frac (float): When to stop increasing the learning rate.
        min_time_factor (float): Minimum time factor for the flow model.
        max_time_factor (float): Maximum time factor for the flow model.
        max_atom_time (float): Maximum atom time for the flow model.
        w_loss_last_step (float): Weight for the loss at the last time step.
        velocity_scale (float): Scaling factor for the velocity in the loss computation.
        seed (int): Random seed for reproducibility.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.
        run_name (str, optional): Name of the MLflow run. Defaults to None.
        previous_checkpoint (str, optional): Path to a checkpoint to resume training from. Defaults to None.
        skip_validation (bool, optional): Whether to skip validation during training. Defaults to False.
    """
    pl.seed_everything(21)  # For reproducibility
    # ======================================================================
    # Data Preparation
    # ======================================================================
    if data_name == "crossdock_full":
        val_dataset = CrossDocked(root=data_dir, split="val")
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    norm_params = np.load(norm_path, allow_pickle=True)
    mean = torch.tensor(norm_params["mean"])
    std = torch.tensor(norm_params["std"])
    val_dataset._data["x_lig"] = val_dataset._data["x_lig"] * std + mean
    val_dataset._data["x_pocket"] = val_dataset._data["x_pocket"] * std + mean

    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
        val_distribution = get_distributions(
            val_dataset,
            metrics=["qed", "logp", "sas", "lipinski", "gnina"],
            atom_decoder=constants.dataset_params[data_name]["atom_decoder"],
        )
        np.save(val_metric_dist_path, val_distribution)
    return val_distribution


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Topological Flow model.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Directory containing the training data.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="crossdock_full",
        choices=["crossdock_full"],
        help="Name of the dataset to use.",
    )
    args = parser.parse_args()

    train_flow_model(
        data_dir=args.data_dir,
        data_name=args.data_name,
    )
