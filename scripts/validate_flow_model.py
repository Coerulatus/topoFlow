"""Validate a Topological Flow model on a given molecular dataset and log results to MLflow.

This script loads a trained TopoFlow model from a checkpoint and runs the validation
procedure using the specified dataset. It logs the evaluation metrics to MLflow.
"""

import argparse
import os
import psutil
import numpy as np
import torch
import lightning.pytorch as pl
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.flow_models import FlowModel
import topoflow.utils.constants as constants
from topoflow.utils.metrics import get_distributions


def validate_flow_model(
    checkpoint_path,
    data_dir,
    data_name,
    batch_size,
    mlflow_tracking_uri,
    experiment_name,
    run_name=None,
):
    """Validate a Topological Flow model.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        data_dir (str): Directory containing the data.
        data_name (str): Name of the dataset to use.
        batch_size (int): Batch size for validation.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.
        run_name (str, optional): MLflow run name.
    """
    # ======================================================================
    # Data Setup
    # ======================================================================
    val_dataset = CrossDocked(root=data_dir, split="val")
    datamodule = MockDatamodule(
        train_length=1,  # Not used for validation
        val_length=len(val_dataset),
        test_length=1,
        batch_size=batch_size,
    )

    dist_path = os.path.join(data_dir, "processed/size_distribution.npy")
    if not os.path.exists(dist_path):
        raise FileNotFoundError(f"Size distribution not found at {dist_path}")
    distribution_hist = np.load(dist_path, allow_pickle=True)

    p_data = DataLoaderSampleable(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=CosineAlpha(),
        beta=CosineBeta(),
    )

    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")

    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
        norm_params = None
        if os.path.exists(norm_path):
            norm_data = np.load(norm_path, allow_pickle=True)
            norm_params = {"mean": norm_data["mean"], "std": norm_data["std"]}

        val_distribution = get_distributions(
            val_dataset,
            metrics=["qed", "logp", "sas", "lipinski", "gnina"],
            atom_decoder=constants.dataset_params[data_name]["atom_decoder"],
            norm_params=norm_params,
        )
        np.save(val_metric_dist_path, val_distribution)

    # ======================================================================
    # Model Loading
    # ======================================================================
    flow_model = FlowModel.load_from_checkpoint(
        checkpoint_path,
        path=path,
        norm_path=norm_path,
        val_distribution=val_distribution,
    )
    flow_model.eval()
    flow_model.num_workers = 1

    # ======================================================================
    # Validation Setup
    # ======================================================================
    logger = pl.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        run_name=run_name if run_name else f"val_{os.path.basename(checkpoint_path)}",
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=logger,
        enable_progress_bar=True,
    )

    # ======================================================================
    # Start Validation
    # ======================================================================
    trainer.validate(flow_model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate a Topological Flow model.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint.",
    )
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
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for validation."
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="TopoFlow_Validation",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="MLflow run name (optional).",
    )

    args = parser.parse_args()

    validate_flow_model(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        data_name=args.data_name,
        batch_size=args.batch_size,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        run_name=args.run_name,
    )
