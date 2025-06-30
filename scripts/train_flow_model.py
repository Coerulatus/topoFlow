"""Script to train a Topological Flow model on a given dataset."""

import argparse
import os
import torch
import lightning.pytorch as pl
from topoflow.data.dataloader import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
from topoflow.flow.schedulers import LinearAlpha, LinearBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.vector_field_models import TransformerVectorField
from topoflow.flow.flow_models import FlowModel
import topoflow.utils.constants as constants
from topoflow.data.augmentations import RandomRotateTranslate


def train_flow_model(
    data_dir,
    data_name,
    augment,
    vector_field_model,
    hidden_dim,
    checkpoint_dir,
    epochs,
    batch_size,
    learning_rate,
    seed,
    mlflow_tracking_uri,
    experiment_name,
    run_name=None,
):
    """Train a Topological Flow model on the specified dataset.

    Args:
        data_dir (str): Directory containing the training data.
        data_name (str): Name of the dataset to use.
        augment (bool): Whether to apply data augmentations.
        vector_field_model (str): Type of vector field model to use.
        hidden_dim (int): Dimension of the hidden layers in the model.
        checkpoint_dir (str): Directory to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        learning_rate (float): Learning rate for the optimizer.
        seed (int): Random seed for reproducibility.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.
        run_name (str, optional): Name of the MLflow run. Defaults to None.
    """
    pl.seed_everything(seed)  # For reproducibility
    # ======================================================================
    # Data Preparation
    # ======================================================================
    if data_name == "cross_docked":
        train_dataset = CrossDocked(
            root=data_dir,
            split="train",
            transform=RandomRotateTranslate() if augment else None,
        )

    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    datamodule = MockDatamodule(
        train_length=len(train_dataset), val_length=batch_size, batch_size=batch_size
    )

    alpha = LinearAlpha()
    beta = LinearBeta()
    p_data = DataLoaderSampleable(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )
    path = GaussianConditionalProbabilityPath(p_data=p_data, alpha=alpha, beta=beta)
    # ======================================================================
    # Model Initialization
    # ======================================================================
    if vector_field_model == "transformer":
        vector_field = TransformerVectorField(
            x_dim=p_data.dim, h_dim=hidden_dim, num_heads=4, num_layers=2, dropout=0.1
        )
    else:
        raise ValueError(f"Unsupported vector field model: {vector_field_model}")

    flow_model = FlowModel(
        vector_field_model=vector_field,
        path=path,
        n_val_samples=0,
        device="cuda" if torch.cuda.is_available() else "cpu",
        lr=learning_rate,
    )
    # ======================================================================
    # Training Setup
    # ======================================================================
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="topoflow-{epoch:02d}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
    )
    logger = pl.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        run_name=run_name,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        check_val_every_n_epoch=1,
    )
    # ======================================================================
    # Start Training
    # ======================================================================
    trainer.fit(flow_model, datamodule=datamodule)


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
        default="cross_docked",
        choices=["cross_docked"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--augment",
        type=bool,
        default=True,
        help="Whether to apply data augmentations.",
    )
    parser.add_argument(
        "--vector_field_model",
        type=str,
        default="transformer",
        choices=["transformer"],
        help="Type of vector field model to use.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Dimension of the hidden layers in the model.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/topoflow/",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default="file:./mlruns",
        help="MLflow tracking URI.",
    )
    parser.add_argument(
        "--mlflow_experiment_name",
        type=str,
        default="TopologicalFlow",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the MLflow run (optional).",
    )

    args = parser.parse_args()

    train_flow_model(
        data_dir=args.data_dir,
        data_name=args.data_name,
        augment=args.augment,
        vector_field_model=args.vector_field_model,
        hidden_dim=args.hidden_dim,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.run_name if args.run_name else None,
    )
