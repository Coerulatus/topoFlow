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
from topoflow.flow.schedulers import LinearAlpha, LinearBeta, SineAlpha, CosineBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.vector_field_models import (
    TransformerVectorField,
    MLPVectorField,
    TransformerVectorFieldLigand,
    GATVectorField,
)
from topoflow.flow.flow_models_ligand import FlowModel
import topoflow.utils.constants as constants
from topoflow.flow.utils import InverseTimeFactor, InverseTimeFactorNoClamping
from topoflow.data.augmentations import RandomRotateTranslate
from topoflow.utils.metrics import get_distributions


def train_flow_model(
    data_dir,
    data_name,
    augment,
    vector_field_model,
    hidden_dim,
    evaluator,
    checkpoint_dir,
    epochs,
    batch_size,
    start_lr,
    max_lr,
    min_lr,
    max_frac,
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
        evaluator (str): Evaluator to use for model evaluation. Either 'rdkit' or a path to a GATEvaluator checkpoint.
        checkpoint_dir (str): Directory to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        start_lr (float): Initial learning rate for the optimizer.
        max_lr (float): Maximum learning rate for the scheduler.
        min_lr (float): Minimum learning rate for the scheduler.
        max_frac (float): Fraction of the maximum learning rate to use.
        seed (int): Random seed for reproducibility.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.
        run_name (str, optional): Name of the MLflow run. Defaults to None.
    """
    pl.seed_everything(seed)  # For reproducibility
    # ======================================================================
    # Data Preparation
    # ======================================================================
    if data_name == "crossdock_full":
        train_dataset = CrossDocked(
            root=data_dir,
            split="train",
            transform=RandomRotateTranslate() if augment else None,
        )
        val_dataset = CrossDocked(root=data_dir, split="val")

    else:
        raise ValueError(f"Unsupported dataset: {data_name}")
    datamodule = MockDatamodule(
        train_length=len(train_dataset),
        val_length=len(val_dataset),
        batch_size=batch_size,
        test_length=100,
    )
    # alpha = LinearAlpha()
    # beta = LinearBeta()
    alpha = SineAlpha()
    beta = CosineBeta()

    p_data = DataLoaderSampleable(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=len(psutil.Process().cpu_affinity()),
    )
    path = GaussianConditionalProbabilityPath(p_data=p_data, alpha=alpha, beta=beta)

    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
        val_distribution = get_distributions(
            val_dataset,
            metrics=["qed", "logp", "sas"],
            atom_decoder=constants.dataset_params[data_name]["atom_decoder"],
        )
        np.save(val_metric_dist_path, val_distribution)
    # ======================================================================
    # Model Initialization
    # ======================================================================
    if vector_field_model == "transformer":
        vector_field = TransformerVectorFieldLigand(
            h_dim=hidden_dim, num_heads=32, num_layers=3, dropout=0.1
        )
    elif vector_field_model == "mlp":
        vector_field = MLPVectorField(
            input_dim=14,  # Assuming 14 features as per the dataset
            output_dim=14,  # Assuming 3D vector field output
            hidden_dims=[hidden_dim, hidden_dim],
            activation=torch.nn.SiLU,
            additional_features="add",  # Use 'add' to incorporate time and conditioning features
        )
    elif vector_field_model == "gat":
        vector_field = GATVectorField(
            h_dim=hidden_dim,
            num_heads=4,
            num_layers=4,
            dropout=0.1,
            add_time_to="ligand",  # "ligand", "both", or "none"
        )
    else:
        raise ValueError(f"Unsupported vector field model: {vector_field_model}")

    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization parameters not found at {norm_path}. Please run the preprocessing script."
        )

    time_factor = InverseTimeFactorNoClamping(
        max_value=10.0,
        min_value=0.05,
    )

    flow_model = FlowModel(
        vector_field_model=vector_field,
        path=path,
        evaluator=evaluator,
        max_batched_samples=batch_size,
        val_distribution=val_distribution,
        sample_every=1,
        atom_decoder=constants.dataset_params[data_name]["atom_decoder"],
        norm_path=norm_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        start_lr=start_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        max_epochs=epochs,
        max_frac=max_frac,
        time_factor=time_factor,
    )
    # ======================================================================
    # Training Setup
    # ======================================================================
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir + "/run_" + (run_name if run_name else "default"),
        filename="topoflow-{epoch:02d}",
        monitor="train/loss",  # validation is just copying train samples
        mode="min",
        save_top_k=-1,
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
        check_val_every_n_epoch=5,
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        limit_train_batches=1.0,
        num_sanity_val_steps=0,
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
        default="crossdock_full",
        choices=["crossdock_full"],
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
        default="gat",
        choices=["transformer", "mlp", "gat"],
        help="Type of vector field model to use.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of the hidden layers in the model.",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="rdkit",
        help="Evaluator to use for model evaluation. Either 'rdkit' or a path to a GATEvaluator checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/topoflow/",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1000, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training."
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default=0.01,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.04,
        help="Maximum learning rate for the optimizer.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.002,
        help="Minimum learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max_frac_lr",
        type=float,
        default=0.1,
        help="Fraction of epochs to use for warmup of learning rate.",
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
        default="TopologicalFlowGAT",
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
        evaluator=args.evaluator,
        checkpoint_dir=args.checkpoint_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_lr=args.start_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        max_frac=args.max_frac_lr,
        seed=args.seed,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.run_name if args.run_name else None,
    )
