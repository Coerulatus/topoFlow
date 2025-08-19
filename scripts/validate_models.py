"""Script to validate a Topological Flow model using a folder of checkpoints.

This script loads each checkpoint from a specified directory, runs a single
validation epoch, and logs the results to a single, continuous MLflow run.
This allows for easy visualization of how model performance evolves over training.
"""

import argparse
import os
import psutil
import torch
import numpy as np
import mlflow
import lightning.pytorch as pl
from topoflow.data.dataloader import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
from topoflow.flow.schedulers import LinearAlpha, LinearBeta, SineAlpha, CosineBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.vector_field_models import TransformerVectorField, MLPVectorField
from topoflow.flow.flow_models import FlowModel
import topoflow.utils.constants as constants
from topoflow.flow.utils import InverseTimeFactorNoClamping
from topoflow.data.augmentations import RandomRotateTranslate
from topoflow.utils.metrics import get_distributions


def validate_checkpoints(
    checkpoint_dir,
    data_dir,
    data_name,
    evaluator,
    mlflow_tracking_uri,
    experiment_name,
    run_name=None,
    batch_size=4,
):
    """Validates a series of checkpoints and logs results to a single MLflow run.

    Args:
        checkpoint_dir (str): Directory containing the model checkpoints.
        data_dir (str): Directory containing the validation data.
        data_name (str): Name of the dataset.
        evaluator (str): Evaluator to use for model evaluation.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.
        run_name (str, optional): Name of the MLflow run. Defaults to None.
        batch_size (int, optional): Batch size for validation. Defaults to 4.
    """
    # ----------------------------------------------------------------------
    # MLflow Setup
    # ----------------------------------------------------------------------
    # Find or create the MLflow experiment
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    # Start a single MLflow run to log all checkpoint results
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"Starting MLflow run with ID: {run_id}")

        # ----------------------------------------------------------------------
        # Data and Model Component Preparation
        # ----------------------------------------------------------------------
        # This part re-initializes the data and model components
        # that are necessary for the validation loop.
        if data_name == "crossdock_full":
            val_dataset = CrossDocked(root=data_dir, split="val")
        else:
            raise ValueError(f"Unsupported dataset: {data_name}")

        datamodule = MockDatamodule(
            train_length=100,  # These values are arbitrary for validation
            val_length=len(val_dataset),
            batch_size=batch_size,
            test_length=100,
        )

        # Initialize core components used in the FlowModel
        alpha = SineAlpha()
        beta = CosineBeta()
        p_data = DataLoaderSampleable(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=len(psutil.Process().cpu_affinity()),
        )
        path = GaussianConditionalProbabilityPath(p_data=p_data, alpha=alpha, beta=beta)

        # Load validation distribution for evaluation metrics
        val_metric_dist_path = os.path.join(
            data_dir, "processed", "val_distribution.npy"
        )
        if os.path.exists(val_metric_dist_path):
            val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
        else:
            raise FileNotFoundError(
                f"Validation distribution not found at {val_metric_dist_path}. Cannot perform validation."
            )

        norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
        if not os.path.exists(norm_path):
            raise FileNotFoundError(
                f"Normalization parameters not found at {norm_path}. Please run the preprocessing script."
            )

        # ----------------------------------------------------------------------
        # Validation Loop
        # ----------------------------------------------------------------------
        # Get all checkpoint files and sort them by epoch number
        checkpoint_files = [
            f for f in os.listdir(checkpoint_dir) if f.endswith(".ckpt")
        ]
        # Sort by the epoch number in the filename (e.g., topoflow-epoch=01.ckpt)
        checkpoint_files.sort(key=lambda x: int(x.split("-epoch=")[-1].split(".")[0]))

        if not checkpoint_files:
            print(f"No checkpoints found in the specified directory: {checkpoint_dir}")
            return

        for checkpoint_file in checkpoint_files:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_file)
            print(f"Validating checkpoint: {checkpoint_file}")

            # Load the model from the checkpoint. We need to pass the arguments
            # that were used to initialize the model initially.
            try:
                model = FlowModel.load_from_checkpoint(
                    checkpoint_path,
                    vector_field_model=TransformerVectorField(
                        h_dim=128,
                        num_heads=32,
                        num_layers=3,
                        dropout=0.1,
                        combine="add",
                    ),
                    path=path,
                    evaluator=evaluator,
                    val_distribution=val_distribution,
                    atom_decoder=constants.dataset_params[data_name]["atom_decoder"],
                    norm_path=norm_path,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
            except Exception as e:
                print(f"Failed to load checkpoint {checkpoint_file}: {e}")
                continue

            # Create a PyTorch Lightning Trainer for validation
            # We must use the MLflowLogger with the same run_id
            logger = pl.loggers.MLFlowLogger(
                experiment_name=experiment_name,
                tracking_uri=mlflow_tracking_uri,
                run_id=run_id,  # This is the crucial part
            )
            trainer = pl.Trainer(
                accelerator="auto",
                devices=1 if torch.cuda.is_available() else None,
                logger=logger,
                limit_val_batches=2,  # Limit to 2 batches for a quick check
            )

            # Run a single validation epoch
            trainer.validate(model, datamodule=datamodule)

    print("Validation for all checkpoints complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate checkpoints and log to a single MLflow run."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/run1/",
        help="Directory containing the model checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/CrossDocked/",
        help="Directory containing the validation data.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="crossdock_full",
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="rdkit",
        help="Evaluator to use for model evaluation.",
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
        default="ValidationRun",
        help="Name of the MLflow run (optional).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for validation."
    )

    args = parser.parse_args()

    validate_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        data_dir=args.data_dir,
        data_name=args.data_name,
        evaluator=args.evaluator,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.run_name,
        batch_size=args.batch_size,
    )
