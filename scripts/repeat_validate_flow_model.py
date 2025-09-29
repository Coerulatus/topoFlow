"""Script to run multiple validations on a trained Topological Flow model."""

import argparse
import os
import numpy as np
import torch
import lightning.pytorch as pl
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

# Assuming the model and data files are in the same directory or accessible in the python path
from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.flow.schedulers import SineAlpha, CosineBeta


def run_multiple_validations(
    checkpoint_path,
    data_dir,
    data_name,
    batch_size,
    n_runs,
    mlflow_tracking_uri,
    experiment_name,
    run_name_prefix,
    output_file,
):
    """Runs multiple validation loops on a trained model and aggregates the results.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.ckpt).
        data_dir (str): Directory containing the training data.
        data_name (str): Name of the dataset to use.
        batch_size (int): Batch size for validation.
        n_runs (int): Number of validation runs to perform.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name for validation runs.
        run_name_prefix (str): Prefix for naming MLflow runs.
        output_file (str): File path to save the aggregated results.
    """
    pl.seed_everything(42)  # for reproducibility of data loading

    # ======================================================================
    # Data Preparation
    # ======================================================================
    print("Setting up validation datamodule...")
    val_dataset = CrossDocked(root=data_dir, split="val")
    # MockDatamodule is used because we only need the validation dataloader
    datamodule = MockDatamodule(
        train_length=1,  # Not used
        val_length=len(val_dataset),
        test_length=1,  # Not used
        batch_size=batch_size,
    )
    alpha = SineAlpha()
    beta = CosineBeta()

    p_data = DataLoaderSampleable(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
    )

    distribution_hist = np.load(
        "./data/CrossDocked/processed/size_distribution.npy", allow_pickle=True
    )
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=alpha,
        beta=beta,
        target_dim=14,
        distribution_hist=distribution_hist,
    )

    # ======================================================================
    # Load Model
    # ======================================================================
    print(f"Loading model from checkpoint: {checkpoint_path}")
    # The hparams saved in the checkpoint will be used to initialize the model
    model = FlowModel.load_from_checkpoint(
        checkpoint_path,
        path=path,
        norm_path="./data/CrossDocked/processed/normalization_params.npz",
    )
    model.eval()
    model.max_batched_samples = batch_size

    # The val_distribution and atom_decoder are needed for the validation step.
    # They should be saved as hyperparameters, but we ensure they are set correctly.
    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
        model.val_distribution = val_distribution
    else:
        raise FileNotFoundError(
            f"Validation distribution file not found at {val_metric_dist_path}. "
            "Please ensure this file was generated during training."
        )
    model.atom_decoder = constants.dataset_params[data_name]["atom_decoder"]

    # ======================================================================
    # Multiple Validation Runs
    # ======================================================================
    run_ids = []
    print(f"Starting {n_runs} validation runs...")
    for i in range(n_runs):
        run_name = f"{run_name_prefix}_{i+1}"
        print(f"--- Running Validation {i+1}/{n_runs} ({run_name}) ---")

        logger = pl.loggers.MLFlowLogger(
            experiment_name=experiment_name,
            tracking_uri=mlflow_tracking_uri,
            run_name=run_name,
        )

        trainer = pl.Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            logger=logger,
            enable_progress_bar=True,
            enable_model_summary=False,
        )

        trainer.validate(model, datamodule=datamodule)
        run_ids.append(logger.run_id)
        # Finalize the logger to push all logs
        logger.finalize("FINISHED")

    print("All validation runs completed.")

    # ======================================================================
    # Fetch Results from MLflow and Aggregate
    # ======================================================================
    print("Fetching results from MLflow and aggregating...")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()

    all_metrics = {}
    for run_id in run_ids:
        run = client.get_run(run_id)
        for metric_name, value in run.data.metrics.items():
            if metric_name not in all_metrics:
                all_metrics[metric_name] = []
            all_metrics[metric_name].append(value)

    # Calculate mean and std
    results = {}
    for metric_name, values in all_metrics.items():
        if len(values) > 0:
            mean = np.mean(values)
            std = np.std(values) if len(values) > 1 else 0.0
            results[metric_name] = f"{mean:.4f} ± {std:.4f}"

    # ======================================================================
    # Save Results
    # ======================================================================
    root_folder = os.path.dirname(output_file)
    if root_folder and not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print(f"Saving aggregated results to {output_file}")
    with open(output_file, "w") as f:
        f.write("# Validation Results\n\n")
        f.write("| Metric | Value (mean ± std) |\n")
        f.write("|--------|----------------------|\n")
        # Sort results for consistent output
        for metric_name in sorted(results.keys()):
            f.write(f"| {metric_name} | {results[metric_name]} |\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple validations on a trained model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/runs_cluster/1000e_09_09_ligand.ckpt",
        help="Path to the trained model checkpoint (.ckpt file).",
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
        choices=["crossdock_full"],
        help="Name of the dataset to use.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for validation.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of validation runs to perform.",
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
        default="TopoFlow_Validation",
        help="MLflow experiment name for validation runs.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="validation_run",
        help="Prefix for naming MLflow validation runs.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./results/results3.md",
        help="File to save the aggregated results.",
    )

    args = parser.parse_args()

    run_multiple_validations(
        checkpoint_path=args.checkpoint_path,
        data_dir=args.data_dir,
        data_name=args.data_name,
        batch_size=args.batch_size,
        n_runs=args.n_runs,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name_prefix=args.run_name_prefix,
        output_file=args.output_file,
    )
