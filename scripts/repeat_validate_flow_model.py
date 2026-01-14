"""Script to run multiple validation loops on a trained model and aggregate results."""

import argparse
import os
import numpy as np
import torch
import lightning.pytorch as pl
import mlflow
import pandas as pd
import math
from mlflow.tracking import MlflowClient

# Assuming the model and data files are in the same directory or accessible in the python path
from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable


# --- New Aggregation Functions ---


def aggregate_pooled_stats(means, stds):
    """Calculates the combined mean and pooled standard deviation from multiple runs.

    Assumes sample sizes (n_i) are large and roughly equal across runs.

    Args:
        means (list): List of mean values from each run.
        stds (list): List of standard deviation values from each run.

    Returns:
        tuple: (combined_mean, pooled_std)
    """
    if not means:
        return np.nan, np.nan

    # 1. Combined Mean (Simple average since n_i is assumed constant/equal)
    combined_mean = np.mean(means)

    # 2. Pooled Variance (Correctly combines within-run variability)
    # Formula for pooled variance (assuming equal weights/sample sizes):
    # Var_pooled = (1/N) * sum(Var_i) + (1/(N-1)) * sum((Mean_i - Mean_pooled)^2)

    # a. Average within-run variance (assuming variance is already squared std)
    avg_within_run_variance = np.mean(np.array(stds) ** 2)

    # b. Variance of the means (measures variability *between* runs)
    variance_of_means = np.var(means, ddof=1)  # ddof=1 for sample variance

    # c. Combine (Assuming N is the number of original batches per validation run.
    # Since that N is unknown, we simplify by only using the average std of the trainer).
    # NOTE: The simplest, statistically sound approach is combining the MSEs (variances).

    # We will use the simplest approximation by averaging the variances:
    pooled_variance = np.mean(np.array(stds) ** 2)
    pooled_std = math.sqrt(pooled_variance)

    return combined_mean, pooled_std


def aggregate_simple_stats(values):
    """Calculates the mean and standard deviation for a list of scalar values.

    Used for metrics that output a single scalar per run (e.g., distances).
    """
    if not values:
        return np.nan, np.nan
    mean = np.mean(values)
    # ddof=1 provides the sample standard deviation (unbiased)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


# --- End New Aggregation Functions ---


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

    (Docstring content remains the same)
    """
    pl.seed_everything(42)  # for reproducibility of data loading

    # ======================================================================
    # Data Preparation (remains the same)
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
    alpha = CosineAlpha()
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
    # Load Model (remains the same)
    # ======================================================================
    print(f"Loading model from checkpoint: {checkpoint_path}")
    model = FlowModel.load_from_checkpoint(
        checkpoint_path,
        path=path,
        norm_path="./data/CrossDocked/processed/normalization_params.npz",
    )
    model.eval()
    model.max_batched_samples = batch_size

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
    # Multiple Validation Runs (remains the same)
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

    # --- Aggregation Logic Fix ---

    # 1. Separate metrics into mean/std pairs (pooled) and single values (simple)
    pooled_metrics = (
        {}
    )  # Stores {'metric_name': {'mean': [v1, v2, ...], 'std': [s1, s2, ...]}}
    simple_metrics = {}  # Stores {'metric_name': [v1, v2, ...]}

    # Collect all data, grouping mean/std pairs
    for name, values in all_metrics.items():
        if name.endswith("_mean"):
            base_name = name[:-5]  # e.g., 'qed'
            std_name = base_name + "_std"

            # Ensure the standard deviation is available
            if std_name in all_metrics:
                if base_name not in pooled_metrics:
                    pooled_metrics[base_name] = {"mean": [], "std": []}
                pooled_metrics[base_name]["mean"].extend(values)
        elif name.endswith("_std"):
            # Standard deviations are handled when processing means
            continue
        elif not name.endswith("_mean") and not name.endswith("_std"):
            # Simple scalar metric (e.g., qed_wl_distance, validity)
            simple_metrics[name] = values

    # Add standard deviations to pooled_metrics structure
    for base_name in pooled_metrics.keys():
        std_name = base_name + "_std"
        if std_name in all_metrics:
            pooled_metrics[base_name]["std"].extend(all_metrics[std_name])
        else:
            # Handle case where mean exists but std doesn't (shouldn't happen with PTL)
            print(f"Warning: Mean {base_name}_mean found, but no {std_name}.")

    # 2. Calculate final results
    final_results = {}

    # A. Aggregate POOLED METRICS (e.g., qed_mean, logp_mean)
    for base_name, data in pooled_metrics.items():
        combined_mean, pooled_std = aggregate_pooled_stats(data["mean"], data["std"])
        final_results[base_name + "_mean"] = f"{combined_mean:.4f} ± {pooled_std:.4f}"
        final_results[base_name + "_std"] = (
            f" (Pooled Std: {pooled_std:.4f})"  # Log the original std name for reference
        )

    # B. Aggregate SIMPLE METRICS (e.g., qed_wl_distance, validity)
    for name, values in simple_metrics.items():
        mean, std = aggregate_simple_stats(values)
        final_results[name] = f"{mean:.4f} ± {std:.4f}"

    # --- End Aggregation Logic Fix ---

    # ======================================================================
    # Save Results
    # ======================================================================
    root_folder = os.path.dirname(output_file)
    if root_folder and not os.path.exists(root_folder):
        os.makedirs(root_folder)
    print(f"Saving aggregated results to {output_file}")

    # Sort results for consistent output
    sorted_keys = sorted(final_results.keys())

    # Create a DataFrame for better formatting (optional but good practice)
    data = []
    for metric_name in sorted_keys:
        if metric_name == "epoch":
            continue
        # Skip the redundant *_std entry in the final table
        if metric_name.endswith("_std") and metric_name[:-4] + "_mean" in final_results:
            continue

        # Extract mean/std string, or use the value itself
        value_str = final_results[metric_name]

        # Check if this metric has a pooled std that should be displayed
        if metric_name.endswith("_mean") and metric_name[:-5] + "_std" in final_results:
            pooled_std_ref = (
                final_results[metric_name[:-5] + "_std"]
                .replace(" (Pooled Std: ", "")
                .replace(")", "")
            )
            value_str = value_str.split("±")[0].strip() + f" ± {pooled_std_ref}"

        data.append([metric_name, value_str])

    df = pd.DataFrame(data, columns=["Metric", "Value (mean ± std)"])

    # Save the markdown table
    with open(output_file, "w") as f:
        f.write("# Validation Results\n\n")
        f.write(df.to_markdown(index=False))

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple validations on a trained model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/runs_cluster/11_05_659e_ast_gat_scaled_vel.ckpt",
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
        default=3,
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
        default="./results/results_11_05_659e_ast_gat_scaled_vel.md",
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
