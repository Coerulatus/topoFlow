"""Script to run multiple validation loops on a trained model and aggregate results."""

import argparse
import os
import numpy as np
import torch
import lightning.pytorch as pl
import mlflow
import pandas as pd
import math
import scipy.stats
from rdkit import Chem
from tqdm import tqdm
from mlflow.tracking import MlflowClient

# Assuming the model and data files are in the same directory or accessible in the python path
from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.utils.metrics import scorer_dict

# --- Aggregation Functions ---


def aggregate_pooled_stats(means, stds):
    """Calculates the combined mean and pooled standard deviation from multiple runs."""
    if not means:
        return np.nan, np.nan

    combined_mean = np.mean(means)
    pooled_variance = np.mean(np.array(stds) ** 2)
    pooled_std = math.sqrt(pooled_variance)

    return combined_mean, pooled_std


def aggregate_simple_stats(values):
    """Calculates the mean and standard deviation for a list of scalar values."""
    if not values:
        return np.nan, np.nan
    mean = np.mean(values)
    std = np.std(values, ddof=1) if len(values) > 1 else 0.0
    return mean, std


def save_results_to_file(final_results, output_file):
    """Helper to format and save the results dictionary to a markdown text file."""
    sorted_keys = sorted(final_results.keys())
    data = []

    for metric_name in sorted_keys:
        if metric_name == "epoch":
            continue
        if metric_name.endswith("_std") and metric_name[:-4] + "_mean" in final_results:
            continue

        value_str = final_results[metric_name]

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

    print(f"Saved aggregated results to {output_file}\n")


# --- Offline Evaluation Function ---


def process_offline_folder(val_samples_dir, val_distribution, output_file):
    """Processes existing val_samples directories locally when no .ckpt is present."""
    print(f"--- Offline Processing: {val_samples_dir} ---")
    run_dirs = [
        os.path.join(val_samples_dir, d)
        for d in os.listdir(val_samples_dir)
        if d.startswith("run_")
    ]

    if not run_dirs:
        print(f"  No run directories found. Skipping.")
        return

    all_run_metrics = {}

    for run_dir in run_dirs:
        sdfs = [f for f in os.listdir(run_dir) if f.endswith(".sdf")]
        run_scores = {
            "qed": [],
            "logp": [],
            "sas": [],
            "lipinski": [],
            "vina": [],
            "cnn_affinity": [],
        }

        for sdf in tqdm(sdfs, desc=f"  Evaluating {os.path.basename(run_dir)}"):
            # 1. RDKit Metrics
            suppl = Chem.SDMolSupplier(os.path.join(run_dir, sdf))
            for mol in suppl:
                if mol is None:
                    continue
                for m in ["qed", "logp", "sas", "lipinski"]:
                    try:
                        score = scorer_dict[m](mol)
                        val = score.item() if hasattr(score, "item") else float(score)
                        run_scores[m].append(val)
                    except Exception:
                        pass

            # 2. GNINA Metrics (Requires .pt file)
            pt_path = os.path.join(run_dir, sdf.replace(".sdf", ".pt"))
            if os.path.exists(pt_path):
                try:
                    data_list = torch.load(
                        pt_path, map_location="cpu", weights_only=False
                    )
                    for data_obj in data_list:
                        try:
                            score = scorer_dict["gnina"](data_obj)
                            vals = (
                                score.tolist()
                                if isinstance(score, torch.Tensor)
                                else score
                            )
                            run_scores["vina"].append(float(vals[0]))
                            run_scores["cnn_affinity"].append(float(vals[1]))
                        except Exception:
                            pass
                except Exception:
                    pass

        # Aggregate logic per run
        for metric, values in run_scores.items():
            if not values:
                continue

            values_flat = np.array(values).flatten().tolist()
            mean_val = np.mean(values_flat)
            std_val = np.std(values_flat, ddof=1) if len(values_flat) > 1 else 0.0

            # Calculate WL Distance
            val_dist_key = metric
            val_vals = val_distribution.get(val_dist_key, [])
            if hasattr(val_vals, "tolist"):
                val_vals = val_vals.tolist()

            if len(val_vals) > 0:
                wl_dist = scipy.stats.wasserstein_distance(
                    values_flat, np.array(val_vals).flatten().tolist()
                )
            else:
                wl_dist = np.nan

            # Log metrics to offline dict matching the MLflow format
            base_name = f"val/{metric}"

            if base_name + "_mean" not in all_run_metrics:
                all_run_metrics[base_name + "_mean"] = []
                all_run_metrics[base_name + "_std"] = []
                all_run_metrics[base_name + "_wl_distance"] = []

            all_run_metrics[base_name + "_mean"].append(mean_val)
            all_run_metrics[base_name + "_std"].append(std_val)
            all_run_metrics[base_name + "_wl_distance"].append(wl_dist)

    # Combine across multiple offline runs
    final_results = {}
    for metric_key in all_run_metrics.keys():
        if metric_key.endswith("_mean"):
            base_name = metric_key[:-5]
            means = all_run_metrics[base_name + "_mean"]
            stds = all_run_metrics[base_name + "_std"]
            comb_mean, pool_std = aggregate_pooled_stats(means, stds)
            final_results[base_name + "_mean"] = f"{comb_mean:.4f} ± {pool_std:.4f}"
            final_results[base_name + "_std"] = f" (Pooled Std: {pool_std:.4f})"
        elif metric_key.endswith("_wl_distance"):
            wls = all_run_metrics[metric_key]
            comb_mean, comb_std = aggregate_simple_stats(wls)
            final_results[metric_key] = f"{comb_mean:.4f} ± {comb_std:.4f}"

    save_results_to_file(final_results, output_file)


# --- Main Validation Logic ---


def run_multiple_validations(
    checkpoint_path,
    data_dir,
    data_name,
    batch_size,
    n_runs,
    mlflow_tracking_uri,
    experiment_name,
    run_name_prefix,
):
    """Runs validations via MLflow for .ckpt files OR aggregates existing val_samples offline."""
    pl.seed_everything(42)

    # ======================================================================
    # Load Validation Distribution (Used by both MLflow and Offline flows)
    # ======================================================================
    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
        raise FileNotFoundError(
            f"Validation distribution file not found at {val_metric_dist_path}. "
            "Please ensure this file was generated during training."
        )

    # ======================================================================
    # Task Discovery (.ckpt vs offline folders)
    # ======================================================================
    ckpt_tasks = []
    offline_tasks = []

    base_dir = (
        checkpoint_path
        if os.path.isdir(checkpoint_path)
        else os.path.dirname(checkpoint_path)
    )

    if os.path.isdir(checkpoint_path):
        all_files = os.listdir(base_dir)
        ckpt_files = [f for f in all_files if f.endswith(".ckpt")]
        val_dirs = [
            d
            for d in all_files
            if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("val_samples_")
        ]

        for ckpt in ckpt_files:
            stem = ckpt[:-5]
            result_name = f"result_{stem}.txt"
            if not os.path.exists(os.path.join(base_dir, result_name)):
                ckpt_tasks.append((stem, os.path.join(base_dir, ckpt)))
            else:
                print(f"Skipping {ckpt} as {result_name} already exists.")

        # For folders, check if they don't have a ckpt task and don't have a result file
        ckpt_stems = [stem for stem, _ in ckpt_tasks]
        for d in val_dirs:
            stem = d.replace("val_samples_", "")
            result_name = f"result_{stem}.txt"
            # Proceed offline only if there's no result file AND no active .ckpt task
            if stem not in ckpt_stems and not os.path.exists(
                os.path.join(base_dir, result_name)
            ):
                offline_tasks.append((stem, os.path.join(base_dir, d)))
    else:
        # A single file was passed
        stem = os.path.splitext(os.path.basename(checkpoint_path))[0]
        result_name = f"result_{stem}.txt"
        if not os.path.exists(os.path.join(base_dir, result_name)):
            ckpt_tasks.append((stem, checkpoint_path))
        else:
            print(f"Skipping {checkpoint_path} as {result_name} already exists.")

    if not ckpt_tasks and not offline_tasks:
        print("\nNo pending evaluations found (all results exist). Exiting.")
        return

    # ======================================================================
    # Execute Offline Tasks
    # ======================================================================
    if offline_tasks:
        print(
            f"\nDiscovered {len(offline_tasks)} model(s) with sample folders but no checkpoint. Processing offline..."
        )
        for stem, folder_path in offline_tasks:
            out_file = os.path.join(base_dir, f"result_{stem}.txt")
            process_offline_folder(folder_path, val_distribution, out_file)

    # ======================================================================
    # Execute Checkpoint Tasks
    # ======================================================================
    if ckpt_tasks:
        print("\nSetting up validation datamodule for model evaluations...")
        val_dataset = CrossDocked(root=data_dir, split="val")
        datamodule = MockDatamodule(
            train_length=1,
            val_length=len(val_dataset),
            test_length=1,
            batch_size=batch_size,
        )
        alpha = CosineAlpha()
        beta = CosineBeta()

        p_data = DataLoaderSampleable(
            val_dataset, batch_size=batch_size, shuffle=True, num_workers=8
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

        for stem, ckpt_path in ckpt_tasks:
            print(f"\nLoading model from checkpoint: {ckpt_path}")
            model = FlowModel.load_from_checkpoint(
                ckpt_path,
                path=path,
                norm_path="./data/CrossDocked/processed/normalization_params.npz",
            )
            if "ast_gat" in os.path.basename(ckpt_path):
                model.vector_field_model.use_gat = True

            model.eval()
            model.max_batched_samples = batch_size
            model.val_distribution = val_distribution
            model.atom_decoder = constants.dataset_params[data_name]["atom_decoder"]
            model.metrics = ["qed", "logp", "sas", "lipinski", "gnina"]

            run_ids = []
            print(f"Starting {n_runs} validation runs...")
            for i in range(n_runs):
                run_name = f"{run_name_prefix}_{i+1}"
                print(f"--- Running Validation {i+1}/{n_runs} ({run_name}) ---")

                run_dir = os.path.join(base_dir, f"val_samples_{stem}", f"run_{i+1}")
                os.makedirs(run_dir, exist_ok=True)
                model.current_run_dir = run_dir

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
                logger.finalize("FINISHED")

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

            # --- Aggregation Logic ---
            pooled_metrics = {}
            simple_metrics = {}

            for name, values in all_metrics.items():
                if name.endswith("_mean"):
                    base_name = name[:-5]
                    std_name = base_name + "_std"
                    if std_name in all_metrics:
                        if base_name not in pooled_metrics:
                            pooled_metrics[base_name] = {"mean": [], "std": []}
                        pooled_metrics[base_name]["mean"].extend(values)
                elif name.endswith("_std"):
                    continue
                elif not name.endswith("_mean") and not name.endswith("_std"):
                    simple_metrics[name] = values

            for base_name in pooled_metrics.keys():
                std_name = base_name + "_std"
                if std_name in all_metrics:
                    pooled_metrics[base_name]["std"].extend(all_metrics[std_name])

            final_results = {}

            # A. Aggregate POOLED METRICS
            for base_name, data in pooled_metrics.items():
                combined_mean, pooled_std = aggregate_pooled_stats(
                    data["mean"], data["std"]
                )
                final_results[base_name + "_mean"] = (
                    f"{combined_mean:.4f} ± {pooled_std:.4f}"
                )
                final_results[base_name + "_std"] = f" (Pooled Std: {pooled_std:.4f})"

            # B. Aggregate SIMPLE METRICS
            for name, values in simple_metrics.items():
                mean, std = aggregate_simple_stats(values)
                final_results[name] = f"{mean:.4f} ± {std:.4f}"

            out_file = os.path.join(base_dir, f"result_{stem}.txt")
            save_results_to_file(final_results, out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple validations on a trained model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/runs_cluster/to_evaluate/",
        help="Path to the trained model checkpoint (.ckpt file) or directory.",
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
    )
