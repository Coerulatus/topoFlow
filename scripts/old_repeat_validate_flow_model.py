"""Run multiple validation loops on a trained model and aggregate statistics across runs.

This script automates the process of running multiple validation loops (e.g., for
different random seeds) to obtain stable performance metrics. It can either load
a model checkpoint and run new sampling/validation tasks via PyTorch Lightning
and MLflow, or process existing generation output folders "offline." It calculates
aggregated means and pooled standard deviations for all metrics.

Parser Arguments:
    --checkpoint_path (str): Path to a .ckpt file or a directory containing checkpoints.
    --data_dir (str): Path to the CrossDocked data.
    --n_runs (int): Number of validation iterations.
    --molecules_per_pocket (int): If > 0, performs custom sampling for every pocket in the split.
    --split (str): Split to use ("val" or "test").
    --mlflow_tracking_uri (str): MLflow tracking URI.
    --mlflow_experiment_name (str): MLflow experiment name for evaluation runs.

Files Produced:
    - result_{checkpoint_name}.txt: A markdown file containing aggregated evaluation results.
    - {split}_samples_{stem}/: Directories containing generated SDFs and PyG objects.
"""

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
from torch_geometric.data import Batch

# Assuming the model and data files are in the same directory or accessible in the python path
from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.utils.metrics import scorer_dict, get_distributions
from topoflow.utils.molecules import data_to_rdmol, data_to_list

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


def save_results_to_file(final_results, output_file, split="val"):
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
        f.write(f"# {split.capitalize()} Results\n\n")
        f.write(df.to_markdown(index=False))

    print(f"Saved aggregated results to {output_file}\n")


# --- Offline Evaluation Function ---


def process_offline_folder(val_samples_dir, val_distribution, output_file, split="val"):
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

    # Get reference size for bucketing
    ref_size = len(next(iter(val_distribution.values())))
    print(
        f"Reference distribution size: {ref_size}. Generated molecules will be split into buckets of this size."
    )

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

            values_flat = np.array(values).flatten()
            mean_val = np.mean(values_flat)
            std_val = np.std(values_flat, ddof=1) if len(values_flat) > 1 else 0.0

            # Bucketed WL Distance Calculation
            n_gen = len(values_flat)
            n_buckets = n_gen // ref_size

            wl_distances = []
            val_vals = np.array(val_distribution.get(metric, [])).flatten()

            if n_buckets > 0 and len(val_vals) > 0:
                # Shuffle generated values for random bucketing
                np.random.shuffle(values_flat)
                for b in range(n_buckets):
                    bucket_vals = values_flat[b * ref_size : (b + 1) * ref_size]
                    wl_dist = scipy.stats.wasserstein_distance(bucket_vals, val_vals)
                    wl_distances.append(wl_dist)

                wl_mean, wl_std = aggregate_simple_stats(wl_distances)
            else:
                # Fallback if too few samples
                if len(val_vals) > 0 and len(values_flat) > 0:
                    wl_mean = scipy.stats.wasserstein_distance(values_flat, val_vals)
                    wl_std = 0.0
                else:
                    wl_mean = np.nan
                    wl_std = np.nan

            # Log metrics to offline dict
            base_name = f"{split}/{metric}"

            if base_name + "_mean" not in all_run_metrics:
                all_run_metrics[base_name + "_mean"] = []
                all_run_metrics[base_name + "_std"] = []
                all_run_metrics[base_name + "_wl_distance"] = []

            all_run_metrics[base_name + "_mean"].append(mean_val)
            all_run_metrics[base_name + "_std"].append(std_val)

            # Store ALL bucket WL distances to get a proper standard deviation during final aggregation
            if n_buckets > 0 and len(val_vals) > 0:
                all_run_metrics[base_name + "_wl_distance"].extend(wl_distances)
            elif not np.isnan(wl_mean):
                all_run_metrics[base_name + "_wl_distance"].append(wl_mean)

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

    save_results_to_file(final_results, output_file, split=split)


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
    split="test",
    molecules_per_pocket=0,
):
    """Runs validations via MLflow for .ckpt files OR aggregates existing samples offline."""
    pl.seed_everything(42)

    # ======================================================================
    # Data Loading
    # ======================================================================
    print(f"\nSetting up {split} datamodule for model evaluations...")
    val_dataset = CrossDocked(root=data_dir, split=split)
    datamodule = MockDatamodule(
        train_length=1,
        val_length=len(val_dataset),
        test_length=1,
        batch_size=batch_size,
    )

    # ======================================================================
    # Load Reference Distribution
    # ======================================================================
    val_metric_dist_path = os.path.join(
        data_dir, "processed", f"{split}_distribution.npy"
    )
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
        print(
            f"Distribution file not found at {val_metric_dist_path}. Generating it from {split} dataset..."
        )
        # Load Normalization Params for correct GNINA scoring
        norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
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
            if os.path.isdir(os.path.join(base_dir, d))
            and d.startswith(f"{split}_samples_")
        ]

        # Fallback to val_samples_ if split is test but only val_samples_ exist
        if split == "test" and not val_dirs:
            val_dirs = [
                d
                for d in all_files
                if os.path.isdir(os.path.join(base_dir, d))
                and d.startswith("val_samples_")
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
            stem = d.replace(f"{split}_samples_", "").replace("val_samples_", "")
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
            process_offline_folder(folder_path, val_distribution, out_file, split=split)

    # ======================================================================
    # Execute Checkpoint Tasks
    # ======================================================================
    if ckpt_tasks:
        alpha = CosineAlpha()
        beta = CosineBeta()

        # IMPORTANT: shuffle=False ensures we cycle through all unique pockets in order
        p_data = DataLoaderSampleable(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=8
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

            if molecules_per_pocket > 0:
                print(
                    f"Starting custom sampling: {molecules_per_pocket} molecules per pocket..."
                )
                # Load names from .npz
                split_npz_path = os.path.join(data_dir, "processed", f"{split}.npz")
                names = np.load(split_npz_path, allow_pickle=True)["names"]

                # We'll use a standard DataLoader to iterate one by one
                from torch_geometric.data import DataLoader as PyGDataLoader

                loader = PyGDataLoader(val_dataset, batch_size=1, shuffle=False)

                run_dir = os.path.join(
                    base_dir, f"{split}_samples_{stem}", "generation"
                )
                os.makedirs(run_dir, exist_ok=True)

                for idx, pocket in enumerate(tqdm(loader, desc="Sampling Pockets")):
                    # Extract pocket name and sanitize
                    full_name = names[idx]
                    # Expected format: pocket_rel_path_ligand_rel_path
                    # We take the pocket part and replace slashes with underscores
                    pocket_part = full_name.split(".pdb")[0]
                    pocket_name = pocket_part.replace("/", "_")

                    # Replicate pocket for batch sampling
                    # Ensure we don't exceed max_batched_samples
                    n_left = molecules_per_pocket
                    mols_all = []
                    pyg_samples_all = []

                    while n_left > 0:
                        n_batch = min(n_left, model.max_batched_samples)
                        repeated_pocket = Batch.from_data_list([pocket] * n_batch).to(
                            model.device
                        )

                        # Sample
                        samples = model.sample(
                            n_samples=n_batch, pocket=repeated_pocket
                        )

                        # Convert to RDKit molecules
                        mols = data_to_rdmol(
                            samples, atom_decoder=model.atom_decoder, sanitize=True
                        )
                        mols_all.extend(mols)

                        # Convert to individual PyG objects (one molecule + pocket)
                        pyg_samples = data_to_list(samples)
                        pyg_samples_all.extend(pyg_samples)

                        n_left -= n_batch

                    # Save SDF
                    sdf_path = os.path.join(run_dir, f"{pocket_name}.sdf")
                    writer = Chem.SDWriter(sdf_path)
                    for m in mols_all:
                        if m is not None:
                            writer.write(m)
                    writer.close()

                    # Save PyG objects
                    pt_path = os.path.join(run_dir, f"{pocket_name}.pt")
                    torch.save(pyg_samples_all, pt_path)

                print(f"Finished sampling for {stem}. Results saved in {run_dir}")
                continue

            run_ids = []
            print(f"Starting {n_runs} {split} runs...")
            for i in range(n_runs):
                run_name = f"{run_name_prefix}_{i+1}"
                print(
                    f"--- Running {split.capitalize()} {i+1}/{n_runs} ({run_name}) ---"
                )

                run_dir = os.path.join(
                    base_dir, f"{split}_samples_{stem}", f"run_{i+1}"
                )
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
            save_results_to_file(final_results, out_file, split=split)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run multiple evaluations on a trained model."
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
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=3,
        help="Number of evaluation runs to perform.",
    )
    parser.add_argument(
        "--molecules_per_pocket",
        type=int,
        default=10,
        help="Number of molecules to generate per pocket. If > 0, skips standard validation and saves generated samples.",
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
        default="TopoFlow_Evaluation",
        help="MLflow experiment name for evaluation runs.",
    )
    parser.add_argument(
        "--run_name_prefix",
        type=str,
        default="test_run",
        help="Prefix for naming MLflow evaluation runs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["val", "test"],
        help="Dataset split to use for evaluation.",
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
        split=args.split,
        molecules_per_pocket=args.molecules_per_pocket,
    )
