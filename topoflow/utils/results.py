"""Utility script to analyze and summarize MLflow runs for GAT model hyperparameter tuning."""

import mlflow
import pandas as pd
import numpy as np  # For statistical calculations
import warnings

# --- Configuration (should match your training script) ---
MLFLOW_TRACKING_URI = "file:./mlruns"
EXPERIMENT_NAME = "gat_evaluator_experiment"
METRIC_TO_ANALYZE = "test/mse"  # Or "val/mse", "test/mae", etc.

# Set the tracking URI for the MLflow client
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Get the experiment by name
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

# --- Define Grouping Keys ---
grouping_keys = ["lr"]


if experiment is None:
    print(
        f"Error: Experiment '{EXPERIMENT_NAME}' not found at '{MLFLOW_TRACKING_URI}'."
    )
    print("Please ensure your training script ran and created this experiment.")
else:
    experiment_id = experiment.experiment_id
    print(f"Found Experiment: '{EXPERIMENT_NAME}' with ID: {experiment_id}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        # filter_string="params.dropout = '0.5'", # Optional: filter if you have too many runs
        order_by=["start_time ASC"],  # Order by start time to see progression if any
        output_format="pandas",
    )

    if runs_df.empty:
        print(f"No runs found in experiment '{EXPERIMENT_NAME}'.")
    else:
        print(f"\nFound {len(runs_df)} runs in total.")

        # --- Data Cleaning & Preparation ---
        # Rename columns for easier access (remove 'params.' and 'metrics.' prefixes)
        param_cols = [col for col in runs_df.columns if col.startswith("params.")]
        metric_cols = [col for col in runs_df.columns if col.startswith("metrics.")]

        # Create new column names without prefixes
        new_param_cols = {col: col.replace("params.", "") for col in param_cols}
        new_metric_cols = {col: col.replace("metrics.", "") for col in metric_cols}

        runs_df = runs_df.rename(columns={**new_param_cols, **new_metric_cols})

        # Convert relevant parameter columns to numeric if they are strings (MLflow logs them as strings)
        numeric_params = [
            "lr",
            "hidden_channels",
            "num_layers",
            "random_seed",
            "batch_size",
        ]
        for param in numeric_params:
            if param in runs_df.columns:
                runs_df[param] = pd.to_numeric(runs_df[param], errors="coerce")
                runs_df.dropna(subset=[param], inplace=True)
            else:
                warnings.warn(
                    f"Parameter '{param}' not found in runs_df. Skipping conversion."
                )

        # Ensure all grouping keys exist in the DataFrame
        missing_keys = [key for key in grouping_keys if key not in runs_df.columns]
        if missing_keys:
            raise ValueError(
                f"Missing grouping keys in DataFrame: {missing_keys}. Check MLflow logging."
            )

        # --- Grouping and Averaging ---
        print(
            f"\n--- Aggregating '{METRIC_TO_ANALYZE}' for each hyperparameter combination ---"
        )

        # Group by the combination of hyperparameters
        grouped_results = runs_df.groupby(grouping_keys)[METRIC_TO_ANALYZE].agg(
            ["mean", "std"]
        )

        print(grouped_results)

        # Optional: Reset index to make the grouping keys regular columns
        grouped_results = grouped_results.reset_index()
        print("\nAggregated Results (DataFrame format):")
        print(grouped_results)

        # Optional: Sort by the mean metric to find best combinations
        print(
            f"\nTop combinations by mean {METRIC_TO_ANALYZE} (sorted by best performance):"
        )
        # Assuming lower MSE is better
        print(grouped_results.sort_values(by="mean", ascending=True))

        # --- Further Analysis (Example: Finding the best combination) ---
        best_combination_row = grouped_results.loc[grouped_results["mean"].idxmin()]
        print(f"\nBest performing combination (lowest mean {METRIC_TO_ANALYZE}):")
        print(best_combination_row)

        # You can now save `grouped_results` to CSV or plot it
        # grouped_results.to_csv("grid_search_summary.csv", index=False)
        # import matplotlib.pyplot as plt
        # grouped_results.plot(x='lr', y='mean', kind='bar', rot=45) # Example plot
        # plt.title(f'Mean {METRIC_TO_ANALYZE} by Learning Rate')
        # plt.show()
