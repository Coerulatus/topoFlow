"""Utility functions for summarizing and printing MLflow run results."""

import mlflow
import pandas as pd
import numpy as np
import os
import warnings

# Helper function to get the MLflow client
client = mlflow.tracking.MlflowClient()


def my_round(value, precision, is_std=False):
    """Rounds a value to a specified precision.

    If is_std is True, it applies a more appropriate rounding for standard deviation.
    """
    if pd.isna(value):
        return value
    if is_std:
        # Round std to one or two significant figures if it's not too small
        if value == 0:
            return 0
        order_of_magnitude = np.floor(np.log10(abs(value)))
        # Adjust precision for std based on its magnitude
        if (
            order_of_magnitude < -1
        ):  # For very small stds, keep precision, e.g. 0.000123 -> 0.0001
            return round(value, max(precision, -int(order_of_magnitude) + 1))
        else:  # For larger stds, aim for 2 significant figures
            return round(value, -int(order_of_magnitude) + 1)
    return round(value, precision)


def get_metric_value_at_step(history_list, target_step):
    """Helper to get a metric value at a specific step."""
    for m in history_list:
        if m.step == target_step:
            return m.value
    return np.nan  # Return NaN if value for target_step is not found


def print_run(
    mlflow_uri,
    experiment_name,
    metric="val/mse",
    precision=None,
    ascending=True,
    grouping_keys=None,
    filter_string="",
    numeric_params=[],
    additional_metrics=[],
    verbose=2,
    best=False,
    out_path=None,
):
    """Function to print and summarize MLflow runs for a given experiment.

    Finds the epoch with the best primary metric for each run and saves the values
    of the primary metric and additional metrics corresponding to that epoch in the dataframe.

    Args:
        mlflow_uri (str): The URI of the MLflow tracking server.
        experiment_name (str): The name of the MLflow experiment to analyze.
        metric (str): The primary metric to summarize (default: 'val/mse').
        precision (int, optional): Number of decimal places to round the metric values (default: None).
        ascending (bool): Whether to sort the results in ascending order (True for lower is better, False for higher is better) (default: True).
        grouping_keys (list or 'all' or None): Keys to group the results by. If 'all', groups by all columns. If None, does not group (default: None).
        filter_string (str): Filter string to apply to the runs (default: "").
        numeric_params (list): List of parameter names to convert to numeric (default: []).
        additional_metrics (list): List of other metric names to log at the best epoch (default: []).
        verbose (int): Verbosity level (0: minimal output, 1: summary, 2: detailed) (default: 2).
        best (bool): Whether to print the best combinations based on the primary metric (default: False).
        out_path (str): Path to save the results as a CSV file (default: None).

    Returns:
        pd.DataFrame: DataFrame containing the summarized results.
    """
    mlflow.set_tracking_uri(mlflow_uri)

    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is None:
        print(f"Error: Experiment '{experiment_name}' not found at '{mlflow_uri}'.")
        print("Please ensure your training script ran and created this experiment.")
        return None
    experiment_id = experiment.experiment_id
    if verbose > 1:
        print(f"Found Experiment: '{experiment_name}' with ID: {experiment_id}")

    runs_df = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=filter_string,
        order_by=["start_time ASC"],
        output_format="pandas",
    )

    if runs_df.empty:
        print(f"No runs found in experiment '{experiment_name}'.")
        return None

    if verbose > 1:
        print(f"Found {len(runs_df)} runs in experiment '{experiment_name}'.")

    # Rename columns for easier access (remove 'params.' and 'metrics.' prefixes)
    param_cols = [col for col in runs_df.columns if col.startswith("params.")]
    metric_cols = [col for col in runs_df.columns if col.startswith("metrics.")]

    new_param_cols = {col: col.replace("params.", "") for col in param_cols}
    new_metric_cols = {col: col.replace("metrics.", "") for col in metric_cols}

    runs_df = runs_df.rename(columns={**new_param_cols, **new_metric_cols})

    # Convert relevant parameter columns to numeric
    default_numeric_params = [
        "lr",
        "hidden_channels",
        "num_layers",
        "random_seed",
        "batch_size",
        "epochs",  # Often useful to have as numeric
    ]
    all_numeric_params = list(
        set(default_numeric_params + numeric_params)
    )  # Use set to avoid duplicates

    for param in all_numeric_params:
        if param in runs_df.columns:
            if not pd.api.types.is_numeric_dtype(runs_df[param]):
                runs_df[param] = pd.to_numeric(runs_df[param], errors="coerce")
                runs_df.dropna(subset=[param], inplace=True)
        else:
            if verbose > 1:
                warnings.warn(
                    f"Parameter '{param}' not found in runs_df. Skipping conversion."
                )

    if type(grouping_keys) is list:
        missing_keys = [key for key in grouping_keys if key not in runs_df.columns]
        if missing_keys:
            raise ValueError(
                f"Missing grouping keys in DataFrame: {missing_keys}. Check MLflow logging."
            )

    # --- Find the best metric value and corresponding epoch for each run ---
    primary_metric_best_values = []
    primary_metric_best_epochs = []
    # Dictionary to hold lists for additional metrics' values at the best epoch
    additional_metrics_best_values = {m: [] for m in additional_metrics}

    if verbose > 1:
        print(
            f"Retrieving full '{metric}' history and additional metrics for each run to find best epoch..."
        )

    for run_id in runs_df["run_id"]:
        try:
            # Get history for the primary metric
            primary_history = client.get_metric_history(run_id, metric)

            if primary_history:
                primary_history_df = pd.DataFrame(
                    [{"step": m.step, "value": m.value} for m in primary_history]
                )

                if (
                    primary_history_df.empty
                ):  # Handle case where history is returned but empty
                    raise ValueError(
                        f"Primary metric '{metric}' history is empty for run {run_id}."
                    )

                if ascending:  # Lower is better (e.g., MSE, loss)
                    best_row_primary = primary_history_df.loc[
                        primary_history_df["value"].idxmin()
                    ]
                else:  # Higher is better (e.g., accuracy, R2)
                    best_row_primary = primary_history_df.loc[
                        primary_history_df["value"].idxmax()
                    ]

                best_epoch_step = best_row_primary["step"]

                primary_metric_best_values.append(best_row_primary["value"])
                primary_metric_best_epochs.append(best_epoch_step)

                # Fetch and store values for additional metrics at this best epoch
                for add_m in additional_metrics:
                    add_m_history = client.get_metric_history(run_id, add_m)
                    value_at_best_epoch = get_metric_value_at_step(
                        add_m_history, best_epoch_step
                    )
                    additional_metrics_best_values[add_m].append(value_at_best_epoch)
            else:
                raise ValueError(
                    f"Primary metric '{metric}' history not found for run {run_id}."
                )

        except Exception as e:
            primary_metric_best_values.append(np.nan)
            primary_metric_best_epochs.append(np.nan)
            for add_m in additional_metrics:
                additional_metrics_best_values[add_m].append(np.nan)
            if verbose > 1:
                warnings.warn(
                    f"Could not retrieve primary metric history for run {run_id} ({e}). Filling with NaN for all metrics."
                )

    runs_df[f"best_{metric}_value"] = primary_metric_best_values
    runs_df[f"best_{metric}_epoch"] = primary_metric_best_epochs

    for add_m in additional_metrics:
        runs_df[f"best_{add_m}_value"] = additional_metrics_best_values[add_m]

    # Drop runs where the primary best_metric_value couldn't be retrieved
    runs_df.dropna(subset=[f"best_{metric}_value"], inplace=True)

    if runs_df.empty:
        print(f"No runs remaining after finding best metric value for '{metric}'.")
        return None

    if verbose > 1:
        print(
            f"\n--- Aggregating primary and additional metrics for each hyperparameter combination ---"
        )

    # Define all columns to aggregate
    all_metrics_for_aggregation_cols = [f"best_{metric}_value"] + [
        f"best_{m}_value" for m in additional_metrics
    ]

    if grouping_keys == "all":
        aggregated_data = {}
        for m_col in all_metrics_for_aggregation_cols:
            original_metric_name = m_col.replace("best_", "").replace("_value", "")
            mean_val = runs_df[m_col].mean()
            std_val = runs_df[m_col].std()
            aggregated_data[f"{original_metric_name}_mean"] = [mean_val]
            aggregated_data[f"{original_metric_name}_std"] = [std_val]
        grouped_results = pd.DataFrame(aggregated_data)
        grouped_results.index = ["all"]
    elif type(grouping_keys) is list:
        agg_dict = {col: ["mean", "std"] for col in all_metrics_for_aggregation_cols}
        grouped_results = runs_df.groupby(grouping_keys).agg(agg_dict)

        # Flatten the MultiIndex columns
        new_cols = []
        for col_level1, col_level2 in grouped_results.columns:
            original_metric_name = col_level1.replace("best_", "").replace("_value", "")
            new_cols.append(f"{original_metric_name}_{col_level2}")
        grouped_results.columns = new_cols
    elif grouping_keys is None:
        # For no grouping, we return individual run details
        display_cols = (
            ["run_id", f"best_{metric}_value", f"best_{metric}_epoch"]
            + [f"best_{m}_value" for m in additional_metrics]
            + [col for col in all_numeric_params if col in runs_df.columns]
        )

        # Filter display_cols to only include those actually present in runs_df
        display_cols = [col for col in display_cols if col in runs_df.columns]
        grouped_results = runs_df[
            display_cols
        ].copy()  # Use .copy() to avoid SettingWithCopyWarning

        # Rename 'best_{metric}_value' and 'best_{add_m}_value' columns for cleaner output
        rename_map = {f"best_{metric}_value": metric}
        for add_m in additional_metrics:
            rename_map[f"best_{add_m}_value"] = add_m
        grouped_results = grouped_results.rename(columns=rename_map)
    else:
        raise ValueError(
            "grouping_keys must be 'all', a list of column names, or None."
        )

    # Reset index to make the grouping keys regular columns (if not already flat)
    if not isinstance(grouped_results.index, pd.RangeIndex):
        grouped_results = grouped_results.reset_index()

    if precision is not None:
        # Apply precision to the relevant columns
        cols_to_round = []
        if grouping_keys is None:
            if metric in grouped_results.columns:
                cols_to_round.append(metric)
            for add_m in additional_metrics:
                if add_m in grouped_results.columns:
                    cols_to_round.append(add_m)
        else:  # Grouped results will have _mean and _std columns
            for m_name in [metric] + additional_metrics:
                if f"{m_name}_mean" in grouped_results.columns:
                    cols_to_round.append(f"{m_name}_mean")
                if f"{m_name}_std" in grouped_results.columns:
                    cols_to_round.append(f"{m_name}_std")

        for col in cols_to_round:
            if "_std" in col:  # Apply std specific rounding
                grouped_results[col] = grouped_results[col].apply(
                    lambda x: my_round(x, precision, is_std=True)
                )
            else:  # Apply normal rounding for means or raw values
                grouped_results[col] = grouped_results[col].apply(
                    lambda x: my_round(x, precision)
                )

    if verbose > 0:
        print("\nResults:")
        print(grouped_results)

    if best:
        print(f"\nTop combinations by mean {metric} (sorted by best performance):")
        sort_col = metric if grouping_keys is None else f"{metric}_mean"
        if sort_col in grouped_results.columns:
            print(grouped_results.sort_values(by=sort_col, ascending=ascending))
        else:
            warnings.warn(f"Cannot sort by '{sort_col}'. Column not found in results.")

    if out_path:
        if verbose > 0:
            print(f"\nSaving results to {out_path}")
        out_dir = os.path.dirname(out_path)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        comment_line = f"# Results for experiment '{experiment_name}'. Primary metric '{metric}' (best epoch value). Additional metrics: {additional_metrics}. Grouped by {grouping_keys if grouping_keys is not None else 'individual runs'}.\n"

        with open(out_path, "w") as f:
            f.write(comment_line)

        # Columns that need float formatting for CSV export
        cols_for_csv_float_format = []
        if grouping_keys is None:
            if metric in grouped_results.columns:
                cols_for_csv_float_format.append(metric)
            for add_m in additional_metrics:
                if add_m in grouped_results.columns:
                    cols_for_csv_float_format.append(add_m)
        else:
            for m_name in [metric] + additional_metrics:
                if f"{m_name}_mean" in grouped_results.columns:
                    cols_for_csv_float_format.append(f"{m_name}_mean")
                if f"{m_name}_std" in grouped_results.columns:
                    cols_for_csv_float_format.append(f"{m_name}_std")

        if precision is None:
            grouped_results.to_csv(out_path, mode="a", index=False, header=True)
        else:
            # Create a copy to format strings without altering the original DataFrame for potential further use
            df_to_save_formatted = grouped_results.copy()
            for col in cols_for_csv_float_format:
                if col in df_to_save_formatted.columns:
                    # Apply f-string formatting, handling NaN
                    df_to_save_formatted[col] = df_to_save_formatted[col].apply(
                        lambda x: f"%.{precision}f" % x if pd.notna(x) else ""
                    )
            df_to_save_formatted.to_csv(out_path, mode="a", index=False, header=True)

    return grouped_results
