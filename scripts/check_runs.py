"""Check and print the results of the GAT evaluator runs."""

from topoflow.utils.results import print_run

mlflow_uri = "file:./mlruns"
experiment_name = "gat_evaluator_experiment"

print_run(
    mlflow_uri=mlflow_uri,
    experiment_name=experiment_name,
    metric="val/mse",
    additional_metrics=["val/qed_mse", "val/logp_mse", "val/sascore_mse"],
    precision=4,
    ascending=True,
    grouping_keys=["lr"],
    filter_string="",
    numeric_params=["lr", "hidden_channels", "num_layers", "random_seed", "batch_size"],
    verbose=1,
    best=True,
    out_path=f"./results/{experiment_name}.csv",
)
