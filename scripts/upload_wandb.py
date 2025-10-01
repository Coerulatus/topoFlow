"""Upload MLflow runs into Weights & Biases using wandb.apis.importers.mlflow."""

#!/usr/bin/env python3
import argparse
import os
import sys


def main():
    """Upload MLflow runs into Weights & Biases using wandb.apis.importers.mlflow."""
    parser = argparse.ArgumentParser(
        description="Import MLflow runs into Weights & Biases using wandb.apis.importers.mlflow"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        default="mlruns",
        help="MLflow tracking URI or local path (use file:/path/to/mlruns for local).",
    )
    parser.add_argument(
        "--entity",
        required=True,
        help="W&B entity (team/org or username) to import into.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="W&B API key (or set WANDB_API_KEY environment variable).",
    )
    parser.add_argument(
        "--project", required=True, help="W&B project name to import into."
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Only import runs from this MLflow experiment name.",
    )
    parser.add_argument(
        "--mlflow-filter",
        default=None,
        help='Optional MLflow search_runs filter string, e.g., "metrics.acc > 0.9".',
    )
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip importing artifacts to speed up import or reduce storage.",
    )
    parser.add_argument(
        "--run-limit",
        type=int,
        default=None,
        help="Maximum number of runs to import (useful for testing).",
    )
    parser.add_argument(
        "--rename-illegal",
        action="store_true",
        help="Sanitize run names (replace characters illegal in some contexts).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Collect and list runs but do not import.",
    )
    args = parser.parse_args()

    # Defer heavy imports until after parsing
    import wandb
    from wandb.apis.importers import Namespace
    from wandb.apis.importers.mlflow import MlflowImporter, MlflowRun
    import mlflow

    tracking_uri = args.mlflow_tracking_uri
    # Normalize local path to MLflow file URI if a plain path is provided
    if os.path.isdir(tracking_uri) and not tracking_uri.startswith(
        ("file:", "http://", "https://", "databricks")
    ):
        tracking_uri = f"file:{os.path.abspath(tracking_uri)}"

    dst_base_url = os.environ.get(
        "WANDB_BASE_URL", "https://api.wandb.ai"
    )  # or your on-prem URL
    if args.api_key:
        dst_api_key = args.api_key.strip()
    else:
        dst_api_key = os.environ.get("WANDB_API_KEY", "").strip()

    wandb.login(key=dst_api_key, relogin=True, host=dst_base_url)

    importer = MlflowImporter(
        dst_base_url=dst_base_url,
        dst_api_key=dst_api_key,
        mlflow_tracking_uri=tracking_uri,
    )

    client = mlflow.tracking.MlflowClient(tracking_uri)
    runs_iter = []

    # Resolve experiment IDs to search
    if args.experiment_name is not None:
        exp = client.get_experiment_by_name(args.experiment_name)
        if exp is None:
            print(
                f"No MLflow experiment named '{args.experiment_name}' found at {tracking_uri}",
                file=sys.stderr,
            )
            sys.exit(1)
        exp_ids = [exp.experiment_id]
    else:
        exps = client.search_experiments()
        exp_ids = [e.experiment_id for e in exps]

    # Build MLflow search and wrap into MlflowRun objects
    for exp_id in exp_ids:
        # MLflow paginates; search_runs handles paging internally
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=args.mlflow_filter or "",
            max_results=args.run_limit or 50000,
            order_by=[],
        )
        for r in runs:
            # Optionally sanitize names that may cause import issues in some contexts
            if args.rename_illegal and r.data.tags.get("mlflow.runName"):
                name = r.data.tags["mlflow.runName"]
                safe = name.replace("/", "_").replace(":", "_")
                if safe != name:
                    client.set_tag(r.info.run_id, "mlflow.runName", safe)
            runs_iter.append(MlflowRun(r, client))
            if args.run_limit and len(runs_iter) >= args.run_limit:
                break
        if args.run_limit and len(runs_iter) >= args.run_limit:
            break

    print(f"Collected {len(runs_iter)} MLflow runs from {tracking_uri}")

    if args.dry_run:
        # Print a small preview
        for i, r in enumerate(runs_iter[:10]):
            print(
                f"- run_id={r.run.info.run_id} status={r.run.info.status} start={r.run.info.start_time} end={r.run.info.end_time}"
            )
        if len(runs_iter) > 10:
            print(f"... ({len(runs_iter)-10} more)")
        return

    ns = Namespace(args.entity, args.project)
    # artifacts=False skips artifact uploads; otherwise importer attempts to copy logs, models, etc.
    importer.import_runs(runs_iter, namespace=ns, artifacts=not args.no_artifacts)
    print("Import complete.")


if __name__ == "__main__":
    main()
