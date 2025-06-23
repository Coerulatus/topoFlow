"""Train an evaluator for the QM9 dataset using PyTorch Geometric."""

import itertools

from topoflow.data.graphs import QM9WithRDKitMetrics
from topoflow.data.graphs import MolecularDataModule
from topoflow.evaluators import GATEvaluator

import torch

torch.set_float32_matmul_precision("medium")
import lightning.pytorch as pl
from pytorch_lightning.loggers import MLFlowLogger


def run_experiment(hparams, dataset):
    """Run a training experiment with the given hyperparameters."""
    seed = hparams["random_seed"]
    lr = hparams["lr"]
    hidden_channels = hparams["hidden_channels"]

    print(f"\n{'='*50}")
    print(f"--- Starting Training Run with SEED: {seed} ---")
    print(f"{'='*50}\n")

    pl.seed_everything(seed)

    datamodule = MolecularDataModule(
        dataset=dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        batch_size=512,
        num_workers=4,
    )

    evaluator = GATEvaluator(
        in_channels=dataset.num_features,
        edge_in_channels=dataset.num_edge_features,
        hidden_channels=64,
        out_channels=3,
        num_layers=3,
        heads=4,
        dropout=0.5,
        lr=1e-4,
        monitor_metric="mse",
        y_cols=["qed_mse", "logp_mse", "sascore_mse"],
        norm_params=dataset.norm_params,
    )

    callbacks = [
        pl.callbacks.ModelCheckpoint(
            monitor="val/mse",
            mode="min",
            save_top_k=1,
            filename=f"best_model_seed_{seed}",
            dirpath=f"./checkpoints/seed_{seed}",
        ),
        pl.callbacks.EarlyStopping(
            monitor="val/mse",
            patience=20,
            mode="min",
        ),
    ]

    mlflow_logger = MLFlowLogger(
        experiment_name=hparams["experiment_name"],
        tracking_uri="file:./mlruns",
        run_name=f"run_seed_{seed}",
    )

    trainer = pl.Trainer(
        max_epochs=200,
        accelerator="auto",
        devices=1,
        logger=mlflow_logger,
        callbacks=callbacks,
    )

    trainer.logger.log_hyperparams({"random_seed": seed})

    print(f"Fitting model for seed {seed}...")
    trainer.fit(
        evaluator,
        datamodule=datamodule,
    )

    print(f"Testing model for seed {seed}...")
    trainer.test(
        datamodule=datamodule,
        ckpt_path="best",
    )

    print(f"\n--- Training Run with SEED: {seed} COMPLETED ---")
    print(
        f"Results for this run are logged in MLflow UI under Experiment 'gat_evaluator_experiment', Run 'Run_Seed_{seed}'"
    )


if __name__ == "__main__":
    data_path = "./data/QM9_RDKit"
    dataset = QM9WithRDKitMetrics(root=data_path)

    random_seeds = [1, 3, 5, 7, 9]
    lrs = [2e-3, 1e-3, 5e-4]
    hidden_channels = [64, 128, 256]

    param_grid = {
        "random_seed": random_seeds,
        "lr": lrs,
        "hidden_channels": hidden_channels,
    }
    keys = param_grid.keys()
    combinations = itertools.product(*param_grid.values())

    for i, combo in enumerate(combinations):
        hparams = dict(zip(keys, combo))
        hparams["experiment_name"] = f"gat_evaluator_experiment2"
        print(f"\n\n{'='*50}")
        print(f"--- Starting Experiment {i+1} with Hyperparameters: {hparams} ---")
        print(f"{'='*50}\n")

        run_experiment(hparams, dataset)

    print(f"\n\n{'='*50}")
    print("ALL TRAINING RUNS COMPLETE!")
    print(
        "To view your results, navigate to the directory containing './mlruns' (or your specified tracking_uri)"
    )
    print(
        "and run: `mlflow ui` in your terminal. Then open http://localhost:5000 in your browser."
    )
    print(f"{'='*50}")
