"""Train an evaluator for the QM9 dataset using PyTorch Geometric."""

# Ensure topoflow is importable, assuming it's in your Python path
from topoflow.data.graphs import QM9WithRDKitMetrics
from topoflow.data.graphs import MolecularDataModule
from topoflow.evaluators import GATEvaluator

import torch

torch.set_float32_matmul_precision("medium")
import lightning.pytorch as pl
from pytorch_lightning.loggers import MLFlowLogger


# --- Global Dataset Loading (Loaded once) ---
data_path = "./data/QM9_RDKit"
dataset = QM9WithRDKitMetrics(root=data_path)

# --- Define Random Seeds for Experimentation ---
# This list will determine how many independent training runs are performed.
random_seeds = [1, 3, 5, 7, 9]  # Example seeds. Add or remove as needed.

# --- Loop Through Each Random Seed ---
for seed in random_seeds:
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
        lr=2e-4,
        monitor_metric="mse",
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
            patience=10,
            mode="min",
        ),
    ]

    mlflow_logger = MLFlowLogger(
        experiment_name="gat_evaluator_experiment",
        tracking_uri="file:./mlruns",
        run_name=f"Run_Seed_{seed}",
    )

    trainer = pl.Trainer(
        max_epochs=100,
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

print(f"\n\n{'='*50}")
print("ALL TRAINING RUNS COMPLETE!")
print(
    "To view your results, navigate to the directory containing './mlruns' (or your specified tracking_uri)"
)
print(
    "and run: `mlflow ui` in your terminal. Then open http://localhost:5000 in your browser."
)
print(f"{'='*50}")
