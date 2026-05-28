"""Profile a Topological Flow model during training.

This script is a modified version of train_flow_model.py designed to profile
the model's performance using the PyTorch Lightning profiler. It uses the
validation split for training to save memory and exports the profiling results
to a format compatible with Chrome's trace viewer.

Logs are saved in the 'profiler_logs' directory.
"""

import argparse
import os
import psutil
import numpy as np
import torch
import lightning.pytorch as pl
from lightning.pytorch.profilers import PyTorchProfiler
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.vector_field_models import (
    MLPVectorField,
    TransformerVectorFieldOld,
    GATVectorField,
    AllSetTransformerVectorField,
    TransformerDistanceVectorField,
)
from topoflow.flow.flow_models import FlowModel
import topoflow.utils.constants as constants
from topoflow.flow.utils import InverseTimeFactorNoClamping
from topoflow.data.augmentations import RandomRotateTranslate
from topoflow.utils.metrics import get_distributions


class WidePyTorchProfiler(PyTorchProfiler):
    """Custom PyTorch Profiler that allows for wider columns in the summary table."""

    def summary(self) -> str:
        """Override the summary method to adjust column widths for better readability."""
        # 1. Let the parent class handle all the version-specific internal
        # variables, kwargs, and parsing. We ignore the truncated string it returns.
        _ = super().summary()

        # 2. Safely retrieve the parsed events that the parent class just cached.
        # Lightning exposes this publicly as 'function_events', but we include
        # a fallback to '_function_events' just in case.
        events = getattr(self, "function_events", None)
        if events is None:
            events = getattr(self, "_function_events", None)

        # 3. Re-render the table with the custom column width
        if events is not None:
            return events.table(
                sort_by=self._sort_by_key,
                row_limit=self._row_limit,
                max_name_column_width=150,
            )

        return "Warning: Could not retrieve parsed profiler events."


def profile_flow_model(
    device,
    data_dir,
    data_name,
    augment,
    vector_field_model,
    hidden_dim,
    evaluator,
    epochs,
    batch_size,
    start_lr,
    max_lr,
    min_lr,
    max_frac,
    min_time_factor,
    max_time_factor,
    max_atom_time,
    w_loss_last_step,
    velocity_scale,
    seed,
    use_geometric_ast=False,
    limit_train_batches=10,
):
    """Profile a Topological Flow model.

    Args:
        device (str): Device to use for profiling (e.g., 'cuda' or 'cpu').
        data_dir (str): Directory containing the training data.
        data_name (str): Name of the dataset to use.
        augment (bool): Whether to apply data augmentations.
        vector_field_model (str): Type of vector field model to use.
        hidden_dim (int): Dimension of the hidden layers in the model.
        evaluator (str): Evaluator to use for model evaluation.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        start_lr (float): Initial learning rate.
        max_lr (float): Maximum learning rate.
        min_lr (float): Minimum learning rate.
        max_frac (float): Fraction of epochs for LR warmup.
        min_time_factor (float): Minimum time factor.
        max_time_factor (float): Maximum time factor.
        max_atom_time (float): Maximum atom time.
        w_loss_last_step (float): Weight for loss at the last step.
        velocity_scale (float): Scaling factor for velocity.
        seed (int): Random seed.
        use_geometric_ast (bool): Use geometric AST.
        limit_train_batches (int): Limit number of batches per epoch for profiling.
    """
    pl.seed_everything(seed)

    # ======================================================================
    # Data Preparation (Using 'val' split for training as requested)
    # ======================================================================
    if data_name == "crossdock_full":
        # Using 'val' split for training to save memory
        train_dataset = CrossDocked(
            root=data_dir,
            split="val",
            transform=RandomRotateTranslate() if augment else None,
        )
        val_dataset = CrossDocked(root=data_dir, split="val")
    else:
        raise ValueError(f"Unsupported dataset: {data_name}")

    datamodule = MockDatamodule(
        train_length=len(train_dataset),
        val_length=len(val_dataset),
        batch_size=batch_size,
        test_length=100,
    )

    alpha = CosineAlpha()
    beta = CosineBeta()
    if max_atom_time is not None and max_atom_time != 1.0:
        alpha_atom = CosineAlpha(t_max=max_atom_time)
        beta_atom = CosineBeta(t_max=max_atom_time)
    else:
        alpha_atom = None
        beta_atom = None

    p_data = DataLoaderSampleable(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(8, len(psutil.Process().cpu_affinity())),
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=3,
    )
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=alpha,
        beta=beta,
        alpha_atom=alpha_atom,
        beta_atom=beta_atom,
    )

    # Load distribution for metrics (if exists, or compute if needed)
    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
        val_distribution = None  # Skip if not found for profiling

    # ======================================================================
    # Model Initialization
    # ======================================================================
    if vector_field_model == "transformer":
        vector_field = TransformerVectorFieldOld(
            h_dim=hidden_dim, num_heads=32, num_layers=3, dropout=0.1
        )
    elif vector_field_model == "mlp":
        vector_field = MLPVectorField(
            input_dim=14,
            output_dim=14,
            hidden_dims=[hidden_dim, hidden_dim],
            activation=torch.nn.SiLU,
            additional_features="add",
        )
    elif vector_field_model == "gat":
        vector_field = GATVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",
        )
    elif vector_field_model == "ast":
        vector_field = AllSetTransformerVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",
            use_gat=False,
            use_geometric_ast=use_geometric_ast,
        )
    elif vector_field_model == "ast_gat":
        vector_field = AllSetTransformerVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",
            use_gat=True,
            use_geometric_ast=use_geometric_ast,
        )
    elif vector_field_model == "distance_transformer":
        vector_field = TransformerDistanceVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",
        )
    else:
        raise ValueError(f"Unsupported vector field model: {vector_field_model}")

    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    time_factor = InverseTimeFactorNoClamping(
        max_value=max_time_factor,
        min_value=min_time_factor,
    )

    flow_model = FlowModel(
        vector_field_model=vector_field,
        path=path,
        evaluator=evaluator,
        max_batched_samples=batch_size,
        val_distribution=val_distribution,
        sample_every=1,
        atom_decoder=constants.dataset_params[data_name]["atom_decoder"],
        norm_path=norm_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        start_lr=start_lr,
        max_lr=max_lr,
        min_lr=min_lr,
        max_epochs=epochs,
        max_frac=max_frac,
        time_factor=time_factor,
        w_loss_last_step=w_loss_last_step,
        velocity_scale=velocity_scale,
    )

    # ======================================================================
    # Profiler Setup
    # ======================================================================
    profiler_dir = "profiler_logs"
    os.makedirs(profiler_dir, exist_ok=True)
    if device == "cuda":
        filename = "gpu_profile"
        sort_key = "cuda_time_total"
    else:
        filename = "cpu_profile"
        sort_key = "cpu_time_total"

    profiler = WidePyTorchProfiler(
        dirpath="profiler_logs",
        filename=filename,
        export_to_chrome=False,
        row_limit=100,
        sort_by_key=sort_key,
    )

    # ======================================================================
    # Trainer Setup
    # ======================================================================
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        profiler=profiler,
        enable_progress_bar=True,
        limit_train_batches=limit_train_batches,
        limit_val_batches=0.0,  # Disable validation for profiling
        num_sanity_val_steps=0,
        logger=False,  # Disable logging to MLflow for profiling
        enable_checkpointing=False,  # Disable checkpointing to avoid pickling errors
        precision="16-mixed",  # Use mixed precision on GPU for better profiling results
    )

    # ======================================================================
    # Start Profiling
    # ======================================================================
    print(f"Starting profiling. Logs will be saved in: {profiler_dir}")
    trainer.fit(flow_model, datamodule=datamodule)
    if (
        hasattr(flow_model.path.p_data, "_iterator")
        and flow_model.path.p_data._iterator is not None
    ):
        flow_model.path.p_data._iterator._shutdown_workers()
        del flow_model.path.p_data._iterator
    # # 1. Force the Lightning profiler to build the summary if it hasn't already
    # trainer.profiler.summary()

    # # 2. Access the cached PyTorch EventList
    # events = trainer.profiler.function_events

    # if events is not None:
    #     # 3. Generate the table using the cached events
    #     detailed_table = events.table(
    #         sort_by="cuda_time_total",
    #         row_limit=50,
    #         max_name_column_width=150,
    #     )

    #     # 4. Write to disk
    #     with open(os.path.join(profiler_dir, "readable_gpu_profile.txt"), "w") as f:
    #         f.write(detailed_table)
    #     print("Detailed profiler table saved successfully.")
    # else:
    #     print("Warning: Profiler events were not cached.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile a Topological Flow model.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for profiling (e.g., 'cuda' or 'cpu').",
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
        "--augment",
        type=bool,
        default=True,
        help="Whether to apply data augmentations.",
    )
    parser.add_argument(
        "--vector_field_model",
        type=str,
        default="ast_gat",
        choices=["ast_gat", "transformer", "mlp", "gat", "ast", "distance_transformer"],
        help="Type of vector field model to use.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="Dimension of the hidden layers in the model.",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs for profiling."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for profiling."
    )
    parser.add_argument(
        "--limit_train_batches",
        type=int,
        default=5,
        help="Number of batches to run for profiling.",
    )
    parser.add_argument(
        "--start_lr", type=float, default=0.0002, help="Initial learning rate."
    )
    parser.add_argument(
        "--max_lr", type=float, default=0.0008, help="Maximum learning rate."
    )
    parser.add_argument(
        "--min_lr", type=float, default=0.00004, help="Minimum learning rate."
    )
    parser.add_argument(
        "--max_frac_lr",
        type=float,
        default=0.1,
        help="Fraction of epochs for LR warmup.",
    )
    parser.add_argument(
        "--min_time_factor", type=float, default=0.05, help="Minimum time factor."
    )
    parser.add_argument(
        "--max_time_factor", type=float, default=10.0, help="Maximum time factor."
    )
    parser.add_argument(
        "--max_atom_time", type=float, default=1.0, help="Maximum atom time."
    )
    parser.add_argument(
        "--w_loss_last_step", type=float, default=0.2, help="Weight for last step loss."
    )
    parser.add_argument(
        "--velocity_scale", type=float, default=1.0, help="Scaling factor for velocity."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--use_geometric_ast",
        action="store_true",
        help="Whether to use geometric AllSet Transformer.",
    )

    args = parser.parse_args()

    profile_flow_model(
        device=args.device,
        data_dir=args.data_dir,
        data_name=args.data_name,
        augment=args.augment,
        vector_field_model=args.vector_field_model,
        hidden_dim=args.hidden_dim,
        evaluator="rdkit",
        epochs=args.epochs,
        batch_size=args.batch_size,
        start_lr=args.start_lr,
        max_lr=args.max_lr,
        min_lr=args.min_lr,
        max_frac=args.max_frac_lr,
        min_time_factor=args.min_time_factor,
        max_time_factor=args.max_time_factor,
        max_atom_time=args.max_atom_time,
        w_loss_last_step=args.w_loss_last_step,
        velocity_scale=args.velocity_scale,
        seed=args.seed,
        use_geometric_ast=args.use_geometric_ast,
        limit_train_batches=args.limit_train_batches,
    )
