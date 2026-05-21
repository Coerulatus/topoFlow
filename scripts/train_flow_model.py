"""Train a Topological Flow model on a given molecular dataset (e.g., CrossDocked).

This is the main training script for TopoFlow. It sets up the PyTorch Lightning
trainer, handles data loading and augmentation, and initializes the continuous
normalizing flow (CNF) with a specified vector field architecture (Transformer,
GAT, MLP, AllSetTransformer, etc.). The model learns to map a Gaussian prior
distribution to the target molecular distribution conditioned on protein pocket
features. It uses MLflow for experiment tracking and logging.

Parser Arguments:
    --data_dir (str): Directory containing the training data.
    --vector_field_model (str): Choice of architecture (e.g., "ast_gat", "distance_transformer").
    --hidden_dim (int): Hidden layer size.
    --epochs (int): Total training iterations.
    --batch_size (int): Number of samples per batch.
    --start_lr, --max_lr, --min_lr (float): Learning rate schedule parameters.
    --velocity_scale (float): Factor to scale the predicted velocity field.
    --checkpoint_dir (str): Directory to save model checkpoints.

Files Produced:
    - checkpoints/topoflow/run_{name}/topoflow-epoch=xx.ckpt: Model weights and training state.
    - MLflow logs in the specified tracking URI.
"""

import argparse
import os
import psutil
import numpy as np
import torch
import lightning.pytorch as pl
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
from topoflow.flow.schedulers import LinearAlpha, LinearBeta, CosineAlpha, CosineBeta
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.vector_field_models import (
    MLPVectorField,
    TransformerVectorFieldLigand,
    GATVectorField,
    AllSetTransformerVectorField,
    TransformerDistanceVectorField,
)
from topoflow.flow.flow_models import FlowModel
import topoflow.utils.constants as constants
from topoflow.flow.utils import InverseTimeFactor, InverseTimeFactorNoClamping
from topoflow.data.augmentations import RandomRotateTranslate
from topoflow.utils.metrics import get_distributions


def train_flow_model(
    data_dir,
    data_name,
    augment,
    vector_field_model,
    hidden_dim,
    evaluator,
    checkpoint_dir,
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
    mlflow_tracking_uri,
    experiment_name,
    run_name=None,
    previous_checkpoint=None,
    skip_validation=False,
):
    """Train a Topological Flow model on the specified dataset.

    Args:
        data_dir (str): Directory containing the training data.
        data_name (str): Name of the dataset to use.
        augment (bool): Whether to apply data augmentations.
        vector_field_model (str): Type of vector field model to use.
        hidden_dim (int): Dimension of the hidden layers in the model.
        evaluator (str): Evaluator to use for model evaluation. Either 'rdkit' or a path to a GATEvaluator checkpoint.
        checkpoint_dir (str): Directory to save the trained model.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        start_lr (float): Initial learning rate for the optimizer.
        max_lr (float): Maximum learning rate for the scheduler.
        min_lr (float): Minimum learning rate for the scheduler.
        max_frac (float): When to stop increasing the learning rate.
        min_time_factor (float): Minimum time factor for the flow model.
        max_time_factor (float): Maximum time factor for the flow model.
        max_atom_time (float): Maximum atom time for the flow model.
        w_loss_last_step (float): Weight for the loss at the last time step.
        velocity_scale (float): Scaling factor for the velocity in the loss computation.
        seed (int): Random seed for reproducibility.
        mlflow_tracking_uri (str): MLflow tracking URI.
        experiment_name (str): MLflow experiment name.
        run_name (str, optional): Name of the MLflow run. Defaults to None.
        previous_checkpoint (str, optional): Path to a checkpoint to resume training from. Defaults to None.
        skip_validation (bool, optional): Whether to skip validation during training. Defaults to False.
    """
    pl.seed_everything(seed)  # For reproducibility
    # ======================================================================
    # Data Preparation
    # ======================================================================
    if data_name == "crossdock_full":
        train_dataset = CrossDocked(
            root=data_dir,
            split="train",
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
    # alpha = LinearAlpha()
    # beta = LinearBeta()
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
        num_workers=len(psutil.Process().cpu_affinity()),
    )
    path = GaussianConditionalProbabilityPath(
        p_data=p_data,
        alpha=alpha,
        beta=beta,
        alpha_atom=alpha_atom,
        beta_atom=beta_atom,
    )

    val_metric_dist_path = os.path.join(data_dir, "processed", "val_distribution.npy")
    if os.path.exists(val_metric_dist_path):
        val_distribution = np.load(val_metric_dist_path, allow_pickle=True).item()
    else:
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
    # Model Initialization
    # ======================================================================
    if vector_field_model == "transformer":
        vector_field = TransformerVectorFieldLigand(
            h_dim=hidden_dim, num_heads=32, num_layers=3, dropout=0.1
        )
    elif vector_field_model == "mlp":
        vector_field = MLPVectorField(
            input_dim=14,  # Assuming 14 features as per the dataset
            output_dim=14,  # Assuming 3D vector field output
            hidden_dims=[hidden_dim, hidden_dim],
            activation=torch.nn.SiLU,
            additional_features="add",  # Use 'add' to incorporate time and conditioning features
        )
    elif vector_field_model == "gat":
        vector_field = GATVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",  # "ligand", "both", or "none"
        )
    elif vector_field_model == "ast":
        vector_field = AllSetTransformerVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",  # "ligand", "both", or "none"
            use_gat=False,
        )
    elif vector_field_model == "ast_gat":
        vector_field = AllSetTransformerVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",  # "ligand", "both", or "none"
            use_gat=True,
        )
    elif vector_field_model == "distance_transformer":
        vector_field = TransformerDistanceVectorField(
            h_dim=hidden_dim,
            num_heads=16,
            num_layers=2,
            dropout=0.1,
            add_time_to="ligand",  # "ligand", "both", or "none"
        )
    else:
        raise ValueError(f"Unsupported vector field model: {vector_field_model}")

    norm_path = os.path.join(data_dir, "processed", "normalization_params.npz")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(
            f"Normalization parameters not found at {norm_path}. Please run the preprocessing script."
        )

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
    # Training Setup
    # ======================================================================
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=checkpoint_dir + "/run_" + (run_name if run_name else "default"),
        filename="topoflow-{epoch:02d}",
        monitor="train/loss",  # validation is just copying train samples
        mode="min",
        save_top_k=-1,
    )
    logger = pl.loggers.MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=mlflow_tracking_uri,
        run_name=run_name,
    )

    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else 4,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        check_val_every_n_epoch=5,
        limit_val_batches=1.0 if not skip_validation else 0.0,
        # gradient_clip_val=0.5,
        # gradient_clip_algorithm="norm",
        limit_train_batches=1.0,
        num_sanity_val_steps=0,
    )
    # ======================================================================
    # Start Training
    # ======================================================================
    trainer.fit(
        flow_model,
        datamodule=datamodule,
        ckpt_path=previous_checkpoint if previous_checkpoint is not None else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Topological Flow model.")
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
        default="distance_transformer",
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
        "--evaluator",
        type=str,
        default="rdkit",
        help="Evaluator to use for model evaluation. Either 'rdkit' or a path to a GATEvaluator checkpoint.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints/topoflow/",
        help="Directory to save the trained model.",
    )
    parser.add_argument(
        "--previous_checkpoint",
        type=str,
        default=None,
        help="Path to a checkpoint to resume training from (optional).",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for training."
    )
    parser.add_argument(
        "--start_lr",
        type=float,
        default=0.0002,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max_lr",
        type=float,
        default=0.0008,
        help="Maximum learning rate for the optimizer.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=0.00004,
        help="Minimum learning rate for the optimizer.",
    )
    parser.add_argument(
        "--max_frac_lr",
        type=float,
        default=0.1,
        help="Fraction of epochs to use for warmup of learning rate.",
    )
    parser.add_argument(
        "--min_time_factor",
        type=float,
        default=0.05,
        help="Minimum time factor for the flow model.",
    )
    parser.add_argument(
        "--max_time_factor",
        type=float,
        default=10.0,
        help="Maximum time factor for the flow model.",
    )
    parser.add_argument(
        "--max_atom_time",
        type=float,
        default=1.0,
        help="Maximum time factor for atom type updates in the flow model.",
    )
    parser.add_argument(
        "--w_loss_last_step",
        type=float,
        default=0.2,
        help="Weight for the loss at the last time step.",
    )
    parser.add_argument(
        "--velocity_scale",
        type=float,
        default=1.0,
        help="Scaling factor for the velocity in the loss computation.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
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
        default="debug",
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Name of the MLflow run (optional).",
    )
    parser.add_argument(
        "--skip_validation",
        action="store_true",
        help="Whether to skip validation during training.",
    )
    args = parser.parse_args()

    train_flow_model(
        data_dir=args.data_dir,
        data_name=args.data_name,
        augment=args.augment,
        vector_field_model=args.vector_field_model,
        hidden_dim=args.hidden_dim,
        evaluator=args.evaluator,
        checkpoint_dir=args.checkpoint_dir,
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
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.mlflow_experiment_name,
        run_name=args.run_name if args.run_name else None,
        previous_checkpoint=args.previous_checkpoint,
        skip_validation=args.skip_validation,
    )
