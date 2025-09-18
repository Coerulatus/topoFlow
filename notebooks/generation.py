"""Test script for evaluating the generation capabilities of a FlowModel on the CrossDocked dataset."""

import numpy as np
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import LinearAlpha, LinearBeta
from topoflow.utils.visualization import (
    visualize_data_rdkit,
    visualize_data,
    visualize_trajectory,
    visualize_data_matplotlib,
)
from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.data.datasets import CrossDocked
from topoflow.flow.schedulers import LinearAlpha, LinearBeta, SineAlpha, CosineBeta

batch_size = 32
data_dir = "./data/CrossDocked"
train_dataset = CrossDocked(
    root=data_dir,
    split="train",
)
val_dataset = CrossDocked(root=data_dir, split="val")

alpha = SineAlpha()
beta = CosineBeta()

p_data = DataLoaderSampleable(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
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
checkpoint_path = "./checkpoints/topoflow/run_200e_16_09_gat/topoflow-epoch=199.ckpt"

# Load the model from the checkpoint
model = FlowModel.load_from_checkpoint(
    checkpoint_path,
    path=path,
    norm_path="./data/CrossDocked/processed/normalization_params.npz",
)


res_traj = model.sample_with_trajectory(1, n_steps=100)

visualize_trajectory(
    res_traj,
    image_path="outputs/trajectory/mol_traj.png",
    gif_path="outputs/trajectory/mol_traj.gif",
)
# visualize_data_matplotlib(res_traj[0].to("cpu"))
