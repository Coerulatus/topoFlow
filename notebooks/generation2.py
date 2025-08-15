"""This file is part of the TopoFlow project. Used to check generation of molecules."""

import numpy as np
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import LinearAlpha, LinearBeta
from topoflow.utils.visualization import (
    visualize_data_rdkit,
    visualize_data,
    visualize_trajectory,
    visualize_data_matplotlib,
)
from topoflow.flow.flow_models import FlowModel
from topoflow.data.dataloader import DataLoaderSampleable
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
checkpoint_path = "./checkpoints/topoflow/topoflow-epoch=840.ckpt"

# Load the model from the checkpoint
model = FlowModel.load_from_checkpoint(
    checkpoint_path,
    path=path,
    norm_path="./data/CrossDocked/processed/normalization_params.npz",
)

res = model.sample(2, n_steps=20)
