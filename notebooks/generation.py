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
from topoflow.flow.flow_models import FlowModel

alpha = LinearAlpha()
distribution_hist = np.load(
    "./data/CrossDocked/processed/size_distribution.npy", allow_pickle=True
)
beta = LinearBeta()
p_data = None
path = GaussianConditionalProbabilityPath(
    p_data=p_data,
    alpha=alpha,
    beta=beta,
    target_dim=14,
    distribution_hist=distribution_hist,
)
checkpoint_path = "./checkpoints/topoflow/topoflow-1k-e783.ckpt"

# Load the model from the checkpoint
model = FlowModel.load_from_checkpoint(
    checkpoint_path,
    path=path,
    norm_path="./data/CrossDocked/processed/normalization_params.npz",
)

res_traj = model.sample_with_trajectory(1, n_steps=20)

visualize_data(
    res_traj[0],
    image_path="outputs/trajectory/0.png",
    session_path="outputs/trajectory/0.pse",
)
visualize_data_matplotlib(res_traj[0].to("cpu"))
