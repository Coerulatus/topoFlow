"""Test script for evaluating the FlowModel on the CrossDocked dataset."""

import os
import numpy as np
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import LinearAlpha, LinearBeta
from topoflow.flow.flow_models import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.utils.molecules import data_to_rdmol
import topoflow.utils.constants as constants
from topoflow.utils.metrics import qed_scorer, logp_scorer, sas_scorer


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
checkpoint_path = "./checkpoints/topoflow/topoflow-100k.ckpt"

# Load the model from the checkpoint
model = FlowModel.load_from_checkpoint(checkpoint_path, path=path)

test_distribution_path = "./data/CrossDocked/processed/test_distribution.npy"
atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]
if os.path.exists(test_distribution_path):
    test_distribution = np.load(test_distribution_path, allow_pickle=True).item()
else:
    test_dataset = CrossDocked(
        root="./data/CrossDocked",
        split="test",
        transform=None,
    )
    test_mols = [
        data_to_rdmol(data, atom_decoder=atom_decoder, sanitize=True)
        for data in test_dataset
    ]
    test_distribution = {
        "qed": [qed_scorer(mol) for mol in test_mols],
        "logp": [logp_scorer(mol) for mol in test_mols],
        "sas": [sas_scorer(mol) for mol in test_mols],
    }
    np.save(test_distribution_path, test_distribution)

model.test_distribution(
    test_distribution=test_distribution,
    metrics=["qed", "logp", "sas"],
    atom_decoder=atom_decoder,
)
