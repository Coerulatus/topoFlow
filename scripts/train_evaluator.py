"""Train an evaluator for the QM9 dataset using PyTorch Geometric."""

from topoflow.data.graphs import QM9WithRDKitMetrics

# data_path = "./data/gdb9_pygmolecular_dataset_9.pkl"
# dataset = PyGMolecularDataset.load(data_path)

data_path = "./data/QM9_RDKit"
dataset = QM9WithRDKitMetrics(root=data_path)
pass
