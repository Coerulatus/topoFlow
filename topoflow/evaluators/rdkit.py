"""Implementation of an RDKit-based evaluator for molecular metrics."""

import rdkit
import rdkit.Chem
import torch
from topoflow.utils.metrics import qed_scorer, logp_scorer, sas_scorer
from topoflow.utils.molecules import data_to_rdmol
import topoflow.utils.constants as constants


class RDKitEvaluator(torch.nn.Module):
    """Evaluator for RDKit-based metrics."""

    def __init__(self, metrics=["qed", "logp", "sas"], dataset_name="crossdock_full"):
        """Initialize the RDKit evaluator.

        Args:
            metrics (list): List of metrics to compute. Supported metrics are "qed", "logp", and "sas".
            dataset_name (str): Name of the dataset. Currently, only "crossdock_full" is supported.
        """
        super(RDKitEvaluator, self).__init__()
        self.metrics = metrics
        self.metric_functions = {}
        for metric in self.metrics:
            if metric == "qed":
                self.metric_functions["qed"] = qed_scorer
            elif metric == "logp":
                self.metric_functions["logp"] = logp_scorer
            elif metric == "sas":
                self.metric_functions["sas"] = sas_scorer
            else:
                raise ValueError(
                    f"Unsupported metric: {metric}. Supported metrics are: qed, logp, sas."
                )

        assert dataset_name in [
            "crossdock_full"
        ], "Dataset name must be 'crossdock_full'."
        self.atom_decoder = constants.dataset_params[dataset_name]["atom_decoder"]

    def evaluate(self, data):
        """Compute the RDKit metrics.

        Args:
            data (torch_geometric.data.Data): Input data containing molecular information.

        Returns:
            torch.Tensor: Computed RDKit metric.
        """
        mols = data_to_rdmol(data, self.atom_decoder)
        mols_valid = []
        for mol in mols:
            try:
                rdkit.Chem.SanitizeMol(mol)
                mols_valid.append(mol)
            except Exception as e:
                print(f"Invalid molecule encountered: {e}")

        if not mols_valid:
            res = {metric: torch.tensor(0.0) for metric in self.metrics}
            res["validity"] = torch.tensor(0.0)
            return res

        validity = torch.tensor(len(mols_valid) / len(mols), dtype=torch.float)
        res = {"validity": validity}
        for metric in self.metrics:
            metric_function = self.metric_functions[metric]
            scores = [metric_function(mol) for mol in mols_valid]
            res[metric] = torch.tensor(scores, dtype=torch.float).mean()
        return res
