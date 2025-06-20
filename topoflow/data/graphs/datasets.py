"""QM9 dataset with RDKit-calculated molecular properties (QED, SAS, LogP)."""

import sys
import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

# Import RDKit modules
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import QED

# For Synthetic Accessibility Score (SAS), it's in rdkit.Chem.rdSAScore
# Ensure you have rdkit-pypi installed, which includes this contrib module.
sys.path.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "RDKit", "Contrib"))
from SA_Score import sascorer


class QM9WithRDKitMetrics(InMemoryDataset):
    """A custom PyTorch Geometric dataset that extends the QM9 dataset to include RDKit-calculated properties."""

    def __init__(self, root, transform=None, pre_transform=None, normalize=True):
        """A custom PyTorch Geometric dataset that extends QM9.

        This class adds RDKit-calculated molecular properties (QED, SAS, LogP) as new attributes to each Data object.
        """
        self.original_qm9_root = os.path.join(root, "QM9_original")
        self.normalize = normalize
        self.qed_min, self.qed_max = 0.0, 1.0
        self.logp_min, self.logp_max = -6.0, 6.0
        self.sas_min, self.sas_max = 1.0, 10.0
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        """QM9 dataset does not have raw files in the traditional sense."""
        return []

    @property
    def processed_file_names(self):
        """Filename for the processed dataset with RDKit metrics."""
        return "qm9_rdkit_metrics.pt"

    def download(self):
        """Download the original QM9 dataset if it doesn't exist."""
        QM9(self.original_qm9_root)

    def process(self):
        """Process the original QM9 dataset and calculate RDKit metrics."""
        original_dataset = QM9(self.original_qm9_root)

        data_list = []
        # Iterate through each molecule in the original QM9 dataset
        for i, data in enumerate(
            tqdm(original_dataset, desc="Calculating RDKit metrics")
        ):
            # Get the SMILES string (QM9 stores it in `data.smiles` usually)
            smiles = (
                data.smiles[0] if isinstance(data.smiles, torch.Tensor) else data.smiles
            )

            # Convert SMILES to RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                print(
                    f"Warning: Could not parse SMILES for molecule {i}: {smiles}. Skipping."
                )
                continue

            # Calculate RDKit Descriptors
            # 1. Quantitative Estimation of Drug-likeness (QED)
            qed_score = torch.tensor([QED.default(mol)], dtype=torch.float)

            # 2. LogP (Partition Coefficient, related to solubility/lipophilicity)
            logp_score = torch.tensor([Descriptors.MolLogP(mol)], dtype=torch.float)

            # 3. Synthetic Accessibility Score (SAS)
            # SAS score typically ranges from 1 (easy to synthesize) to 10 (hard to synthesize)
            sas_score = torch.tensor([sascorer.calculateScore(mol)], dtype=torch.float)

            if self.normalize:
                if qed_score < self.qed_min or qed_score > self.qed_max:
                    raise ValueError(
                        f"QED score {qed_score.item()} out of bounds [{self.qed_min}, {self.qed_max}]"
                    )
                if logp_score < self.logp_min or logp_score > self.logp_max:
                    raise ValueError(
                        f"LogP score {logp_score.item()} out of bounds [{self.logp_min}, {self.logp_max}]"
                    )
                if sas_score < self.sas_min or sas_score > self.sas_max:
                    raise ValueError(
                        f"SAS score {sas_score.item()} out of bounds [{self.sas_min}, {self.sas_max}]"
                    )
                qed_score = (qed_score - self.qed_min) / (self.qed_max - self.qed_min)
                logp_score = (logp_score - self.logp_min) / (
                    self.logp_max - self.logp_min
                )
                sas_score = (sas_score - self.sas_min) / (self.sas_max - self.sas_min)

            new_y = torch.cat([qed_score, logp_score, sas_score], dim=0)
            data.y = new_y

            # We only care about the atom types and positions for the graph structure
            data.x = torch.cat([data.x[:, :5], data.pos], dim=1)
            del data.pos  # Remove position if not needed

            data_list.append(data)

        # Save the processed data
        if self.pre_filter is not None:
            data_list = self.pre_filter(data_list)
        if self.pre_transform is not None:
            data_list = self.pre_transform(data_list)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# --- How to use the function ---
if __name__ == "__main__":
    # Define the root directory for storing the dataset
    # This will create 'data/QM9_RDKit' for the new dataset
    # and 'data/QM9_original' for the base QM9 dataset if not present
    dataset_root = "data/QM9_RDKit"

    print(
        f"Processing QM9 dataset and calculating RDKit metrics. This may take a while..."
    )
    print(
        f"Original QM9 dataset will be downloaded to: {os.path.join(dataset_root, 'QM9_original')}"
    )
    print(
        f"Processed dataset with RDKit metrics will be saved to: {os.path.join(dataset_root, 'processed', 'qm9_rdkit_metrics.pt')}"
    )

    # Initialize the custom dataset. This will trigger the download and process methods.
    dataset = QM9WithRDKitMetrics(root=dataset_root)

    print("\nDataset processing complete!")
    print(f"Number of molecules in the dataset: {len(dataset)}")

    # Example: Accessing properties of the first molecule
    first_molecule = dataset[0]
    print("\nExample for the first molecule:")
    print(f"Original QM9 targets (data.y): {first_molecule.y}")
    print(f"SMILES: {first_molecule.smiles}")
    print(f"QED Score: {first_molecule.y[0].item()}")
    print(f"LogP Score: {first_molecule.y[1].item()}")
    print(f"SAS Score: {first_molecule.y[2].item()}")
