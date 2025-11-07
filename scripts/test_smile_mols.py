"""Script to generate ligand molecules from SMILES and save them as SDF and PNG files. Used to check scaffold structure."""

import os
from tqdm import tqdm
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
import torch
from torch_geometric.data import Data

from topoflow.flow.flow_models_ligand import FlowModel
from topoflow.data.datasets import CrossDocked
from topoflow.data import MockDatamodule
import topoflow.utils.constants as constants
from topoflow.flow.paths import GaussianConditionalProbabilityPath
from topoflow.flow.schedulers import CosineAlpha, CosineBeta
from topoflow.data.dataloader_adj import DataLoaderSampleable
from topoflow.utils.molecules import (
    pdb_to_data_object,
    make_mol_openbabel,
    mol_to_data,
    get_largest_fragment_info,
)
from topoflow.utils.visualization import visualize_trajectory

base_ligand_smile = "c1ccc2c(c1)[nH]cc2"

output_folder = "./outputs/generated_ligands_wwp2_c2/"
atom_encoder = constants.dataset_params["crossdock_full"]["atom_encoder"]
atom_decoder = constants.dataset_params["crossdock_full"]["atom_decoder"]

rdkit_ligand = Chem.MolFromSmiles(base_ligand_smile)
AllChem.Compute2DCoords(rdkit_ligand)
coords, one_hot, batch = mol_to_data(rdkit_ligand, atom_encoder)
print(coords)
rdmol = make_mol_openbabel(coords, one_hot, atom_decoder)

base_name = f"temp"
sdf_file_path = os.path.join(output_folder, f"{base_name}.sdf")
png_file_path = os.path.join(output_folder, f"{base_name}.png")

writer = Chem.SDWriter(sdf_file_path)
writer.write(rdmol)
writer.close()

AllChem.Compute2DCoords(rdmol)
Draw.MolToFile(rdmol, png_file_path, size=(300, 300))
