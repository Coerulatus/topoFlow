import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from topoflow.utils.molecules import pdb_to_data_object
import torch

sdf_path = "test_outputs/sdfs/5mma-A-rec-4ztf-x2p-lig-tt-min-0-pocket10.sdf"
pdb_path = "test_pockets/5mma-A-rec-4ztf-x2p-lig-tt-min-0-pocket10.pdb"

suppl = Chem.SDMolSupplier(sdf_path)
mols = [m for m in suppl if m is not None]

print(f"Loaded {len(mols)} molecules from {sdf_path}")

from topoflow.utils.constants import dataset_params

atom_encoder = dataset_params["crossdock_full"]["atom_encoder"]
pocket_data = pdb_to_data_object(pdb_path, atom_encoder)
pocket_coords = pocket_data.x_pocket + pocket_data.original_centroid
pocket_center = pocket_coords.mean(dim=0)
print(f"Pocket center of mass (uncentered): {pocket_center}")
print(f"Original centroid: {pocket_data.original_centroid.squeeze()}")

for i, mol in enumerate(mols):
    num_atoms = mol.GetNumAtoms()
    # Get coordinates
    conf = mol.GetConformer()
    coords = torch.tensor(conf.GetPositions(), dtype=torch.float32)
    center = coords.mean(dim=0)
    
    # Distance from pocket center
    dist_to_pocket = torch.norm(center - pocket_center).item()
    
    # Closest atom to pocket
    dist_matrix = torch.cdist(coords, pocket_coords)
    closest_dist = dist_matrix.min().item()
    
    print(f"Mol {i+1}: {num_atoms} atoms. Center: {center}. Dist to pocket center: {dist_to_pocket:.2f}. Closest pocket atom dist: {closest_dist:.2f}")

