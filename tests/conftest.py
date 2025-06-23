"""Utility function to create toy molecules for testing purposes."""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def get_toy_molecule(molecule_name: str) -> dict:
    """Utility function to create toy molecules using RDKit.

    Returns a dictionary containing molecular data (RDKit molecule,
    positions, atom types, and atom decoder) for common toy molecules.

    Args:
        molecule_name (str): The name of the molecule ('water', 'benzene', or 'invalid').

    Returns:
        dict: A dictionary with the following keys:
            - 'rdkit_mol': An RDKit Mol object.
            - 'positions': A NumPy array of atom positions.
            - 'atom_types': A list of integer atom types (indices for atom_decoder).
            - 'atom_decoder': A dictionary mapping integer types to atom symbols.

    Raises:
        ValueError: If an unsupported molecule_name is provided.
    """
    if molecule_name.lower() == "water":
        # RDKit Molecule for Water
        mol = Chem.MolFromSmiles("O")
        mol = Chem.AddHs(mol)  # Add hydrogens for 3D structure
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())  # Generate 3D coordinates
        AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry

        # Get actual positions from RDKit molecule after optimization
        conf = mol.GetConformer()
        positions_rdkit = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        )

        atom_types = []
        atom_decoder = {}
        type_counter = 0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in atom_decoder.values():
                atom_decoder[type_counter] = symbol
                type_counter += 1
            atom_types.append([k for k, v in atom_decoder.items() if v == symbol][0])

        return {
            "rdkit_mol": mol,
            "positions": positions_rdkit,
            "atom_types": atom_types,
            "atom_decoder": atom_decoder,
        }

    elif molecule_name.lower() == "benzene":
        # RDKit Molecule for Benzene
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)

        # Positions, Atom Types, and Decoder for Benzene
        conf = mol.GetConformer()
        positions_rdkit = np.array(
            [list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]
        )

        atom_types = []
        atom_decoder = {}
        type_counter = 0
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            if symbol not in atom_decoder.values():
                atom_decoder[type_counter] = symbol
                type_counter += 1
            atom_types.append([k for k, v in atom_decoder.items() if v == symbol][0])

        return {
            "rdkit_mol": mol,
            "positions": positions_rdkit,
            "atom_types": atom_types,
            "atom_decoder": atom_decoder,
        }
    elif molecule_name.lower() == "invalid":
        # Example of an invalid molecule (not a real chemical structure)
        return {
            "rdkit_mol": None,
            "positions": np.array([[0, 0, 0]]),
            "atom_types": [0],
            "atom_decoder": {0: "X"},  # Placeholder for an invalid atom
        }
    else:
        raise ValueError(
            f"Unsupported molecule_name: {molecule_name}. Choose 'water' or 'benzene'."
        )
