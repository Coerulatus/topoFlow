"""Test cases for creating RDKit molecules from atom positions and types using Open Babel."""

import unittest
from rdkit import Chem
from tests.conftest import get_toy_molecule
from topoflow.utils.molecules import make_mol_openbabel


class TestMolecules(unittest.TestCase):
    """Test cases for creating RDKit molecules from atom positions and types using Open Babel."""

    def setUp(self):
        """Set up test cases with toy molecules."""
        self.water_molecule = get_toy_molecule("water")
        self.benzene_molecule = get_toy_molecule("benzene")
        self.invalid_molecule = get_toy_molecule("invalid")

    def test_make_mol_openbabel_water(self):
        """Test the Open Babel molecule creation for water."""
        mol = make_mol_openbabel(
            self.water_molecule["positions"],
            self.water_molecule["atom_types"],
            self.water_molecule["atom_decoder"],
        )
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 3)
        self.assertEqual(mol.GetNumBonds(), 2)

    def test_make_mol_openbabel_benzene(self):
        """Test the Open Babel molecule creation for benzene."""
        mol = make_mol_openbabel(
            self.benzene_molecule["positions"],
            self.benzene_molecule["atom_types"],
            self.benzene_molecule["atom_decoder"],
        )
        self.assertIsNotNone(mol)
        self.assertEqual(mol.GetNumAtoms(), 12)
        self.assertEqual(mol.GetNumBonds(), 12)

        double_bonds = sum(
            1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0
        )
        self.assertEqual(double_bonds, 3)

        Chem.SanitizeMol(mol)  # Force sanitization for better perception
        print(f"Benzene aromaticity: {mol.GetAtomWithIdx(0).GetIsAromatic()}")
