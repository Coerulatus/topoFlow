"""Utility functions to search for scaffold presence in molecules."""

from rdkit import Chem
from rdkit.Chem import rdFMCS


def check_scaffold_presence(generated_mol, scaffold_mol, relaxed=True):
    """Checks if a reference scaffold is present in a generated molecule.

    Args:
        generated_mol (rdkit.Chem.Mol): The generated molecule to check.
        scaffold_mol (rdkit.Chem.Mol): The reference scaffold molecule.
        relaxed (bool): If True, uses a relaxed matching criteria. Defaults to True.

    Returns:
        bool: True if the scaffold is present in the generated molecule, False otherwise.
    """
    mol_to_check = Chem.RemoveHs(generated_mol)
    ref_scaffold = Chem.RemoveHs(scaffold_mol)

    if relaxed:
        bondCompare = rdFMCS.BondCompare.CompareAny
    else:
        bondCompare = rdFMCS.BondCompare.CompareOrderExact

    mcs_result = rdFMCS.FindMCS(
        [mol_to_check, ref_scaffold], bondCompare=bondCompare, ringMatchesRingOnly=True
    )

    scaffold_size = ref_scaffold.GetNumAtoms()
    mcs_size = mcs_result.numAtoms

    return mcs_size == scaffold_size
