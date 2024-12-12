import rdkit
from rdkit import Chem


def find_equivalent_atoms(mol:rdkit.Chem.rdchem.Mol)->dict:
    """
    Identifies equivalent atoms in a molecule using RDKit's symmetry classes.

    Parameters:
    - mol: RDKit molecule object.

    Returns:
    - dict: A dictionary where keys are symmetry class indices, and values are lists of atom indices that are equivalent.
    """
    # Compute symmetry classes for all atoms
    symmetry_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
    
    # Group atoms by symmetry class
    equivalent_atoms = {}
    for atom_idx, sym_class in enumerate(symmetry_classes):
        if sym_class not in equivalent_atoms:
            equivalent_atoms[sym_class] = []
        equivalent_atoms[sym_class].append(atom_idx)
    
    return equivalent_atoms

def are_there_equ_atoms(dict_symmetry:dict)->bool:
    """
    Check if there are equivalent atoms in the molecule.

    Parameters:
    - dict_symmetry: dictionary containing the symmetry classes of the atoms in the molecule.

    Returns:
    - bool: True if there are equivalent atoms, False otherwise.
    """
    # Check if there are symmetry classes with more than one atom
    if max(list(map(len, dict_symmetry.values())))>1:
        return True
    else:
        return False

def get_indices_of_equ_atoms(dict_symmetry:dict)->list:
    """
    Get the indices of the equivalent atoms in the molecule.

    Parameters:
    - dict_symmetry: dictionary containing the symmetry classes of the atoms in the molecule.

    Returns:
    - list: list of lists of atom indices that are equivalent.
    """
    # Return the keys and values only for the equivalent atoms
    if are_there_equ_atoms(dict_symmetry):
        return {key: value for key, value in dict_symmetry.items() if len(value) > 1}
    else: 
        return {}
    

## examples with tert-butyls
#tertbutyl_example0 = 'CCCC(C)=O'
#tertbutyl_example1 = 'CC(C)(C)CC(C)=O'
#tertbutyl_example2 = 'CC(C)(C)C(C)(C)C(C)=O'
#
#tertbutyl_example0_mol = Chem.MolFromSmiles(tertbutyl_example0)
#tertbutyl_example1_mol = Chem.MolFromSmiles(tertbutyl_example1)
#tertbutyl_example2_mol = Chem.MolFromSmiles(tertbutyl_example2)
#
#example0_dict = find_equivalent_atoms(tertbutyl_example0_mol)
#example1_dict = find_equivalent_atoms(tertbutyl_example1_mol)
#example2_dict = find_equivalent_atoms(tertbutyl_example2_mol)
#print(example0_dict)
#print(example1_dict)
#print(example2_dict)
#
#bool_ex0 = are_there_equ_atoms(example0_dict)
#bool_ex1 = are_there_equ_atoms(example1_dict)
#bool_ex2 = are_there_equ_atoms(example2_dict)
#print(bool_ex0)
#print(bool_ex1)
#print(bool_ex2)
#
#dict_ind0 = get_indices_of_equ_atoms(example0_dict)
#dict_ind1 = get_indices_of_equ_atoms(example1_dict)
#dict_ind2 = get_indices_of_equ_atoms(example2_dict)
#print(dict_ind0)
#print(dict_ind1)
#print(dict_ind2)