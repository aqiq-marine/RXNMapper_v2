import os
import rdkit
from rdkit import Chem
from concurrent.futures import ThreadPoolExecutor

# MOL -> SMILES
def apply_mol_to_smiles(mol):
    try:
        return Chem.MolToSmiles(mol)
    except:
        return ''
 
# MOL -> MAPPED SMILES
def atom_map_molecule(mol:rdkit.Chem.rdchem.Mol)->rdkit.Chem.rdchem.Mol:
    """Assigns unique atom map numbers to each atom."""
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetAtomMapNum(i + 1)  # Map numbers start at 1
    return mol

def get_mapped_smile_from_mol(mol:rdkit.Chem.rdchem.Mol)->rdkit.Chem.rdchem.Mol:
    try:
        mapped_mol = atom_map_molecule(mol)
        return Chem.MolToSmiles(mapped_mol)
    except:
        return ''

def get_mapped_smile_from_mol_list(mol_list):
    ncpus = os.cpu_count()-2
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        results = list(executor.map(get_mapped_smile_from_mol, mol_list))
    return results

# MAPPED SMILES -> SMILES
def get_unmapped_smiles(smi:str)->str:
    try:
        # Convert the mapped SMILES to an RDKit molecule
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            raise ValueError("Invalid SMILES")

        # Remove atom mapping
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(0)  

        # Generate the unmapped SMILES
        unmapped_smiles = Chem.MolToSmiles(mol)
        return unmapped_smiles
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_unmapped_smiles_list(smiles_list:list[str])->list[str]:
    ncpus = os.cpu_count()-2
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        results = list(executor.map(get_unmapped_smiles, smiles_list))
    return results

# ADD EXPLICIT HYDROGENS TO MOL
def add_Hs_to_mol(mol:rdkit.Chem.rdchem.Mol)->rdkit.Chem.rdchem.Mol:
    return Chem.AddHs(mol)

def add_Hs_to_mol_list(mol_list):
    ncpus = os.cpu_count()-2
    with ThreadPoolExecutor(max_workers=ncpus) as executor:
        results = list(executor.map(add_Hs_to_mol, mol_list))
    return results

# CHOOSE BONDS GIVEN CERTAIN CRITERIA
# --> NO H ATOM
def get_non_H_bonds_indices(mol:rdkit.Chem.rdchem.Mol)->tuple[int,int]:
    """
    Returns all bonds indices from the molecule that do not involve a hydrogen atom.
    
    Parameters:
    - mol: RDKit molecule object.
    
    Returns:
    - tuple: Indices of the two atoms composing the randomly selected bond.
    """
    # Filter bonds to exclude those involving hydrogen atoms
    non_hydrogen_bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if bond.GetBeginAtom().GetAtomicNum() != 1 and bond.GetEndAtom().GetAtomicNum() != 1
    ]
    
    # Check if there are any valid bonds
    if not non_hydrogen_bonds:
        return ''

    return non_hydrogen_bonds

# --> R-H BONDS (with R != H)
def get_R_H_bonds_indices(mol)->tuple[int, int]:
    """
    Returns all bonds indices from the molecule that involve exactly one hydrogen atom.
    
    Parameters:
    - mol: RDKit molecule object.
    
    Returns:
    - tuple: Indices of the two atoms composing the randomly selected bond.
    """
    mol = Chem.AddHs(mol)
    # Filter bonds to exclude those involving hydrogen atoms
    hydrogen_bonds = [
        (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in mol.GetBonds()
        if (bond.GetBeginAtom().GetAtomicNum() == 1) ^ (bond.GetEndAtom().GetAtomicNum() == 1)
    ]
    
    # Check if there are any valid bonds
    if not hydrogen_bonds:
        return ''
        
    return hydrogen_bonds

    ## i. check to avoid trying to bond atoms already bonded 

# CHECKS FOR TWO TUPLES ON ATOMS' PROPERTIES
# --> ARE ATOMS NOT NEIGHBORS
def are_atoms_not_neigbors(t1:tuple[int,int], t2:tuple[int,int], mol:rdkit.Chem.rdchem.Mol)->bool:
    return mol.GetBondBetweenAtoms(t1[0], t2[0]) is None and mol.GetBondBetweenAtoms(t1[1], t2[1]) is None
    
# --> IS THERE AN AROMATIC ATOM
def is_there_aromatic_atom(t1:tuple[int,int], t2:tuple[int,int], mol:rdkit.Chem.rdchem.Mol)->bool:
    return mol.GetAtomWithIdx(t1[0]).GetIsAromatic() or mol.GetAtomWithIdx(t1[1]).GetIsAromatic() or mol.GetAtomWithIdx(t2[0]).GetIsAromatic() or mol.GetAtomWithIdx(t2[1]).GetIsAromatic()

