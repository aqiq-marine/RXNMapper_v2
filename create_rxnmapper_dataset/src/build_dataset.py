import os
import sys
import rdkit
import numpy as np
import pandas as pd
from itertools import product
from rdkit import Chem
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor



sys.path.append('../')
from src.rdkit_utils import *
from src.reaction_series_utils import *
from src.data_handler import DataHandler

class AtomsToBond:

    def __init__(self,
    mol_list:list[rdkit.Chem.rdchem.Mol],  
    dict_substructs:dict[int,str], 
    dict_conditions:dict[int,dict[int,int]], 
    ):
        self.mol_list = mol_list
        self.dict_substructs = dict_substructs
        self.dict_conditions = dict_conditions

    def get_dict_count_substructs(self, mol):
        '''
        for a given molecule (rdkit.mol object) and a dictionary of substructures (dict{smarts}), returns a dictionary 
        where keys are the index of the substructure and the value is the number of times this substructure is found in the molecule 

        Inputs:
        - mol: rdkit.mol object             : molecule for which we want to count the number of substructures
        - dict_substructs: dict{smarts}   : dictionary of substructures we want to count in the molecule. (ex: {0:'[#6h]', 1:'[#7h]'})

        Outputs:
        dict{int:int}                       : dictionary where keys are the index of the substructure and the value is the number of times this substructure 
                                            is found in the molecule. (ex: {0:2, 1:0})
        '''
        dict_count_substructs = {}
        for key, value in self.dict_substructs.items():
            dict_count_substructs[key] = len(list(mol.GetSubstructMatches(Chem.MolFromSmarts(value))))
        return dict_count_substructs

    @staticmethod
    def check_biggerorequal_condition(dict_count_substructs:dict[int,int], condition:dict[int,dict[int,int]])->bool:     
        '''
        for a given dictionary  counting how many times each substructure appears in a molecule and a dictionary of conditions,

        Inputs
        - dict_count_substructs: dict{int:int} : dictionary where keys are the index of the substructure and the value is 
                                                the number of times this substructure is found in the molecule. (ex: {0: 7, 1: 0})
        - condition: dict{int:int}              : dictionary of conditions we want to check. (ex: {0:2,1:0})

        Outputs
        bool                                    : True if all the conditions are satisfied, False otherwise
        '''
        return all(a >= b for a, b in zip(list(dict_count_substructs.values()), list(condition.values())))

    def check_all_conditions_on_mol(self, dict_count_substructs)->list:
        return [self.check_biggerorequal_condition(dict_count_substructs, condition) for condition in self.dict_conditions.values()]

    @staticmethod
    def get_true_index(list_bool:list)->list:
        return [i for i, x in enumerate(list_bool) if x]
    
    def is_candidate(self, mol:rdkit.Chem.rdchem.Mol)->[bool,list[int]]:
        '''
        For a given molecule, a dictionary of substructures and a dictionary of conditions, returns True if the molecule respects
        the conditions on the substructures and a list with the keys (int) of the conditions that are respected.
        '''
        dict_count_substructs = self.get_dict_count_substructs(mol, self.dict_substructs)
        conditions_satisfied_list = self.check_all_conditions_on_mol(dict_count_substructs)
        index_true = self.get_true_index(conditions_satisfied_list)
        if len(index_true) > 0:
            return True, index_true
        else:
            return False, []

    def get_index_atoms_in_substruct(self, mol:rdkit.Chem.rdchem.Mol)->list:
        '''
        For a given molecule and a substructure, returns the indices of the atoms in the substructure i in dict, in index_atoms[i]
        '''
        index_atoms_in_substructs = []
        for key, value in self.dict_substructs.items():
            index_atoms_in_substructs.append(list(mol.GetSubstructMatches(Chem.MolFromSmarts(value))))

        return index_atoms_in_substructs
    
    def get_atoms_indices_in_condition(self, index_atoms_in_substructs:list, index_cond)->list:
        indices_list = []
        for key, value in self.dict_conditions[index_cond].items():
            if value > 0:
                indices_list.append(index_atoms_in_substructs[key])
            else:
                continue
        return indices_list
    
    @staticmethod
    def get_adjacency_df(mol:rdkit.Chem.rdchem.Mol)->pd.DataFrame:
        adjacency_matrix = Chem.GetAdjacencyMatrix(mol)
        return pd.DataFrame(adjacency_matrix)        

    def get_possible_bond_pairs(self, adjacency_df:pd.DataFrame, ind_0:list, ind_1:list)->set:
        '''
        For a given adjacency matrix and two lists of atom indices, returns all the possible pairs of indices that can be bonded together.
        '''
        sub_df = adjacency_df.loc[ind_0, ind_1]
        zero_positions = [
            (row_idx, col_idx) 
            for row_idx in sub_df.index
            for col_idx in sub_df.columns
            if sub_df.loc[row_idx, col_idx] == 0 and row_idx != col_idx
        ]
        return {tuple(sorted(pair)) for pair in zero_positions}
    
    def get_indices_of_atoms_to_bond_mol(self, mol:rdkit.Chem.rdchem.Mol):
        '''
        return the indices of the atoms that should be bonded together in a molecule
        '''
        dict_count_substructs = self.get_dict_count_substructs(mol)
        conditions_satisfied = self.check_all_conditions_on_mol(dict_count_substructs)
        index_true = self.get_true_index(conditions_satisfied)
        index_atoms_in_substructs = self.get_index_atoms_in_substruct(mol)
        adjacency_df = self.get_adjacency_df(mol)

        for i in index_true:
            # initialize the final results
            results_list = []
            ind_atoms_condition = self.get_atoms_indices_in_condition(index_atoms_in_substructs, i)
            if len(ind_atoms_condition) == 1:
                # adjacency matrix analysis giving indices for two atoms with same substruct (*)
                ind_0 = [i[0] for i in ind_atoms_condition[0]]

                result = self.get_possible_bond_pairs(adjacency_df, ind_0, ind_0)
            
            elif len(ind_atoms_condition)==2:
                # adjacency matrix analysis giving indices for two atoms with different substruct (**)
                ind_0 = [i[0] for i in ind_atoms_condition[0]]
                ind_1 = [i[0] for i in ind_atoms_condition[1]]
                result = self.get_possible_bond_pairs(adjacency_df, ind_0, ind_1)            
            results_list.append(result)
            if results_list:
                return results_list[0]
            else:
                return ''

    
    def get_indices_of_atoms_to_bond_mol_list(self):
        # '''
        # return the indices of the atoms that should be bonded together in a list of molecules
        # '''
        # results_list = []
        # for mol in self.mol_list:
        #     results_list.append(self.get_indices_of_atoms_to_bond_mol(mol))
        # return results_list
        """
        Apply `get_indices_of_atoms_to_bond_mol` on the list of molecules in parallel.
        """
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_indices_of_atoms_to_bond_mol, self.mol_list))
        return results


class BondAtoms:
    def __init__(self, 
        mol_list:list[rdkit.Chem.rdchem.Mol],  
        smiles_list:list[str],
        atoms_indices: list[list[tuple[int,int]]], 
        outputs_path:str='', 
        ):
        self.mol_list=mol_list
        self.smiles_list=smiles_list
        self.atoms_indices=atoms_indices

        if outputs_path == '':
            self.outputs_path='../results/cyclization/'
        else:
            self.outputs_path=outputs_path

        if os.path.exists(self.outputs_path) == False:
            os.makedirs(self.outputs_path)  


    @staticmethod
    def cyclization_mol(mol, index_atoms:tuple[int, int])->rdkit.Chem.rdchem.Mol:
        mol = Chem.AddHs(mol)
        editable_mol = Chem.EditableMol(mol)
        if index_atoms is None:
            return ''
        else:
            hydrogens_to_bond_idx = []
            for el in range(len(index_atoms)):
                # Identify the hydrogen to remove for the first atom
                for neighbor in mol.GetAtomWithIdx(index_atoms[el]).GetNeighbors():
                    if neighbor.GetAtomicNum() == 1:  # Check if it's a hydrogen
                        hydrogen_idx = neighbor.GetIdx()
                        break

                # Remove the C-H bond
                editable_mol.RemoveBond(index_atoms[el], hydrogen_idx)
                # store the index of the hydrogen
                hydrogens_to_bond_idx.append(hydrogen_idx)

            # create bond between the two atoms
            editable_mol.AddBond(index_atoms[0], index_atoms[1], Chem.BondType.SINGLE)
            # create bond between the two hydrogens
            editable_mol.AddBond(hydrogens_to_bond_idx[0], hydrogens_to_bond_idx[1], Chem.BondType.SINGLE)
            cyclic_mol = editable_mol.GetMol()
            return Chem.RemoveHs(cyclic_mol)

    @staticmethod
    def atom_map_molecule(mol:rdkit.Chem.rdchem.Mol)->rdkit.Chem.rdchem.Mol:
        """Assigns unique atom map numbers to each atom."""
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)  # Map numbers start at 1
        return mol
    
    def get_mapped_smile_from_mol(self, mol:rdkit):
        try:
            mapped_mol = self.atom_map_molecule(mol)
            return Chem.MolToSmiles(mapped_mol)
        except:
            return ''
    
    def get_mapped_smile_from_mol_list(self):
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(self.get_mapped_smile_from_mol, self.mol_list))
        return results

    def build_df(self):
        '''
        build a df of each original smiles together with all the different atom indices
        '''
        rows = [
        {"SMILES": smiles, "MAPPED SMILES": mapped_smiles, "index_atoms": ind if ind else None}
        for smiles, mapped_smiles, sets in zip(self.smiles_list, self.get_mapped_smile_from_mol_list(), self.atoms_indices)
        for ind in (sets or [None])
        ]
        df = pd.DataFrame(rows)
        return df

    def apply_cyclization(self, row):
        mol = Chem.MolFromSmiles(row['SMILES'])
        return self.cyclization_mol(mol, row['index_atoms'])

    def apply_cyclization_on_map(self, row):
        mol = Chem.MolFromSmiles(row['MAPPED SMILES'])
        mapped_mol = self.atom_map_molecule(mol)
        return self.cyclization_mol(mapped_mol, row['index_atoms'])
    
    def apply_mol_to_smiles(self, mol):
        try:
            return Chem.MolToSmiles(mol)
        except:
            return ''
        
    
    def get_df_cyclized_mols(self):
        print('build dataframe structure..')
        df = self.build_df()
        print('dataframe structure ready.')

        print('starting cyclic mols obtention..')
        df['cycled_product'] = df.apply(self.apply_cyclization, axis=1)

        print('cyclic mols obtained, converting to smiles..')
        df['cycled_product'] = df['cycled_product'].apply(self.apply_mol_to_smiles)

        print('starting mapped cyclic mols obtention..')
        df['mapped_cycled_product'] = df.apply(self.apply_cyclization_on_map, axis=1)

        print('mapped cyclic mols obtained, converting to smiles..')
        df['mapped_cycled_product'] = df['mapped_cycled_product'].apply(self.apply_mol_to_smiles)

        print('dataframe fully ready')
        return df
    
    @staticmethod
    def build_reaction_series(reactant_series:pd.Series, product_series:pd.Series):
        return reactant_series + ">>" + product_series
    
    def get_df_cyclized_mols_with_reaction(self, df):
        print('preparing reactions..')
        df['reaction'] = self.build_reaction_series(df['SMILES'], df['cycled_product'])

        print('preparing mapped reactions..')
        df['mapped_reaction'] = self.build_reaction_series(df['MAPPED SMILES'], df['mapped_cycled_product'])

        print('finished reactions')
        return df
    
    def create_and_save_cyclization_df(self, filename:str='df_cyclized_mols.csv'):
        df = self.get_df_cyclized_mols()
        print('Adding reactions to dataframe')
        df = self.get_df_cyclized_mols_with_reaction(df)
        df = df[df['cycled_product'] != '']
        print('saving dataframe..')
        DataHandler(df, f'{self.outputs_path}{filename}')._save_to_csv()
        return df




class SimpleRearrangement():
    '''
    Class creating fictive rearrangement reactions for a list of molecules. For any given molecule, will take every
    bond A-B (with A and B non H atoms) and every R-H bond (with R a non H atom) and create a fictive rearrangement
    reaction where we will obtain A-H and R-B.
    '''
    def __init__(self, 
        mol_list:list[rdkit.Chem.rdchem.Mol],  
        smiles_list:list[str],
        outputs_path:str='', 
        ):
        self.mol_list=mol_list
        self.smiles_list = smiles_list
        self.mapped_smiles_list = get_mapped_smile_from_mol_list(self.mol_list)
        self.molH_list = add_Hs_to_mol_list(self.mol_list)
        self.ncpus = os.cpu_count()-2
        self.bonds_indices = self.get_all_bonds_indices_of_mol_list(self.mol_list)

        if outputs_path == '':
            self.outputs_path='../results/rearrangement/'
        else:
            self.outputs_path=outputs_path

        if os.path.exists(self.outputs_path) == False:
            os.makedirs(self.outputs_path)  
    
    @staticmethod
    def are_elements_from_tuples_all_diff(t1:tuple[int,int], t2:tuple[int,int])->bool:
        """
        Check if all elements from two tuples are different.
        
        Parameters:
        - t1: Tuple of two integers.
        - t2: Tuple of two integers.
        
        Returns:
        - bool: True if all elements are different, False otherwise.
        """
        return len(set(list(t1)).intersection(set(list(t2)))) == 0

    def build_all_indices_combinations_for_rearrangement(self, list1:list[tuple[int,int]], list2:list[tuple[int,int]])->list[list[tuple[int,int], tuple[int,int]]]:
        valid_combinations = [
            [t1, t2] for t1, t2 in product(list1, list2)
            if self.are_elements_from_tuples_all_diff(t1, t2)
        ]
        return valid_combinations

    def get_all_bonds_indices_of_mol(self, mol:rdkit.Chem.rdchem.Mol)->list[tuple[int,int]]:
        non_H_bonds_indices = get_non_H_bonds_indices(mol)
        R_H_bonds_indices = get_R_H_bonds_indices(mol)
        return self.build_all_indices_combinations_for_rearrangement(non_H_bonds_indices, R_H_bonds_indices)

    def get_all_bonds_indices_of_mol_list(self, mol_list):
        with ThreadPoolExecutor(max_workers=self.ncpus) as executor:
            results = list(executor.map(self.get_all_bonds_indices_of_mol, mol_list))
        return results
    
    def build_df(self):
        smiles_list_expanded        = np.repeat(self.smiles_list, [len(sublist) for sublist in self.bonds_indices])
        mapped_smiles_list_expanded = np.repeat(self.mapped_smiles_list, [len(sublist) for sublist in self.bonds_indices])
        map_mol_list_expanded       = np.repeat(self.molH_list, [len(sublist) for sublist in self.bonds_indices])
        all_bonds_indices_expanded  = [item for sublist in self.bonds_indices for item in sublist]

        # Create the DataFrame
        df = pd.DataFrame({
            'SMILES': smiles_list_expanded,
            'MAPPED SMILES': mapped_smiles_list_expanded,
            'MAPPED MOL': map_mol_list_expanded,
            'INDICES': all_bonds_indices_expanded
        })
        return df
    
    @staticmethod
    def rearrangement_mol(mol:rdkit.Chem.rdchem.Mol, index_bonds:list[tuple[int, int]])->rdkit.Chem.rdchem.Mol:
        """
        Rearranges the molecule by breaking two bonds and forming two new bonds.
        
        Parameters:
        - mol: RDKit molecule object.
        - index_bonds: List of two tuples, each containing the indices of two atoms.
        
        Returns:
        - mol: RDKit molecule object after rearrangement.
        """

        AB_bond = list(index_bonds[0])
        RH_bond = list(index_bonds[1])
        mol = Chem.AddHs(mol)
        editable_mol = Chem.EditableMol(mol)
        editable_mol.RemoveBond(AB_bond[0], AB_bond[1])
        editable_mol.RemoveBond(RH_bond[0], RH_bond[1])
        
        editable_mol.AddBond(AB_bond[0], RH_bond[0], order=Chem.rdchem.BondType.SINGLE)
        editable_mol.AddBond(AB_bond[1], RH_bond[1], order=Chem.rdchem.BondType.SINGLE)
        rearr_mol = editable_mol.GetMol()
        
        return Chem.RemoveHs(rearr_mol) 
       
    def apply_rearrangement_on_map_mol(self, row):
        if are_atoms_not_neigbors(row['INDICES'][0], row['INDICES'][1], row['MAPPED MOL']) and not is_there_aromatic_atom(row['INDICES'][0], row['INDICES'][1], row['MAPPED MOL']):
            return self.rearrangement_mol(row['MAPPED MOL'], row['INDICES'])
        else:
            return ''



    def process(self, filename= 'df_rearrangement_test.csv'):

        print('building dataframe...')
        df = self.build_df()

        print('applying rearrangement on mapped molecules...')
        df['REARR MAPPED PROD'] = df.apply(self.apply_rearrangement_on_map_mol, axis=1)

        print('converting rearranged mapped molecules to smiles...')
        df['REARR MAPPED PROD'] = df['REARR MAPPED PROD'].apply(apply_mol_to_smiles)

        print('remove rows with empty rearranged product...')
        ind_to_keep = get_ind_non_empty_elements(df['REARR MAPPED PROD'])
        df = df.loc[ind_to_keep]
        df.reset_index(drop=True, inplace=True)

        print('adding mapped reactions to dataframe...')
        df['MAPPED REACTION'] = build_reaction_series(df['MAPPED SMILES'], df['REARR MAPPED PROD'])

        print('add unmapped reactions to dataframe...')
        df['UNMAPPED SMILES'] = get_unmapped_smiles_list(df['MAPPED SMILES'])
        df['UNMAPPED PROD'] = get_unmapped_smiles_list(df['REARR MAPPED PROD'])
        df['UNMAPPED REACTION'] = build_reaction_series(df['UNMAPPED SMILES'], df['UNMAPPED PROD'])        
        
        print('removing unnecessary columns...')
        df.drop(columns=['MAPPED SMILES', 'MAPPED MOL', 'REARR MAPPED PROD', 'UNMAPPED SMILES', 'UNMAPPED PROD'], inplace=True)

        print('saving dataframe...')
        DataHandler(df, f'{self.outputs_path}{filename}')._save_to_csv()
        return df

