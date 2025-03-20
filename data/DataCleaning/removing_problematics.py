# this piece of code was taken from Chen et al. LocalMapper work, you can find their Git repo here: https://github.com/snu-micc/LocalMapper and their article here: https://doi.org/10.1038/s41467-024-46364-y
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

class RemovingProblematicRxns:
    def __init__(self, df:pd.DataFrame, column_name:str):
        self.df = df
        self.column_name = column_name

        self.problematic_rxns_dict = {}
        self.problematic_rxns_set = set()
        self.filtered_df = pd.DataFrame()

    def invalid_smiles_map(self, product):
        mol = Chem.MolFromSmiles(product)
        atom_maps = []
        for i, atom in enumerate(mol.GetAtoms()):
            atom_map = atom.GetAtomMapNum()
            if atom_map == 0 or atom_map in atom_maps:
                return True
            else:
                atom_maps.append(atom_map)
        return False

    def Tanimoto_Sim(self, mol1, mol2, radius = 2):
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius)
        return DataStructs.TanimotoSimilarity(fp1, fp2)

    def contain_confusing_reagent(self, reactants, product):
        atom_mapping_pattern = r'\[[A-Za-z0-9\+\-]+:\d+\]'
        product_mol = Chem.MolFromSmiles(product)
        for reactant in reactants.split('.'):
            reactant_mol = Chem.MolFromSmiles(reactant)
            if not re.search(atom_mapping_pattern, reactant) and self.Tanimoto_Sim(reactant_mol, product_mol) >= 0.5:
                return True
        return False

    def get_problematic_rxns_dict(self):   
        problematic_rxns = {'invalid product map': [], 'confusing reagent': []}
        problematic_list = []
        invalid_product_cnt = 0
        confusing_reagent_cnt = 0
        problematic_cnt = 0
        for i, rxn in tqdm(enumerate(self.df[self.column_name]), total=len(self.df)):
            reactants, products = rxn.split('>>')
            if self.invalid_smiles_map(products):
                problematic_rxns['invalid product map'].append(rxn)
                problematic_list.append('invalid product map')
                invalid_product_cnt += 1
                problematic_cnt += 1
                continue
            if self.contain_confusing_reagent(reactants, products):
                problematic_rxns['confusing reagent'].append(rxn)
                problematic_list.append('confusing reagent')
                confusing_reagent_cnt += 1
                problematic_cnt += 1
            else:
                problematic_list.append('none')
        print ('# of problematic reactions: %d (invalid product: %d, confusing reagent: %d)' % (problematic_cnt, invalid_product_cnt, confusing_reagent_cnt))
        self.problematic_rxns_dict = problematic_rxns
    
    def unwrap_dict_values_into_set(self):
        problematic_set = set()
        for reactions in self.problematic_rxns_dict.values():
            for rxn in reactions:
                problematic_set.update(rxn.split(","))
        self.problematic_rxns_set = problematic_set

    def remove_problematic_rxns_from_df(self):
        return self.df[~self.df[self.column_name].isin(self.problematic_rxns_set)].reset_index(drop=True)

    def main(self):
        self.get_problematic_rxns_dict()
        self.unwrap_dict_values_into_set()
        self.filtered_df = self.remove_problematic_rxns_from_df()