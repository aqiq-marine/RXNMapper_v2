import os
import sys
import pandas as pd
from rdkit import Chem
from rdkit.Chem import DataStructs
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect

from rapidfuzz.distance import Levenshtein
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from rxnmapper.data_handler import DataHandler
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class CalcDistances:
    def __init__(self, 
                 file_path, 
                 rxn_col='rxn',
                 mapped = False):
        
        self.file_path = file_path
        self.rxn_col = rxn_col
        self.mapped = mapped

        self.num_workers = os.cpu_count()-2

    @staticmethod
    def smiles_to_mol(smiles:str):
        try:
            return MolFromSmiles(smiles)
        except:
            return None
    
    @staticmethod
    def canonicalize_smiles(smiles:str)->str:
        try:
            mol = MolFromSmiles(smiles)
            return MolToSmiles(mol, canonical=True) if mol else None
        except:
            return None

    @staticmethod
    def parallel_apply_batches(series, func, batch_size=100_000, num_workers=8):
        results = []
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in range(0, len(series), batch_size):
                batch = series.iloc[i:i+batch_size]
                results.extend(executor.map(func, batch))
        return pd.Series(results, index=series.index)

    @staticmethod
    def split_rxns_series(rxn_series:pd.Series)->pd.Series:
        split_series = rxn_series.str.split('>>', expand=True)
        return split_series[0], split_series[1]

    @staticmethod
    def calc_lvshtn_dist_strings(str1,str2):
        return Levenshtein.distance(str1,str2)

    @staticmethod    
    def compute_morgan_fp(mol, radius=2, nBits=2048):
        return GetMorganFingerprintAsBitVect(mol, radius, nBits)

    @staticmethod
    def calc_tanimoto_sim(pair):
        return DataStructs.TanimotoSimilarity(pair[0], pair[1])

    def canonicalize_rxn(self, rxn):
        try:
            reactants, products = rxn.split('>>')
            reactants = sorted([self.canonicalize_smiles(r) for r in reactants.split('.')])
            products = sorted([self.canonicalize_smiles(p) for p in products.split('.')])
            return '>>'.join(['.'.join(reactants), '.'.join(products)])
        except:
            return None    
    
    def calc_lvshtn_dist_pair(self, pair):
        return self.calc_lvshtn_dist_strings(pair[0], pair[1])

    def rxn_to_mol_reac_and_prods(self, rxn):
        try:
            reactants, products = rxn.split('>>')
            reactants = self.smiles_to_mol(reactants)
            products = self.smiles_to_mol(products)
            return reactants, products
        except:
            return None

    @staticmethod
    def get_reac_reag_rightside_rxn(reaction_smiles:str)->tuple:
        '''
        Separate reactants, reagents and products from a reaction mapped SMILES string
        '''
        try:
            # split the different reactants/reagents and products
            left_smiles, right_smiles = reaction_smiles.split(">>")

            # split both reaction sides into individual molecules
            left_molecules = left_smiles.split(".")
            right_molecules = right_smiles.split(".")

            # convert right molecules (=products) into RDKit mol objects
            right_mols = [Chem.MolFromSmiles(smiles) for smiles in right_molecules]

            # extract atom mapping from the product molecules
            mapped_atoms_in_products = set()
            for mol in right_mols:
                if mol:
                    for atom in mol.GetAtoms():
                        if atom.GetAtomMapNum() > 0:
                            mapped_atoms_in_products.add(atom.GetAtomMapNum())

            # classify left molecules into reactants and reagents
            reactants = []
            reagents = []
            for smiles in left_molecules:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    # check if any atom in the molecule is mapped to the products
                    has_mapped_atoms = any(
                        atom.GetAtomMapNum() in mapped_atoms_in_products for atom in mol.GetAtoms()
                    )
                    if has_mapped_atoms:
                        reactants.append(smiles)
                    else:
                        reagents.append(smiles)
            return reactants, reagents, right_molecules
        except:
            return None

    def get_reac_reag_rightside_rxn_list(self, rxn_list:list)->list:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.get_reac_reag_rightside_rxn, rxn_list))
        return results

    @staticmethod
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

    def get_unmapped_smiles_list(self, smiles_list:list)->list:
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(self.get_unmapped_smiles, smiles_list))
        return results
    
    @staticmethod
    def rebuild_rxn_series(reactants_series, products_series):
        return reactants_series.str.cat([">>"] * len(reactants_series)).str.cat(products_series)

    def remove_reagents_from_mapped_rxn_list(self, rxn_list):

        # separate reactants, reagents and products from the mapped reaction SMILES
        results = self.get_reac_reag_rightside_rxn_list(rxn_list)

        # create a dataframe from the results
        df_reac_reag_prod = pd.DataFrame(results, columns=['reactants', 'reagents', 'products'])
        df_reac_reag_prod = df_reac_reag_prod.applymap(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else x)

        df_reac_reag_prod['reactants_nomap'] = self.get_unmapped_smiles_list(df_reac_reag_prod['reactants'].values.tolist())
        df_reac_reag_prod['products_nomap'] = self.get_unmapped_smiles_list(df_reac_reag_prod['products'].values.tolist())
        
        reactions = self.rebuild_rxn_series(df_reac_reag_prod['reactants_nomap'], df_reac_reag_prod['products_nomap'])

        return reactions

    def calc_levenshtein_dist_df(self):
        
        # load dataframe 
        print(f'Loading data from {self.file_path}')
        data_handler = DataHandler(file_path=self.file_path)
        df = data_handler.load_data()
        
        # get the reactions without reagents if mapped
        if self.mapped:
            print('Removing reagents from the reactions...')
            df[self.rxn_col] = self.remove_reagents_from_mapped_rxn_list(df[self.rxn_col].values.tolist())

        # canonicalize + sort each side of the reaction
        print('Canonicalizing reactions...')
        df['can_rxn'] = self.parallel_apply_batches(df[self.rxn_col], self.canonicalize_rxn, num_workers=self.num_workers)
        
        # drop rows with None values
        print('Dropping rows with None values...')
        df.dropna(subset=['can_rxn'], inplace=True, ignore_index=True)

        # split reactions into reactants and products series
        print('Splitting reactions into reactants and products...')
        df['reactants'], df['products'] = self.split_rxns_series(df['can_rxn'])

        # # create a series of both reactants and products series
        print('Creating the series of reactants and products...')
        zip_series = pd.Series(zip(df['reactants'], df['products']))

        # calculate levenshtein distance between reactants and products
        print('Calculating Levenshtein distance...')
        df['lev_dist'] = self.parallel_apply_batches(zip_series, self.calc_lvshtn_dist_pair, num_workers=self.num_workers)

        avg_lev_dist = df['lev_dist'].mean()
        med_lev_dist = df['lev_dist'].median()
        print(f'Average Levenshtein distance: {avg_lev_dist.__round__(3)}')
        print(f'Median Levenshtein distance: {med_lev_dist.__round__(3)}')

        # save the dataframe with levenshtein distances

    def calc_tanimoto_sim_df(self):

        # load dataframe 
        print(f'Loading data from {self.file_path}')
        data_handler = DataHandler(file_path=self.file_path)
        df = data_handler.load_data()

        # get the reactions without reagents if mapped
        if self.mapped:
            print('Removing reagents from the reactions...')
            df[self.rxn_col] = self.remove_reagents_from_mapped_rxn_list(df[self.rxn_col].values.tolist())

        # split reactions into reactants and products series
        print('Splitting reactions into reactants and products...')
        df['reactants'], df['products'] = self.split_rxns_series(df[self.rxn_col])

        # calculate mol objects for reactants and products
        print('Calculating mol objects for reactants...')
        df['reactants_mol'] = self.parallel_apply_batches(df['reactants'], self.smiles_to_mol, num_workers=self.num_workers)
        print('Calculating mol objects for products...')
        df['products_mol'] = self.parallel_apply_batches(df['products'], self.smiles_to_mol, num_workers=self.num_workers)

        # remove None values from the dataframe
        df.dropna(how='any', inplace=True, ignore_index=True)

        # calculate morgan fingerprints for reactants and products
        print('Calculating Morgan fingerprints for reactants...')
        df['reac_fp'] = self.parallel_apply_batches(df['reactants_mol'], self.compute_morgan_fp, num_workers=self.num_workers)
        print('Calculating Morgan fingerprints for products...')
        df['pro_fp'] = self.parallel_apply_batches(df['products_mol'], self.compute_morgan_fp, num_workers=self.num_workers)
        
        # create a series of both reactants and products series
        zip_series = pd.Series(zip(df['reac_fp'], df['pro_fp']))

        # calculate tanimoto similarity
        print('Calculating Tanimoto similarity...')
        df['tanimoto_sim'] = self.parallel_apply_batches(zip_series, self.calc_tanimoto_sim, num_workers=self.num_workers)

        avg_tanimoto_sim = df['tanimoto_sim'].mean()
        med_tanimoto_sim = df['tanimoto_sim'].median()

        print(f'Average Tanimoto similarity: {avg_tanimoto_sim.__round__(3)}')
        print(f'Median Tanimoto similarity: {med_tanimoto_sim.__round__(3)}')

        # save the dataframe with tanimoto similarities   