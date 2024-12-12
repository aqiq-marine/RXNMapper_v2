import os 
import sys
import time
import argparse

# Append the parent directory to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations and modules

from config_run_cyclization import (BASE_DIR, SMILES_FILE_PATH, MOL_FILE_PATH, DICT_SUBSTRUCTS, DICT_CONDITIONS, OUTPUT_FILE_PATH, OUTPUT_FILE_NAME)
from src.data_handler import DataHandler
from src.build_dataset import AtomsToBond
from src.build_dataset import BondAtoms

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a dataset of cyclization reactions from a SMILES file and its corresponding mol-file')

    parser.add_argument('--smiles-file-path', type=str, default=SMILES_FILE_PATH, help='Path to the SMILES file (txt file)')
    parser.add_argument('--mol-file-path', type=str, default=MOL_FILE_PATH, help='Path to the mol-file (pickle file)')
    parser.add_argument('--dict-substructs', type=dict, default=DICT_SUBSTRUCTS, help='Dictionary of substructures to be used for the cyclization')
    parser.add_argument('--dict-conditions', type=dict, default=DICT_CONDITIONS, help='Dictionary of conditions to be used for the cyclization')
    parser.add_argument('--output-file-path', type=str, default=OUTPUT_FILE_PATH, help='Path of the output folder')
    parser.add_argument('--output-file-name', type=str, default=OUTPUT_FILE_NAME, help='Name of the output file')
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_arguments()

    # load the mol-file to get the indices of the atoms to bond to each other
    datahandler = DataHandler(file_path=args.mol_file_path)
    mol_list = datahandler.load_data()

    # initialize the AtomsToBond class
    atomstobond = AtomsToBond(mol_list, args.dict_substructs, args.dict_conditions)

    # obtain the list of tuples ( of atoms' indices) for each molecule in the dataset
    ind_tuple = atomstobond.get_indices_of_atoms_to_bond_mol_list()

    # load the smiles list to create the cyclization reactions
    datahandler = DataHandler(file_path=args.smiles_file_path)
    smiles_list = datahandler.load_data()

    # initialize the BondAtoms class
    bondatoms = BondAtoms(mol_list=mol_list, smiles_list=smiles_list, atoms_indices=ind_tuple, outputs_path=args.output_file_path)

    # create and save the dataframe containing the cyclization reactions
    _ = bondatoms.create_and_save_cyclization_df(filename=args.output_file_name)

    print(f'Cyclization reactions saved in {args.output_file_path}{args.output_file_name}')


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds')

