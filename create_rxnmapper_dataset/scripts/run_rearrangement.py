import os 
import sys
import time
import argparse

# Append the parent directory to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import configurations and modules
from config_run_rearrangement import (BASE_DIR, SMILES_FILE_PATH, MOL_FILE_PATH, OUTPUT_FILE_PATH, OUTPUT_FILE_NAME)

from src.data_handler import DataHandler
from src.build_dataset import SimpleRearrangement

def parse_arguments():
    parser = argparse.ArgumentParser(description='Create a dataset of rearrangement reactions from a SMILES file and its corresponding mol-file')

    parser.add_argument('--smiles-file-path', type=str, default=SMILES_FILE_PATH, help='Path to the SMILES file (txt file)')
    parser.add_argument('--mol-file-path', type=str, default=MOL_FILE_PATH, help='Path to the mol-file (pickle file)')
    parser.add_argument('--output-file-path', type=str, default=OUTPUT_FILE_PATH, help='Path of the output folder')
    parser.add_argument('--output-file-name', type=str, default=OUTPUT_FILE_NAME, help='Name of the output file')
    return parser.parse_args()


def main():
    # parse arguments
    args = parse_arguments()

    # load the mol-file 
    datahandler = DataHandler(file_path=args.mol_file_path)
    mol_list = datahandler.load_data()

    # load the smiles list
    datahandler = DataHandler(file_path=args.smiles_file_path)
    smiles_list = datahandler.load_data()

    # initialize the SimpleRearrangement class
    simplerearrangement = SimpleRearrangement(mol_list=mol_list, smiles_list=smiles_list, outputs_path=args.output_file_path)

    _ = simplerearrangement.process(filename=args.output_file_name)
    
    print(f'Rearrangement reactions saved in {args.output_file_path}{args.output_file_name}')

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print(f'Execution time: {end_time - start_time} seconds')