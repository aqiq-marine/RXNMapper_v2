import os
import sys
import time
import logging
import argparse

# Append the parent directory to sys.path to import modules from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from configs.config_calc_levenshtein_tanimoto import BASE_DIR, DF_PATH, REACTION_COL_NAME
from rxnmapper.levenshtein_tanimoto_calc import CalcDistances

def parse_arguments():
    parser = argparse.ArgumentParser(description='Calculate Levenshtein distance and Tanimoto similarity between reactions reactants and products')

    # Optional arguments to override config settings
    parser.add_argument('--file_path', type=str, default=DF_PATH, help='Path to the dataframe containing the reactions')
    parser.add_argument('--rxn_col', type=str, default=REACTION_COL_NAME, help='Name of the column containing the reactions')
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_arguments()

    # Initialize logger
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)
    
    # initialize the CalcDistances class
    calculate_distances = CalcDistances(args.file_path, args.rxn_col)


    start_time = time.time()
    calculate_distances.calc_levenshtein_dist_df()
    logger.info(f'Levenshtein distance calculation completed in {time.time() - start_time:.2f} seconds')

    calculate_distances.calc_tanimoto_sim_df()
    logger.info(f'Tanimoto similarity calculation completed in {time.time() - start_time:.2f} seconds')

if __name__ == '__main__':
    main()