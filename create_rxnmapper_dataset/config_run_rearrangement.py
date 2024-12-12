import os

# =======================
# General Configuration
# =======================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SMILES_FILE_PATH = './data/USPTO_rand/USPTO_rand_1.txt'

MOL_FILE_PATH = './data/USPTO_rand/USPTO_rand_1_mol.pkl'

OUTPUT_FILE_PATH = './results/rearrangement/USPTO_rand/'

OUTPUT_FILE_NAME = 'USPTO_rand_1_rearrangement.csv'