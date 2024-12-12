import os

# =======================
# General Configuration
# =======================

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SMILES_FILE_PATH = './data/USPTO_rand/USPTO_rand_1.txt'

MOL_FILE_PATH = './data/USPTO_rand/USPTO_rand_1_mol.pkl'

DICT_SUBSTRUCTS = {0:'[Ch]', 1:'[Nh]'} 

DICT_CONDITIONS = {0:{0:2,1:0}, 1:{0:1, 1:1}}

OUTPUT_FILE_PATH = './results/cyclization/USPTO_rand/'

OUTPUT_FILE_NAME = 'USPTO_rand_1_cyclization.csv'
