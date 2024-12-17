#!/bin/bash

# Within the finetuning_baseline_MLM.yaml file, change the following: 
# model_path: Path to the saved model you want to use. 
# You may want to choose the layer/head you use for the mapping. 



# Run Model USPTO Dataset: 
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/finetuning_baseline_MLM.yaml --file_path data/RXNMapperData/Test/test_schneider.json -o experiments/MLM_finetuning/v1_baseline_model_schneider_testset.json

python scripts/validation_script.py --ground_truth_path data/RXNMapperData/Test/test_schneider.json --prediction_path experiments/MLM_finetuning/v1_baseline_model_schneider_testset.json


# Run Model On cyclisation Dataset 10/10/80 SPLIT:
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/finetuning_baseline_MLM.yaml --file_path data/Cyclisation/processed/test.csv -o experiments/MLM_finetuning/v1_baseline_model_cyclisation_101080_testset.json

python scripts/validation_script.py --ground_truth_path data/Cyclisation/processed/test.csv --prediction_path  experiments/MLM_finetuning/v1_baseline_model_cyclisation_101080_testset.json


# Run Model on Rearrangement Dataset 10/10/80 SPLIT:
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/finetuning_baseline_MLM.yaml --file_path data/Rearrangement/processed/test.csv -o experiments/MLM_finetuning/v1_baseline_model_rearrangement_101080_testset.json

python scripts/validation_script.py --ground_truth_path data/Rearrangement/processed/test.csv --prediction_path  experiments/MLM_finetuning/v1_baseline_model_rearrangement_101080_testset.json


# Run Model On cyclisation Dataset 50/50 SPLIT:
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/finetuning_baseline_MLM.yaml --file_path data/Cyclisation/processed_50-50_train_test/test.csv -o experiments/MLM_finetuning/v1_baseline_model_cyclisation_5050_testset.json

python scripts/validation_script.py --ground_truth_path data/Cyclisation/processed_50-50_train_test/test.csv --prediction_path  experiments/MLM_finetuning/v1_baseline_model_cyclisation_5050_testset.json


# Run Model on Rearrangement Dataset 50/50 SPLIT:
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/finetuning_baseline_MLM.yaml --file_path data/Rearrangement/processed_50-50_train_test/test.csv -o experiments/MLM_finetuning/v1_baseline_model_rearrangement_5050_testset.json

python scripts/validation_script.py --ground_truth_path data/Rearrangement/processed_50-50_train_test/test.csv --prediction_path  experiments/MLM_finetuning/v1_baseline_model_rearrangement_5050_testset.json





