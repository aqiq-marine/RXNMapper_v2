#!/bin/bash

# Baseline Model USPTO Dataset: 
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/rxnmapper_baseline.yaml --file_path data/RXNMapperData/Test/test_schneider.json -o experiments/V1_Benchmarks/v1_baseline_model_schneider_testset.json

python scripts/validation_script.py --ground_truth_path data/RXNMapperData/Test/test_schneider.json --prediction_path experiments/V1_Benchmarks/v1_baseline_model_schneider_testset.json


# Baseline Model On cyclisation Dataset:
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/rxnmapper_baseline.yaml --file_path data/Cyclisation/processed/test.csv -o experiments/V1_Benchmarks/v1_baseline_model_cyclisation_101080_testset.json

python scripts/validation_script.py --ground_truth_path data/Cyclisation/processed/test.csv --prediction_path  experiments/V1_Benchmarks/v1_baseline_model_cyclisation_101080_testset.json


# Baseline Model on Rearrangement Dataset:
python scripts/run_rxnmapper_on_dataset.py --config_path  configs/rxnmapper_baseline.yaml --file_path data/Rearrangement/processed/test.csv -o experiments/V1_Benchmarks/v1_baseline_model_rearrangement_101080_testset.json

python scripts/validation_script.py --ground_truth_path data/Rearrangement/processed/test.csv --prediction_path  experiments/V1_Benchmarks/v1_baseline_model_rearrangement_101080_testset.json







