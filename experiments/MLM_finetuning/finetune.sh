#!/bin/bash


# Adapt the finetuning_baseline_AtomMaps.yaml file as you like - Choose your datasets and pretrained model. 
# you might want to choose another model you would like to tune - For now its just the V1 model. 
python scripts/train_mlm_alberta.py -c configs/finetuning_baseline_MLM.yaml

