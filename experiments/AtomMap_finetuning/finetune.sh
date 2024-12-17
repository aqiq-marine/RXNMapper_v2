#!/bin/bash


# Adapt the finetuning_baseline_AtomMaps.yaml file as you like
# you might want to choose another model you would like to tune - For now its just the V1 model. 
python scripts/finetune_model_with_atom_map_xent.py -c configs/finetuning_baseline_AtomMaps.yaml

