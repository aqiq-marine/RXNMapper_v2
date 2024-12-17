# Script to finetune a baseline Model using custom Atom Mapping Loss function.

## To preprocess:

1. Make sure you are using the cyclisation data 
    - There are still some bugs with the processing of the rearrangement data. :(
2. Within the scripts/setup_atom_maps_cyclisation.py file, input the path to the CSV of your data which contains atom maps. 
3. Run that script. 

## To Finetune: 

1. Choose the dataset to finetune on - For now should only either be Cyclisation_5050 / Cyclisation_101080 :(
2. Choose the base model to use. 
3. Change Hyperparameters within the configs/finetuning_baseline_AtomMaps.yaml file. 
4. Adapt the layers/heads to use for finetuning. 


Simply run from the base directory: 

```bash
bash experiments/AtomMap_finetuning/finetune.sh
```


## To Validate

1. Make sure to add the model checkpoint you want to use within the configs/finetuning_baseline_AtomMaps.yaml file.

2. You may have to adapt the heads/layers use depending on the base model youve used. It should be the same as the baseline model. 

Simply run from the base directory: 

```bash
bash experiments/AtomMap_finetuning/generate_and_test.sh
```
