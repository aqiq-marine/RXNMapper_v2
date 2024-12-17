# Script to finetune a baseline Model using MLM. 

## To Finetune: 

1. Choose the dataset to finetune on.
2. Choose the base model to use. 
3. Change Hyperparameters within the configs/finetuning_baseline_MLM.yaml file. 

Simply run from the base directory: 

```bash
bash experiments/MLM_finetuning/finetune.sh
```


## To Validate

1. Make sure to add the model checkpoint you want to use within the configs/finetuning_baseline_MLM.yaml file.

2. You may have to adapt the heads/layers use depending on the base model youve used. It should be the same as the baseline model. 


Simply run from the base directory: 

```bash
bash experiments/MLM_finetuning/generate_and_test.sh
```
