# Adaptation of Extraction of organic chemistry grammar from unsupervised learning of chemical reactions - Version 2 :) 

Enable robust atom mapping on valid reaction SMILES. The atom-mapping information was learned by an ALBERT model trained in an unsupervised fashion on a large dataset of chemical reactions - Second round :) 

## What this version includes: 
  1. Training scripts
  2. Evaluation Scripts
  3. New Datasets :) 
  4. And other experiments. 


## Installation

### From github
You can install the package and setup the environment directly from github using:

```console
git clone https://github.com/yvsgrndjn/RXNMapper_v2 
cd RXNMapper_v2
conda create -n rxnmapper_v2 python=3.8 -y
conda activate rxnmapper_v2
pip install -e .
```

## Usage

### Basic usage

```python
from rxnmapper import RXNMapper
rxn_mapper = RXNMapper()
rxns = ['CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F', 'C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12']
results = rxn_mapper.get_attention_guided_atom_maps(rxns)
```

The results contain the mapped reactions and confidence scores:

```python
[{'mapped_rxn': 'CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[CH3:1][CH:2]([CH3:3])[SH:4].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]',
  'confidence': 0.9565619900376546},
 {'mapped_rxn': 'C1COCCO1.CC(C)(C)[O:3][C:2](=[O:1])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12.Cl>>[O:1]=[C:2]([OH:3])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12',
  'confidence': 0.9704424331552834}]
```

To account for batching and error handling automatically, you can use `BatchedMapper` instead:
```python
from rxnmapper import BatchedMapper
rxn_mapper = BatchedMapper(batch_size=32)
rxns = ['CC[O-]~[Na+].BrCC>>CCOCC', 'invalid>>reaction']

# The following calls work with input of arbitrary size. Also, they do not raise 
# any exceptions but will return ">>" or an empty dictionary for the second reaction.
results = list(rxn_mapper.map_reactions(rxns))  # results as strings directly
results = list(rxn_mapper.map_reactions_with_info(rxns))  # results as dictionaries (as above)
```

## Examples

To learn more see the [examples](./examples).

## Data 
### Original IBM data (USPTO 2.8M reactions + USPTO-50K)
1. Data can be found at: https://ibm.box.com/v/RXNMapperData
2. Download the files and place them in the data-directory
3. Unzip the zip file: 

```bash
cd data
unzip RXNMapperData.zip
rm RXNMapperData.zip
cd ..
```

4. Process the data: 

- This will clean the USPTO data (Only keep rxns within the model size)
- Split Cyclisation data
- Split Rearrangement data
- Combine all the training data into one big pre-training dataset. 

```bash
python scripts/process_data.py
 ```

### Localmapper USPTO version (1.06M reactions)
1. Data can be found at: figshare <https://figshare.com/articles/dataset/USPTO_reaction_datasets_remapped_by_LocalMapper/25046471/1>
2. Download the files and place them in the newly created data/USPTO_FULL-directory
```bash
cd data
mkdir USPTO_FULL
```
3. Unzip the zip file
```bash
cd USPTO_FULL
unzip USPTO_full_and_50K_remapped.zip
rm USPTO_full_and_50K_remapped.zip
cd ..
```
4. Preprocess the data:
```bash
python scripts/remove_mapping.py
```
5. Obtain the Golden dataset from Localmapper <https://github.com/snu-micc/LocalMapper/tree/main/data/Golden> and save it under data/Golden/test_data.csv

6. Go to data/DataCleaning/cleaning_dataset.ipynb and get the cleaned versions of USPTO-49K (test set) and the validation set (USPTO-1K)

7. Process USPTO (1.06M rxns) for training
```bash
python scripts/process_localmapper_data.py
```

<!-- 

## Validating the model:

The current validation algorithm does not account for equivalent carbons. This still needs to be implemented :). 

1. Run RXNMapper on the validation data: 

```bash
python scripts/run_rxnmapper_on_dataset.py --config_path configs/rxnmapper_baseline.yaml --file_path data/RXNMapperData/Validation/eval_schneider.json -o results/v1_baseline_validation_dataset.json
```

Or to run on the cyclisation/Rearangement datasets: 

```bash
python scripts/run_rxnmapper_on_dataset.py --config_path configs/rxnmapper_baseline.yaml --file_path data/Cyclisation/processed/test.csv -o results/v1_baseline_cyclisation_test_dataset.json
```

```bash
python scripts/run_rxnmapper_on_dataset.py --config_path configs/rxnmapper_baseline.yaml --file_path data/Rearrangement/processed/test.csv -o results/v1_baseline_rearrangement_test_dataset.json
```


2. Run Validation of results: 

```bash
python scripts/validation_script.py --ground_truth_path data/RXNMapperData/Validation/eval_schneider_processed.json --prediction_path results/v1_baseline_validation_dataset.json
```

Or to run on the cyclisation/Rearangement datasets:

```bash
python scripts/validation_script.py --ground_truth_path data/Cyclisation/processed/test.csv --prediction_path results/v1_baseline_cyclisation_test_dataset.json
```
```bash
python scripts/validation_script.py --ground_truth_path data/Rearrangement/processed/test.csv --prediction_path results/v1_baseline_rearrangement_test_dataset.json -->
<!-- ``` -->
 <!-- --> 

<!-- ## Training a model: 

1. Once the training data has been processed, Simply run: 


```bash
python scripts/train_mlm_alberta.py -c configs/training_baseline_full_pt.yaml
```

2. To finetune the original model with MLM on new data, run : 

```bash
python scripts/train_mlm_alberta.py -c configs/finetuning_baseline.yaml
```

3. To finetune with Atom Mapping (Script is not fully-automated) and you will have to digest the code and simply just run the main function. 

TO be cleaned at a later point: 
  1. Adapt the Traner class to only take in the Atom-map dataset. 


 -->


<!-- ## Finetuning for atom mapping:  -->

## Running Experiments: 

Head over to the Experiments Directory, and have a look at the different scripts which can be used: 

1. Run Benchmarks on V1 model
2. Train Model from scratch
3. Finetune a model with MLM
4. Finetune a model with a Custom Atom-Mapping loss function. 


<!-- # Current Benchmarks:

## V1 Model: 

USPTO Schneider: 

- 96.2%

Cyclisation 10/10/80:

- 93.4%

Cyclisation 50/50:

- 93.5%

Rearrangement 10/10/80: 

- 82.8%

Rearrangement 50/50:

- 82.9%

# V2 Model: Pretrained from scratch with USPTO + 10% Cyclisation (3k) + 10% Rearrangement (13k)


USPTO Schneider Test Set: 

- 95.8%

Cyclisation Test Set: 

- 95.3%

Rearrangement Test Set: 

-xx



## Cyclisation data: 10/10/80 Split:

RXNmapper V1 - No finetuning: 

- 93.4%

RXNMapper V1 + MLM finetuning on Cyclisation-only Data (3,074 Training samples):

- 95.4%

RXNMapper V1 + Atom-Map finetuning on Cyclisation-only Data:

- 92.1%
- Worst results than MLM finetuning.

## Rearrangement Data 10/10/80 Split:

RXNmapper V1 - No finetuning:

- 82.8%

RXNMapper V1 + MLM finetuning on Cyclisation-only Data (13,000 Training samples):

- 91.3%



## Cyclisation data: 50/50 Split:

Test Size - Samples: 13,665

RXNmapper V1 - No finetuning: 

- 93.5%

RXNMapper V1 + MLM finetuning on Cyclisation-only Data (13,665 samples) :

- 96.5%

RXNMapper V1 + Atom-map Finetuning (13,665 samples):

- 96.5%


## Rearrangement data: 50/50 Split

Test Size -  samples: 57,783

RXNMapper V1 - No finetuning:

- 82.9%

RXNMapper V1 + MLM finetuning on Rearrangement-only Data (57,783):

- 92.1% -->




# Current ISSUES: 

- Processing of Atom maps has BUG for the Rearrangement RXNs - TO FIX
- Would like to see WHERE the atom map is wrong - Is it on the reacting atoms or scattered across reaction.
- Clean code so easily reproducible



## Future experiments: 

1. Run with randomised smiles -> Larger Levenshtein distance
2. Run with LOADS of Extra cyclisation and rearrangement reactions
3. Finetune on Reacting atoms only -> These could be the hardest for RXNMapper to map. 








## Citation

```
@article{schwaller2021extraction,
  title={Extraction of organic chemistry grammar from unsupervised learning of chemical reactions},
  author={Schwaller, Philippe and Hoover, Benjamin and Reymond, Jean-Louis and Strobelt, Hendrik and Laino, Teodoro},
  journal={Science Advances},
  volume={7},
  number={15},
  pages={eabe4166},
  year={2021},
  publisher={American Association for the Advancement of Science}
}
```
