# Adaptation of Extraction of organic chemistry grammar from unsupervised learning of chemical reactions - Round 2 :) 

Enable robust atom mapping on valid reaction SMILES. The atom-mapping information was learned by an ALBERT model trained in an unsupervised fashion on a large dataset of chemical reactions - Second round :) 

## What this version includes: 
  1. Training scripts
  2. Evaluation Scripts
  3. New Datasets :) 
  4. And other experiments. 


## Installation

### From pip
```console
conda create -n rxnmapper_v2 python=3.8 -y
conda activate rxnmapper_v2
pip install -e .
pip install rdkit
pip install transformers[torch]
pip install wandb
```

### From github
You can install the package and setup the environment directly from github using:

```console
git clone https://github.com/yvsgrndjn/RXNMapper_v2 
cd RXNMapper_v2
conda create -n rxnmapper_v2 python=3.8 -y
conda activate rxnmapper_v2
pip install rxnmapper_v2
pip install rdkit
pip install transformers[torch]
pip install wandb
```


## Usage

### Basic usage

```python
from rxnmapper_v2 import RXNMapper
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
from rxnmapper_v2 import BatchedMapper
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

Data can be found at: https://ibm.box.com/v/RXNMapperData

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
