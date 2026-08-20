# RXNMapper v2

**Updated fast and rule-free atom mapping for chemical reactions with a complete, reproducible pipeline to train and select your own model.**

RXNMapper v2 builds on the original [RXNMapper](https://github.com/rxn4chemistry/rxnmapper) (Schwaller *et al.*, *Science Advances* 2021), which showed that atom mapping can be learned **unsupervised** from reaction SMILES: an ALBERT model trained via Masked Language Modeling (MLM) on unmapped reaction SMILES grasps which atoms in the reactants correspond to which atoms in the products.

This version adds:

1. **Retrained models** released openly on the Hugging Face Hub.
2. **A full training-from-scratch pipeline** build a vocabulary, preprocess data, pretrain an ALBERT model, and **select the best model on your own validation set**.
3. **Graph-isomorphism evaluation** of mapping quality via [`atommap-eval`](https://pypi.org/project/atommap-eval/).

---

## Contents

- [Installation](#installation)
- [Quickstart: map reactions in Python](#quickstart-map-reactions-in-python)
- [Map an entire dataset](#map-an-entire-dataset)
- [Pretrained models](#pretrained-models)
- [Train your own model from scratch](#train-your-own-model-from-scratch)
- [Evaluate models on a validation set](#evaluate-models-on-a-validation-set)
- [Data](#data)
- [Repository structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

---

## Installation

Clone the repository and install it in editable mode. Python ≥ 3.8 is supported; 3.11 is recommended.

**Using [uv](https://docs.astral.sh/uv/) (recommended):**
```bash
git clone https://github.com/yvsgrndjn/RXNMapper_v2
cd RXNMapper_v2
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

**Using conda:**
```bash
git clone https://github.com/yvsgrndjn/RXNMapper_v2
cd RXNMapper_v2
conda create -n rxnmapper_v2 python=3.11 -y
conda activate rxnmapper_v2
pip install -e .
```

---

## Quickstart: map reactions in Python

`RXNMapper` loads a packaged default model and maps a list of reaction SMILES:

```python
from rxnmapper import RXNMapper

rxn_mapper = RXNMapper()
rxns = [
    'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F',
    'C1COCCO1.CC(C)(C)OC(=O)CONC(=O)NCc1cccc2ccccc12.Cl>>O=C(O)CONC(=O)NCc1cccc2ccccc12',
]
results = rxn_mapper.get_attention_guided_atom_maps(rxns)
```

Each result contains the mapped reaction and a confidence score:

```python
[{'confidence': 0.6908274969652951,
  'mapped_rxn': '[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]'},
 {'confidence': 0.9687064395754957,
  'mapped_rxn': 'C1COCCO1.CC(C)(C)[O:3][C:2](=[O:1])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12.Cl>>[O:1]=[C:2]([OH:3])[CH2:4][O:5][NH:6][C:7](=[O:8])[NH:9][CH2:10][c:11]1[cH:12][cH:13][cH:14][c:15]2[cH:16][cH:17][cH:18][cH:19][c:20]12'}]
```

By default this uses the **`alberta-uspto-2800k`** model (head 3, layer 10). To use the original RXNMapper model instead, pass `model_name="original"`:

```python
rxn_mapper = RXNMapper(config={"model_name": "original"})   # original model (head 5, layer 10)
```

For large inputs and automatic error handling, use `BatchedMapper`. It never raises on invalid input, but returns `">>"` (or an empty dict) for reactions it cannot map:

```python
from rxnmapper import BatchedMapper

rxn_mapper = BatchedMapper(batch_size=32)                   # alberta-uspto-2800k by default
# rxn_mapper = BatchedMapper(batch_size=32, model_name="original")  # original model
rxns = ['CC[O-]~[Na+].BrCC>>CCOCC', 'invalid>>reaction']

results = list(rxn_mapper.map_reactions(rxns))            # mapped reaction strings
results = list(rxn_mapper.map_reactions_with_info(rxns))  # dicts with confidence (as above)
```

---

## Map an entire dataset

`scripts/run_rxnmapper_on_dataset.py` maps a `.csv`, `.tsv`, or `.json` file that has a `rxn` column, and writes a JSON of results.

```bash
python scripts/run_rxnmapper_on_dataset.py \
    --file_path path/to/reactions.csv \
    --output_path out.json \
    --batch_size 8
```

| Option | Alias | Default | Description |
| --- | --- | --- | --- |
| `--file_path` | `-f` | *required* | Input `.csv` / `.tsv` / `.json` with a `rxn` column |
| `--output_path` | `-o` | *required* | Output JSON path |
| `--batch_size` | `-bs` | `1` | Reactions per batch |
| `--canonicalize/--no_canonicalize` | | `True` | Canonicalize inputs before mapping |
| `--detailed` | `-d` | `False` | Include detailed per-atom output |

---

## Pretrained models

`RXNMapper()` with no arguments uses the packaged **default** model (`alberta-uspto-2800k`, head 3, layer 10); pass `model_name="original"` for the original RXNMapper model. Both ship inside the package, so they work offline. To use a different model (e.g. one you downloaded from the Hub or trained yourself), point the scripts at its folder via the `model_path` field in the relevant config.

| Model | Training data | Hub |
| --- | --- | --- |
| `alberta-uspto-2800k` | USPTO (~2.8M reactions), pretrained from scratch | [yvsgrndjn/alberta-uspto-2800k](https://huggingface.co/yvsgrndjn/alberta-uspto-2800k) |


The repository hosts all 150 training checkpoints (steps 10k–1.5M) as `checkpoint-<step>/` subfolders, plus the recommended `checkpoint-1360000` (head 3, layer 10) at the root. Download it with:

```bash
hf download yvsgrndjn/alberta-uspto-2800k --local-dir alberta-uspto-2800k
```

The downloaded folder matches the layout the scripts expect: point `model_path` at a checkpoint folder and `vocab_path` at the root `vocab.txt`.

---

## Train your own model from scratch

Retrain a model when the default vocabulary or chemistry does not match your data. Every step is driven by a small YAML config in `configs/`; copy the matching `*_example.yaml`, edit the paths, and run.

### 1. Build a vocabulary tailored to your data

```bash
python scripts/create_vocab.py -c configs/create_vocab_example.yaml
```
Builds a token vocabulary from the `rxn` column of your dataset, optionally merging it with the default vocabulary.

### 2. Preprocess by token length

```bash
python scripts/preprocess_by_token_len.py -c configs/preprocess_by_token_len_example.yaml
```
Tokenizes the reactions and drops those longer than the model's `max_position_embeddings` (default 512), which the model cannot fit.

### 3. Pretrain the ALBERT model (masked language modeling)

```bash
python scripts/train_mlm_alberta_stream.py -c configs/training_from_scratch_example.yaml
```
Pretrains an ALBERT model by masked language modeling, saving checkpoints periodically. Use `train_mlm_alberta_stream.py` for datasets streamed from disk (large data); use `train_mlm_alberta.py` (same config) when the dataset fits in memory.

### 4. Select the best head / layer

The atom-mapping signal lives in **one attention head of one layer**. This step searches head/layer combinations against your validation set (a CSV with an unmapped `rxn` column and a ground-truth `mapped_rxn` column) and reports the best one (focus on layers 9 and 10 first):

```bash
python scripts/find_best_head_layer.py -c configs/find_best_head_layer_example.yaml
```

Validation accuracy can be noisy, so several head/layer/checkpoint combinations may score similarly; this finds a strong local optimum. To then compare candidate **checkpoints** (or any set of models) head-to-head on your validation set, list them in `configs/models_to_eval_example.yaml` and run the evaluation below: the highest-scoring entry is your selected model.

### 5. Select the training checkpoint

At last, once the best head/layer combination was found given the validation set, the best training checkpoint can be determined by modifying `configs/find_best_checkpoint_example.yaml` and running:
```bash
python scripts/find_best_training_checkpoint.py -c configs/find_best_checkpoint_example.yaml
```

## Find the best model for your data
if the vocab from your specific data is similar to the one provided in this repo, it will probably be faster to:
1. design a broad validation set with trustworthy atom-maps.
2. screen the pretrained models to find the best head/layer/checkpoint given your validation set.
3. depending on the performance, consider some finetuning using the best model so far. (not shown in this repo)

---

## Evaluate models on a validation set

`scripts/evaluate_models.py` maps a validation set with one or more model / head / layer configurations and scores each against ground-truth atom maps using reaction-graph isomorphism ([`atommap-eval`](https://pypi.org/project/atommap-eval/)).

```bash
python scripts/evaluate_models.py -c configs/models_to_eval_example.yaml
```

The config declares the validation set, an optional `defaults:` block inherited by every model, and the `models:` to compare:

```yaml
dataset_path: "/path/to/val_set.csv"
rxn_col: "rxn"                 # unmapped reactions column
ref_col: "mapped_rxn"          # ground-truth atom-mapped column
batch_size: 32
output_path: "results/model_eval.csv"

defaults:                      # inherited by every model unless overridden
  model_type: "albert"
  vocab_path: "/path/to/vocab.txt"
  attention_multiplier: 90

models:
  original:
    model_path: "/path/to/albert_heads_8_uspto_all_1310k"
    head: 5
    layer: 10
  uspto_2800k:
    model_path: "/path/to/alberta_uspto_2800k/checkpoint-1360000"
    head: 3
    layer: 10
```

Outputs:
- **`output_path`** detailed per-model predictions in long (tidy) format (`model, index, ground_truth_rxn, predicted_rxn, success`).
- **`<output_path stem>_summary.csv`** accuracy per model, best first.
- The same accuracy summary is printed to the console.

---

## Data

The dataset used to pretrain both original and updated model `alberta-uspto-2800k` (USPTO, ~2.8M reactions) is available at [IBM Box](https://ibm.box.com/v/RXNMapperData).

---

## Repository structure

```
rxnmapper/        # the installable Python package (mapper, tokenizer, evaluators)
scripts/          # command-line entry points for each pipeline step
configs/          # *_example.yaml templates for the scripts
tests/            # test suite
```

---

## Citation

If you use this work, please cite our work:

```bibtex
@article{grandjeanRXNMapperv2UpdatedValidation2026
      title={RXNMapperv2: updated validation and evaluation on expanded chemical space}, 
      author={Grandjean, Yves and Raffaud, Sacha, and Westerlund, Annie M. and Schwaller, Philippe and Genheden, Samuel and Reymond, Jean-Louis},
      journal={ChemRxiv}, 
      year={2026},
      doi={10.26434/chemrxiv.15005247/v1}
      URL={https://chemrxiv.org/doi/full/10.26434/chemrxiv.15005247/v1},
      note= {Preprint}
}
```
and the original RXNMapper paper:

```bibtex
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

## License

Released under the MIT License : see [LICENSE](./LICENSE).
