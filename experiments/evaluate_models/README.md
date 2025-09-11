# Evaluate several models on a given dataset with provided ground truth

Add all the AlBERT models to evaluate in `configs/models_to_eval.yaml` similarly to the examples.

Run (ideally with allocated gpu for inference, cpus for evaluation):
```bash
python scripts/evaluate_models.py \
--csv-file path/to/csv \
--rxn-col column with rxn to map \
--ref-col gorund truth column \
--models-yaml ./configs/models_to_eval.yaml \
--out-csv path/output/csv
```
