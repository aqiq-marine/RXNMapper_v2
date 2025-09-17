Filters a reactions CSV by tokenized length of the 'rxn' column.

- Uses SmilesTokenizer with provided vocab
- Drops rows where tokenized length is too long


**Usage**
```bash
    python scripts/preprocess_by_token_len.py -c configs/preprocess_by_token_len_example.yaml
```

