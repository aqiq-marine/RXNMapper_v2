"""
Filters a reactions CSV by tokenized length of the 'rxn' column.

- Uses SmilesTokenizer with provided vocab
- Drops rows where tokenized length is too long

Usage:
    python scripts/preprocess_by_token_len.py -c configs/preprocess_by_token_len.yaml
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Dict, Tuple

import click
from datasets import load_dataset
from transformers import AlbertConfig, AlbertForMaskedLM

from rxnmapper.config_utils import parse_preprocess_tok_len_config
from rxnmapper.tokenization_smiles import SmilesTokenizer

LOGGER = logging.getLogger(__name__)


def _validate_dataset_has_rxn_header(dataset_path: Path, delimiter: str) -> Tuple[list, int]:
    with dataset_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Dataset file is empty: {dataset_path}")
    if "rxn" not in header:
        raise ValueError(f"Dataset missing required 'rxn' column. Found columns: {header}")
    return header, header.index("rxn")

def _derive_output_path(input_path: Path, max_input_len: int) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_preprocessed_len_{max_input_len}{input.suffix}")

def _get_max_input_len(config) -> int:
    max_pos = None
    if hasattr(config, "alberta_config") and isinstance(config.alberta_config, dict):
        max_pos = config.alberta_config.get("max_position_embeddings")
    if max_pos is None:
        raise ValueError(
            "Could not determine model maximum length, provide alberta_config.max_position_embeddings."
        )
    return max_pos - 2 # Take into account oth special tokens [CLS] and [SEP]

def filter_dataset_by_token_len(config) -> Path:
    """
    Stream the dataset, keep rows with len(tokenize(ex['rxn'])) <= max_input_len, write to output path.
    Returns the output file path.
    """
    dataset_path = Path(getattr(config, "dataset_path"))
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")

    vocab_path = Path(getattr(config, "vocab_path"))
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocab not found at path: {vocab_path}")

    delimiter = getattr(config, "delimiter", ",")
    header, rxn_idx = _validate_dataset_has_rxn_header(dataset_path, delimiter)

    max_input_len = _get_max_input_len(config)

    save_path = getattr(config, "save_processed_data_path", "") or ""
    if not save_path:
        out_path = _derive_output_path(dataset_path, max_input_len)
    else:
        out_path = Path(save_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading tokenizer with vocab: %s", vocab_path)
    tokenizer = SmilesTokenizer(
        str(vocab_path), 
        model_max_length=max_input_len + 2,  # +2 for CLS/SEP if used internally
        padding="longest",
        )

    kept = 0
    dropped = 0
    processed = 0

    LOGGER.info("Filtering with max_input_len=%d (excludes special tokens).", max_input_len)
    with dataset_path.open("r", newline="", encoding="utf-8") as fin, \
         out_path.open("w", newline="", encoding="utf-8") as fout:
        reader = csv.DictReader(fin, delimiter=delimiter)
        if "rxn" not in reader.fieldnames:
            # should already be caught by header check
            raise ValueError(f"'rxn' not found in columns: {reader.fieldnames}")
        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames, delimiter=",")
        writer.writeheader()

        for row in reader:
            processed += 1
            rxn = row["rxn"]
            # NOTE: SmilesTokenizer.tokenize returns a list of tokens
            tok_len = len(tokenizer.tokenize(rxn))
            if tok_len <= max_input_len:
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1
            if processed % 100000 == 0:
                LOGGER.info("Processed=%d kept=%d dropped=%d", processed, kept, dropped)

    LOGGER.info("Done. Processed=%d kept=%d dropped=%d Output=%s", processed, kept, dropped, out_path)
    return out_path    



@click.command()
@click.option(
    "--config_path",
    "-c",
    required=True,
    help="Path to preprocessing config (can be the training config).",
    type=str,
)
def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_preprocess_tok_len_config(config_path=config_path)
    filter_dataset_by_token_len(config)


if __name__ == "__main__":
    main()