"""
Generate a vocab based on a given dataset of reaction SMILES (column: 'rxn').
Optionally merge with an existing/default vocab.


Usage:
    python scripts/create_vocab.py -c configs/create_vocab_example.yaml
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable

import click
from datasets import load_dataset

from rxnmapper.config_utils import parse_create_vocab_config
from rxnmapper.tokenization_smiles import (build_vocab_from_data_stream, 
                                           merge_ordered_vocabs, 
                                           list_to_ord_dict, 
                                           load_vocab, 
                                           save_vocabulary
)

LOGGER = logging.getLogger(__name__)


def _validate_paths_and_config(config) -> None:
    dataset_path = Path(config.dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found at path: {dataset_path}")
    
    if not getattr(config, "save_vocab_path", ""):
        raise ValueError("Config 'save_vocab_path' must be a non-empty file path.")
    save_path = Path(config.save_vocab_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(config, "merge_new_with_default", False):
        default_path = getattr(config, "default_vocab_path", "")
        if not default_path:
            raise ValueError(
                "merge_new_with_default=True but 'default_vocab_path' is empty. "
                "Provide a valid vocab path."
            )
        default_file = Path(default_path)
        
        if not default_file.is_file():
            raise FileNotFoundError(
                f"merge_new_with_default=True but default vocab file does not exist: {default_file}"
            )

def _validate_dataset_has_rxn_header(dataset_path: Path, delimiter: str) -> None:
    # read only header
    with dataset_path.open("r", newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Dataset file is empty: {dataset_path}")
    if "rxn" not in header:
        raise ValueError(
            f"Dataset missing required 'rxn' column, Found columns: {header}"
        )
    
def _rxn_stream(dataset_path: Path, delimiter: str) -> Iterable[str]:
    ds_stream = load_dataset(
        "csv",
        data_files=str(dataset_path),
        split="train",
        streaming=True,
        delimiter=delimiter,
    )
    for ex in ds_stream:
        yield ex["rxn"]

def create_vocab_from_config(config) -> Path:
    """
    Core function: builds a vocab from the dataset's 'rxn' column and optionally merges it 
    with an existing vocab. Returns the path to the saved vocab file.
    """
    dataset_path = Path(config.dataset_path)
    delimiter = getattr(config, "delimiter", ",")
    save_path = Path(config.save_vocab_path)
    merge_with_default = bool(getattr(config, "merge_new_with_default", False))

    _validate_paths_and_config(config)
    _validate_dataset_has_rxn_header(dataset_path=dataset_path, delimiter=delimiter)

    LOGGER.info("Building vocab from dataset '%s'...", dataset_path.name)

    # new vocab
    new_vocab_list = build_vocab_from_data_stream(_rxn_stream(dataset_path=dataset_path, delimiter=delimiter))
    new_vocab_ord_dict = list_to_ord_dict(new_vocab_list)

    if merge_with_default:
        default_vocab_path = Path(config.default_vocab_path)
        LOGGER.info("Merging with default vocab: %s", default_vocab_path.name)
        default_vocab = load_vocab(str(default_vocab_path))
        merged_vocab = merge_ordered_vocabs(default_vocab, new_vocab_ord_dict)
        LOGGER.info("Saving merged vocab to: %s", save_path)
        save_vocabulary(merged_vocab, str(save_path))
    else:
        LOGGER.info("saving new vocab to: %s", save_path)
        save_vocabulary(new_vocab_ord_dict, str(save_path))
    
    LOGGER.info("Done.")
    return save_path


@click.command()
@click.option(
    "--config_path",
    "-c",
    required=True,
    help="Path to create_vocab config file",
    type=str,
)

def main(config_path: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = parse_create_vocab_config(config_path=config_path)
    create_vocab_from_config(config=config)


if __name__ == "__main__":
    main()