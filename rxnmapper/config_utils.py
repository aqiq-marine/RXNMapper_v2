"""

Config files utility functions

"""

import yaml
from dataclasses import dataclass

@dataclass
class CreateVocabConfig:
    """
    Dataclass for create_vocab configuration parameters
    """
    dataset_path: str
    delimiter: str
    save_vocab_path: str
    merge_new_with_default: bool
    default_vocab_path: str

def parse_create_vocab_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return CreateVocabConfig(**config)


@dataclass
class PreprocessTokLenConfig:
    """
    Dataclass for preprocess_by_token_len configuration parameters
    """
    dataset_path: str
    vocab_path: str
    delimiter: str
    save_processed_data_path: str
    # Model config:
    alberta_config: dict

def parse_preprocess_tok_len_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return PreprocessTokLenConfig(**config)