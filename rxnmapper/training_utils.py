"""

Training utility functions:

"""

from dataclasses import dataclass
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils.import_utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import Trainer
from transformers import DataCollatorWithPadding
from typing import Any, Dict, List


@dataclass
class Config:
    """
    Dataclass for configuration parameters
    """

    # Basic params:
    dataset_name: str
    checkpoint_path: str
    dataset_path: str
    delimiter: str
    preprocess_dataset: bool
    save_processed_data_path: str

    # Vocab arguments:
    vocab_path: str
    default_vocab_path: str
    merge_new_with_default: bool
    save_vocab_path: str
    auto_vocab: bool #old, remove once nothing else asks for it
    vocab_output_path: str #old, remove once nothing else asks for it

    # Model config:
    alberta_config: dict

    # Atom mapping parameters:
    model_path: str
    model_type: str
    attention_multiplier: float
    head: int
    layers: list

    # Masked language modelling parameters:
    mlm_probability: float

    # Training parameters:
    output_dir: str
    logging_dir: str
    report_to: list
    run_name: str

    learning_rate: float
    weight_decay: float
    batch_size: int
    accum_steps: int

    max_steps: int
    warmup_steps: int
    logging_steps: int
    save_step: int

    eval_strategy: str
    save_strategy: str
    logging_first_step: bool
    save_total_limit: int
    save_only_model: bool


def parse_config(config_path): # modified to take into account boolean "resume_from_checkpoint" argument in YAML config file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # REMOVE resume_from_checkpoint before instantiating the Config class
    config.pop("resume_from_checkpoint", None)

    return Config(**config)

