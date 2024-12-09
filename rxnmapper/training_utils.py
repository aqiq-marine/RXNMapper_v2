"""

Training utility functions:

"""

from dataclasses import dataclass
import yaml


@dataclass
class Config:
    """
    Dataclass for configuration parameters
    """

    # Paths:
    alberta_config_path: str
    dataset_name: str
    checkpoint_path: str

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


def parse_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Config(**config)
