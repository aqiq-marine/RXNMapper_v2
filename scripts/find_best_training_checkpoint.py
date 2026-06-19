"""
find best checkpoint of a model given a validation set

How to run:

python scripts/find_best_training_checkpoint.py -c configs/find_best_checkpoint_example.yaml
"""

import os
import click
import yaml
import pandas as pd
from huggingface_hub import snapshot_download
from rxnmapper.find_best_training_checkpoint import FindBestTrainingCheckpoint

DEFAULT_HF_REPO = "yvsgrndjn/alberta-uspto-2800k"


def parse_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def resolve_checkpoints_folder(value: str) -> str:
    """Resolve the config value to a local folder containing checkpoint-* subdirs.

    - An existing local directory is used as-is.
    - "DEFAULT"/"HF" (or any other value, treated as a Hub repo id) is downloaded
      from the Hugging Face Hub and the local snapshot path is returned.
    """
    if value not in ("DEFAULT", "HF") and os.path.isdir(value):
        return value
    repo_id = DEFAULT_HF_REPO if value in ("DEFAULT", "HF") else value
    return snapshot_download(repo_id=repo_id, repo_type="model")


@click.command()
@click.option(
    "--config_path",
    "-c",
    required=True,
    help="Path to config file",
    type=str,
)
def main(config_path: str)-> None:

    if config_path:
        config = parse_config(config_path)

    checkpoint_folder = resolve_checkpoints_folder(config["checkpoints_folder_path"])

    # Checkpoint subfolders carry no vocab.txt (it lives at the folder root), so
    # resolve it here and pass it through; otherwise the tokenizer silently falls
    # back to the packaged default vocab, which is wrong for custom-vocab models.
    vocab_path = config.get("vocab_path") or os.path.join(checkpoint_folder, "vocab.txt")
    vocab_path = vocab_path if os.path.exists(vocab_path) else None

    val_set_df = pd.read_csv(config["dataset_path"])
    rxn_col = config.get("rxn_col", "rxn")
    ref_col = config.get("ref_col", "mapped_rxn")
    for col in (rxn_col, ref_col):
        if col not in val_set_df.columns:
            raise ValueError(
                f"Column '{col}' not found in {config['dataset_path']}. "
                f"Available columns: {list(val_set_df.columns)}"
            )
    rxns_list = val_set_df[rxn_col].tolist()
    ref_rxns_list = val_set_df[ref_col].tolist()

    findbesttrainingcheckpoint = FindBestTrainingCheckpoint(
        head=config["head"],
        layer=config["layer"],
        folder_path=checkpoint_folder,
        model_name=config.get("model_name", "model"),
        rxns_list=rxns_list,
        ref_rxns_list=ref_rxns_list,
        limit_checkpoints=config.get("limit_checkpoints"),
        vocab_path=vocab_path,
        batch_size=config.get("batch_size", 32),
    )

    _ = findbesttrainingcheckpoint.get_config_dict_to_eval()
    _ = findbesttrainingcheckpoint.map_rxns_from_models_in_dict()
    _ = findbesttrainingcheckpoint.evaluate_results()

if __name__ == "__main__":
    main()
