"""
Evaluate one or more model / head / layer configurations on a validation set
using reaction-graph isomorphism (atommap-eval).

How to run:

python scripts/evaluate_models.py -c configs/models_to_eval_example.yaml
"""

import click
import yaml
import pandas as pd
from pathlib import Path
from rxnmapper.mapping_evaluator_base import BaseMappingEvaluator


def parse_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_models_to_eval(config: dict) -> dict:
    """Merge the top-level defaults block into every entry of models.

    Per-model keys override the inherited defaults.
    """
    if "models" not in config or not isinstance(config["models"], dict):
        raise ValueError("Config must contain a top-level 'models' mapping (name -> config).")

    defaults = config.get("defaults", {}) or {}
    models_to_eval = {}
    for name, model_config in config["models"].items():
        merged = {**defaults, **(model_config or {})}
        for required in ("model_path", "head", "layer"):
            if required not in merged:
                raise ValueError(f"Model '{name}' is missing required key '{required}'.")
        models_to_eval[name] = merged
    return models_to_eval


@click.command()
@click.option(
    "--config_path",
    "-c",
    required=True,
    help="Path to config file",
    type=str,
)
def main(config_path: str) -> None:
    config = parse_config(config_path)

    # Load validation set
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

    # Build the set of models to evaluate (defaults inherited by each model)
    models_to_eval = build_models_to_eval(config)

    evaluator = BaseMappingEvaluator(
        rxns_list=rxns_list,
        ref_rxns_list=ref_rxns_list,
        batch_size=config.get("batch_size", 24),
    )
    evaluator.models_to_eval_dict = models_to_eval

    # Inference + evaluation
    evaluator.map_rxns_from_models_in_dict()
    scores, detailed = evaluator.evaluate_results(return_detailed=True)

    out_path = Path(config["output_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    detailed_frames = []
    for name, model_df in detailed.items():
        frame = model_df.copy()
        frame.insert(0, "model", name)
        detailed_frames.append(frame)
    pd.concat(detailed_frames, ignore_index=True).to_csv(out_path, index=False)

    # Save accuracy summary (best first) + print it
    summary_df = (
        pd.DataFrame(
            [
                {"model": name, "accuracy": acc, "n_reactions": len(ref_rxns_list)}
                for name, acc in scores.items()
            ]
        )
        .sort_values("accuracy", ascending=False)
        .reset_index(drop=True)
    )
    summary_path = out_path.with_name(f"{out_path.stem}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print(f"\nAccuracy on {len(ref_rxns_list)} reactions:")
    for _, row in summary_df.iterrows():
        print(f"  {row['model']}: {round(row['accuracy'] * 100, 2)}%")
    print(f"\nDetailed results: {out_path}\nSummary: {summary_path}")


if __name__ == "__main__":
    main()