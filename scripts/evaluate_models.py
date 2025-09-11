import time
import yaml
import argparse
import pandas as pd
from pathlib import Path
from rxnmapper.mapping_evaluator_base import BaseMappingEvaluator


def read_validation(path: Path, rxn_col: str, ref_col: str):
    df = pd.read_csv()
    if rxn_col not in df.columns or ref_col not in df.columns:
        raise ValueError(
            f"Input CSV must contain columns '{rxn_col}' and '{ref_col}'."
            f"Found: {list(df.columns)}" 
        )
    rxns = df[rxn_col].tolist()
    refs = df[ref_col].tolist()
    return rxns, refs

def load_models_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict) or "models" not in cfg or not isinstance(cfg["models"], dict):
        raise ValueError("YAML must have a top-level key 'models' mapping names -> model configs.")
    return cfg

def ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

def export_results(refs: list , rxn_col: str, results: dict, models_to_eval: dict, out_csv: Path):
    df_out = pd.DataFrame({"ground_truth_rxns": refs})
    for name in models_to_eval.keys():
        model_res = results.get(name, {})
        df_out[f"{name}_success"] = model_res.get("success", [])
        df_out[f"{name}_pred"] = model_res.get("predicted_rxn", [])
    ensure_output_path(out_csv)
    df_out.to_csv(out_csv, index=False)


def main():
    parser = argparse.ArgumentParser(description="Evaluates atom-mapping performance of AlBERT models on given dataset.")
    parser.add_argument("--csv-file", type=Path, required=True, help="path to CSV file containing reactions to map and groud truth atom-maps")
    parser.add_argument("--rxn-col", default="rxn", help="Column name for unmapped reactions.")
    parser.add_argument("--ref-col", defaut="ground_truth_rxn", help="Column name for ground truth atom-mapped reaction.")
    parser.add_argument("--models-yaml", type=Path, required=True, default="", help="path to YAML file describing models to evaluate.")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size for inference.")
    parser.add_argument("--out-csv", type=Path, required=True, help="Path to write results CSV.")
    parser.add_argument("-v", "--verbose", action="count", default=1, help="Increase verbosity (-v, -vv).")
    args = parser.parse_args()

    # read inputs
    rxns, refs = read_validation(args.csv_file, args.rxn_col, args.ref_col)

    # load model configs
    models_to_eval = load_models_yaml(args.models_yaml)

    bme = BaseMappingEvaluator(
        rxns_list=rxns,
        ref_rxns_list=refs,
        batch_size=args.batch_size,
    )
    bme.models_to_eval_dict = models_to_eval

    # Inference
    t0 = time.time()
    _pred = bme.map_rxns_from_models_in_dict()
    print(f"Inference done in {time.time()-t0:,.1f}s")

    # Evaluation
    t1 = time.time()
    scores, results = bme.evaluate_results(return_detailed=True)
    print(f"Evaluation done in {time.time()-t1:,.1f}s")

    # Save results
    export_results(refs, rxn_col=args.rxn_col, results=results, models_to_eval=bme.models_to_eval_dict, out_csv=args.out_csv)


if __name__ == "__main__":
    main()