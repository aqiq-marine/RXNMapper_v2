import os
import sys
import json
import pandas as pd
from pathlib import Path

def load_grid_results(model_name, val_set_name, results_root):
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    root = Path(results_root)
    
    # if relative, interpret under project; if absolute, use as-is
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    results_path = root / model_name / val_set_name
    
    print(f"Looking for JSONs in: {results_path.resolve()}")

    all_results = []

    for file in results_path.glob("head*_layer*_ckpt*.json"):
        with open(file) as f:
            try:
                result = json.load(f)
                result["filename"] = file.name
                all_results.append(result)
            except Exception as e:
                print(f"Failed to load {file.name}: {e}")
    
    return pd.DataFrame(all_results)

def check_completeness(df):
    expected = 8 * 3 * 150  # heads × layers × checkpoints
    found = len(df)
    print(f" Found {found} results. Expected: {expected}")
    if found < expected:
        missing = expected - found
        print(f" WARNING: {missing} results are missing!")
    else:
        print(" All combinations present.")

def find_best(df):
    best = df.loc[df["score"].idxmax()]
    print(" Best combination:")
    print(best)
    return best

def save_to_json(df, output_path):
    df_sorted = df.sort_values(["ckpt", "layer", "head"])
    df_sorted.to_json(output_path, orient="records", indent=2)
    print(f" Full result saved to: {output_path}")

# ------------- Run for a specific model/set --------------
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python grid_post_process.py <model_name> <val_set_name> <results_root_path>")
    model = sys.argv[1] # example "alberta_uspto_2800k_mlm_010"
    valset = sys.argv[2] #"eval_schneider_corr"
    results_root = sys.argv[3] #data/results/grid_search

    df = load_grid_results(model, valset, results_root)
    if df.empty:
        print(f"ERROR: No results JSONs found under {results_root}/{model}/{valset}")
        sys.exit(1)
        
    check_completeness(df)
    best = find_best(df)
    
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    root = Path(results_root)
    
    # if relative, interpret under project; if absolute, use as-is
    if not root.is_absolute():
        root = PROJECT_ROOT / root
    results_path = root / model / valset
    save_to_json(df, f"{results_path}/{valset}_consolidated_results.json")
