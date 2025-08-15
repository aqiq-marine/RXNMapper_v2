import os
import sys
import json
import pandas as pd
from pathlib import Path
from rxnmapper.mapping_evaluator_base import BaseMappingEvaluator
from rxnmapper.validation_utils import evaluate_atom_map_equivalence_parallel


GRID_PATH = "configs/grid_combinations.json"

def load_validation_set(path):
    with open(path, "r") as f:
        data = json.load(f)
    return list(data["rxn"].values()), list(data["rxn_atom_mapped"].values())

def main(model_name, val_set_path, vocab_path, chunk_start, chunk_end):
    with open(GRID_PATH, "r") as f:
        grid = json.load(f)

    rxns_list, ref_rxns_list = load_validation_set(val_set_path)
    val_set_name = Path(val_set_path).stem

    for i in range(chunk_start, chunk_end):
        config = grid[i]

        model_path = f"trained_models/{model_name}/checkpoint-{config['checkpoint_number']}/"
        if not Path(model_path).exists():
            print(f"Skipping: model checkpoint not found at {model_path}")
            continue

        evaluator = BaseMappingEvaluator(
            rxns_list=rxns_list,
            ref_rxns_list=ref_rxns_list,
            model_path=model_path,
            model_type="albert",
            vocab_path=vocab_path,
            batch_size=24,
        )

        config.update({
            "model_folder_name": model_name,
            "valid_set_name": val_set_name,
            "output_dir": "data/results/grid_search"
        })

        mapped_rxns = evaluator.map_rxns_from_dict(config=config)
        maps_df = pd.DataFrame(mapped_rxns)

        if len(rxns_list) != len(maps_df['mapped_rxn']):
            print(f"Mismatch at index {i}: skipping")
            continue

        model_results = evaluate_atom_map_equivalence_parallel(
            ground_truth_rxns=ref_rxns_list,
            predicted_rxns=maps_df['mapped_rxn'].tolist(),
            canonicalize=False,
            num_workers=os.cpu_count()-2,
            chunksize=24
        )
        df_model_results = pd.DataFrame(model_results)
        score = round(df_model_results['success'].mean(), 6)

        # Save result
        output_dir = Path(config["output_dir"]) / model_name / val_set_name
        output_dir.mkdir(parents=True, exist_ok=True)
        result_path = output_dir / f"head{config['head']}_layer{config['layer']}_ckpt{config['checkpoint_number']}.json"

        with open(result_path, "w") as f:
            json.dump({
                "head": config["head"],
                "layer": config["layer"],
                "ckpt": config["checkpoint_number"],
                "score": score
            }, f)        
             

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python evaluate_model_config.py <model_name> <val_set_path> <vocab_path> <chunk_start> <chunk_end>")
        sys.exit(1)

    model_name = sys.argv[1]
    val_set_path = sys.argv[2]
    vocab_path = sys.argv[3]
    chunk_start = int(sys.argv[4])
    chunk_end = int(sys.argv[5])

    main(model_name, val_set_path, vocab_path, chunk_start, chunk_end)
