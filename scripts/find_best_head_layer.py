"""
find best head/layer of a model given a validation set

How to run: 

python scripts/find_best_head_layer.py -c configs/find_best_head_layer_example.yaml
"""

import click
import yaml
import pandas as pd
from rxnmapper.find_best_head_layer import FindBestHeadLayerCombination

def parse_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

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
    
    val_set_df = pd.read_csv(config["dataset_path"])
    rxns_list = val_set_df['rxn'].tolist()
    ref_rxns_list = val_set_df['mapped_rxn'].tolist()

    fbhlc = FindBestHeadLayerCombination(
        heads=config["heads"],
        layers=config["layers"],
        model_name="original",
        rxns_list=rxns_list, 
        ref_rxns_list=ref_rxns_list,
        model_path=config["model_path"],
        model_type=config["model_type"],
        vocab_path=config["vocab_path"],
        batch_size=config["batch_size"],
        )

    _ = fbhlc.get_config_dict_to_eval()
    _ = fbhlc.map_rxns_from_models_in_dict()
    _ = fbhlc.evaluate_results()

if __name__ == "__main__":
    main()

