import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from rxnmapper import BatchedMapper, RXNMapper
from rxnmapper.mapping_evaluator_base import BaseMappingEvaluator
#from rxnmapper.smiles_utils import compare_pred_maps_w_ground_truth

def parse_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def open_json(json_path):
    with open(json_path) as f:
        json_file = json.load(f)
        return json_file

class FindBestHeadLayerCombination(BaseMappingEvaluator):
    """
    Class to find the best attention head/layer combination for atom-mapping performance using RXNMapper.
    """

    def __init__(self, 
                 rxns_list:List[str], 
                 ref_rxns_list: List[str], 
                 heads: List[int], 
                 layers: List[int], 
                 model_path: str, 
                 model_name: str = "model", 
                 batch_size: int = 32, 
                 model_type: str = "albert", 
                 attention_multiplier: float = 90.0, 
                 canonicalize: bool = True, 
                 placeholder_for_invalid : str = ">>",  
    ):  
        """      
        Args:
            rxns_list (List[str]): List of unmapped SMILES reactions.
            heads (List[int]): Attention heads to evaluate.
            layers (List[int]): Transformer layers to evaluate.
            model_path (str): Path to trained model directory.
            model_name (str): Custom name for the model (used in result keys).
            batch_size (int): Batch size for BatchedMapper.
            model_type (str): Type of transformer model (e.g., 'albert').
            attention_multiplier (float): Weight multiplier for attention maps.
            canonicalize (bool): Whether to canonicalize reactions.
            placeholder_for_invalid (str): Placeholder for failed mappings.
        """
        self.rxns_list = rxns_list
        self.ref_rxns_list = ref_rxns_list
        self.heads = heads
        self.layers = layers
        self.model_path = model_path
        self.model_name = model_name
        self.batch_size = batch_size
        self.model_type = model_type
        self.attention_multiplier = attention_multiplier
        self.canonicalize = canonicalize
        self.placeholder_for_invalid = placeholder_for_invalid

    def get_config_dict_to_eval(self) -> Dict[str, Dict[str, int]]:
        """
        Builds dictionary of all model configurations to evaluate.

        Returns:
            Dict[str, Dict]: Dictionary where keys are model IDs and values contain head/layer indices.
        """
    
        assert os.path.isdir(self.model_path), "The entered model path does not exist" 
        config_path = os.path.join(self.model_path, 'config.json')
        assert os.path.exists(config_path), "Missing config.json file in model folder"
        self.model_config = open_json(config_path)

        assert all(x in range(self.model_config['num_attention_heads']) for x in self.heads), f"Invalid heads:{self.heads}"
        assert all(x in range(self.model_config['num_hidden_layers']) for x in self.layers), f"Invalid layers:{self.layers}"
        
        self.models_to_eval_dict = {}
        for layer in self.layers:
            for head in self.heads:
                key = f"{self.model_config['model_type']}_{self.model_name}_head_{head}_layer_{layer}"
                self.models_to_eval_dict[key] = {'head': head, 'layer': layer, 'model_path': self.model_path, 'model_type': self.model_type}        
        return self.models_to_eval_dict


