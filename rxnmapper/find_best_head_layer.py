import os
import json
import yaml
from typing import List, Dict
from rxnmapper.mapping_evaluator_base import BaseMappingEvaluator

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

    def __init__(
        self,
        heads: List[int],
        layers: List[int],
        model_name: str = "model",
        *args,
        **kwargs,
    ):
        """
        Args:
            heads: List of attention heads to test.
            layers: List of layers to test.
            model_name: Optional model name for result labeling.
            *args, **kwargs: All arguments forwarded to BaseMappingEvaluator
        """
        super().__init__(*args, **kwargs)

        self.heads = heads
        self.layers = layers
        self.model_name = model_name

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


