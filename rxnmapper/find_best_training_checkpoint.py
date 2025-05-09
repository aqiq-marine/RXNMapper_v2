import os
import re
import json
import pandas as pd
from tqdm import tqdm
from typing import List, Dict
from rxnmapper import BatchedMapper
from rxnmapper.mapping_evaluator_base import BaseMappingEvaluator

def open_json(json_path):
    with open(json_path) as f:
        json_file = json.load(f)
        return json_file


def list_checkpoints(folder_path):
    """
    Lists subfolders in folder_path that match the 'checkpoint-{number}' pattern.

    Args:
        folder_path (str): Path to the folder to scan.
    
    Returns:
        List[Tuple[int, str]]: List of (step_number, full_path) tuples, sorted by step.
    """
    checkpoint_pattern = re.compile(r"^checkpoint-(\d+)$")

    matches = []

    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        match = checkpoint_pattern.match(name)
        if os.path.isdir(full_path) and match:
            step = int(match.group(1))
            matches.append((step, full_path))
    matches.sort(key=lambda x: x[0])
    
    return matches


class FindBestTrainingCheckpoint(BaseMappingEvaluator):
    """
    Class to find the best training checkpoint of a given model for atom-mapping performance using RXNMapper.
    """
    def __init__(
        self,
        head: int,
        layer: int,
        folder_path: str,
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

        self.head = head
        self.layer = layer
        self.folder_path = folder_path
        self.model_name = model_name
    

    def get_config_dict_to_eval(self) -> Dict[str, Dict[str, int]]:
        """
        Builds dictionary of all model configurations to evaluate.

        Returns:
            Dict[str, Dict]: Dictionary where keys are model IDs and values are the head, layer, and model path.
        """
        assert os.path.isdir(self.folder_path), "The entered folder path does not exist"
        matching_folders = list_checkpoints(self.folder_path)

        self.models_to_eval_dict = {}
        for step, model_path in matching_folders:
            key = f"{self.model_type}_{self.model_name}_head_{self.head}_layer_{self.layer}_checkpoint_{step}"
            self.models_to_eval_dict[key] = {'head': self.head, 'layer': self.layer, 'model_path': model_path, 'model_type': self.model_type}

        return self.models_to_eval_dict

