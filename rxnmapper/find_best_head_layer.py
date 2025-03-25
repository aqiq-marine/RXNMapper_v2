import os
import json
import yaml
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Tuple
from rxnmapper import BatchedMapper, RXNMapper
from rxnmapper.smiles_utils import (
    process_reaction_with_product_maps_atoms,
    generate_atom_mapped_reaction_atoms,
)

def parse_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def open_json(json_path):
    with open(json_path) as f:
        json_file = json.load(f)
        return json_file

class FindBestHeadLayerCombination:
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
                 canonicalize: bool = False, 
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
        print('Model config validated.')
        
        self.models_to_eval_dict = {}
        for layer in tqdm(self.layers, desc="Layers"):
            for head in tqdm(self.heads, desc="Heads"):
                key = f"{self.model_config['model_type']}_{self.model_name}_head_{head}_layer_{layer}"
                self.models_to_eval_dict[key] = {'head': head, 'layer': layer}        
        return self.models_to_eval_dict


    def map_rxns_from_dict(self, dict_:Dict)->List[Dict]:
        """
        Maps reactions using a given model configuration.

        Args:
            dict_ (Dict): Dictionary with 'head' and 'layer' keys.

        Returns:
            List[Dict]: List of mapping results with confidence scores.
        """
        assert "head" in dict_ and "layer" in dict_, "Dictionary must contain 'head' and 'layer'"
        mapper = BatchedMapper(
                batch_size=self.batch_size, 
                model_path=self.model_path, 
                head=dict_['head'], 
                layer=dict_['layer'], 
                model_type=self.model_type, 
                attention_multiplier=self.attention_multiplier, 
                canonicalize=self.canonicalize, 
                placeholder_for_invalid=self.placeholder_for_invalid, 
                )
        mapped_results = []
        for result in mapper.map_reactions_with_info(self.rxns_list):
            mapped_results.append({
                "mapped_rxn": result.get("mapped_rxn", self.placeholder_for_invalid), 
                "confidence": result.get("confidence")
            })
        return mapped_results
        

    def map_rxns_from_models_in_dict(self) -> pd.DataFrame: 
        """
        Runs mapping for all models in self.models_to_eval_dict.

        Returns:
            pd.DataFrame: Combined DataFrame of all predictions with model identifiers.
        """
        all_results = []
        for model_id, model_config in tqdm(self.models_to_eval_dict.items(), desc="Models"):
            model_results = self.map_rxns_from_dict(model_config)

            for r in model_results:
                r["model_id"] = model_id
            all_results.extend(model_results)
        self.pred_results_df = pd.DataFrame(all_results)
        return self.pred_results_df


    def evaluate_results(self) -> Dict[str, int]:
        """
        Evaluates predictions against ground truth reactions.

        Returns:
            Dict[str, int]: Dictionary of correct predictions per model ID.
        """
        result_dicts = {}
        for model_id, _ in (self.models_to_eval_dict.items()):
            predictions = self.pred_results_df[self.pred_results_df['model_id']==model_id]['mapped_rxn'].tolist()
            num_matches = self.compare_pred_maps_w_ground_truth(ground_truth_rxns=self.ref_rxns_list, predicted_rxns=predictions)
            result_dicts.update({model_id:num_matches})
        max_val = max(result_dicts.values())
        max_key = max(result_dicts, key=result_dicts.get)
        print(f"Best head/layer combination for model {max_key}, getting {max_val} matches out of {len(self.ref_rxns_list)} --> {round(max_val/len(self.ref_rxns_list)*100,2)}% accuracy")
        return result_dicts


    def compare_pred_maps_w_ground_truth(self, ground_truth_rxns:List[str], predicted_rxns:List[str]) -> int:
        """
        Compares predicted and ground-truth atom maps using canonical forms.

        Args:
            ground_truth_rxns (List[str]): Ground-truth atom-mapped reactions.
            predicted_rxns (List[str]): Predicted atom-mapped reactions.

        Returns:
            int: Number of correct predictions.
        """
        correct_count = 0
        for counter, (rxn, predicted_rxn) in enumerate(tqdm(zip(ground_truth_rxns, predicted_rxns), total=len(ground_truth_rxns))):
            try:
                rxn_can = process_reaction_with_product_maps_atoms(rxn)
                pred_can = process_reaction_with_product_maps_atoms(predicted_rxn)

                rxn_can, _ = generate_atom_mapped_reaction_atoms(*rxn_can, canonical=True)
                pred_can, _ = generate_atom_mapped_reaction_atoms(*pred_can, canonical=True)

                rxn_str = self.create_rxn_smiles_from_rxn_equation(rxn_can)
                pred_str = self.create_rxn_smiles_from_rxn_equation(pred_can)

                if rxn_str == pred_str:
                    correct_count += 1

            except Exception:
                print(f"Error at index: {counter}")
                continue

        return correct_count

# -----------
# UTILITIES  
# -----------
    def create_rxn_smiles_from_rxn_equation(self, rxn_equation) -> str: # utility, could be moved elsewhere
        """
        Converts a parsed reaction equation object into a SMILES string.

        Args:
            rxn_equation: Parsed reaction equation with reactants/products.

        Returns:
            str: Reaction SMILES string.
        """
        reactants = rxn_equation.reactants
        products = rxn_equation.products
        reactants = ".".join(reactants)
        products = ".".join(products)
        return f"{reactants}>>{products}"