import os
import pandas as pd
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple, Union
from rxnmapper import BatchedMapper
from rxnmapper.validation_utils import evaluate_atom_map_equivalence, evaluate_atom_map_equivalence_parallel

class BaseMappingEvaluator:
    """
    Base class for evaluating atom-mapped reaction predictions
    """
    def __init__(self, 
                 rxns_list: List[str], 
                 ref_rxns_list: Optional[List[str]] = None, 
                 model_path: Optional[str] = None, 
                 model_type: str = 'albert', 
                 vocab_path: Optional[str] = None, 
                 batch_size: int = 1, 
                 attention_multiplier: float = 90.0, 
                 canonicalize: bool = True, 
                 placeholder_for_invalid: str = ">>", 
                 canonical_eval: bool = False,
                 ):
        """
        Base class for evaluating or predicting atom-mapped reactions.

        Args:
            rxns_list: List of input (unmapped) reaction SMILES.
            ref_rxns_list: List of ground truth reactions for evaluation (optional).
            model_path: Default model path (used if not passed in config).
            model_type: Model type, e.g., 'albert'.
            head: head to use for predictions.
            layer: layer to use for predictions.
            batch_size: Batch size for BatchedMapper.
            attention_multiplier: Multiplier for attention scores.
            canonicalize: Whether to canonicalize reactions before mapping.
            placeholder_for_invalid: Reaction placeholder for failed mappings.
        """
        self.rxns_list = rxns_list
        self.ref_rxns_list = ref_rxns_list
        self.model_path = model_path
        self.model_type = model_type
        self.vocab_path = vocab_path
        self.batch_size = batch_size
        self.attention_multiplier = attention_multiplier
        self.canonicalize = canonicalize
        self.placeholder_for_invalid = placeholder_for_invalid
        self.canonical_eval = canonical_eval
        self.num_workers = os.cpu_count() - 2


    def _validate_attrs(self, required_attrs: List[str]):
        """
        Check that required attributes are defined in the subclass instance.

        Args:
            required_attrs (List[str]): List of attribute names to validate.

        Raises:
            AttributeError: If any required attribute is missing.
        """
        for attr in required_attrs:
            if not hasattr(self, attr):
                raise AttributeError(
                    f"Missing required attribute '{attr}' in {self.__class__.__name__}."
                )


    def map_rxns_from_dict(self, config: Dict) -> List[Dict]:
        """
        Maps reactions using a given model configuration.

        Args:
            config (Dict): Dictionary with at least 'head' and 'layer'. Optionally 'model_path' and 'model_type'.

        Returns:
            List[Dict]: Mapping results with confidence scores
        """
        assert "head" in config and "layer" in config, "Config must contain 'head' and 'layer'"

        model_path = config["model_path"] if "model_path" in config else self.model_path
        model_type = config["model_type"] if "model_type" in config else self.model_type
        vocab_path = config["vocab_path"] if "vocab_path" in config else self.vocab_path

        mapper = BatchedMapper(
            batch_size=self.batch_size, 
            model_path=model_path, 
            head=config["head"], 
            layer=config["layer"], 
            model_type=model_type, 
            attention_multiplier=self.attention_multiplier, 
            canonicalize=self.canonicalize, 
            placeholder_for_invalid=self.placeholder_for_invalid, 
            vocab_path = vocab_path
        )

        mapped_results = []
        for result in mapper.map_reactions_with_info(self.rxns_list):
            mapped_results.append({
                "mapped_rxn": result.get("mapped_rxn", self.placeholder_for_invalid), 
                "confidence": result.get("confidence")
            })
        return mapped_results


    def map_rxns_from_models_in_dict(self, config_dict: Optional[Dict[str, Dict]] = None) -> pd.DataFrame: 
        """
        Runs mapping for all models in self.models_to_eval_dict.

        Args:
            config_dict (Optional[Dict[str, Dict]]): A dict of model configs. Defaults to self.models_to_eval_dict.
        
        Returns:
            pd.DataFrame: Combined DataFrame of all predictions with model identifiers.
        """
        config_dict = config_dict or self.models_to_eval_dict
        all_results = []

        for model_id, model_config in tqdm(config_dict.items(), desc="Models"):
            model_results = self.map_rxns_from_dict(model_config)

            for r in model_results:
                r["model_id"] = model_id
            all_results.extend(model_results)
        self.pred_results_df = pd.DataFrame(all_results)
        return self.pred_results_df


    def evaluate_results(self,
                         chunksize: int = 10,
                         return_detailed: bool = False
        ) -> Union[Dict, Tuple[Dict, Dict[str, pd.DataFrame]]]:
        """
        Evaluates predictions against ground truth reactions. Can only be run after 
        `map_rxns_from_models_in_dict`.

        Args:
            chunksize (int):    Number of reactions per parallel chunk
            return_detailed (bool): Whether to also return the full per-model DataFrames.

        Returns:
            If return_detailed is False:
                Dict[str, float]: Accuracy per model ID
            If return_detailed is True:
                Tuple[
                    Dict[str, float], # accuracy summary
                    Dict[str, pd.DataFrame] # per-model detailed results    
            ]
        """
        self._validate_attrs(["models_to_eval_dict", "pred_results_df", "ref_rxns_list"])

        result_dicts = {}
        detailed_dfs = {}

        for model_id, _ in tqdm(self.models_to_eval_dict.items(), desc="Evaluating models"):
            predictions = self.pred_results_df[
                self.pred_results_df['model_id'] == model_id
            ]['mapped_rxn'].tolist()

            model_results = evaluate_atom_map_equivalence_parallel(
                ground_truth_rxns=self.ref_rxns_list,
                predicted_rxns=predictions,  
                canonicalize=self.canonical_eval,
                num_workers=self.num_workers,
                chunksize=chunksize,
            )
            df_model_results = pd.DataFrame(model_results)
            result_dicts[model_id] = df_model_results['success'].mean()
            if return_detailed:
                detailed_dfs[model_id] = df_model_results

        max_key = max(result_dicts, key=result_dicts.get)
        max_val = result_dicts[max_key]
        print(f"Best configuration: {max_key} → Accuracy: {round(max_val * 100, 2)}% on {len(self.ref_rxns_list)} reactions")
        
        return (result_dicts, detailed_dfs) if return_detailed else result_dicts