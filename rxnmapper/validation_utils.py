from typing import List, Dict, Tuple, Optional
from atommap_eval.evaluator import are_atom_maps_equivalent
from atommap_eval.parallel import evaluate_pairs_in_parallel


def convert_to_tuple_pair(list1: List, list2: List) -> List[Tuple[str,str]]:
    return [(el_list1, el_list2) for el_list1, el_list2 in zip(list1, list2)]


def evaluate_atom_map_equivalence(
        ground_truth_rxns: List[str],
        predicted_rxns: List[str],
        canonicalize: bool = True
        ) -> List[Dict]:
    """
    Compares predicted and ground-truth atom maps using graph-based
    isomorphism evaluation.

    Args:
        ground_truth_rxns (List[str]): Ground-truth atom-mapped reactions.
        predicted_rxns (List[str]): Predicted atom-mapped reactions.
        canonicalize (bool):    If all reactions should be canonicalized
        (SMILES and atom-map indices)

    Returns:
        List[Dict]: Detailed comparison info for each reaction.
    """
    results = []

    for index, (ground_truth_rxn, predicted_rxn) in enumerate(zip(ground_truth_rxns, predicted_rxns)):
        try:
            success = are_atom_maps_equivalent(ground_truth_rxn, predicted_rxn, canonicalize=canonicalize)
        except Exception as e:
            success = False
            print(f"Error at index {index}: {e}")


        results.append({
            "index": index, 
            "ground_truth_rxn": ground_truth_rxn, 
            "predicted_rxn": predicted_rxn, 
            "success": success, 
        })

    return results


def evaluate_atom_map_equivalence_parallel(
        ground_truth_rxns: List[str],
        predicted_rxns: List[str],
        canonicalize: bool = False,
        num_workers: Optional[int] = None,
        chunksize: int = 10,
        ) -> List[Dict]:
    """
    Compares predicted and ground-truth atom maps using graph-based
    isomorphism evaluation with multi-cpu performance.

    Args:
        ground_truth_rxns (List[str]): Ground-truth atom-mapped reactions.
        predicted_rxns (List[str]): Predicted atom-mapped reactions.
        canonicalize (bool):    If all reactions should be canonicalized
        (SMILES and atom-map indices)
        num_workers (Optional[int]): number of CPUs to use in parallel
        chunksize (int):    number of reactions to submit at the same time

    Returns:
        List[Dict]: Detailed comparison info for each reaction.
    """
    gt_pred_tuple_list = convert_to_tuple_pair(ground_truth_rxns, predicted_rxns)

    # Run parallel evaluation
    results_bool = evaluate_pairs_in_parallel(
        pairs = gt_pred_tuple_list,
        canonicalize = canonicalize,
        num_workers=num_workers,
        chunksize=chunksize,
        )
    
    # Build result dicts
    results = []
    for index, ((gt, pred), success) in enumerate(zip(gt_pred_tuple_list, results_bool)):
        results.append({
            "index": index,
            "ground_truth_rxn": gt,
            "predicted_rxn": pred,
            "success": success,
        })

    return results
