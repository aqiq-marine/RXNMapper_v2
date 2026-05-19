from typing import List, Dict, Tuple, Optional
import pandas as pd
from atommap_eval.preprocess import preprocess_dataset
from atommap_eval.data_models import ReactionPair
from atommap_eval.pair_evaluation import evaluate_pairs_batched

def convert_to_tuple_pair(list1: List, list2: List) -> List[Tuple[str,str]]:
    return [(el_list1, el_list2) for el_list1, el_list2 in zip(list1, list2)]


def convert_rxn_lists_to_df(
        gt_rxn_list: List[str],
        pred_rxn_list: List[str]
        ) -> pd.DataFrame:
    """
    Takes lists of ground truth and predicted reactions and arranges them in a
    pd.DataFrame to match API of atommap-eval.

    Args:
        ground_truth_rxns (List[str]): Ground-truth atom-mapped reactions.
        predicted_rxns (List[str]): Predicted atom-mapped reactions.
    
    Returns:
        pd.DataFrame: DataFrame with Nx2 shape, with N beign the length of both lists.
    """

    assert len(gt_rxn_list)==len(pred_rxn_list), f"Both input lists must be the same length, now {len(gt_rxn_list)}, {len(pred_rxn_list)}"
    return pd.DataFrame(
        {
            "ground_truth_rxn": gt_rxn_list,
            "predicted_rxn" : pred_rxn_list,
        }
    )


def evaluate_atom_map_equivalence(
        ground_truth_rxns: List[str],
        predicted_rxns: List[str],
        preprocessing: bool = True,
        num_workers: int = 8,
        batch_per_worker: int = 32,
        ) -> List[Dict]:
    """
    Compares predicted and ground-truth atom maps using graph-based
    isomorphism evaluation.

    Args:
        ground_truth_rxns (List[str]): Ground-truth atom-mapped reactions.
        predicted_rxns (List[str]): Predicted atom-mapped reactions.
        preprocessing (bool): Should all reactions be preprocessed (ensures format is respected)

    Returns:
        List[Dict]: Detailed comparison info for each reaction.
    """
    
    pairs = []
    if preprocessing:
        input_df = convert_rxn_lists_to_df(ground_truth_rxns,predicted_rxns)
        preproc_df = preprocess_dataset(df=input_df)
        
        for _, row in preproc_df.iterrows():
            pairs.append(
                ReactionPair(
                    row.preproc_ref,
                    row.preproc_pred,
                    row.pair_index,
                )
            )
    elif not preprocessing:
        raise NotImplementedError

    # run evaluation
    eval_results = evaluate_pairs_batched(pairs=pairs, num_workers=num_workers, batch_per_worker=batch_per_worker)
    
    # prepare the results summary output
    ## add eval_results to the preprocessed_df
    preproc_df["eval_result"] = [eval_results[el][0] for el in range(len(eval_results))]
    results = []
    for _,row in preproc_df.iterrows():
        results.append({
            "index": row.pair_index,
            "ground_truth_rxn": row.ground_truth_rxn,
            "predicted_rxn": row.predicted_rxn,
            "success": row.eval_result,
        })
    return results
