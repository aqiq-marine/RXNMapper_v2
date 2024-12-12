import json
from rxnmapper.smiles_utils import (
    process_reaction_with_product_maps_atoms,
    generate_atom_mapped_reaction_atoms,
)
from tqdm import tqdm
import os
import click
import pandas as pd


def load_json(path):
    """
    Simply load a json file from a given path.
    """
    assert os.path.exists(path), f"Path {path} does not exist"
    with open(path, "r") as f:
        data = json.load(f)
    return data


def create_rxn_smiles_from_rxn_equation(rxn_equation):
    """
    Create a reaction SMILES from a reaction equation.
    """
    reactants = rxn_equation.reactants
    products = rxn_equation.products
    reactants = ".".join(reactants)
    products = ".".join(products)
    return f"{reactants}>>{products}"

    # return f"{reactants[0]}>>{products[0]}"


def main_no_maps(gt_rxns, predicted_rxns):
    """

    Calculate accuracy of predicted mappings comparing their canonical forms.

    Only check with first atom map.

    """
    correct_count = 0
    wrong_count = 0
    error_count = 0

    counter = 0
    for rxn, predicted_rxn in tqdm(zip(gt_rxns, predicted_rxns), total=len(gt_rxns)):
        try:
            rxn_can = process_reaction_with_product_maps_atoms(rxn)
            predicted_rxn_can = process_reaction_with_product_maps_atoms(predicted_rxn)

            # Convert to true atom map with canonicalisation:
            rxn_can, _ = generate_atom_mapped_reaction_atoms(
                *rxn_can,
                canonical=True,
            )
            predicted_rxn_can, _ = generate_atom_mapped_reaction_atoms(
                *predicted_rxn_can,
                canonical=True,
            )

            rxn_can = create_rxn_smiles_from_rxn_equation(rxn_can)
            predicted_rxn_can = create_rxn_smiles_from_rxn_equation(predicted_rxn_can)

            if rxn_can == predicted_rxn_can:  # Only check the atom maps
                correct_count += 1
            else:
                wrong_count += 1

        except:
            print(f"Error at index: {counter}")
            error_count += 1
            continue

        counter += 1

    print(f"Correct Count: {correct_count}")
    print(f"Percentage: {correct_count / len(gt_rxns) * 100:.3}%")

    print(f"Wrong count: {wrong_count}")
    print(f"Wrong percentage: {wrong_count / len(gt_rxns) * 100:.3}%")

    print(f"Error count: {error_count}")
    print(f"Error percentage: {error_count / len(gt_rxns) * 100:.3}%")


@click.command()
@click.option(
    "--ground_truth_path",
    "-f",
    required=True,
    help='Input file path to json with "rxn_atom_mapped" key',
)
@click.option(
    "--prediction_path",
    "-o",
    required=True,
    help="Output file path from model",
)
def main(
    ground_truth_path,
    prediction_path,
):
    """
    Main function to run the evaluation script.
    """

    assert os.path.exists(ground_truth_path), f"Path {ground_truth_path} does not exist"
    assert os.path.exists(prediction_path), f"Path {prediction_path} does not exist"
    # assert ground_truth_path.endswith(".json"), "Ground truth path must be a json file"
    # assert prediction_path.endswith(".json"), "Prediction path must be a json file"

    # Load the data:
    if ground_truth_path.endswith(".json"):
        gt_data = load_json(ground_truth_path)
        assert (
            "rxn_atom_mapped" in gt_data.keys()
        ), "Ground truth data must have atom maps"
        gt_rxns = list(gt_data["rxn_atom_mapped"].values())

    elif ground_truth_path.endswith(".csv"):
        gt_data = pd.read_csv(ground_truth_path)
        # print(gt_data.columns.to_list())
        assert (
            "rxn_atom_mapped" in gt_data.columns
        ), "Ground truth data must have atom maps"
        gt_rxns = gt_data["rxn_atom_mapped"].tolist()

    predicted_data = load_json(prediction_path)

    assert "mapped_rxn" in predicted_data.keys(), "Predicted data must have atom maps"

    predicted_rxns = list(predicted_data["mapped_rxn"].values())

    assert len(gt_rxns) == len(
        predicted_rxns
    ), "Number of reactions must be the same - Pls check your data."

    # Run eval:
    main_no_maps(gt_rxns, predicted_rxns)


if __name__ == "__main__":
    main()
