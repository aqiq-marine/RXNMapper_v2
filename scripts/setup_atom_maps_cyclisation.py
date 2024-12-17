"""

Script to setup the atom maps for Finetuning of the cyclisation dataset.

"""

import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np
from rxnmapper.smiles_utils import process_reaction_with_product_maps_atoms
from tqdm import tqdm
import os


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction into individual atoms and bonds.
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens), "SMILES string did not match tokenized pattern."
    return " ".join(tokens)


def load_json(path):
    """
    Load data from a JSON file.
    """
    return pd.read_json(path)


def load_csv(path):
    """
    Load data from a CSV file.
    """
    return pd.read_csv(path)


def check_explicit_hydrogens(data):
    """Method to check that we do not have any explicit hydrogens"""

    for i, row in data.iterrows():
        rxn_atom_mapped = row["rxn_atom_mapped"]

        # Check that we do not have any explicit hydrogens
        assert (
            "[H]" not in rxn_atom_mapped
        ), "The reaction has explicit hydrogens. Please make sure no Explicit hydrogens are present in your data."


def get_atom_maps(data):
    """
    Processes the 'rxn_atom_mapped' column in the DataFrame, extracting
    processed reactions and atom maps.

    Args:
        data (pd.DataFrame): Input DataFrame containing a column 'rxn_atom_mapped'.

    Returns:
        pd.DataFrame: DataFrame with two new columns:
                      'rxn' for processed reactions and 'atom_maps' for atom maps.
    """

    def process_row(rxn_atom_mapped):
        """Process a single row and return rxn and atom map."""
        try:
            return process_reaction_with_product_maps_atoms(rxn_atom_mapped)
        except Exception as e:
            print(f"Error processing reaction: {rxn_atom_mapped}. Error: {e}")
            return None, None  # Return None for both if processing fails

    # Wrap apply with tqdm for progress tracking
    tqdm.pandas(desc="Processing reactions")

    # Apply the processing function to all rows with a progress bar
    results = data["rxn_atom_mapped"].progress_apply(process_row)
    data["rxn"], data["atom_maps"] = zip(*results)
    return data


#################### Atom Mapping Functions - Very Clingy ####################

NON_ATOMIC_TOKENS = "0123456789().=-#/\\%"


def split_rxn(rxn):
    """Split rxn into reactants and products using the >> separator."""
    return rxn.split(">>")


def tokenize(smi):
    """Tokenize SMILES string."""
    return smi_tokenizer(smi).split(" ")


def create_atom_mapping_dict(reactants_tokenized):
    """
    1. Create absolute position of reactant tokens.
    2. Create a mapping between atomic token number (first reactant atom - where is it in the absolute reactant tokens?)
        - We need this as to remap the product atom tokens to actual tokens and not just atom number.
    """
    atom_positions = list(range(len(reactants_tokenized)))
    atomic_counter = 0
    mapping = {}

    for idx, token in enumerate(reactants_tokenized):
        if token not in NON_ATOMIC_TOKENS:
            mapping[atomic_counter] = idx
            atomic_counter += 1

    return mapping, atom_positions


def remap_products(
    products_tokenized,
    atom_map,
    mapping_dict,
):
    """
    Remaps the tokenized product SMILES to their corresponding absolute positions in the reactant.

    Args:
        products_tokenized (list): List of tokenized SMILES representing the product.
        atom_map (list): List of atom maps, linking product tokens to reactant tokens.
        mapping_dict (dict): Dictionary mapping atomic tokens to their absolute positions in the reactant.

    Returns:
        list: List of absolute positions corresponding to the tokenized product, with non-atomic tokens assigned -100.

    Notes:
        - Non-atomic tokens are assigned -100 in the mapping (to be adjusted in a later step).
        - Atomic tokens are mapped based on the atom map and then converted to absolute positions using `mapping_dict`.
    """
    mapping = []
    current_number = 0

    # Map atomic tokens to reactant numbers
    for token in products_tokenized:
        if token in NON_ATOMIC_TOKENS:
            mapping.append(-100)
        else:
            mapping.append(str(atom_map[current_number]))
            current_number += 1

    # Convert to absolute positions using the mapping dictionary
    absolute_positions = []
    for num in mapping:
        if num == -100:
            absolute_positions.append(-100)
        else:
            absolute_positions.append(mapping_dict[int(num)])
    return absolute_positions


def adjust_for_cls_token(
    reactants_numbered,
    products_remapped,
    apply_attn_sink=True,
):
    """
    Assume CLS token is loaded during model training and inference.
    We need to increment all reactant tokens by 1 and
    replace -100 tokens with 0 to point to CLS token (Attention sink manipulation).


    This could also be adapted so that the -100 tokens point to themselves.
    """
    reactants = [i + 1 for i in reactants_numbered]
    if apply_attn_sink:
        products = [i + 1 if i != -100 else 0 for i in products_remapped]
    else:  # TODO: This is a bit sketchy - Would have to test this downstream a little more :)
        raise NotImplementedError("This is not implemented yet.")
        products = [i + 1 if i != -100 else i + 1 for i in products_remapped]
    return reactants, products


def map_reactants_to_products(
    reactants_numbered,
    product_mapping,
    apply_attention_sink=True,
):
    """
    Here we want to map the REACTANT atoms to those of the product. This is so we have a triangular matrix.

    Add length of reactants + 1 (for CLS token + 1 for >> token) to all product tokens (as they are still absolute for now.)

    For each reactant number present in product_map, use index + offset. Otherwise point to CLS
    """
    # Add 1 to account for the >> token:
    offset = max(reactants_numbered) + 1
    mapping = []

    for reactant_num in reactants_numbered:
        if reactant_num in product_mapping:
            mapping.append(product_mapping.index(reactant_num) + offset)
        else:
            if apply_attention_sink:
                mapping.append(0)
            else:
                raise NotImplementedError("This is not implemented yet.")
                mapping.append(reactant_num + 1)
    return mapping


def create_atom_map(rxn, atom_mapping):
    """
    Process rxn SMILES, create atom mapping of products, annotate reactant tokens,
    create dictionary of reactant atomic tokens and their absolute positions, remap product tokens to account for
    relative position of reactant tokens, then map the reactant tokens to their product counterparts.
    """
    try:
        # Split and tokenize reaction
        reactants, products = split_rxn(rxn)
        reactants, products = tokenize(reactants), tokenize(products)

        # Create mappings
        mapping_dict, reactants_numbered = create_atom_mapping_dict(reactants)

        product_mapping = remap_products(products, atom_mapping, mapping_dict)

        # Adjust for CLS token
        reactants_numbered, product_mapping = adjust_for_cls_token(
            reactants_numbered, product_mapping
        )

        # Create full mapping
        reactant_mapping = map_reactants_to_products(
            reactants_numbered, product_mapping
        )
        reactant_mapping = [0] + reactant_mapping + [0]  # Add CLS and >> token mappings
        reactant_tokens = ["<cls>"] + reactants + [">>"]

        # Add the [SEP] token to the end of the product tokens
        product_mapping = product_mapping + [0]
        products = products + ["<sep>"]

        rxn_tokens = reactant_tokens + products
        full_attn_map = reactant_mapping + product_mapping

        assert len(rxn_tokens) == len(
            full_attn_map
        ), "Length of tokens and attn map do not match."

        return rxn_tokens, full_attn_map

    except Exception as e:
        print(f"Error processing reaction: {rxn}. Error: {e}")
        exit()


##############################################################################


#################################### MISC ####################################


def plot_reaction_mapping(rxn_tokens, atom_map):
    """Create matrix of zeros and fill with atom mapping.
    Plot as heatmap with token labels."""
    matrix = np.zeros((len(rxn_tokens), len(rxn_tokens)))

    for idx, atom in enumerate(atom_map):
        matrix[idx, atom] = 1

    plt.figure(figsize=(20, 20))
    plt.imshow(matrix, cmap="viridis")
    plt.xticks(range(len(rxn_tokens)), rxn_tokens, rotation=90)
    plt.yticks(range(len(rxn_tokens)), rxn_tokens)
    plt.tight_layout()
    plt.show()


##############################################################################


def main(path_to_data_csv):

    # Quick Assertions:
    assert os.path.exists(
        path_to_data_csv
    ), f"Path to data does not exist: {path_to_data_csv}"

    assert path_to_data_csv.endswith(
        ".csv"
    ), "Please provide a path to a CSV file containing the data. - SORRY THIS IS hard coded for now."

    # Load the data:
    data = load_csv(path_to_data_csv)

    assert (
        "rxn_atom_mapped" in data.columns
    ), "The data does not have the column 'rxn_atom_mapped'."

    data = data[["rxn_atom_mapped"]]

    # Check that we do not have any explicit hydrogens:
    check_explicit_hydrogens(data)

    # Get the canonical rxn and a list of the atom maps:
    print("Processing the atom maps for the dataset...")
    data = get_atom_maps(data)

    print("Creating full attention maps for the dataset...")
    # Create the atom maps:
    attention_maps = []
    for _, row in tqdm(data.iterrows(), total=len(data), desc="Processing Atom Maps"):
        rxn = row["rxn"]
        atom_map = row["atom_maps"]
        _, attn_map = create_atom_map(rxn, atom_map)
        attention_maps.append(attn_map)

    # Save to the dataset:
    data["full_rxn_maps"] = attention_maps

    # Drop the atom_maps column:
    data = data.drop(columns=["atom_maps"])

    # Save the data to JSON format:
    base_dir = os.path.dirname(path_to_data_csv)
    save_path = os.path.join(base_dir, "cleaned_data_with_attention_maps.json")
    data.to_json(save_path)
    print("Data saved to:", save_path)

    # Also save as csv:
    save_path_csv = os.path.join(base_dir, "cleaned_data_with_attention_maps.csv")
    data.to_csv(save_path_csv)
    print("Data saved to:", save_path_csv)


if __name__ == "__main__":
    print("Setting up the atom maps for the cyclisation dataset")

    path_to_data_to_atom_map = (
        "data/Cyclisation/processed/train.csv"  # "data/Cyclisation/processed/train.csv"
    )

    main(path_to_data_csv=path_to_data_to_atom_map)
