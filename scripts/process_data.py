"""

Script to process data downloaded from IBM Dropbox,
Split cyclisation and rearrangement data into train/val/test sets and save the processed data
and Create large pretraining files which include the USPTO, Cyclisation and Rearrangement data.



1. Process the USPTO data: 
    - Convert it to CSV + Keep all samples OR only keep those below size 510

2. Process the validation data:
    - Rename the original_rxn column to rxn_atom_mapped to match other datasets

3. Process the cyclisation / Rearangement data:
    - Split into train/val/test sets with 10/10/80 split

4. Create pretraining datasets:
    - USPTO + Cyclisation
    - USPTO + Rearrangement
    - Full pretraining dataset (USPTO + Cyclisation + Rearrangement)


"""

import pandas as pd
import re
from tqdm import tqdm
import os
from sklearn.model_selection import train_test_split


def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction into individual atoms and bonds.
    """
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == "".join(tokens), "SMILES string did not match tokenized pattern."
    return " ".join(tokens)


def split_data(
    data,
    train_size=0.2,
):
    """
    1. Split the data into train/val/test sets
    """
    train, test = train_test_split(
        data,
        train_size=train_size,
        random_state=42,
    )
    train, val = train_test_split(
        train,
        test_size=0.5,
        random_state=42,
    )
    return train, val, test


def save_splits(train, val, test, data_path):
    """
    Save the processed data to a new directory.
    """
    save_dir = os.path.dirname(data_path)
    save_dir = os.path.join(save_dir, "processed")
    os.makedirs(save_dir, exist_ok=True)
    train.to_csv(os.path.join(save_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(save_dir, "val.csv"), index=False)
    test.to_csv(os.path.join(save_dir, "test.csv"), index=False)
    print(f"Processed data saved to {save_dir}")


def check_tokenized_rxn_sizes(tokenized_rxn, model_size):
    """
    Check if all tokenized reactions are within the model size.
    """
    max_input_size = model_size - 2

    valid_rxns = []
    for rxn in tokenized_rxn:
        if len(rxn.split()) < max_input_size:
            valid_rxns.append(rxn)

    assert len(valid_rxns) == len(
        tokenized_rxn
    ), "Not all samples are within the model size."


def process_uspto_data(train_data_path, save_path, model_size=512):
    """
    1. Load the USPTO data
    2. Tokenize the SMILES strings
    3. Only keep samples with a tokenised length less than 510 (Accounting for the [CLS] and [SEP] tokens)
    4. Save the processed data
    """
    data = pd.read_csv(train_data_path)
    data.columns = ["rxn"]

    # Tokenize the SMILES strings
    rxns = data["rxn"].tolist()
    print("Tokenising SMILES strings...")
    rxns_tokenised = [smi_tokenizer(rxn) for rxn in tqdm(rxns)]

    max_len = max([len(rxn.split()) for rxn in rxns_tokenised])
    print(f"\nMax tokenised length in the training Set: {max_len:,}")

    # Only keep samples with a tokenised length less than 510 (Accounting for the [CLS] and [SEP] tokens)
    max_input_size = model_size - 2  # Account for the [CLS] and [SEP] tokens
    valid_rxns = []

    for rxn in rxns_tokenised:
        if len(rxn.split()) < max_input_size:
            valid_rxns.append(rxn)

    print(
        f"Number of samples with tokenised length less than {model_size}: {len(valid_rxns):,}"
    )
    print(f"Total number of samples: {len(rxns_tokenised):,}")
    print(f"Percentage of samples kept: {len(valid_rxns)/len(rxns_tokenised):.2%}")

    print("Saving samples...")
    valid_rxns = [rxn.replace(" ", "") for rxn in valid_rxns]
    valid_rxns = pd.DataFrame(valid_rxns, columns=["rxn"])
    valid_rxns.to_csv(save_path, index=False)
    print(f"Samples saved to {save_path}")


def process_validation_data(val_data_path):
    """
    1. Load the validation data
    2. Rename the original_rxn column to rxn_atom_mapped
    3. Save the processed data
    """
    val_save_dir = os.path.dirname(val_data_path)
    val_data_save_path = os.path.join(val_save_dir, "eval_schneider_processed.json")
    val_data = pd.read_json(val_data_path)
    val_data = val_data.rename(columns={"original_rxn": "rxn_atom_mapped"})

    # Save the processed data:
    val_data.to_json(val_data_save_path)
    print(f"Processed validation data saved to {val_data_save_path}")


def process_cyclisation_data(cyclisation_data_path, model_size=512):
    """
    1. Load the cyclisation data
    2. Tokenize the SMILES strings
    3. Split into train/val/test sets
    4. Save the processed data
    """
    cyclisation_data = pd.read_csv(cyclisation_data_path, index_col=0)

    # Check if the .[H][H] product is in each reaction.
    full_count = 0
    hh_count = 0
    for rxn in cyclisation_data.reaction:
        full_count += 1
        if "[H][H]" in rxn:
            hh_count += 1
    assert (
        full_count == hh_count
    ), "Not all reactions contain Explicit hydrogen products - Code should be adapted as we currently remove the explicit hydrogen product."

    # Remove ".[H][H]" and "[H][H]." from the products
    cyclisation_data["reaction"] = cyclisation_data["reaction"].apply(
        lambda x: x.replace(".[H][H]", "")
    )
    cyclisation_data["reaction"] = cyclisation_data["reaction"].apply(
        lambda x: x.replace("[H][H].", "")
    )

    # Tokenize the SMILES strings:
    rxns_tokenised = [smi_tokenizer(rxn) for rxn in tqdm(cyclisation_data.reaction)]

    max_len = max([len(rxn.split()) for rxn in rxns_tokenised])
    print(f"\nMax tokenised length in the Cyclisation Set: {max_len:,}")

    # Only keep samples with a tokenised length less than 510 (Accounting for the [CLS] and [SEP] tokens)
    check_tokenized_rxn_sizes(rxns_tokenised, model_size)

    # Only keep rxn columns and mapped_reaction columns:
    cyclisation_data = cyclisation_data[["reaction", "mapped_reaction"]]
    cyclisation_data.rename(
        columns={"reaction": "rxn", "mapped_reaction": "rxn_atom_mapped"}, inplace=True
    )

    # Split into train/val/test sets: 10/10/80:
    train, val, test = split_data(cyclisation_data, train_size=0.1)

    print("Cyclisation Splits:")
    print(f"Train size: {len(train):,}")
    print(f"Val size: {len(val):,}")
    print(f"Test size: {len(test):,}")

    save_splits(train, val, test, cyclisation_data_path)


def process_rearrangement_data(rearrangement_data_path, model_size=512):
    """
    1. Load the rearrangement data
    2. Tokenize the SMILES strings
    3. Split into train/val/test sets
    4. Save the processed data
    """
    rearrangement_data = pd.read_csv(rearrangement_data_path)

    # Check if the [H][H] product is in some of the reactions.
    full_count = 0
    hh_count = 0
    for rxn in rearrangement_data["UNMAPPED REACTION"]:
        full_count += 1
        if "[H][H]" in rxn:
            hh_count += 1

    assert (
        hh_count == 0
    ), "Some explicit hydrogen products are still present in the rearrangement data. - Code should be adapted."

    # Tokenize the SMILES strings:
    rxns_tokenised = [
        smi_tokenizer(rxn) for rxn in tqdm(rearrangement_data["UNMAPPED REACTION"])
    ]

    # Check all sizes are valid:
    check_tokenized_rxn_sizes(rxns_tokenised, model_size)

    # Only keep rxn columns and mapped_reaction columns:
    rearrangement_data = rearrangement_data[["UNMAPPED REACTION", "MAPPED REACTION"]]

    # Rename columns:
    rearrangement_data.rename(
        columns={"UNMAPPED REACTION": "rxn", "MAPPED REACTION": "rxn_atom_mapped"},
        inplace=True,
    )

    # Split into train/val/test sets: 10/10/80:
    train, val, test = split_data(rearrangement_data, train_size=0.1)

    print("Rearrangement Splits:")
    print(f"Train size: {len(train):,}")
    print(f"Val size: {len(val):,}")
    print(f"Test size: {len(test):,}")

    save_splits(train, val, test, rearrangement_data_path)


# Create a new pretraining_dataset which includes the USPTO, Cyclisation and Rearrangement data.
def create_cyclisation_pretraining_dataset():
    """

    Create a a pretraining dataset which includes the USPTO, and Cyclisation data.

    """
    uspto_data_path = (
        "data/RXNMapperData/Training/uspto_all_reactions_512_max_input_len.csv"
    )
    cyclisation_data_path = "data/Cyclisation/processed/train.csv"

    uspto_data = pd.read_csv(uspto_data_path)
    cyclisation_data = pd.read_csv(cyclisation_data_path)

    uspto_data = uspto_data["rxn"]
    cyclisation_data = cyclisation_data["rxn"]

    # Combine all the data:
    pretraining_data = pd.concat([uspto_data, cyclisation_data])

    # Save the pretraining data:
    save_path = "data/uspto_cyclisation_pretraining_data.csv"
    pretraining_data.to_csv(save_path, index=False)


def create_rearrangement_pretraining_dataset():
    """

    Create a a pretraining dataset which includes the USPTO and Rearrangement data.

    """
    uspto_data_path = (
        "data/RXNMapperData/Training/uspto_all_reactions_512_max_input_len.csv"
    )
    rearrangement_data_path = "data/Rearrangement/processed/train.csv"

    uspto_data = pd.read_csv(uspto_data_path)
    rearrangement_data = pd.read_csv(rearrangement_data_path)

    uspto_data = uspto_data["rxn"]
    rearrangement_data = rearrangement_data["rxn"]

    # Combine all the data:
    pretraining_data = pd.concat([uspto_data, rearrangement_data])

    # Save the pretraining data:
    save_path = "data/uspto_rearrangement_pretraining_data.csv"
    pretraining_data.to_csv(save_path, index=False)


def create_full_pretraining_dataset():
    """

    Create a a pretraining dataset which includes the USPTO, Cyclisation and Rearrangement data.

    """
    uspto_data_path = (
        "data/RXNMapperData/Training/uspto_all_reactions_512_max_input_len.csv"
    )
    cyclisation_data_path = "data/Cyclisation/processed/train.csv"
    rearrangement_data_path = "data/Rearrangement/processed/train.csv"

    uspto_data = pd.read_csv(uspto_data_path)
    cyclisation_data = pd.read_csv(cyclisation_data_path)
    rearrangement_data = pd.read_csv(rearrangement_data_path)

    uspto_data = uspto_data["rxn"]
    cyclisation_data = cyclisation_data["rxn"]
    rearrangement_data = rearrangement_data["rxn"]

    # Combine all the data:
    pretraining_data = pd.concat([uspto_data, cyclisation_data, rearrangement_data])

    # Save the pretraining data:
    save_path = "data/full_pretraining_data.csv"
    pretraining_data.to_csv(save_path, index=False)


def main(model_size=512):
    """
    Args:
        - model_size: Max size of inputs.
    """
    # Setup USPTO
    train_data_path = "data/RXNMapperData/Training/uspto_all_reactions_training.txt"
    save_path = f"data/RXNMapperData/Training/uspto_all_reactions_{model_size}_max_input_len.csv"

    assert os.path.exists(
        train_data_path
    ), "USPTO data not found. Please download the data from IBM Dropbox."

    print("\n", "-" * 50)
    print("Processing USPTO data...")
    process_uspto_data(train_data_path, save_path, model_size=model_size)

    # Setup Schneiders Validation:
    print("\n", "-" * 50)
    print("Processing validation data...")
    val_data_path = "data/RXNMapperData/Validation/eval_schneider.json"
    process_validation_data(val_data_path)

    # Setup Cyclisation data:
    print("\n", "-" * 50)
    print("Processing Cyclisation data...")
    cyclisation_data_path = "data/Cyclisation/df_cyclization_1506_first_uspto_mol.csv"
    assert os.path.exists(
        cyclisation_data_path
    ), "Cyclisation data not found. Please download the data."
    process_cyclisation_data(cyclisation_data_path, model_size=model_size)

    # Setup Rearrangement data:
    print("\n", "-" * 50)
    print("Processing rearrangement data...")
    rearrangement_data_path = (
        "data/Rearrangement/df_simple_rearrangement_1506_first_uspto_mol.csv"
    )
    assert os.path.exists(
        rearrangement_data_path
    ), "Rearrangement data not found. Please download the data."
    process_rearrangement_data(rearrangement_data_path, model_size=model_size)

    print("Creating Pretraining datasets...")
    # Setup the full pretraining dataset:
    create_full_pretraining_dataset()

    # Setup USPTO + Cyclisation dataset:
    create_cyclisation_pretraining_dataset()

    # Setup USPTO + Rearrangement dataset:
    create_rearrangement_pretraining_dataset()


if __name__ == "__main__":
    main()
