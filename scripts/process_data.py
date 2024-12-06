"""

Script to process data downloaded from IBM Dropbox. 


Steps: 


"""

import pandas as pd
import re
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


def main():

    tain_data_path = "data/RXNMapperData/Training/uspto_all_reactions_training.txt"
    save_path = (
        "data/RXNMapperData/Training/uspto_all_reactions_training_less_than_510.csv"
    )
    assert os.path.exists(tain_data_path), f"Path {tain_data_path} does not exist."

    data = pd.read_csv(tain_data_path)
    data.columns = ["rxn"]

    # Tokenize the SMILES strings
    rxns = data["rxn"].tolist()

    print("Tokenising SMILES strings...")
    rxns_tokenised = [smi_tokenizer(rxn) for rxn in tqdm(rxns)]

    max_len = max([len(rxn.split()) for rxn in rxns_tokenised])
    print(f"Max tokenised length in the training Set: {max_len:,}")

    # Only keep samples with a tokenised length less than 510 (Accounting for the [CLS] and [SEP] tokens)
    theshold_len = 510
    valid_rxns = []

    for rxn in rxns_tokenised:
        if len(rxn.split()) < theshold_len:
            valid_rxns.append(rxn)

    print(
        f"Number of samples with tokenised length less than {theshold_len}: {len(valid_rxns):,}"
    )
    print(f"Total number of samples: {len(rxns_tokenised):,}")
    print(f"Percentage of samples kept: {len(valid_rxns)/len(rxns_tokenised):.2%}")

    print("Saving samples...")
    valid_rxns = [rxn.replace(" ", "") for rxn in valid_rxns]
    valid_rxns = pd.DataFrame(valid_rxns, columns=["rxn"])
    valid_rxns.to_csv(save_path, index=False)
    print(f"Samples saved to {save_path}")

    # Process the validation data and rename the original_rxn column to rxn_atom_mapped
    print("\n\nProcessing validation data...")
    val_data_path = "data/RXNMapperData/Validation/eval_schneider.json"
    val_data_save_path = "data/RXNMapperData/Validation/eval_schneider_processed.json"
    val_data = pd.read_json(val_data_path)
    val_data = val_data.rename(columns={"original_rxn": "rxn_atom_mapped"})

    # Save the processed data:
    val_data.to_json(val_data_save_path)
    print(f"Processed validation data saved to {val_data_save_path}")


if __name__ == "__main__":
    main()
