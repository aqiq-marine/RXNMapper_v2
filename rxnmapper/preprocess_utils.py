from datasets import Dataset
from rxnmapper.tokenization_smiles import SmilesTokenizer


def preprocess_rxn_smi(rxn_smi:str, tokenizer: SmilesTokenizer) -> str:
    '''
    Convert a reaction smiles into space-separated tokens of the reaction
    '''
    tokens = tokenizer.basic_tokenizer.tokenize(rxn_smi)
    return tokenizer.convert_tokens_to_string(tokens)


def preprocess_dataset(dataset: Dataset, tokenizer: SmilesTokenizer) -> Dataset:
    '''
    Take a Dataset object and convert all reaction smiles from its 'rxn' field
    into space-separated tokens in 'rxn_tokenized' field
    '''
    def preprocess(example):
        return {"rxn_tokenized": preprocess_rxn_smi(example["rxn"], tokenizer)}
    return dataset.map(preprocess)