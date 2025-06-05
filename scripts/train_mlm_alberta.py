"""

Training script:

Takes as input: 

1. A config file with the model configuration + Training arguments


Example run of training: 

python scripts/train_mlm_alberta.py -c configs/training_baseline.yaml
"""
import pandas as pd
from transformers import AlbertForMaskedLM, AlbertConfig, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from rxnmapper.tokenization_smiles import SmilesTokenizer, build_vocab_from_data, merge_ordered_vocabs, list_to_ord_dict, load_vocab, save_vocabulary
import click
import yaml
from rxnmapper.training_utils import parse_config
from rxnmapper.preprocess_utils import preprocess_dataset


@click.command()
@click.option(
    "--config_path",
    "-c",
    required=True,
    help="Path to training config file",
    type=str,
)

def main(config_path: str):

    config = parse_config(config_path)
    
    # load the "resume_from_checkpoint" argument from YAML config file
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    resume = raw_config.get("resume_from_checkpoint", False)
    
    # Setup model:
    model_config = AlbertConfig(**config.alberta_config)
    model = AlbertForMaskedLM(model_config)
    
    # vocab path
    default_vocab_path = "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k/vocab.txt"
    final_vocab_path = config.vocab_output_path
    
    if config.auto_vocab: ## Create a vocab adapted to the new data + merge to the default
        print("Auto-vocab enabled: building vocab from data...")
        dataset = load_dataset(
            "csv",
            data_files=config.dataset_path,
            split="train",
        )
        smiles_list = dataset['rxn']
        new_vocab = build_vocab_from_data(smiles_list)
        new_vocab_ord_dict = list_to_ord_dict(new_vocab)

        default_vocab = load_vocab(default_vocab_path)
        merged_vocab = merge_ordered_vocabs(default_vocab, new_vocab_ord_dict)

        _ = save_vocabulary(merged_vocab, final_vocab_path)
        vocab_path = final_vocab_path

    else:
        vocab_path = default_vocab_path
    
    # Setup tokenizer:
    tokenizer = SmilesTokenizer(
        vocab_path,
        model_max_length=model_config.max_position_embeddings,
        padding="longest",
    )

    # Preprocessing of the dataset if needed
    if config.preprocess_dataset: ## To tokenize each reaction and remove too long data points
        print('Preprocessing enabled: tokenizing and filtering reactions for max model size...')
        if "dataset" not in locals():
            dataset = load_dataset(
                "csv",
                data_files=config.dataset_path,
                split="train",
            )
        if 'rxn' not in dataset.column_names:
            raise ValueError(f"Expected 'rxn' column in dataset, but found: {dataset.column_names}")
        
        tok_dataset = preprocess_dataset(dataset, tokenizer)

        max_input_size = model_config.max_position_embeddings - 2 # account for [CLS] and [SEP]

        def length_filter(example):
            return (len(example["rxn_tokenized"].split())) < max_input_size

        filtered_dataset = tok_dataset.filter(length_filter)

        config.dataset_path = config.dataset_path.replace(
            '.csv',
            f'_preprocessed_{model_config.max_position_embeddings}.csv'
            )
        filtered_dataset.to_csv(config.dataset_path)


    # Tokenize the dataset with truncation logic
    def tokenize_function(examples):
        # Tokenize the SMILES strings
        tokenized = tokenizer(examples["rxn"], truncation=True)
        return tokenized

    # load the dataset if it does not exist yet
    if "dataset" not in locals():
        dataset = load_dataset(
            "csv",
            data_files=config.dataset_path,
            split="train",
        )
    if 'rxn' not in dataset.column_names:
        raise ValueError(f"Expected 'rxn' column in dataset, but found: {dataset.column_names}")



    ## Dataset setup
    #if config.dataset_name == "USPTO":
    #    dataset_path = f"data/RXNMapperData/Training/uspto_all_reactions_{model_config.max_position_embeddings}_max_input_len.csv"
    #elif config.dataset_name == "USPTO_1065k":
    #    dataset_path = f"data/USPTO_FULL/uspto_1065k_{model_config.max_position_embeddings}_max_input_len.csv"
    #elif config.dataset_name == "full_pretraining":
    #    dataset_path = "data/full_pretraining_data.csv"
    #elif config.dataset_name == "uspto_cyclisation":
    #    dataset_path = "data/uspto_cyclisation_pretraining_data.csv"
    #elif config.dataset_name == "uspto_rearrangement":
    #    dataset_path = "data/uspto_rearrangement_pretraining_data.csv"
    #elif config.dataset_name == "cyclisation_101080":
    #    dataset_path = "data/Cyclisation/processed/train.csv"
    #elif config.dataset_name == "rearrangement_101080":
    #    dataset_path = "data/Rearrangement/processed/train.csv"
    #
    #elif config.dataset_name == "cyclisation_5050":
    #    dataset_path = "data/Cyclisation/processed_50-50_train_test/train.csv"
    #
    #elif config.dataset_name == "rearrangement_5050":
    #    dataset_path = "data/Rearrangement/processed_50-50_train_test/train.csv"
    #else:
    #    raise ValueError(f"{config.dataset_name} Dataset not supported yet")

    print(f"Loading the {config.dataset_name} dataset...")
    dataset = load_dataset(
        "csv",
        data_files=config.dataset_path,
        split="train",
    )

    # Tokenize the dataset
    print("Tokenizing dataset...")
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=24,
    )

    # Load model from checkpoint:
    if config.checkpoint_path:
        print(f"Loading model from checkpoint: {config.checkpoint_path}")
        model = AlbertForMaskedLM.from_pretrained(config.checkpoint_path)

    # Setup the MLM Data collator - This automatically masks tokens within the data-loader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=config.mlm_probability,
    )

    # Training arguments setup
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.accum_steps,
        max_steps=config.max_steps,
        eval_strategy=config.eval_strategy,
        save_strategy=config.save_strategy,
        logging_steps=config.logging_steps,
        save_steps=config.save_step,
        save_total_limit=config.save_total_limit,
        save_only_model=True,
        logging_first_step=True,
        logging_dir=config.logging_dir,
        report_to=config.report_to,
        run_name=config.run_name,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset["train"],
    )

    # Train the model
    trainer.train(resume_from_checkpoint=resume)

    # Save the model:
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
