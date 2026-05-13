"""

Training script:

Takes as input: 

1. A config file with the model configuration + Training arguments


Example run of training: 

python scripts/train_mlm_alberta.py -c configs/training_baseline.yaml
"""
import os
import pandas as pd
from transformers import AlbertForMaskedLM, AlbertConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset
from rxnmapper.tokenization_smiles import SmilesTokenizer, build_vocab_from_data_stream, merge_ordered_vocabs, list_to_ord_dict, load_vocab, save_vocabulary
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
    if config.vocab_path in [None, "", "DEFAULT"]:
        vocab_path = default_vocab_path
        print(f"Using default vocab path: {default_vocab_path}")
    else:
        vocab_path = config.vocab_path
        print(f"Using user-provided vocab_path: {vocab_path}")

    if not os.path.isfile(vocab_path):
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

    final_vocab_path = config.vocab_output_path
    
    if config.auto_vocab: ## Create a vocab adapted to the new data + merge to the default
        print("Auto-vocab enabled: building vocab from data...")
        ds_stream = load_dataset(
            "csv",
            data_files=config.dataset_path,
            split="train",
            streaming=True,
            delimiter="\t",
        )

        def rxn_iter():
            for ex in ds_stream:
                yield ex["rxn"]

        new_vocab = build_vocab_from_data_stream(rxn_iter())
        new_vocab_ord_dict = list_to_ord_dict(new_vocab)

        default_vocab = load_vocab(vocab_path)
        merged_vocab = merge_ordered_vocabs(default_vocab, new_vocab_ord_dict)

        _ = save_vocabulary(merged_vocab, final_vocab_path)
        vocab_path = final_vocab_path

    
    # Setup tokenizer:
    tokenizer = SmilesTokenizer(
        vocab_path,
        model_max_length=model_config.max_position_embeddings,
        padding="longest",
    )

    # Check that vocab size matches between vocab and model config
    if model_config.vocab_size != tokenizer.vocab_size:
        print(f"Vocab size in model ({model_config.vocab_size}) " + 
               "not the same size as " +
               f"tokenizer ({tokenizer.vocab_size}), using the tokenizers'...")
        model_config.vocab_size = tokenizer.vocab_size
        model = AlbertForMaskedLM(model_config)
        print(f"Model updated to vocab size of {model_config.vocab_size}...")

    ## Load model from checkpoint:
    if config.checkpoint_path:
        print(f"Loading model from checkpoint: {config.checkpoint_path}")
        model = AlbertForMaskedLM.from_pretrained(config.checkpoint_path)
        
        if tokenizer.vocab_size != model.config.vocab_size:
            print(f"Resizing model embeddings from {model.config.vocab_size} to {tokenizer.vocab_size}")
            model.resize_token_embeddings(tokenizer.vocab_size)


    # Preprocessing of the dataset if needed
    if config.preprocess_dataset: ## To tokenize each reaction and remove too long data points
        print('Preprocessing enabled: tokenizing and filtering reactions for max model size...')
        if "dataset" not in locals():
            dataset = load_dataset(
                "csv",
                data_files=config.dataset_path,
                split="train",
                delimiter="\t",
            )
        if 'rxn' not in dataset.column_names:
            raise ValueError(f"Expected 'rxn' column in dataset, but found: {dataset.column_names}")
        
        max_input_size = model_config.max_position_embeddings - 2 # account for [CLS] and [SEP]

        def length_filter(ex):
            return len(tokenizer.tokenize(ex["rxn"])) < max_input_size
        dataset = dataset.filter(length_filter, num_proc=8)
        
        config.dataset_path = config.dataset_path.replace(
            '.csv',
            f'_preprocessed_{model_config.max_position_embeddings}.csv'
            )
        dataset.to_csv(config.dataset_path) # save for later use
    

    # load the dataset if it does not exist yet
    if "dataset" not in locals():
        print(f"Loading the {config.dataset_name} dataset...")
        dataset = load_dataset(
            "csv",
            data_files=config.dataset_path,
            split="train",
        )
    if 'rxn' not in dataset.column_names:
        raise ValueError(f"Expected 'rxn' column in dataset, but found: {dataset.column_names}")


    # Tokenize the dataset
    def tokenize_func(ex):
        return tokenizer(ex["rxn"], truncation=True, padding=False)
    dataset = dataset.remove_columns([c for c in dataset.column_names if c != "rxn"])
    print("Tokenizing dataset...")
    dataset.set_transform(tokenize_func)
    

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
        save_only_model=config.save_only_model,
        logging_first_step=config.logging_first_step,
        logging_dir=config.logging_dir,
        report_to=config.report_to,
        run_name=config.run_name,

        remove_unused_columns=False, # importand when using set_transform /lazy tokenization
        # throughput and stability
        dataloader_num_workers=8, 
        dataloader_persistent_workers=True, 
        group_by_length=True,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )

    # Train the model
    trainer.train(resume_from_checkpoint=resume)

    # Save the model:
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
