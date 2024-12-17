"""

Finetune Alberta Model with an AtomMap loss function on an attention head.


TODO: 
    - Have the user choose the head/layer to use for finetuning atom-maps :) 

"""

from transformers import AlbertForMaskedLM, AlbertConfig, TrainingArguments
from datasets import load_dataset
from rxnmapper.training_utils import CustomTrainer, CustomDataLoader, parse_config
from rxnmapper.tokenization_smiles import SmilesTokenizer
import ast
import click


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

    # Setup the model:
    model_config = AlbertConfig(
        **config.alberta_config, attn_implementation="eager"
    )  # Eager attention so that the attn can be outputted.
    model = AlbertForMaskedLM(model_config)

    # Setup the tokenizer:
    vocab_path = (
        "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k/vocab.txt"
    )
    tokenizer = SmilesTokenizer(
        vocab_path,
        model_max_length=model_config.max_position_embeddings,
        padding="longest",
    )

    def tokenize_atom_map_function(examples):
        # Tokenize the SMILES strings
        tokenized = tokenizer(examples["rxn"], truncation=True)

        # Tokenize the atom map:
        full_rxn_map = ast.literal_eval(examples["full_rxn_maps"])

        # Add the full_rxn_map to the tokenized data:
        tokenized["full_rxn_maps"] = full_rxn_map
        return tokenized

        #############################

    # Load the datasets: TODO - Change depending on dataset chosen
    if config.dataset_name == "cyclisation_5050":
        atom_map_data_path = "data/Cyclisation/processed_50-50_train_test/cleaned_data_with_attention_maps.csv"
    elif config.dataset_name == "cyclisation_101080_only":
        atom_map_data_path = (
            "data/Cyclisation/processed/cleaned_data_with_attention_maps.csv"
        )

    elif config.dataset_name == "cyclisation_101080":
        atom_map_data_path = (
            "data/Cyclisation/processed/cleaned_data_with_attention_maps.csv"
        )

    elif config.dataset_name == "rearrangement_5050":
        atom_map_data_path = "data/Rearrangement/processed_50-50_train_test/cleaned_data_with_attention_maps.csv"
        raise NotImplementedError(
            "Rearrangement dataset not implemented yet - Still have some bugs in the extraction of atom-maps"
        )

    else:
        raise ValueError(
            f"{config.dataset_name} Dataset not supported yet. Please choose from: cyclisation_5050_only, cyclisation_101080_only, rearrangement_5050_only"
        )

    print(f"Loading the {config.dataset_name} dataset...")
    atom_map_dataset = load_dataset(
        "csv",
        data_files=atom_map_data_path,
    )

    # Tokenise the atom map dataset:
    atom_map_dataset = atom_map_dataset.map(
        tokenize_atom_map_function,
        batched=False,
        num_proc=24,
    )

    # Load model from checkpoint:
    if config.checkpoint_path:
        print(f"Loading model from checkpoint: {config.checkpoint_path}")
        model = AlbertForMaskedLM.from_pretrained(
            config.checkpoint_path,
            attn_implementation="eager",
        )

    # checkpoint_path = "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k"
    # if checkpoint_path:
    #     # Need to keep the attn implementation as eager for atom mapping to work:
    #     model = AlbertForMaskedLM.from_pretrained(checkpoint_path, config=model_config)

    # Setup the data loader:
    data_collator = CustomDataLoader(tokenizer=tokenizer)

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

    # Trainer setup:
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=atom_map_dataset["train"],
    )

    # Train the model
    trainer.train()

    # Save the model:
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
