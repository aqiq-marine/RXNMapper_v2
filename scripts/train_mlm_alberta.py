"""

Training script:

Takes as input: 

1. A config file with the model configuration + Training arguments


Example run of training: 

python scripts/train_mlm_alberta.py -c configs/training_baseline.yaml
"""

from transformers import AlbertForMaskedLM, AlbertConfig, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from rxnmapper.tokenization_smiles import SmilesTokenizer
import click
from rxnmapper.training_utils import parse_config


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

    # Setup model:
    model_config = AlbertConfig(**config.alberta_config)
    model = AlbertForMaskedLM(model_config)

    # Can stay hard coded for now:
    vocab_path = (
        "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k/vocab.txt"
    )

    # TODO: Fix the tokenizer so that it just truncates SMILES Strings which are too large:
    tokenizer = SmilesTokenizer(
        vocab_path,
        max_length=model_config.max_position_embeddings,
        truncation=True,
        padding="longest",
    )

    # Tokenize the dataset with truncation logic
    def tokenize_function(examples):
        # Tokenize the SMILES strings
        tokenized = tokenizer(examples["rxn"])
        return tokenized

    # Dataset setup
    if config.dataset_name == "USPTO":
        dataset_path = (
            "data/RXNMapperData/Training/uspto_all_reactions_training_less_than_510.csv"
        )
    else:
        raise ValueError(f"{config.dataset_name} Dataset not supported Yet")

    print(f"Loading the {config.dataset_name} dataset...")
    dataset = load_dataset(
        "csv",
        data_files=dataset_path,
    )

    # Tokenize the dataset
    print("Tokenizing dataset...")
    dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=24,
    )

    # Load model from checkpoint:
    checkpoint_path = None
    if config.checkpoint_path:
        print(f"Loading model from checkpoint: {config.checkpoint_path}")
        model = AlbertForMaskedLM.from_pretrained(checkpoint_path)

    # Data collator for MLM, using a mask probability of 15%
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
    trainer.train()

    # Save the model
    trainer.save_model(config.output_dir)
    print(f"Model saved to {config.output_dir}")


if __name__ == "__main__":
    main()
    pass
