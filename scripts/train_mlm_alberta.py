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
import yaml
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
    
    # load the "resume_from_checkpoint" argument from YAML config file
    with open(config_path, "r") as f:
        raw_config = yaml.safe_load(f)
    resume = raw_config.get("resume_from_checkpoint", False)
    
    # Setup model:
    model_config = AlbertConfig(**config.alberta_config)
    model = AlbertForMaskedLM(model_config)

    # Setup tokenizer:
    vocab_path = "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k/vocab.txt"  # Can stay hard coded for now
    tokenizer = SmilesTokenizer(
        vocab_path,
        model_max_length=model_config.max_position_embeddings,
        padding="longest",
    )

    # Tokenize the dataset with truncation logic
    def tokenize_function(examples):
        # Tokenize the SMILES strings
        tokenized = tokenizer(examples["rxn"], truncation=True)
        return tokenized

    # Dataset setup
    if config.dataset_name == "USPTO":
        dataset_path = f"data/RXNMapperData/Training/uspto_all_reactions_{model_config.max_position_embeddings}_max_input_len.csv"
    elif config.dataset_name == "USPTO_1065k":
        dataset_path = f"data/USPTO_FULL/uspto_1065k_{model_config.max_position_embeddings}_max_input_len.csv"
    elif config.dataset_name == "full_pretraining":
        dataset_path = "data/full_pretraining_data.csv"
    elif config.dataset_name == "uspto_cyclisation":
        dataset_path = "data/uspto_cyclisation_pretraining_data.csv"
    elif config.dataset_name == "uspto_rearrangement":
        dataset_path = "data/uspto_rearrangement_pretraining_data.csv"
    elif config.dataset_name == "cyclisation_101080":
        dataset_path = "data/Cyclisation/processed/train.csv"
    elif config.dataset_name == "rearrangement_101080":
        dataset_path = "data/Rearrangement/processed/train.csv"

    elif config.dataset_name == "cyclisation_5050":
        dataset_path = "data/Cyclisation/processed_50-50_train_test/train.csv"

    elif config.dataset_name == "rearrangement_5050":
        dataset_path = "data/Rearrangement/processed_50-50_train_test/train.csv"
    else:
        raise ValueError(f"{config.dataset_name} Dataset not supported yet")

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
