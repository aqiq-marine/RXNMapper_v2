from transformers import AlbertForMaskedLM, AlbertConfig, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from rxnmapper.tokenization_smiles import SmilesTokenizer


# Tokenize the dataset with truncation logic
def tokenize_function(examples):
    # Tokenize the SMILES strings
    tokenized = tokenizer(examples["rxn"])
    return tokenized


# Tokenizer setup
vocab_path = "data/vocab.txt"
max_length = 512  # Maximum sequence length for debugging

# TODO: Fix the tokenizer so that it just truncates SMILES Strings which are too large:
tokenizer = SmilesTokenizer(
    vocab_path,
    # model_max_length=max_length,
    max_length=max_length,
    truncation=True,
    padding="longest",
)


# Dataset setup
# dataset_path = "/home/sacharfd/RXNMapper_2/data/uspto_all_reactions_training.csv"
dataset_path = "data/uspto_all_reactions_training_511.csv"
dataset = load_dataset(
    "csv",
    data_files=dataset_path,
)

# Tokenize the dataset
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=24,
)

# Model setup
alberta_config_path = "configs/config.json"
model_config = AlbertConfig.from_pretrained(alberta_config_path)
model = AlbertForMaskedLM(model_config)

# Load model from checkpoint:
checkpoint_path = None
if checkpoint_path:
    model = AlbertForMaskedLM.from_pretrained(checkpoint_path)

# Data collator for MLM, using a mask probability of 15%
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments setup
training_args = TrainingArguments(
    output_dir="trained_models/alberta_all_rxn_512_limit_save_every_100_steps_for_double_decent_4_big_lr",
    learning_rate=5e-4,
    max_steps=1_500_000,
    weight_decay=0.001,
    logging_steps=100,
    eval_strategy="no",
    save_strategy="steps",
    save_steps=200,
    save_total_limit=500,
    save_only_model=True,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=4,
    logging_dir="logs",
    logging_first_step=True,
    report_to="wandb",
    run_name="alberta_all_rxn",
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
