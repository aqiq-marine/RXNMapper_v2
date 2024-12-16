"""

Finetune Alberta Model with an AtomMap loss function on an attention head.


This code is VERY Proof-of-concepty

Issues: 
    - Still takes as input a MLM dataset, although no loss is calculated on it :)
    - Generates a new dataloader at every forward pass - Clearly not optimal and SLOW
    - No validation set.

"""

from transformers import AlbertForMaskedLM, AlbertConfig, Trainer, TrainingArguments
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from rxnmapper.tokenization_smiles import SmilesTokenizer
from torch.utils.data import DataLoader
import torch
import ast


class CustomTrainer(Trainer):

    def __init__(self, *args, train_atom_map_dataset=None, **kwargs):

        # Initialise the base class:
        super().__init__(*args, **kwargs)

        # Here we can add custom functionalities:
        self.train_atom_map_dataset = train_atom_map_dataset

        print(self.train_atom_map_dataset)

    def get_train_attn_map_dataloader(self, atom_map_batch_size=32):
        """
        Get dataloader for atom mapping training with vectorized collation.
        """
        if self.train_atom_map_dataset is None:
            raise ValueError(
                "Trainer: Training with Atom map requires a train_atom_map_dataset"
            )

        train_attn_map_dataset = self.train_atom_map_dataset.remove_columns(["rxn"])

        def collate_fn(batch):
            # Convert batch to pandas-like structure for vectorized operations
            input_ids = [torch.tensor(x["input_ids"]) for x in batch]
            atom_maps = [torch.tensor(x["full_rxn_maps"]) for x in batch]
            attention_masks = [torch.tensor(x["attention_mask"]) for x in batch]

            # Pad in a vectorized way using torch.nn.utils.rnn.pad_sequence
            input_ids_padded = torch.nn.utils.rnn.pad_sequence(
                input_ids, batch_first=True, padding_value=0
            )
            atom_maps_padded = torch.nn.utils.rnn.pad_sequence(
                atom_maps, batch_first=True, padding_value=0
            )
            attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
                attention_masks, batch_first=True, padding_value=0
            )

            return (input_ids_padded, attention_masks_padded, atom_maps_padded)

        return DataLoader(
            train_attn_map_dataset,
            batch_size=atom_map_batch_size,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=1,
        )

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):

        del inputs
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        # Get the atom map dataloader:
        atom_map_dataloader = self.get_train_attn_map_dataloader(
            atom_map_batch_size=32,
        )

        # randomly sample ;
        input_atom_map = next(iter(atom_map_dataloader))

        inputs_atom_map = self._prepare_inputs(input_atom_map)
        loss = self.compute_atom_map_loss(model, inputs_atom_map)

        del inputs_atom_map

        kwargs = {}

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        self.accelerator.backward(loss, **kwargs)
        # Finally we need to normalize the loss for reporting
        if num_items_in_batch is None:
            return loss.detach() / self.args.gradient_accumulation_steps

        return loss.detach()

    def cross_entropy_loss_softmaxed(self, predictions, labels_, ignore_index=-100):
        """
        Computes cross-entropy loss for softmaxed predictions in PyTorch.

        Args:
            predictions (torch.Tensor): Softmaxed predictions, shape (N, C).
            labels (torch.Tensor): One-hot encoded labels, shape (N, C).

        Returns:
            torch.Tensor: Cross-entropy loss.
        """

        batch_size, seq_len, _ = predictions.shape
        predictions = predictions.reshape(batch_size * seq_len, seq_len)

        # Turn into labels:
        labels = torch.nn.functional.one_hot(
            labels_,
            num_classes=labels_.shape[1],
        )
        labels = labels.reshape(batch_size * seq_len, seq_len)

        # Create mask for non-ignored indices
        mask = (labels_ != ignore_index).view(batch_size * seq_len)

        # Avoid log(0) by clipping
        epsilon = 1e-10
        predictions = torch.clamp(predictions, min=epsilon, max=1.0)

        # Apply mask to labels and predictions
        masked_labels = labels[mask]
        masked_predictions = predictions[mask]

        # Compute cross-entropy loss
        loss = -torch.sum(
            masked_labels * torch.log(masked_predictions)
        ) / masked_predictions.size(0)

        return loss

    def compute_atom_map_loss(self, model, inputs):

        # Setup forward pass:
        outputs = model(
            input_ids=inputs[0],
            attention_mask=inputs[1],
            output_attentions=True,
        )

        # Hard coded heads and layers for now
        layer = 10
        head = 5

        attn_output = outputs.attentions[layer][:, head, :, :]

        loss = self.cross_entropy_loss_softmaxed(
            attn_output,
            inputs[2],
            ignore_index=0,
        )

        del outputs
        del attn_output

        return loss


################################################################################


def tokenize_function(examples):
    # Tokenize the SMILES strings
    tokenized = tokenizer(examples["rxn"])
    return tokenized


def tokenize_atom_map_function(examples):
    # Tokenize the SMILES strings
    tokenized = tokenizer(examples["rxn"])

    # Tokenize the atom map:
    full_rxn_map = ast.literal_eval(examples["full_rxn_maps"])

    # Add the full_rxn_map to the tokenized data:
    tokenized["full_rxn_maps"] = full_rxn_map

    return tokenized


################################################################################


# Tokenizer setup
vocab_path = "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k/vocab.txt"
max_length = 512  # Maximum sequence length for debugging

tokenizer = SmilesTokenizer(
    vocab_path,
    # model_max_length=max_length,
    max_length=max_length,
    truncation=True,
    padding="longest",
)


# Dataset setup

#### USELESS Dataset ####
# Use much smaller dataset - No use here anyways
dataset_path = "data/Cyclisation/processed/cleaned_data_with_attention_maps.csv"
dataset = load_dataset(
    "csv",
    data_files=dataset_path,
)


# Loading the atom-mapping dataset:
atom_map_data_path = "data/Cyclisation/processed/cleaned_data_with_attention_maps.csv"


atom_map_dataset = load_dataset(
    "csv",
    data_files=atom_map_data_path,
)


# Tokenize the dataset
dataset = dataset.map(
    tokenize_function,
    batched=True,
    num_proc=24,
)

# Tokenise the atom map dataset:
atom_map_dataset = atom_map_dataset.map(
    tokenize_atom_map_function,
    batched=False,
    num_proc=24,
)


# Model setup
alberta_config_path = "configs/config.json"
model_config = AlbertConfig.from_pretrained(
    alberta_config_path, attn_implementation="eager"
)
model = AlbertForMaskedLM(model_config)

# Load model from checkpoint:
checkpoint_path = "rxnmapper/models/transformers/albert_heads_8_uspto_all_1310k"
if checkpoint_path:
    # Need to keep the attn implementation as eager for atom mapping to work:
    model = AlbertForMaskedLM.from_pretrained(checkpoint_path, config=model_config)


# Data collator for MLM, using a mask probability of 15%
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Training arguments setup
training_args = TrainingArguments(
    output_dir="trained_models/atom_map_ft_V1_model_cyclisation_101080",
    learning_rate=5e-6,
    max_steps=50_000,
    weight_decay=0.001,
    logging_steps=100,
    eval_strategy="no",
    save_strategy="steps",  # Disable saving model checkpoints
    save_steps=100,
    save_total_limit=10,
    save_only_model=True,
    per_device_train_batch_size=32,
    gradient_accumulation_steps=1,
    logging_dir="logs",
    logging_first_step=True,
    report_to=["none"],  # Disable wandb or any other reporting
    # run_name="alberta_all_rxn",
)


# Trainer setup
trainer = CustomTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    train_atom_map_dataset=atom_map_dataset["train"],
)

# Train the model
trainer.train()
