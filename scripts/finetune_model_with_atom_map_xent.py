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
from transformers import (
    DataCollatorWithPadding,
)
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils.import_utils import is_datasets_available
from transformers.trainer_utils import seed_worker

from rxnmapper.tokenization_smiles import SmilesTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
import ast
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


class CustomTrainer(Trainer):

    def __init__(self, *args, **kwargs):

        # Initialise the base class:
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            # Keep the RXN Maps for training:
            self.label_names.append("full_rxn_maps")
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )
        # exit()

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def training_step(
        self,
        model,
        inputs,
        num_items_in_batch=None,
    ):

        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()

        inputs_prepped = self._prepare_inputs(inputs)

        loss = self.compute_atom_map_loss(model, inputs_prepped)

        del inputs

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
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )

        # Hard coded heads and layers for now
        layer = 10
        head = 5

        attn_output = outputs.attentions[layer][:, head, :, :]

        loss = self.cross_entropy_loss_softmaxed(
            attn_output,
            inputs["full_rxn_maps"],
            ignore_index=0,
        )

        del outputs
        del attn_output

        return loss


### Setup new Data Loader ###


class CustomDataLoader(DataCollatorWithPadding):

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:

        # Get all the atom maps:
        rxn_atom_maps = [torch.tensor(rxn["full_rxn_maps"]) for rxn in features]

        # Get all the other features - not including the full_rxn_maps:
        features = [
            {k: v for k, v in rxn.items() if k != "full_rxn_maps"} for rxn in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # No we can pad the rxn_atom_maps the same length - with 0s:
        rxn_atom_maps = torch.nn.utils.rnn.pad_sequence(
            rxn_atom_maps, batch_first=True, padding_value=0
        )
        max_length = batch["input_ids"].shape[1]
        assert (
            rxn_atom_maps.shape[1] == max_length
        ), "Error in the padding of the Atom Maps :( - Have a look at the code. "

        # Add the rxn_atom_maps to the batch:
        batch["full_rxn_maps"] = rxn_atom_maps

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]
        return batch

    pass


#############################


################################################################################


def tokenize_function(examples):
    # Tokenize the SMILES strings
    tokenized = tokenizer(examples["rxn"], truncation=True)
    return tokenized


def tokenize_atom_map_function(examples):
    # Tokenize the SMILES strings
    tokenized = tokenizer(examples["rxn"], truncation=True)

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
    model_max_length=max_length,
    truncation=True,
    padding="longest",
)


# Dataset setup

#### USELESS Dataset ####
# Use much smaller dataset - No use here anyways
# dataset_path = "data/Cyclisation/processed/cleaned_data_with_attention_maps.csv"
# dataset = load_dataset(
#     "csv",
#     data_files=dataset_path,
# )


# Loading the atom-mapping dataset:
atom_map_data_path = "data/Cyclisation/processed/cleaned_data_with_attention_maps.csv"


atom_map_dataset = load_dataset(
    "csv",
    data_files=atom_map_data_path,
)


# # Tokenize the dataset
# dataset = dataset.map(
#     tokenize_function,
#     batched=True,
#     num_proc=24,
# )

# Tokenise the atom map dataset:
atom_map_dataset = atom_map_dataset.map(
    tokenize_atom_map_function,
    batched=False,
    num_proc=24,
)

# print(atom_map_dataset)
# exit()
print(atom_map_dataset["train"][0])
# exit()


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
data_collator = CustomDataLoader(tokenizer=tokenizer)

# Training arguments setup
training_args = TrainingArguments(
    output_dir="trained_models/testing_new_xent_loss_fxn",
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
    # train_dataset=dataset["train"],
    train_dataset=atom_map_dataset["train"],
)

# Train the model
trainer.train()
