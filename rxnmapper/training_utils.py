"""

Training utility functions:

"""

from dataclasses import dataclass
import yaml
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.data.data_collator import pad_without_fast_tokenizer_warning
from transformers.utils.import_utils import is_datasets_available
from transformers.trainer_utils import seed_worker
from transformers import Trainer
from transformers import DataCollatorWithPadding
from typing import Any, Dict, List


@dataclass
class Config:
    """
    Dataclass for configuration parameters
    """

    # Basic params:
    dataset_name: str
    checkpoint_path: str

    # Model config:
    alberta_config: dict

    # Atom mapping parameters:
    model_path: str
    model_type: str
    attention_multiplier: float
    head: int
    layers: list

    # Masked language modelling parameters:
    mlm_probability: float

    # Training parameters:
    output_dir: str
    logging_dir: str
    report_to: list
    run_name: str

    learning_rate: float
    weight_decay: float
    batch_size: int
    accum_steps: int

    max_steps: int
    warmup_steps: int
    logging_steps: int
    save_step: int

    eval_strategy: str
    save_strategy: str
    logging_first_step: bool
    save_total_limit: int
    save_only_model: bool


def parse_config(config_path):
    """
    Parse the configuration file and return a Config object
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return Config(**config)


#################### Custom Training with Atom maps ####################
class CustomTrainer(Trainer):

    def __init__(self, head_to_train, layer_to_train, *args, **kwargs):

        # Initialise the base class:
        super().__init__(*args, **kwargs)

        self.head = head_to_train
        self.layer = layer_to_train

        print(f"Training the model with head: {self.head} and layer: {self.layer}")

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

    def cross_entropy_loss_softmaxed(
        self,
        predictions,
        labels_,
        ignore_index=-100,
    ):
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

    def cross_entropy_loss_using_torch_fn(
        self,
        predictions,
        labels_,
        ignore_index=-100,
    ):
        """
            Computes cross-entropy loss using PyTorch's built-in function for pre-softmaxed predictions.

        Args:
            predictions (torch.Tensor): Softmaxed predictions, shape (N, T, C).
            labels_ (torch.Tensor): Integer class labels, shape (N, T).
            ignore_index (int): Index to ignore when computing loss.

        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        raise NotImplementedError(
            "This function is not implemented yet. Use the custom cross-entropy loss function."
        )
        # Avoid log(0) by clamping softmaxed predictions

        epsilon = 1e-10
        predictions = torch.clamp(predictions, min=epsilon, max=1.0)

        # Convert softmax probabilities to logits - Necessary for PyTorch's built-in cross-entropy loss
        logits = torch.log(predictions)

        # Reshape logits and labels for cross-entropy

        batch_size, seq_len, num_classes = logits.shape
        logits = logits.view(-1, num_classes)  # Shape (N*T, C)
        labels_ = labels_.view(-1)  # Shape (N*T)

        # Compute cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits,
            labels_,
            ignore_index=ignore_index,
            reduction="mean",  # Compute mean loss over valid elements
        )
        return loss

    def compute_atom_map_loss(
        self,
        model,
        inputs,
        use_torch_fn=False,
    ):
        """
        Calculate a Cross-Entropy loss for the atom maps in the reaction.
        """

        # Forward pass:
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_attentions=True,
        )

        # Extract the attention output for the specified layer and head:
        attn_output = outputs.attentions[self.layer][:, self.head, :, :]

        # Either use Custom Cross entropy (no softmax)
        if use_torch_fn:
            # Or use PyTorch's built-in cross entropy loss with a log operation on the softmaxed logits :)
            loss = self.cross_entropy_loss_using_torch_fn(
                attn_output,
                inputs["full_rxn_maps"],
                ignore_index=0,
            )
        else:
            loss = self.cross_entropy_loss_softmaxed(
                attn_output,
                inputs["full_rxn_maps"],
                ignore_index=0,
            )

        del outputs
        del attn_output

        return loss


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
