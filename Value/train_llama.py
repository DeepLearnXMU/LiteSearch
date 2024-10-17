import sys
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, LlamaForSequenceClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from transformers import Trainer, TrainingArguments
from accelerate import Accelerator, DistributedType
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from torch import nn
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

# Set up training arguments
training_args = TrainingArguments(
    output_dir=sys.argv[5],
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,
    save_steps=0.2,
    save_total_limit=4,
    eval_strategy="steps",
    eval_steps=0.2,
    learning_rate=5e-6,
    fp16=True,
    weight_decay=0.01,
    warmup_steps=100,
    max_steps=0, # note: only for small datasize
    deepspeed=sys.argv[2],
    save_only_model=True
)
HEAD_LR = 5e-5 # special learning rate for head

# Load model, tokenizer, and dataset
model_name = sys.argv[1]
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
config.pad_token_id = config.eos_token_id

train_file = sys.argv[3]
valid_file = sys.argv[4]
dataset = load_dataset(
    "json",
    data_files={"train": train_file,
                "valid": valid_file},
)

input_column = "text"
label_column = "label"
def preprocess_function(examples):
    inputs, targets = [], []
    for i in range(len(examples[input_column])):
        inputs.append(examples[input_column][i])
        targets.append(1. if examples[label_column][i] > 0 else -1.)

    model_inputs = tokenizer(inputs, padding=True, max_length=1024, truncation=True)
    model_inputs["labels"] = targets
    return model_inputs

with training_args.main_process_first(desc="dataset map tokenizer"):
    train_dataset = dataset["train"].map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=[input_column, label_column],
        desc="Running tokenizer on train dataset",
    )

with training_args.main_process_first(desc="dataset map tokenizer"):
    valid_dataset = dataset["valid"].map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=[input_column, label_column],
        desc="Running tokenizer on valid dataset",
    )

if config.pad_token_id is None:
    config.pad_token_id = tokenizer.eos_token_id
if config.num_labels != 1:
    config.num_labels = 1
model = LlamaForSequenceClassification.from_pretrained(model_name, config=config, torch_dtype=torch.float16)

# for param in model.model.parameters():
#     param.requires_grad = False


from transformers.optimization import Adafactor, AdamW
from transformers.trainer_pt_utils import get_parameter_names
from transformers.utils import is_sagemaker_mp_enabled
class MyTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            head_parameters = [name for name, _ in self.model.named_parameters() if "score" in name]
            # print("*** head parameters ***")
            # print(head_parameters)
            decay_parameters = get_parameter_names(self.model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if n in decay_parameters and n not in head_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if n not in decay_parameters and n not in head_parameters],
                    "weight_decay": 0.0,
                    "lr": self.args.learning_rate,
                },
                {
                    "params": [p for n, p in model.named_parameters() if n in head_parameters],
                    "weight_decay": self.args.weight_decay,
                    "lr": HEAD_LR,
                },
            ]
            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            # optimizer_kwargs["lr"] = self.args.learning_rate
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer


def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded

@dataclass
class DataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return batch

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer,
)

# Initialize Trainer
trainer = MyTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()