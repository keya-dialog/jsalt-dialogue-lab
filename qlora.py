"""
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

This is a modified script from the artidoro/qlora repository.
Additional features:
    - loading multi_woz_v22 dataset as:
        - turn prediction word by word conditioned on dialogue history
        - as next word predictions from full dialogues
        For MultiWoz dataset description see the dataset card on HF:
        https://huggingface.co/datasets/multi_woz_v22
    - scripts with sensible parameters for multi_woz_v22
        - finetuning
        - generation
    - scripts and fixes to use pythia (small) models for debugging
    - Merge script of PEFT checkpoints and the base model weights. See merge_peft.py.
    - Ability to define LoRa modules with regexp.

"""
if __name__ == "__main__":
    print(f"HOSTNAME {__import__('socket').gethostname()}", flush=True)
from collections import defaultdict
from pathlib import Path
import re
import glob
import copy
import json
import os
from os.path import exists, join, isdir
from dataclasses import dataclass, field
import sys
from typing import Optional, Dict, Sequence
import numpy as np
from tqdm import tqdm
import logging
import bitsandbytes as bnb
import pandas as pd

import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from datasets import load_dataset, Dataset
import evaluate

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from mwzeval.metrics import Evaluator as MWZEvaluator


torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
# MULTI_WOZ_V22 constants
SYSTEM_SPK_ID = 1  # Multi_woz_v22
BOT_PREFIX = "bot> "
USR_PREFIX = "user> "


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="EleutherAI/pythia-12b")
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "PEFT checkpoint_dir at given step. Note the base model is not there so specify the base modelvia model_name_or_path "
        },
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )


def group_turns_by_dialogue(turns, single_turn=False):
    JSON_SUFFIX_LEN = len(".json")
    d = defaultdict(list)
    for t in turns:
        # dialogue_id compatible with tomiinek/MultiWOZ_Evaluation scripts
        dialogue_id = t["dialogue_id"][:-JSON_SUFFIX_LEN].lower()
        if single_turn:
            t["turn_id"] = 0  # we generate whole dialogues - this is a hack
        d[dialogue_id].append(t)

    for turns in d.values():
        turns = sorted(turns, key=lambda t: t["turn_id"])
    return d


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=2048,
        metadata={
            "help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    target_max_len: int = field(
        default=2048,
        metadata={
            "help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    dataset: str = field(
        default="multi_woz_v22",
        metadata={
            "help": "Which dataset to finetune on. See datamodule for options e.g.: alpaca"
        },
    )
    dataset_format: Optional[str] = field(
        default="multi_woz_v22_turns",
        metadata={
            "help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf|multi_woz_v22_turns|multi_woz_v22_dialogs]"
        },
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )
    print_turns: int = field(default=10)
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    lora_modules: str = field(
        default="linear",
        metadata={
            "help": "'linear' was the default for original qlora script. "
            "You can choose from ['linear', 'fnn', 'attention', 'regexp_YOURPATTERN']. "
            "For example 'attention' value is equivalent to 'regexp_attn|attention|query|key|value'. "
            "This is a heuristic based on the base model's naming convention for its layers. Be careful."
        },
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    output_dir: str = field(
        default="./output", metadata={"help": "The output dir for logs and checkpoints"}
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    max_steps: int = field(
        default=650, metadata={"help": "How many optimizer update steps to take"}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learnign rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=200, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )


@dataclass
class GenerationArguments:
    # Length arguments
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={
            "help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
            "if predict_with_generate is set."
        },
    )
    min_new_tokens: Optional[int] = field(
        default=None, metadata={"help": "Minimum number of new tokens to generate."}
    )

    # Generation strategy
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    # Hyperparameters for logit manipulation
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def find_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            name = name.lower()
            if (
                args.lora_modules == "linear"
                or (
                    args.lora_modules == "attention"
                    and re.search("attn|attention|query|key|value", name)
                )
                or (args.lora_modules == "ffn" and re.search("mlp|ffn", name))
                or (
                    args.lora_modules.startswith("regexp_")
                    and re.search(args.lora_modules[len("regexp_") :], name)
                )
            ):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkpoint...", file=sys.stderr)
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(
                state.best_model_checkpoint, "adapter_model"
            )
        else:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        touch(join(args.output_dir, "completed"))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args, checkpoint_dir):

    n_gpus = torch.cuda.device_count()
    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    if args.full_finetune:
        assert args.bits in [16, 32]

    print(f"loading base model {args.model_name_or_path}...", file=sys.stderr)
    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=args.trust_remote_code,
        use_auth_token=args.use_auth_token,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80, file=sys.stderr)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16",
                file=sys.stderr,
            )
            print("=" * 80, file=sys.stderr)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    if not args.full_finetune:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.", file=sys.stderr)
            model = PeftModel.from_pretrained(
                model, join(checkpoint_dir, "adapter_model"), is_trainable=True
            )
        else:
            lora_modules = find_linear_names(args, model)
            print(f"adding LoRA modules {lora_modules=}", file=sys.stderr)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=lora_modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}",
        file=sys.stderr,
    )


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [
            f"{self.tokenizer.bos_token}{example['input']}" for example in instances
        ]
        targets = [
            f"{example['output']}{self.tokenizer.eos_token}" for example in instances
        ]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + copy.deepcopy(tokenized_target)
                        )
                    )
                else:
                    labels.append(
                        torch.tensor(copy.deepcopy(tokenized_source + tokenized_target))
                    )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    We kept only multi_woz_v22_turns and multi_woz_v22_dialogs format for readibility.
    See github.com/artidoro/qlora for using qlora with other datasets

    Datasets are expected to have the following columns: { `input`, `output`, `dialogue_id` }
    For the turn version we also add the `turn_id` column.
    See the MultiWOZ 2.2 dataset at the HuggingFace dataset viewer
        https://huggingface.co/datasets/multi_woz_v22/viewer/v2.2/train?row=0.
    """

    def load_data(dataset_name):
        return load_dataset("multi_woz_v22", streaming=False)

    def format_dataset(dataset, dataset_format):
        if dataset_format == "multi_woz_v22_turns" and args.dataset == "multi_woz_v22":

            def extract_spks_utts(dialog):
                turns = dialog["turns"]
                speakers = turns["speaker"]
                utterances = turns["utterance"]
                assert len(speakers) == len(utterances), (speaker, utterances)
                dialogue_id = dialog["dialogue_id"]
                return {
                    "dialogue_id": dialogue_id,
                    "speakers": speakers,
                    "utterances": utterances,
                }

            def batched_dial2turns(dialogs):
                batches = {"input": [], "output": [], "dialogue_id": [], "turn_id": []}
                speakerss = dialogs["speakers"]
                utterancess = dialogs["utterances"]
                dialogue_ids = dialogs["dialogue_id"]
                for speakers, utterances, dialogue_id in zip(
                    speakerss, utterancess, dialogue_ids
                ):
                    inps, tgts, dids, tids = dial2turns(
                        speakers, utterances, dialogue_id
                    )
                    batches["input"].extend(inps)
                    batches["output"].extend(tgts)
                    batches["dialogue_id"].extend(dids)
                    batches["turn_id"].extend(tids)
                return batches

            def dial2turns(speakers, utterances, dialogue_id):
                inputs, targets, turn_ids, dialog_history = [], [], [], []
                last_gen_id = -1
                for turn_id, spk_id in enumerate(speakers):
                    if spk_id != SYSTEM_SPK_ID:
                        continue  # for user
                    else:  # system turn: spk_id == 1
                        assert (
                            last_gen_id == turn_id - 2
                        ), f"{last_gen_id} vs {turn_id} for\n{speakers=}\n{utterances=}"
                        last_gen_id = turn_id
                        dialog_history.append(f"{USR_PREFIX}{utterances[turn_id -1]}")
                        dialog_history.append(f"{BOT_PREFIX}")  # prompt for bot
                        dial_hist_str = "\n".join(dialog_history)
                        inputs.append(dial_hist_str)
                        targets.append(dial_hist_str + utterances[turn_id])
                        turn_ids.append(turn_id)
                        dialog_history[-1] = f"{BOT_PREFIX}{utterances[turn_id]}"
                return (
                    inputs,
                    targets,
                    [dialogue_id] * len(turn_ids),
                    turn_ids,
                )

            dataset = (
                dataset.map(extract_spks_utts)
                .remove_columns(("turns", "services"))
                .filter(lambda x: any([x == SYSTEM_SPK_ID for x in x["speakers"]]))
                .map(
                    batched_dial2turns,
                    batched=True,
                    remove_columns=("speakers", "utterances"),
                )
            )
        elif (
            dataset_format == "multi_woz_v22_dialogs"
            and args.dataset == "multi_woz_v22"
        ):
            # See data at https://huggingface.co/datasets/multi_woz_v22/viewer/v2.2/train?row=0
            # replicating traing format as in https://huggingface.co/datasets/timdettmers/openassistant-guanaco

            def multiwoz_full_dial(x):
                turns = x["turns"]
                full_dialog = "\n".join(
                    f"{BOT_PREFIX if spk == SYSTEM_SPK_ID else USR_PREFIX}{utt}"
                    for spk, utt in zip(turns["speaker"], turns["utterance"])
                )
                return {
                    "input": "",
                    "output": full_dialog,
                    "dialogue_id": x["dialogue_id"],
                }

            dataset = dataset.map(multiwoz_full_dial)
            dataset = dataset.remove_columns(
                [
                    col
                    for col in dataset.column_names["train"]
                    if col not in ["input", "output", "dialogue_id"]
                ]
            )
        else:
            raise NotImplementedError("Unsuported dataset")
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        if "test" in dataset and args.dataset == "multi_woz_v22":
            # TODO can I remove the multi_woz_v22 condition. keeping it strict for backward compatability
            eval_dataset = dataset["test"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`",
                file=sys.stderr,
            )
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )
    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


def parse_args():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, GenerationArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        generation_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(
        **vars(generation_args)
    )
    args = argparse.Namespace(
        **vars(model_args),
        **vars(data_args),
        **vars(training_args),
    )
    print(f"INFO: {extra_args=}", file=sys.stderr)
    return args, training_args


def train(args, training_args):
    set_seed(args.seed)

    if args.checkpoint_dir is not None:
        checkpoint_dir = Path(args.checkpoint_dir)
        assert checkpoint_dir.exists(), checkpoint_dir
    else:
        checkpoint_dir = None

    model = get_accelerate_model(args, checkpoint_dir)
    model.config.use_cache = False
    print_trainable_parameters(args, model)
    print("loaded model", file=sys.stderr)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast="pythia"
        in args.model_name_or_path,  # Fast tokenizer giving issues but Pythia models require them
        tokenizer_type="llama"
        if "llama" in args.model_name_or_path
        else None,  # Needed for HF name change
        use_auth_token=args.use_auth_token,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    if "llama" in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print("Adding special tokens.", file=sys.stderr)
        tokenizer.add_special_tokens(
            {
                "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
                "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
                "unk_token": tokenizer.convert_ids_to_tokens(
                    model.config.pad_token_id
                    if model.config.pad_token_id != -1
                    else tokenizer.pad_token_id
                ),
            }
        )

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.dataset_format == "multi_woz_v22_turns":
        e = MWZEvaluator(bleu=True, success=True, richness=True)

        class MWZEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                pass  # TODO

        trainer.add_callback(MWZEvalCallback)

    # Verifying the datatypes.
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes:
            dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total, file=sys.stderr)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        if not training_args.predict_with_generate:
            print("WARNING: setting predict_with_generate", file=sys.stderr)
            training_args.predict_with_generate = True
            trainer = Seq2SeqTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
            )

        prediction_output = trainer.predict(
            test_dataset=data_module["predict_dataset"], metric_key_prefix="predict"
        )
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        turns = []
        for i, example in enumerate(data_module["predict_dataset"]):
            example["prediction_with_input"] = predictions[i].strip()
            example["response"] = predictions[i].replace(example["input"], "").strip()
            turns.append(example)
        predictions_dialogs_json = os.path.join(
            args.output_dir, "predictions_dialogs.json"
        )
        print(
            f"\nSaving responses to\n\t{predictions_dialogs_json}\n",
            flush=True,
            file=sys.stderr,
        )
        diaturns = group_turns_by_dialogue(
            turns, single_turn=args.dataset_format == "multi_woz_v22_dialogs"
        )
        with open(predictions_dialogs_json, "w") as w:
            w.write("{\n")
            w.write(
                ",\n ".join([f'"{k}": {json.dumps(v)}' for k, v in diaturns.items()])
            )
            w.write("\n}")
        if args.dataset_format == "multi_woz_v22_turns":
            for idx, (dialogue_id, turns) in zip(
                range(args.print_turns), diaturns.items()
            ):
                for tidx, turn in enumerate(turns):
                    print(
                        f"=======[{idx}, {dialogue_id}, {tidx}]======\n{turn['prediction_with_input']}\n---vs---\n{turn['output']}",
                        file=sys.stderr,
                    )
            e = MWZEvaluator(bleu=True, success=True, richness=True)
            results = e.evaluate(diaturns)
            for metric, values in results.items():
                if values is not None:
                    print(f"====== {metric.upper()} ======", file=sys.stderr)
                    for k, v in values.items():
                        print(f"{k.ljust(15)}{v}", file=sys.stderr)
                    print("", file=sys.stderr)
            with open(os.path.join(args.output_dir, "turn_results.json"), "w") as f:
                json.dump(results, f)
        else:
            assert args.dataset_format == "multi_woz_v22_dialogs", args.dataset_format
            print(
                "TODO evaluate perplexity(richness) of generated dialogue",
                file=sys.stderr,
            )

        print(prediction_metrics, file=sys.stderr)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    print(
        f"output_dir and PEFT checkpoint folder:\n\t{args.output_dir}\n",
        flush=True,
        file=sys.stderr,
    )
    # The only outputs of the stdout should be the output_dir and the checkpoint
    # so the bash scripts can be easily composed together.
    print(args.output_dir, flush=True)

    checkpoints = glob.glob(f"{args.output_dir}/checkpoint-*")
    print(",".join(checkpoints), flush=True)

    return args.output_dir, checkpoints


if __name__ == "__main__":
    train(*parse_args())
