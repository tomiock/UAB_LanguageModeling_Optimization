"""
Finetuning Qwen3-0.6B-Base on CNN/DailyMail — padding strategy comparison.

Three modes controlled by PADDING_MODE:

  FIXED   — every sequence padded to MAX_SEQ_LEN
             → constant-shape batches, heavy compute waste on pad tokens

  DYNAMIC — pad only to the longest sequence in each batch
             → variable-shape batches, less padding waste

  PACKING — concatenate examples and split into MAX_SEQ_LEN chunks (SFTTrainer)
             → near-zero padding waste, maximum GPU utilisation

Toggle PADDING_MODE and re-run to collect numbers for each strategy.
"""

import time
from enum import Enum

import torch
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID       = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME   = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128
MAX_SEQ_LEN    = MAX_INPUT_LEN + MAX_TARGET_LEN  # upper bound for fixed / packing

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

# ---------------------------------------------------------------------------
# Padding mode — change this to compare strategies
# ---------------------------------------------------------------------------

class PaddingMode(str, Enum):
    FIXED   = "fixed"    # pad to MAX_SEQ_LEN
    DYNAMIC = "dynamic"  # pad to longest in batch
    PACKING = "packing"  # SFTTrainer sequence packing

PADDING_MODE = PaddingMode.DYNAMIC

# ---------------------------------------------------------------------------
# GPU opts
# ---------------------------------------------------------------------------

@dataclass
class GPUOpts:
    bf16: bool = True
    tf32: bool = True
    gradient_checkpointing: bool = False
    torch_compile: bool = False
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 4


GPU_OPTS = GPUOpts()

# ---------------------------------------------------------------------------
# Benchmark mixin — shared by BenchmarkTrainer and BenchmarkSFTTrainer
# ---------------------------------------------------------------------------

class BenchmarkMixin:
    """
    Overrides training_step and log to inject:
        throughput/active_tokens_per_sec
        throughput/samples_per_sec
        memory/peak_vram_gb
    into the standard training logs every logging_steps.
    """

    def _bench_reset(self):
        self._bench_active:  int        = 0
        self._bench_samples: int        = 0
        self._bench_t0:      float|None = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not hasattr(self, "_bench_active"):
            self._bench_reset()

        # Count only non-padding tokens for fair comparison across strategies.
        # attention_mask is 1 for real tokens, 0 for padding.
        if "attention_mask" in inputs:
            self._bench_active += int(inputs["attention_mask"].sum())
        else:
            self._bench_active += inputs["input_ids"].numel()
        self._bench_samples += inputs["input_ids"].shape[0]

        if self._bench_t0 is None:
            torch.cuda.synchronize()
            self._bench_t0 = time.perf_counter()

        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict, start_time: float | None = None) -> None:
        if not hasattr(self, "_bench_active"):
            self._bench_reset()

        if self._bench_t0 is not None:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._bench_t0

            logs["throughput/active_tokens_per_sec"] = round(self._bench_active  / elapsed)
            logs["throughput/samples_per_sec"]       = round(self._bench_samples / elapsed, 2)
            logs["memory/peak_vram_gb"]              = round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            )

            self._bench_reset()
            torch.cuda.reset_peak_memory_stats()

        super().log(logs, start_time)


class BenchmarkTrainer(BenchmarkMixin, Trainer):
    pass


class BenchmarkSFTTrainer(BenchmarkMixin, SFTTrainer):
    pass


# ---------------------------------------------------------------------------
# Tokenizer & model
# ---------------------------------------------------------------------------

def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if GPU_OPTS.bf16 else torch.float32,
        attn_implementation="sdpa",
    )
    if GPU_OPTS.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _raw_streaming(split: str):
    return load_dataset(
        DATASET_NAME, DATASET_VERSION,
        split=split, streaming=True, trust_remote_code=True,
    )


def _make_tokenize_fn(tokenizer, padding: str):
    """
    padding="max_length"  → fixed padding to MAX_SEQ_LEN
    padding=False         → no padding (collator handles it for dynamic)
    """
    def tokenize(example):
        prompt = PROMPT_TEMPLATE.format(article=example["article"])
        target = example["highlights"] + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt,
            truncation=True,
            max_length=MAX_INPUT_LEN,
            add_special_tokens=True,
        )["input_ids"]

        target_ids = tokenizer(
            target,
            truncation=True,
            max_length=MAX_TARGET_LEN,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        # Loss only on the summary tokens
        labels = [-100] * len(prompt_ids) + target_ids

        if padding == "max_length":
            real_len  = len(input_ids)
            pad_len   = MAX_SEQ_LEN - real_len
            input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
            labels    = labels    + [-100] * pad_len
            # Explicit mask: pad_token == eos_token so the collator can't infer
            # padding positions from token IDs — we must pass it explicitly.
            attention_mask = [1] * real_len + [0] * pad_len
            return {"input_ids": input_ids, "labels": labels,
                    "attention_mask": attention_mask}

        return {"input_ids": input_ids, "labels": labels}

    return tokenize


def load_padded_dataset(tokenizer, split: str, padding: str):
    """For FIXED and DYNAMIC modes."""
    ds       = _raw_streaming(split)
    tokenize = _make_tokenize_fn(tokenizer, padding)
    return ds.map(tokenize, remove_columns=["article", "highlights", "id"])


def load_raw_dataset(split: str):
    """For PACKING mode — SFTTrainer handles tokenisation via formatting_func."""
    return _raw_streaming(split)


def formatting_func(example):
    return (
        PROMPT_TEMPLATE.format(article=example["article"])
        + example["highlights"]
        + "\n"   # EOS added by SFTTrainer / tokenizer
    )


# ---------------------------------------------------------------------------
# Shared TrainingArguments fields
# ---------------------------------------------------------------------------

COMMON_ARGS = dict(
    output_dir                  = "./checkpoints",
    per_device_train_batch_size = GPU_OPTS.per_device_train_batch_size,
    gradient_accumulation_steps = GPU_OPTS.gradient_accumulation_steps,
    bf16                        = GPU_OPTS.bf16,
    tf32                        = GPU_OPTS.tf32,
    torch_compile               = GPU_OPTS.torch_compile,
    max_steps                   = 2000,
    warmup_steps                = 100,
    learning_rate               = 2e-5,
    lr_scheduler_type           = "cosine",
    weight_decay                = 0.01,
    logging_steps               = 50,
    eval_strategy               = "steps",
    eval_steps                  = 500,
    save_steps                  = 500,
    save_total_limit            = 2,
    dataloader_num_workers      = 3,
    dataloader_pin_memory       = True,
    remove_unused_columns       = False,
    report_to                   = "none",
)

# ---------------------------------------------------------------------------
# Trainer builders — one per strategy
# ---------------------------------------------------------------------------

def build_fixed_trainer(model, tokenizer):
    """
    FIXED: every batch has shape (B, MAX_SEQ_LEN).
    Padding is applied at tokenisation time so the collator
    receives already-padded sequences of identical length.
    The GPU processes pad tokens in the attention computation —
    they contribute no gradient but consume FLOPs and memory.
    """
    train_ds = load_padded_dataset(tokenizer, "train",      padding="max_length")
    eval_ds  = load_padded_dataset(tokenizer, "validation", padding="max_length")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=False,        # sequences are already padded to MAX_SEQ_LEN
        label_pad_token_id=-100,
    )
    args = TrainingArguments(**COMMON_ARGS)
    return BenchmarkTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, processing_class=tokenizer,
    )


def build_dynamic_trainer(model, tokenizer):
    """
    DYNAMIC: sequences are tokenised without padding and the collator
    pads each batch to its longest sequence.
    Average batch length tracks the data distribution → less wasted compute
    than fixed, but still some padding within each batch.
    """
    train_ds = load_padded_dataset(tokenizer, "train",      padding=False)
    eval_ds  = load_padded_dataset(tokenizer, "validation", padding=False)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding="longest",
        pad_to_multiple_of=8,   # keeps shapes aligned for tensor cores
        label_pad_token_id=-100,
    )
    args = TrainingArguments(**COMMON_ARGS)
    return BenchmarkTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=collator, processing_class=tokenizer,
    )


def build_packing_trainer(model, tokenizer):
    """
    PACKING (SFTTrainer): examples are concatenated into MAX_SEQ_LEN chunks
    separated by EOS tokens. Every token in the batch carries a gradient
    signal — near-zero padding waste. The trade-off is that the loss is
    computed across all tokens (no prompt masking) and sequences from different
    examples can appear in the same chunk.
    """
    train_ds = load_raw_dataset("train")
    eval_ds  = load_raw_dataset("validation")

    args = SFTConfig(
        **COMMON_ARGS,
        max_length      = MAX_SEQ_LEN,
        packing         = True,
        eval_packing    = False,  # keep eval unpacked for consistent loss reporting
        dataset_num_proc= 3,
    )
    return BenchmarkSFTTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=eval_ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if GPU_OPTS.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print(f"\nPadding mode: {PADDING_MODE.value}\n")

    tokenizer = build_tokenizer()
    model     = load_model()

    builders = {
        PaddingMode.FIXED:   build_fixed_trainer,
        PaddingMode.DYNAMIC: build_dynamic_trainer,
        PaddingMode.PACKING: build_packing_trainer,
    }
    trainer = builders[PADDING_MODE](model, tokenizer)

    trainer.train()
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Done. Model saved to ./final_model")


if __name__ == "__main__":
    main()
