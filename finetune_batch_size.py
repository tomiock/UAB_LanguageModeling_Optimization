"""
Batch-size scaling experiment — Qwen3-0.6B on CNN/DailyMail.

Compares three configurations that double the per-device batch size at each step,
keeping gradient_accumulation_steps=4 fixed.  Effective batch size therefore
doubles too, and the learning rate is scaled linearly (linear scaling rule):

  Config       per_device_bs  grad_accum  effective_bs   lr       VRAM (est.)
  ─────────────────────────────────────────────────────────────────────────────
  bs2_eff8          2             4              8        2e-5     ~13 GB
  bs4_eff16         4             4             16        4e-5     ~21 GB
  bs8_eff32         8             4             32        8e-5     ~38 GB  ← max

Throughput optimizations applied (all active for every config):
  • bf16                   — forward/backward in bfloat16; halves activation memory
                             vs fp32 and doubles tensor-core throughput.
  • tf32                   — fp32 matmuls use TF32 precision (~8× faster than fp32
                             on Ampere/Ada with no significant accuracy loss).
  • adamw_torch_fused      — single fused CUDA kernel for the Adam update; avoids
                             a separate loop over all parameter tensors.
  • torch_compile          — traces the forward+backward graph and emits optimised
                             Triton/CUDA kernels; typically 10-20 % throughput gain.
  • attn_implementation=sdpa — PyTorch scaled-dot-product attention with flash-kernel
                             selection; memory-efficient and compile-friendly.
  • dataloader_prefetch    — 2 batches prefetched per worker so GPU never stalls
                             waiting for the next batch.
  • dataloader_drop_last   — ensures every batch has the same shape, which helps
                             torch.compile avoid recompilation.
  • fixed padding          — constant tensor shapes across all steps; compile only
                             ever sees one input shape → no retracing.

Design note: gradient_checkpointing is intentionally OFF so that activation memory
grows visibly with batch size, making the VRAM comparison meaningful.
"""

import os
import time

import torch
from dataclasses import dataclass, field
from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

os.environ.setdefault("WANDB_PROJECT", "qwen3-batch-size")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128
MAX_SEQ_LEN    = MAX_INPUT_LEN + MAX_TARGET_LEN   # 1152 — fixed padding target

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

# ---------------------------------------------------------------------------
# Batch-size configurations
# (per_device_bs, grad_accum, base_lr)
# lr scales linearly with effective_bs (linear scaling rule, anchored at bs=8 → lr=2e-5)
# ---------------------------------------------------------------------------

@dataclass
class BSConfig:
    per_device_bs: int
    grad_accum:    int
    lr:            float

    @property
    def effective_bs(self) -> int:
        return self.per_device_bs * self.grad_accum

    @property
    def run_name(self) -> str:
        return f"pbs{self.per_device_bs}_ga{self.grad_accum}_eff{self.effective_bs}"


CONFIGS: List[BSConfig] = [
    BSConfig(per_device_bs=2, grad_accum=4, lr=2e-5),   # eff=8,  ~13 GB
    BSConfig(per_device_bs=4, grad_accum=4, lr=4e-5),   # eff=16, ~21 GB
    BSConfig(per_device_bs=8, grad_accum=4, lr=8e-5),   # eff=32, ~38 GB  ← GPU maxed
]

# ---------------------------------------------------------------------------
# Shared training hyperparameters (everything except bs/lr/run_name)
# ---------------------------------------------------------------------------

COMMON = dict(
    max_steps         = 2000,
    warmup_steps      = 200,
    lr_scheduler_type = "cosine",
    weight_decay      = 0.01,
    # --- precision ---
    bf16              = True,
    bf16_full_eval    = True,   # run eval in bf16 too — no need for fp32 eval
    tf32              = True,
    # --- optimizer ---
    optim             = "adamw_torch_fused",   # fused CUDA Adam kernel
    adam_beta1        = 0.9,
    adam_beta2        = 0.999,
    adam_epsilon      = 1e-8,
    # --- compilation ---
    torch_compile     = True,   # Triton/CUDA graph compilation (~10-20% speedup)
    # --- dataloader ---
    dataloader_num_workers   = 3,       # CNN/DM has 3 streaming shards
    dataloader_pin_memory    = True,
    dataloader_prefetch_factor = 2,     # prefetch 2 batches per worker
    dataloader_drop_last     = True,    # constant shapes → no compile retracing
    # --- logging & checkpointing ---
    logging_steps     = 50,
    eval_strategy     = "steps",
    eval_steps        = 500,
    save_steps        = 9999,
    save_total_limit  = 1,
    remove_unused_columns = False,
    report_to         = "wandb",
    output_dir        = "./checkpoints_bs",
)

# ---------------------------------------------------------------------------
# BenchmarkTrainer — logs active tokens/s, samples/s, peak VRAM
# ---------------------------------------------------------------------------

class BenchmarkTrainer(Trainer):

    def _bench_reset(self):
        self._bench_tokens:  int        = 0
        self._bench_samples: int        = 0
        self._bench_t0:      float|None = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not hasattr(self, "_bench_tokens"):
            self._bench_reset()
        if "attention_mask" in inputs:
            self._bench_tokens += int(inputs["attention_mask"].sum())
        else:
            self._bench_tokens += inputs["input_ids"].numel()
        self._bench_samples += inputs["input_ids"].shape[0]
        if self._bench_t0 is None:
            torch.cuda.synchronize()
            self._bench_t0 = time.perf_counter()
        return super().training_step(model, inputs, num_items_in_batch)

    def log(self, logs: dict, start_time: float | None = None) -> None:
        if not hasattr(self, "_bench_tokens"):
            self._bench_reset()
        if self._bench_t0 is not None:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._bench_t0
            logs["throughput/active_tokens_per_sec"] = round(self._bench_tokens  / elapsed)
            logs["throughput/samples_per_sec"]       = round(self._bench_samples / elapsed, 2)
            logs["memory/peak_vram_gb"]              = round(
                torch.cuda.max_memory_allocated() / 1024**3, 3
            )
            self._bench_reset()
            torch.cuda.reset_peak_memory_stats()
        super().log(logs, start_time)

# ---------------------------------------------------------------------------
# Tokenizer & model
# ---------------------------------------------------------------------------

def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return model

# ---------------------------------------------------------------------------
# Dataset — fixed padding to MAX_SEQ_LEN with explicit attention_mask
# ---------------------------------------------------------------------------

def _make_tokenize_fn(tokenizer):
    def tokenize(example):
        prompt = PROMPT_TEMPLATE.format(article=example["article"])
        target = example["highlights"] + tokenizer.eos_token

        prompt_ids = tokenizer(
            prompt, truncation=True, max_length=MAX_INPUT_LEN,
            add_special_tokens=True,
        )["input_ids"]
        target_ids = tokenizer(
            target, truncation=True, max_length=MAX_TARGET_LEN,
            add_special_tokens=False,
        )["input_ids"]

        input_ids = prompt_ids + target_ids
        labels    = [-100] * len(prompt_ids) + target_ids

        # Explicit mask required: pad_token == eos_token so the collator
        # cannot distinguish padding from real EOS by token ID alone.
        real_len       = len(input_ids)
        pad_len        = MAX_SEQ_LEN - real_len
        input_ids      = input_ids + [tokenizer.pad_token_id] * pad_len
        labels         = labels    + [-100] * pad_len
        attention_mask = [1] * real_len + [0] * pad_len

        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": attention_mask}

    return tokenize


def load_dataset_fixed(tokenizer, split: str):
    ds = load_dataset(DATASET_NAME, DATASET_VERSION, split=split, streaming=True)
    return ds.map(_make_tokenize_fn(tokenizer),
                  remove_columns=["article", "highlights", "id"])

# ---------------------------------------------------------------------------
# Training loop — runs all CONFIGS sequentially
# ---------------------------------------------------------------------------

def run_one(cfg: BSConfig, tokenizer, train_ds, eval_ds, collator):
    print(f"\n{'='*60}")
    print(f"  Config : {cfg.run_name}")
    print(f"  eff_bs : {cfg.effective_bs}   lr: {cfg.lr:.1e}")
    print(f"  est VRAM: {4.5 + cfg.per_device_bs * 4.1:.1f} GB")
    print(f"{'='*60}\n", flush=True)

    args = TrainingArguments(
        **COMMON,
        run_name                    = cfg.run_name,
        per_device_train_batch_size = cfg.per_device_bs,
        gradient_accumulation_steps = cfg.grad_accum,
        learning_rate               = cfg.lr,
    )

    trainer = BenchmarkTrainer(
        model            = load_model(),   # fresh model per run
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        data_collator    = collator,
        processing_class = tokenizer,
    )

    trainer.train()

    # Print a one-line summary
    final_log = trainer.state.log_history
    train_logs = [l for l in final_log if "loss" in l and "eval_loss" not in l]
    if train_logs:
        loss_start = float(train_logs[0]["loss"])
        loss_end   = float(train_logs[-1]["loss"])
        vram_vals  = [float(l["memory/peak_vram_gb"])
                      for l in train_logs if "memory/peak_vram_gb" in l]
        vram_avg   = sum(vram_vals) / len(vram_vals) if vram_vals else 0
        print(f"\n  [{cfg.run_name}]  loss {loss_start:.3f} → {loss_end:.3f}"
              f"  |  peak VRAM avg {vram_avg:.2f} GB\n", flush=True)


def main():
    tokenizer = build_tokenizer()
    train_ds  = load_dataset_fixed(tokenizer, "train")
    eval_ds   = load_dataset_fixed(tokenizer, "validation")

    # Collator receives already-padded sequences — just stack them
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=False, label_pad_token_id=-100,
    )

    for cfg in CONFIGS:
        run_one(cfg, tokenizer, train_ds, eval_ds, collator)

    print("All configs done. Compare loss curves at:")
    print(f"  https://wandb.ai/{os.environ.get('WANDB_ENTITY', '<your-entity>')}/qwen3-batch-size")


if __name__ == "__main__":
    main()
