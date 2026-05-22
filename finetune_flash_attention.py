"""
Finetuning Qwen3-0.6B-Base on CNN/DailyMail — Flash Attention 2 showcase.

Standard scaled-dot-product attention (SDPA) computes the full N×N attention
matrix and stores it in GPU memory before applying softmax.  Memory grows as
O(seq_len²) per head, and each element of that matrix must be read and written
twice (once for the softmax, once for the weighted sum).

Flash Attention 2 (Dao et al. 2023) fuses the entire attention operation into a
single tiled CUDA kernel that never materialises the full N×N matrix.  It tiles
the Q/K/V tensors into SRAM-resident blocks and computes the output in a single
pass over K and V.  The result:

  Memory:     O(N) instead of O(N²) for the attention buffer.
  Speed:      2–4× faster on long sequences due to fewer HBM round-trips.
  Constraint: requires bf16 or fp16; does NOT support arbitrary 4D masks
              (so varlen packing is incompatible).

The speedup grows with sequence length because the quadratic attention buffer
that SDPA must allocate gets proportionally larger.  CNN/DailyMail articles are
long (median ~700 tokens after prompt), which makes this a good benchmark.

Requires:  pip install flash-attn --no-build-isolation

Toggle GPU_OPTS.flash_attention and compare the two runs side-by-side in W&B.
"""

import os
import time

import torch
from dataclasses import dataclass
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

os.environ.setdefault("WANDB_PROJECT", "qwen3-flash-attention")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128

# Larger batch makes the O(N²) vs O(N) memory difference more visible
PER_DEVICE_BS  = 4
GRAD_ACCUM     = 4

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

# ---------------------------------------------------------------------------
# Attention configuration — flip this to showcase the effect
# ---------------------------------------------------------------------------

@dataclass
class AttnOpts:
    flash_attention: bool = True   # False → sdpa (standard),  True → flash_attention_2

ATTN_OPTS = AttnOpts()

# ---------------------------------------------------------------------------
# Trainer — logs active tokens/s, samples/s, peak VRAM
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
    attn_impl = "flash_attention_2" if ATTN_OPTS.flash_attention else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    print(f"\n  Attention implementation : {attn_impl}\n", flush=True)
    return model

# ---------------------------------------------------------------------------
# Dataset — dynamic padding (pad to longest in batch)
# Dynamic padding shows FA2's memory advantage more clearly: real batches have
# variable lengths so the average N is lower than the worst-case MAX_SEQ_LEN,
# but FA2 still avoids the N×N buffer entirely.
# ---------------------------------------------------------------------------

def make_tokenize_fn(tokenizer):
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
        return {"input_ids": input_ids, "labels": labels}

    return tokenize


def load_streaming_dataset(tokenizer, split: str):
    ds = load_dataset(DATASET_NAME, DATASET_VERSION, split=split, streaming=True)
    return ds.map(make_tokenize_fn(tokenizer),
                  remove_columns=["article", "highlights", "id"])

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    if ATTN_OPTS.flash_attention:
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            raise SystemExit(
                "flash-attn is not installed.\n"
                "Install with:  pip install flash-attn --no-build-isolation"
            )

    attn_tag = "fa2" if ATTN_OPTS.flash_attention else "sdpa"
    run_name = f"attn_{attn_tag}_bs{PER_DEVICE_BS * GRAD_ACCUM}_bf16"
    print(f"\nRun: {run_name}\n", flush=True)

    tokenizer = build_tokenizer()
    model     = load_model()

    train_dataset = load_streaming_dataset(tokenizer, "train")
    eval_dataset  = load_streaming_dataset(tokenizer, "validation")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding="longest", pad_to_multiple_of=16,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir = "./checkpoints_fa",
        run_name   = run_name,
        # --- optimizer ---
        optim         = "adamw_torch_fused",
        learning_rate = 2e-5,
        adam_beta1    = 0.9,
        adam_beta2    = 0.999,
        adam_epsilon  = 1e-8,
        weight_decay  = 0.01,
        # --- batch / accumulation ---
        per_device_train_batch_size = PER_DEVICE_BS,
        gradient_accumulation_steps = GRAD_ACCUM,
        # --- precision ---
        bf16 = True,
        tf32 = False,
        torch_compile = True,
        # --- schedule ---
        max_steps         = 2000,
        warmup_steps      = 200,
        lr_scheduler_type = "cosine",
        # --- dataloader ---
        dataloader_num_workers   = 3,
        dataloader_pin_memory    = True,
        dataloader_prefetch_factor = 2,
        dataloader_drop_last     = True,
        # --- logging & saving ---
        logging_steps    = 50,
        eval_strategy    = "steps",
        eval_steps       = 500,
        save_steps       = 9999,
        save_total_limit = 1,
        report_to        = "wandb",
        # --- misc ---
        remove_unused_columns = False,
    )

    trainer = BenchmarkTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        data_collator    = collator,
        processing_class = tokenizer,
    )

    trainer.train()

    final_log  = trainer.state.log_history
    train_logs = [l for l in final_log if "loss" in l and "eval_loss" not in l]
    if train_logs:
        loss_start = float(train_logs[0]["loss"])
        loss_end   = float(train_logs[-1]["loss"])
        vram_vals  = [float(l["memory/peak_vram_gb"])
                      for l in train_logs if "memory/peak_vram_gb" in l]
        tps_vals   = [float(l["throughput/active_tokens_per_sec"])
                      for l in train_logs if "throughput/active_tokens_per_sec" in l]
        print(f"\n  [{run_name}]")
        print(f"  loss      : {loss_start:.3f} → {loss_end:.3f}")
        if vram_vals:
            print(f"  peak VRAM : avg {sum(vram_vals)/len(vram_vals):.2f} GB  "
                  f"(max {max(vram_vals):.2f} GB)")
        if tps_vals:
            print(f"  tok/s     : avg {sum(tps_vals)/len(tps_vals):.0f}")
        print(flush=True)


if __name__ == "__main__":
    main()
