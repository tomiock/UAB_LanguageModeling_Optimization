"""
Finetuning Qwen3-0.6B-Base on CNN/DailyMail — fused AdamW optimizer showcase.

Standard PyTorch AdamW (adamw_torch) launches one CUDA kernel per parameter
tensor for the momentum update.  With 600 M parameters spread over hundreds of
named tensors, the per-kernel launch overhead accumulates to a measurable cost
every optimizer step.

adamw_torch_fused merges all of those per-tensor kernel launches into a single
fused CUDA kernel that iterates over every parameter in one pass.  The result is
a 5–10 % throughput gain at no cost in memory or numerical behaviour — it is a
pure execution-scheduling improvement.

The fused kernel is available from PyTorch ≥ 2.0 and is selected by passing
optim="adamw_torch_fused" to TrainingArguments.  No extra dependencies are
needed; bitsandbytes is not required.

Toggle GPU_OPTS.fused_adam and compare the two runs side-by-side in W&B.
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

os.environ.setdefault("WANDB_PROJECT", "qwen3-summarization")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

# ---------------------------------------------------------------------------
# GPU optimizations — flip these to showcase their effect
# ---------------------------------------------------------------------------

@dataclass
class GPUOpts:
    bf16: bool = True
    tf32: bool = False
    flash_attention: bool = False
    gradient_checkpointing: bool = False
    torch_compile: bool = True
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 4
    fused_adam: bool = True


GPU_OPTS = GPUOpts()

# ---------------------------------------------------------------------------
# Memory profiling
# ---------------------------------------------------------------------------

PROFILE_MEMORY          = True
PROFILE_AT_STEP         = 3
PROFILE_NUM_STEPS       = 3
PROFILE_SNAPSHOT_PREFIX = "memory_snapshot"

# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class BenchmarkTrainer(Trainer):
    """
    Adds three capabilities on top of the stock Trainer:

    1. Throughput metrics (every logging_steps):
         throughput/tokens_per_sec   — all tokens processed (incl. prompt)
         throughput/samples_per_sec  — sequences per second
         memory/peak_vram_gb         — peak VRAM in the logging window

    2. Optimizer-state memory report (printed once at training start):
         Shows measured optimizer VRAM vs the float32-AdamW baseline so
         the memory saving from 8-bit quantisation is immediately visible.

    3. Forward / backward memory split table (PROFILE_NUM_STEPS steps):
         Printed once and saved as {PROFILE_SNAPSHOT_PREFIX}.pickle for the
         PyTorch memory visualiser (https://pytorch.org/memory_viz).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bench_tokens:  int        = 0
        self._bench_samples: int        = 0
        self._bench_t0:      float|None = None
        self._profile_done:         bool = False
        self._profile_recording:    bool = False
        self._profile_micro_steps:  int  = 0
        self._profiling_this_step:  bool = False
        self._profile_rows:         list = []
        self._mem_before_fwd:       int  = 0
        self._mem_after_fwd:        int  = 0
        self._mem_peak_fwd:         int  = 0

    # -- optimizer creation --------------------------------------------------

    def create_optimizer(self):
        """
        Delegates to HuggingFace Trainer (which reads self.args.optim).
        Optimizer states are allocated lazily on the first step, so VRAM and
        state dtypes are reported after that step via _report_optimizer_info().
        """
        print("[DEBUG] create_optimizer: start", flush=True)
        optimizer = super().create_optimizer()
        print(f"[DEBUG] create_optimizer: done → {type(optimizer).__name__}", flush=True)
        self._state_reported = False
        return optimizer

    def _report_optimizer_info(self) -> None:
        """Called once after the first step when optimizer states are populated."""
        optimizer = self.optimizer
        n_params  = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        state_bytes = sum(
            v.nbytes
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        opt_vram_gb = state_bytes / 1024 ** 3

        state_dtypes = {}
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and k not in state_dtypes:
                    state_dtypes[k] = str(v.dtype)
        dtypes_str = ", ".join(f"{k}: {v}" for k, v in state_dtypes.items())

        fused = getattr(optimizer, "fused", False) or "fused" in type(optimizer).__name__.lower()

        W = 60
        print("\n" + "=" * W)
        print(f"  Optimizer   : {type(optimizer).__name__}")
        print(f"  Fused kernel: {'yes — single CUDA kernel for all tensors' if fused else 'no  — one kernel per parameter tensor'}")
        print(f"  Params      : {n_params / 1e6:.1f} M trainable")
        print(f"  State dtypes: {dtypes_str}")
        print(f"  State VRAM  : {opt_vram_gb:.2f} GB")
        print("=" * W + "\n", flush=True)
        self._state_reported = True

    # -- forward / backward memory split -------------------------------------

    def compute_loss(self, model, inputs, **kwargs):
        loss = super().compute_loss(model, inputs, **kwargs)
        if self._profiling_this_step:
            torch.cuda.synchronize()
            self._mem_after_fwd = torch.cuda.memory_allocated()
            self._mem_peak_fwd  = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return loss

    # -- main step -----------------------------------------------------------

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._bench_tokens  += inputs["input_ids"].numel()
        self._bench_samples += inputs["input_ids"].shape[0]
        if self._bench_t0 is None:
            print(f"[DEBUG] training_step: first step — input_ids shape {inputs['input_ids'].shape}", flush=True)
            torch.cuda.synchronize()
            self._bench_t0 = time.perf_counter()

        total_micro = PROFILE_NUM_STEPS * self.args.gradient_accumulation_steps
        if self._profile_recording:
            self._profiling_this_step = True
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._mem_before_fwd = torch.cuda.memory_allocated()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if not getattr(self, "_state_reported", True) and self.optimizer.state:
            self._report_optimizer_info()

        if self._profile_recording:
            torch.cuda.synchronize()
            self._profile_rows.append({
                "opt_step":   self.state.global_step,
                "micro_step": self._profile_micro_steps % self.args.gradient_accumulation_steps + 1,
                "before_fwd": self._mem_before_fwd,
                "after_fwd":  self._mem_after_fwd,
                "peak_fwd":   self._mem_peak_fwd,
                "after_bwd":  torch.cuda.memory_allocated(),
                "peak_bwd":   torch.cuda.max_memory_allocated(),
            })
            self._profile_micro_steps += 1
            self._profiling_this_step  = False
            if self._profile_micro_steps >= total_micro:
                torch.cuda.memory._dump_snapshot(f"{PROFILE_SNAPSHOT_PREFIX}.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
                self._profile_recording = False
                self._profile_done      = True
                self._print_memory_table()

        return loss

    # -- logging -------------------------------------------------------------

    def log(self, logs: dict, start_time: float | None = None) -> None:
        if self._bench_t0 is not None:
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - self._bench_t0

            logs["throughput/tokens_per_sec"]  = round(self._bench_tokens  / elapsed)
            logs["throughput/samples_per_sec"] = round(self._bench_samples / elapsed, 2)
            logs["memory/peak_vram_gb"]        = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 3
            )

            self._bench_tokens  = 0
            self._bench_samples = 0
            self._bench_t0      = None
            torch.cuda.reset_peak_memory_stats()

        super().log(logs, start_time)

    # -- memory table --------------------------------------------------------

    def _print_memory_table(self) -> None:
        def fmt(b: int) -> str:
            return f"{b / 1024**3:.3f}"

        ga       = self.args.gradient_accumulation_steps
        last_opt = PROFILE_AT_STEP + PROFILE_NUM_STEPS - 1
        header   = (
            f"{'step':>5}  {'μ':>4}  "
            f"{'bef-fwd':>8}  {'aft-fwd':>8}  {'pk-fwd':>8}  "
            f"{'aft-bwd':>8}  {'pk-bwd':>8}  (GB)"
        )
        sep = "-" * len(header)
        W   = max(len(header), 56)

        print("\n" + "=" * W)
        print(f"  Memory profile — optimizer steps {PROFILE_AT_STEP}–{last_opt}  "
              f"({PROFILE_NUM_STEPS} steps × {ga} micro-steps)")
        print("=" * W)
        print(header)
        print(sep)

        prev_opt = None
        for row in self._profile_rows:
            if prev_opt is not None and row["opt_step"] != prev_opt:
                print(sep)
            print(
                f"{row['opt_step']:>5}  {row['micro_step']:>4}  "
                f"{fmt(row['before_fwd']):>8}  {fmt(row['after_fwd']):>8}  {fmt(row['peak_fwd']):>8}  "
                f"{fmt(row['after_bwd']):>8}  {fmt(row['peak_bwd']):>8}"
            )
            prev_opt = row["opt_step"]

        print("=" * W)
        print("  bef-fwd  weights + optimizer states")
        print("  aft-fwd  + activations cached for backward")
        print("  pk-fwd   + intermediate tensors during forward")
        print("  aft-bwd  + gradients, - activations")
        print("  pk-bwd   worst case during backward pass")
        print("=" * W)
        print(f"  Snapshot → {PROFILE_SNAPSHOT_PREFIX}.pickle")
        print(f"  Visualize → https://pytorch.org/memory_viz")
        print("=" * W + "\n")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


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
    ds = load_dataset(
        DATASET_NAME, DATASET_VERSION,
        split=split, streaming=True,
    )
    return ds.map(make_tokenize_fn(tokenizer),
                  remove_columns=["article", "highlights", "id"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def load_model():
    kwargs = {"dtype": torch.bfloat16 if GPU_OPTS.bf16 else torch.float32}
    if GPU_OPTS.flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    if GPU_OPTS.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def main():
    if GPU_OPTS.tf32:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision   = "tf32"

    opt_tag  = "adamw_fused" if GPU_OPTS.fused_adam else "adamw_standard"
    run_name = (
        f"{opt_tag}"
        f"_bs{GPU_OPTS.per_device_train_batch_size * GPU_OPTS.gradient_accumulation_steps}"
        f"{'_bf16' if GPU_OPTS.bf16 else ''}"
        f"{'_compile' if GPU_OPTS.torch_compile else ''}"
    )
    print(f"\nRun: {run_name}\n", flush=True)

    print("[DEBUG] loading tokenizer...", flush=True)
    tokenizer = build_tokenizer()
    print("[DEBUG] tokenizer loaded", flush=True)

    print("[DEBUG] loading model...", flush=True)
    model = load_model()
    print("[DEBUG] model loaded", flush=True)

    print("[DEBUG] building datasets...", flush=True)
    train_dataset = load_streaming_dataset(tokenizer, "train")
    eval_dataset  = load_streaming_dataset(tokenizer, "validation")
    print("[DEBUG] datasets ready (streaming — no data loaded yet)", flush=True)

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding="max_length",
        max_length=MAX_INPUT_LEN + MAX_TARGET_LEN,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir    = "./checkpoints",
        run_name      = run_name,
        # --- optimizer ---
        optim         = "adamw_torch_fused" if GPU_OPTS.fused_adam else "adamw_torch",
        learning_rate = 2e-5,
        adam_beta1    = 0.9,
        adam_beta2    = 0.999,
        adam_epsilon  = 1e-8,
        weight_decay  = 0.01,
        # --- throughput ---
        per_device_train_batch_size = GPU_OPTS.per_device_train_batch_size,
        gradient_accumulation_steps = GPU_OPTS.gradient_accumulation_steps,
        gradient_checkpointing      = GPU_OPTS.gradient_checkpointing,
        # --- precision ---
        bf16          = GPU_OPTS.bf16,
        # tf32 set via new API in main() — avoids mixing legacy/new cublas setters with torch_compile
        # --- compilation ---
        torch_compile = GPU_OPTS.torch_compile,
        # --- schedule ---
        max_steps         = 10_000,
        warmup_steps      = 100,
        lr_scheduler_type = "cosine",
        # --- logging & saving ---
        logging_steps    = 10,
        eval_strategy    = "steps",
        eval_steps       = 500,
        save_steps       = 1000,
        save_total_limit = 2,
        report_to        = "wandb",
        # --- misc ---
        dataloader_num_workers = 3,
        dataloader_pin_memory  = True,
        remove_unused_columns  = False,
    )

    print("[DEBUG] building trainer...", flush=True)
    trainer = BenchmarkTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        data_collator    = collator,
        processing_class = tokenizer,
    )

    print("[DEBUG] trainer built", flush=True)

    if PROFILE_MEMORY:
        print("[DEBUG] starting memory recording...", flush=True)
        torch.cuda.memory._record_memory_history(max_entries=100_000)
        trainer._profile_recording = True

    print("[DEBUG] calling trainer.train()...", flush=True)
    trainer.train()
    trainer.save_model("./final_model")
    tokenizer.save_pretrained("./final_model")
    print("Done. Model saved to ./final_model")


if __name__ == "__main__":
    main()
