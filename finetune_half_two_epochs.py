"""
Half-dataset × 2 epochs finetuning of Qwen3-0.6B-Base on CNN/DailyMail.

This script is the *counterpart* to finetune_full_epoch.py and is designed to
be run side-by-side in W&B to answer a concrete question:

    Given a fixed compute budget (same total optimizer steps), is it better to
    train once on all the data, or twice on half the data?

Budget equivalence
──────────────────
  finetune_full_epoch.py  : 1 epoch  × 287 113 samples  ≈ 3 589 steps
  this script             : 2 epochs × 143 556 samples  ≈ 3 589 steps

Both runs share identical hyperparameters (batch size, LR, warmup ratio,
optimizer, precision) so the only variable is data repetition.

Launch (same config as the full-epoch run):
    accelerate launch --config_file accelerate_l40s.yaml finetune_half_two_epochs.py
"""

import os
import time

import torch
from dataclasses import dataclass
from accelerate import PartialState
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

os.environ.setdefault("WANDB_PROJECT", "qwen3-summarization")

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

@dataclass
class GPUOpts:
    bf16: bool = True
    tf32: bool = False
    flash_attention: bool = True
    gradient_checkpointing: bool = False
    torch_compile: bool = True
    gradient_accumulation_steps: int = 4
    per_device_train_batch_size: int = 10
    fused_adam: bool = True


GPU_OPTS = GPUOpts()

PROFILE_MEMORY          = True
PROFILE_AT_STEP         = 3
PROFILE_NUM_STEPS       = 3
PROFILE_SNAPSHOT_PREFIX = "memory_snapshot_half2e"

class BenchmarkTrainer(Trainer):

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

    @property
    def _is_main(self) -> bool:
        return self.args.process_index == 0

    def create_optimizer(self):
        if self._is_main:
            print("[DEBUG] create_optimizer: start", flush=True)
        optimizer = super().create_optimizer()
        if self._is_main:
            print(f"[DEBUG] create_optimizer: done → {type(optimizer).__name__}", flush=True)
        self._state_reported = False
        return optimizer

    def _report_optimizer_info(self) -> None:
        if not self._is_main:
            self._state_reported = True
            return
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

    def compute_loss(self, model, inputs, **kwargs):
        loss = super().compute_loss(model, inputs, **kwargs)
        if self._profiling_this_step and self._is_main:
            torch.cuda.synchronize()
            self._mem_after_fwd = torch.cuda.memory_allocated()
            self._mem_peak_fwd  = torch.cuda.max_memory_allocated()
            torch.cuda.reset_peak_memory_stats()
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._bench_tokens  += inputs["input_ids"].numel()
        self._bench_samples += inputs["input_ids"].shape[0]
        if self._bench_t0 is None:
            if self._is_main:
                print(f"[DEBUG] training_step: first step — input_ids shape {inputs['input_ids'].shape}", flush=True)
            torch.cuda.synchronize()
            self._bench_t0 = time.perf_counter()

        if self._profile_recording and self._is_main:
            self._profiling_this_step = True
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            self._mem_before_fwd = torch.cuda.memory_allocated()

        loss = super().training_step(model, inputs, num_items_in_batch)

        if not getattr(self, "_state_reported", True) and self.optimizer.state:
            self._report_optimizer_info()

        if self._profile_recording and self._is_main:
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
            total_micro = PROFILE_NUM_STEPS * self.args.gradient_accumulation_steps
            if self._profile_micro_steps >= total_micro:
                torch.cuda.memory._dump_snapshot(f"{PROFILE_SNAPSHOT_PREFIX}.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
                self._profile_recording = False
                self._profile_done      = True
                self._print_memory_table()

        return loss

    def log(self, logs: dict, start_time: float | None = None) -> None:
        if self._bench_t0 is not None:
            torch.cuda.synchronize()
            elapsed    = time.perf_counter() - self._bench_t0
            world_size = self.args.world_size

            logs["throughput/tokens_per_sec"]  = round(self._bench_tokens  * world_size / elapsed)
            logs["throughput/samples_per_sec"] = round(self._bench_samples * world_size / elapsed, 2)
            logs["memory/peak_vram_gb"]        = round(
                torch.cuda.max_memory_allocated() / 1024 ** 3, 3
            )

            self._bench_tokens  = 0
            self._bench_samples = 0
            self._bench_t0      = None
            torch.cuda.reset_peak_memory_stats()

        super().log(logs, start_time)

    def _print_memory_table(self) -> None:
        if not self._is_main:
            return

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


def load_full_dataset(tokenizer, split: str):
    ds = load_dataset(DATASET_NAME, DATASET_VERSION, split=split)
    return ds.map(
        make_tokenize_fn(tokenizer),
        remove_columns=["article", "highlights", "id"],
        num_proc=4,
        desc=f"Tokenizing {split}",
    )


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


def main():
    state = PartialState()

    opt_tag  = "adamw_fused" if GPU_OPTS.fused_adam else "adamw_standard"
    run_name = (
        f"{opt_tag}"
        f"_bs{GPU_OPTS.per_device_train_batch_size * GPU_OPTS.gradient_accumulation_steps * state.num_processes}"
        f"{'_bf16'    if GPU_OPTS.bf16           else ''}"
        f"{'_fa2'     if GPU_OPTS.flash_attention else ''}"
        f"{'_compile' if GPU_OPTS.torch_compile   else ''}"
        f"_half_2epochs"
        f"_{state.num_processes}gpu"
    )

    if state.is_main_process:
        print(f"\nRun: {run_name}", flush=True)
        print(f"GPUs: {state.num_processes}\n", flush=True)

    if state.is_main_process:
        print("[DEBUG] loading tokenizer...", flush=True)
    tokenizer = build_tokenizer()

    if state.is_main_process:
        print("[DEBUG] loading model...", flush=True)
    model = load_model()

    if state.is_main_process:
        print("[DEBUG] building datasets...", flush=True)
    with state.main_process_first():
        full_train = load_full_dataset(tokenizer, "train")
        eval_dataset = load_full_dataset(tokenizer, "validation")

    half = len(full_train) // 2
    train_dataset = full_train.select(range(half))

    if state.is_main_process:
        print(
            f"[DEBUG] datasets ready — train (half): {len(train_dataset):,}"
            f"  val: {len(eval_dataset):,}",
            flush=True,
        )
        print(
            f"[DEBUG] budget: 2 epochs × {len(train_dataset):,} samples"
            f" ≈ {2 * len(train_dataset) // (GPU_OPTS.per_device_train_batch_size * GPU_OPTS.gradient_accumulation_steps * state.num_processes):,} steps",
            flush=True,
        )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding="max_length",
        max_length=MAX_INPUT_LEN + MAX_TARGET_LEN,
        label_pad_token_id=-100,
    )

    training_args = TrainingArguments(
        output_dir    = "./checkpoints_half2e",
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
        tf32          = False,
        # --- compilation ---
        torch_compile = GPU_OPTS.torch_compile,
        # --- schedule: 2 epochs on half the data ≈ same steps as 1 full epoch ---
        num_train_epochs  = 2,
        warmup_ratio      = 0.03,
        lr_scheduler_type = "cosine",
        # --- logging & saving ---
        logging_steps    = 10,
        eval_strategy    = "steps",
        eval_steps       = 500,
        save_steps       = 1000,
        save_total_limit = 2,
        report_to        = "wandb",
        # --- misc ---
        dataloader_num_workers = 4,
        dataloader_pin_memory  = True,
        remove_unused_columns  = False,
    )

    if state.is_main_process:
        print("[DEBUG] building trainer...", flush=True)
    trainer = BenchmarkTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_dataset,
        eval_dataset     = eval_dataset,
        data_collator    = collator,
        processing_class = tokenizer,
    )

    if PROFILE_MEMORY and state.is_main_process:
        torch.cuda.memory._record_memory_history(max_entries=100_000)
        trainer._profile_recording = True

    if state.is_main_process:
        print("[DEBUG] calling trainer.train()...", flush=True)
    trainer.train()
    trainer.save_model("./final_model_half2e")
    if state.is_main_process:
        tokenizer.save_pretrained("./final_model_half2e")
        print("Done. Model saved to ./final_model_half2e")


if __name__ == "__main__":
    main()
