"""
Progressive optimization — Qwen3-0.6B on CNN/DailyMail.

Runs configs sequentially, each adding one optimization over the previous.
The goal: show the path from a naive baseline to maximum GPU utilisation.

  Step  Config       per_bs  ga  eff_bs  flash  compile   lr
  ────────────────────────────────────────────────────────────
  1     baseline          4   4      16  no     no        2e-5
  2     + FA2             4   4      16  yes    no        2e-5
  3     + compile         4   4      16  yes    yes       2e-5
  4     + bs×3           12   4      48  yes    yes       6e-5
  5     + bs×6           24   4      96  yes    yes       12e-5

Story arc
  Steps 1→3: VRAM falls as FA2 and compile free activation memory.
  Steps 3→5: VRAM climbs back as we spend the freed headroom on
             larger batches — converting memory savings into throughput.

Padding
  Dynamic padding for all configs — the collator pads each batch to
  the longest sequence in that batch.  FA2's varlen kernel skips
  padding tokens entirely, which is the main source of its throughput
  gain over SDPA.  Compile handles dynamic shapes via recompilation on
  shape change, which is acceptable here since CNN/DM articles cluster
  around a few common lengths.

Results saved to metrics/progressive.json.
"""

import json
import os
import time
from dataclasses import dataclass

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset

torch.set_float32_matmul_precision('high')

os.environ.setdefault("WANDB_PROJECT", "qwen3-progressive")

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

METRICS_PATH = "metrics/progressive.json"

@dataclass
class ProgConfig:
    name:          str
    label:         str
    per_device_bs: int
    grad_accum:    int
    lr:            float
    flash_attn:    bool
    compile:       bool
    sac:           bool = False

    @property
    def effective_bs(self) -> int:
        return self.per_device_bs * self.grad_accum

    @property
    def run_name(self) -> str:
        return f"prog_{self.name}"


CONFIGS = [
    ProgConfig("baseline", "Baseline\n(sdpa)",  4,  4,  2e-5, False, False),
    ProgConfig("fa2",      "+FA2",              4,  4,  2e-5, True,  False),
    ProgConfig("compile",  "+compile",          4,  4,  2e-5, True,  True),
    ProgConfig("bs12",     "+bs×3",            12,  4,  6e-5, True,  True),
    ProgConfig("bs12_sac", "+bs×3\n+SAC",      12,  4,  6e-5, True,  True,  True),
    ProgConfig("bs16_sac", "+bs×4\n+SAC",      16,  4,  6e-5, True,  True,  True),
    ProgConfig("bs17_sac", "+bs×5\n+SAC",      17,  4,  6e-5, True,  True,  True),
]

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


def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    tok.pad_token    = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_model(cfg: ProgConfig) -> torch.nn.Module:
    attn_impl = "flash_attention_2" if cfg.flash_attn else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )
    print(f"\n  attn_implementation : {attn_impl}", flush=True)
    return model

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

def run_one(cfg: ProgConfig, tokenizer, train_ds, eval_ds, collator) -> dict:
    print(f"\n{'='*62}")
    print(f"  {cfg.run_name}")
    print(f"  flash={cfg.flash_attn}  compile={cfg.compile}  sac={cfg.sac}")
    print(f"  per_bs={cfg.per_device_bs}  ga={cfg.grad_accum}  eff_bs={cfg.effective_bs}  lr={cfg.lr:.1e}")
    print(f"{'='*62}\n", flush=True)

    # SAC via compiler: budget=0.5 tells the inductor to recompute 50% of
    # activation memory instead of storing it. Reset to 1.0 (no recompute)
    # for non-SAC configs so they are unaffected.
    import torch._functorch.config as functorch_cfg
    functorch_cfg.activation_memory_budget = 0.8 if cfg.sac else 1.0

    model = load_model(cfg)

    args = TrainingArguments(
        output_dir                  = "./checkpoints_progressive",
        run_name                    = cfg.run_name,
        # --- batch ---
        per_device_train_batch_size = cfg.per_device_bs,
        gradient_accumulation_steps = cfg.grad_accum,
        # --- optimizer ---
        optim         = "adamw_torch_fused",
        learning_rate = cfg.lr,
        adam_beta1    = 0.9,
        adam_beta2    = 0.999,
        adam_epsilon  = 1e-8,
        weight_decay  = 0.01,
        # --- precision ---
        bf16           = True,
        bf16_full_eval = True,
        tf32           = False,
        # --- compilation ---
        torch_compile = cfg.compile,
        # --- schedule ---
        max_steps         = 200,
        warmup_steps      = 20,
        lr_scheduler_type = "cosine",
        # --- dataloader ---
        dataloader_num_workers     = 3,
        dataloader_pin_memory      = True,
        dataloader_prefetch_factor = 2,
        dataloader_drop_last       = True,
        # --- logging & saving ---
        logging_steps = 10,
        save_strategy = "no",
        eval_strategy = "no",
        report_to     = "wandb",
        remove_unused_columns = False,
    )

    trainer = BenchmarkTrainer(
        model            = model,
        args             = args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        data_collator    = collator,
        processing_class = tokenizer,
    )
    trainer.train()

    train_logs = [l for l in trainer.state.log_history
                  if "loss" in l and "eval_loss" not in l]
    # Skip first log entry — compile warmup distorts throughput
    stable = train_logs[1:] if len(train_logs) > 1 else train_logs

    tps_vals  = [float(l["throughput/active_tokens_per_sec"])
                 for l in stable if "throughput/active_tokens_per_sec" in l]
    vram_vals = [float(l["memory/peak_vram_gb"])
                 for l in stable if "memory/peak_vram_gb" in l]

    result = {
        "name":          cfg.name,
        "label":         cfg.label,
        "per_device_bs": cfg.per_device_bs,
        "grad_accum":    cfg.grad_accum,
        "effective_bs":  cfg.effective_bs,
        "flash_attn":    cfg.flash_attn,
        "compile":       cfg.compile,
        "sac":           cfg.sac,
        "loss_start":    float(train_logs[0]["loss"])  if train_logs else None,
        "loss_end":      float(train_logs[-1]["loss"]) if train_logs else None,
        "tps_avg":       round(sum(tps_vals)  / len(tps_vals))  if tps_vals  else None,
        "vram_avg":      round(sum(vram_vals) / len(vram_vals), 2) if vram_vals else None,
        "vram_max":      round(max(vram_vals), 2)                  if vram_vals else None,
    }

    print(f"\n  [{cfg.run_name}]  loss {result['loss_start']:.3f} → {result['loss_end']:.3f}"
          f"  |  tps {result['tps_avg']:,}  |  VRAM avg {result['vram_avg']:.2f} GB\n",
          flush=True)
    return result

def main():
    os.makedirs("metrics", exist_ok=True)

    # Load existing results so already-completed configs are skipped
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            results = json.load(f)
    else:
        results = []
    done = {r["name"] for r in results}

    tokenizer = build_tokenizer()
    train_ds  = load_streaming_dataset(tokenizer, "train")
    eval_ds   = load_streaming_dataset(tokenizer, "validation")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=None,
        padding="longest", pad_to_multiple_of=16,
        label_pad_token_id=-100,
    )

    for cfg in CONFIGS:
        if cfg.name in done:
            print(f"  Skipping {cfg.name} (already in {METRICS_PATH})", flush=True)
            continue
        result = run_one(cfg, tokenizer, train_ds, eval_ds, collator)
        results.append(result)
        with open(METRICS_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved → {METRICS_PATH}", flush=True)

    print("\nAll configs done.")
    print("Plot:  python plots/plot_progressive.py")


if __name__ == "__main__":
    main()
