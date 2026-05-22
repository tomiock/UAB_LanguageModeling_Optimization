"""
Sequence packing with a block-diagonal causal attention mask.

Standard SFTTrainer packing concatenates sequences but uses a single causal
mask, letting tokens from different examples attend to each other. This file
fixes that by passing:
  - a 4D block-diagonal causal attention_mask  (1, 1, total_len, total_len)
  - per-sub-sequence restarted position_ids    (1, total_len)

The combination enforces that each sub-sequence only attends to previous
tokens within itself (correct causal masking) and gets correct RoPE encodings.
No custom attention registration or model subclassing required.

attn_implementation="sdpa" is used because PyTorch's SDPA kernel accepts
arbitrary 4D float masks natively. FA2 only supports causal + padding masks.
"""

import time
from dataclasses import dataclass

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"

MAX_INPUT_LEN  = 1024
MAX_TARGET_LEN = 128
MAX_SEQ_LEN    = MAX_INPUT_LEN + MAX_TARGET_LEN

PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"

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
    sequences_per_pack: int = 4


GPU_OPTS = GPUOpts()

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

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
# Benchmark trainer — logs active (non-padding) tokens per second
# ---------------------------------------------------------------------------

class BenchmarkTrainer(Trainer):
    """
    Injects throughput and memory metrics into the standard training logs.
    With varlen packing there is no padding, so every token in the batch is
    an active token — throughput/active_tokens_per_sec equals the raw token
    throughput and is directly comparable with the same metric in
    finetune_padding.py.

    Logged keys (every logging_steps):
        throughput/active_tokens_per_sec  — non-padding tokens per second
        throughput/samples_per_sec        — actual sequences per second
        memory/peak_vram_gb               — peak VRAM over the logging window
    """

    def _bench_reset(self) -> None:
        self._bench_active:  int        = 0
        self._bench_samples: int        = 0
        self._bench_t0:      float|None = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        if not hasattr(self, "_bench_active"):
            self._bench_reset()

        # With varlen packing input_ids has no padding → all tokens are active.
        self._bench_active  += inputs["input_ids"].numel()
        # cu_seqlens has shape (n_seqs + 1,), so actual sequences = len - 1.
        self._bench_samples += len(inputs["cu_seqlens"]) - 1

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

# ---------------------------------------------------------------------------
# Collator — packs N tokenised sequences into one flat tensor
# ---------------------------------------------------------------------------

def _build_block_diag_causal_mask(cu_seqlens: torch.Tensor, total_len: int) -> torch.Tensor:
    """
    Returns (1, 1, total_len, total_len) bool mask:
      True  — token pair is in the same sequence and key_pos <= query_pos
      False — cross-sequence or future position (must not attend)
    """
    # Assign a sequence index to every token position
    seq_ids = torch.zeros(total_len, dtype=torch.long)
    for i in range(len(cu_seqlens) - 1):
        seq_ids[cu_seqlens[i] : cu_seqlens[i + 1]] = i

    # bool mask: True = can attend, False = blocked
    # PyTorch SDPA converts bool → float bias internally, avoiding dtype mismatches.
    same_seq = seq_ids.unsqueeze(1) == seq_ids.unsqueeze(0)
    positions = torch.arange(total_len)
    causal   = positions.unsqueeze(1) >= positions.unsqueeze(0)  # lower-triangular

    return (same_seq & causal).unsqueeze(0).unsqueeze(0)  # (1, 1, total_len, total_len) bool


@dataclass
class VarLenPackingCollator:
    """
    Receives a list of pre-tokenised examples (each with input_ids + labels)
    and returns a single packed batch:

        input_ids      : (1, total_len)              — all sequences concatenated
        labels         : (1, total_len)              — -100 for prompt tokens
        position_ids   : (1, total_len)              — restarted from 0 per sub-sequence
        attention_mask : (1, 1, total_len, total_len)— block-diagonal causal bool mask
        cu_seqlens     : (n_seqs + 1,)               — kept for the benchmark counter
    """
    max_seq_len: int = MAX_SEQ_LEN

    def __call__(self, features: list[dict]) -> dict:
        # Sort longest-first for better kernel utilisation
        features = sorted(features, key=lambda f: len(f["input_ids"]), reverse=True)

        all_ids, all_labels, seqlens = [], [], []
        for f in features:
            ids = f["input_ids"][: self.max_seq_len]
            lbl = f["labels"][: self.max_seq_len]
            all_ids.extend(ids)
            all_labels.extend(lbl)
            seqlens.append(len(ids))

        seqlens_t  = torch.tensor(seqlens, dtype=torch.int32)
        cu_seqlens = torch.zeros(len(seqlens) + 1, dtype=torch.int32)
        cu_seqlens[1:] = seqlens_t.cumsum(0)

        total_len = int(cu_seqlens[-1])

        # position_ids: [0..n1-1, 0..n2-1, ...] — restarts per sub-sequence
        position_ids = torch.cat([
            torch.arange(n, dtype=torch.long) for n in seqlens
        ]).unsqueeze(0)  # (1, total_len)

        return {
            "input_ids":      torch.tensor(all_ids,    dtype=torch.long).unsqueeze(0),
            "labels":         torch.tensor(all_labels, dtype=torch.long).unsqueeze(0),
            "position_ids":   position_ids,
            "attention_mask": _build_block_diag_causal_mask(cu_seqlens, total_len),
            "cu_seqlens":     cu_seqlens,   # used only by BenchmarkTrainer counter
        }

# ---------------------------------------------------------------------------
# Dataset — tokenise without padding (collator handles packing)
# ---------------------------------------------------------------------------

def build_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_streaming_dataset(tokenizer, split: str):
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

    ds = load_dataset(
        DATASET_NAME, DATASET_VERSION,
        split=split, streaming=True, trust_remote_code=True,
    )
    return ds.map(tokenize, remove_columns=["article", "highlights", "id"])

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if GPU_OPTS.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer  = build_tokenizer()
    model      = load_model()
    train_ds   = load_streaming_dataset(tokenizer, "train")
    eval_ds    = load_streaming_dataset(tokenizer, "validation")
    collator   = VarLenPackingCollator(max_seq_len=MAX_SEQ_LEN)

    training_args = TrainingArguments(
        output_dir                  = "./checkpoints_varlen",
        per_device_train_batch_size = GPU_OPTS.sequences_per_pack,
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

    trainer = BenchmarkTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = eval_ds,
        data_collator    = collator,
        processing_class = tokenizer,
    )

    trainer.train()
    trainer.save_model("./final_model_varlen")
    tokenizer.save_pretrained("./final_model_varlen")
    print("Done. Model saved to ./final_model_varlen")


if __name__ == "__main__":
    main()
