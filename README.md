# GPU Training Optimization — Qwen3-0.6B on CNN/DailyMail

Each script fine-tunes **Qwen3-0.6B-Base** on the CNN/DailyMail summarization dataset and isolates one GPU optimization technique so runs can be compared side-by-side in Weights & Biases.

See [PROFILING.md](PROFILING.md) for how to open and interpret the memory snapshots produced during training, and how to inspect `torch.compile` behavior and results.

## Main Scripts 

### `finetune_summarization.py`
**Baseline / fused AdamW optimizer showcase.**

The starting point for most comparisons. Fine-tunes Qwen3-0.6B with a custom `BenchmarkTrainer` that logs tokens/s, samples/s, and peak VRAM every N steps. The key toggle is `GPU_OPTS.fused_adam`: standard AdamW launches one CUDA kernel per parameter tensor, while `adamw_torch_fused` merges them all into a single kernel, yielding a ~5–10% throughput gain at no memory cost. Also captures a per-micro-step memory profile (before/after forward, peak forward, after/peak backward) and saves a `.pickle` snapshot compatible with the PyTorch memory visualizer.

---

### `finetune_half_two_epochs.py`
**Half the data, 2 epochs — same total compute as the full-epoch run.**

The counterpart to `finetune_full_epoch.py`. Selects the first half of the training set and trains for 2 epochs, matching the optimizer step count of the single-epoch run. The experiment tests whether repeated exposure to fewer examples or a single pass over more examples yields better validation loss, illustrating the standard NLP argument against multi-epoch training on large datasets.

## Training scripts


### `finetune_flash_attention.py`
**Flash Attention 2 vs. SDPA.**

Toggle `ATTN_OPTS.flash_attention` to switch between standard PyTorch scaled-dot-product attention (O(N²) memory) and Flash Attention 2 (O(N) memory, 2–4× faster on long sequences). Uses dynamic padding (pad to longest in batch) to make FA2's memory advantage visible in real batches. Requires `pip install flash-attn --no-build-isolation`.

---

### `finetune_padding.py`
**Padding strategy comparison: fixed vs. dynamic vs. packing.**

Three modes controlled by the `PADDING_MODE` enum:
- **FIXED** — every sequence padded to `MAX_SEQ_LEN`; constant tensor shapes help `torch.compile` but wastes FLOPs on pad tokens.
- **DYNAMIC** — pad each batch to its longest sequence; less compute waste.
- **PACKING** — uses TRL's `SFTTrainer` to concatenate examples into `MAX_SEQ_LEN` chunks; near-zero padding waste and maximum GPU utilization.

A shared `BenchmarkMixin` counts only non-padding (active) tokens so throughput numbers are comparable across all three modes.

---

### `finetune_progressive.py`
**Step-by-step optimization story from baseline to maximum throughput.**

Runs seven configs sequentially, each adding one optimization over the previous: baseline (SDPA only) → +FA2 → +`torch.compile` → +larger batch (×3) → +Selective Activation Checkpointing (SAC) at increasing batch sizes. Results are saved to `metrics/progressive.json` and skips already-completed configs on re-run. Demonstrates how memory savings from FA2 and SAC can be recycled into larger batches rather than just reducing VRAM.

---

### `finetune_full_epoch.py`
**Full-epoch training across 2× L40S GPUs with Distributed Data Parallel (DDP).**

Trains for one complete epoch (~4,500 optimizer steps) on the full CNN/DailyMail training set using two GPUs in parallel.

**What DDP does:** each GPU holds a full copy of the model and processes a different slice of the batch. After each backward pass, gradients are all-reduced across GPUs (summed and averaged) so every replica stays in sync. With 2 GPUs the effective batch size doubles — and so does throughput — without any change to the model architecture or training code.

**How it is set up here:** the script is launched with `accelerate launch --config_file accelerate_l40s.yaml finetune_full_epoch.py`. Accelerate wraps HuggingFace Trainer and handles process spawning, device placement, and the gradient all-reduce automatically. Inside the script, `PartialState` from Accelerate exposes `num_processes` (world size) and `is_main_process` so throughput metrics can be scaled by `world_size` and console output restricted to rank 0.

Key differences from the single-GPU scripts: non-streaming (full) dataset load so the epoch boundary is exact, Flash Attention 2 enabled (L40S supports it), larger per-device batch of 10 (48 GB VRAM), and dataset tokenization done on the main process first then read from cache by all ranks. Paired with `finetune_half_two_epochs.py` for a controlled compute-budget comparison.

---

### `evaluate_epoch_vs_repeat.py`
**Test-set evaluation of the two epoch-comparison models.**

Loads `final_model_epoch` and `final_model_half2e` and generates greedy summaries on the CNN/DailyMail test set. Computes ROUGE-1/2/L/Lsum and BERTScore F1/P/R (via RoBERTa-large). Generated summaries are cached to disk so metrics can be recomputed without re-running inference. Prints a side-by-side table with a delta row.

## Plot scripts (`plots/`)

Each script reads from hard-coded numbers or from `metrics/*.json` and saves a `.png` to the same directory.

| Script | What it plots |
|---|---|
| `plot_ac.py` | Activation checkpointing (full AC on/off) and compiler-based Selective AC vs. baseline |
| `plot_compile.py` | `torch.compile` on vs. off — throughput and peak VRAM |
| `plot_epoch_vs_repeat.py` | Validation loss curves: 1 epoch full data vs. 2 epochs half data |
| `plot_flash_attention.py` | SDPA vs. FA2 vs. FA2+compile — active tokens/s and peak VRAM |
| `plot_optimizer.py` | Optimizer memory waterfall across {fp32, bf16} × {AdamW, Adam8bit} |
| `plot_padding.py` | Visual diagram of the three padding strategies (fixed / dynamic / packing) |
| `plot_progressive.py` | 2D scatter of VRAM vs. throughput across the progressive optimization configs |
| `plot_seq_lengths.py` | Histogram of tokenized sequence lengths in the CNN/DailyMail training split |
