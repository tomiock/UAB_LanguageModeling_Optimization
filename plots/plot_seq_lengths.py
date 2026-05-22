"""
Horizontal histogram of tokenised sequence lengths for CNN/DailyMail (train split).
Shows the long-tail distribution with 70th and 95th percentile markers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datasets import load_dataset
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Config — mirrors finetune_padding.py
# ---------------------------------------------------------------------------

MODEL_ID        = "Qwen/Qwen3-0.6B-Base"
DATASET_NAME    = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
MAX_INPUT_LEN   = 1024 - 128
MAX_TARGET_LEN  = 128
MAX_SEQ_LEN     = MAX_INPUT_LEN + MAX_TARGET_LEN
PROMPT_TEMPLATE = "Summarize the following article.\n\n### Article:\n{article}\n\n### Summary:\n"
N_SAMPLES       = 50_000   # enough to characterise the distribution

# ---------------------------------------------------------------------------
# Load & tokenise
# ---------------------------------------------------------------------------

print("Loading tokenizer…")
tok = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Streaming {N_SAMPLES} examples from {DATASET_NAME}…")
ds = load_dataset(DATASET_NAME, DATASET_VERSION,
                  split="train", streaming=True, trust_remote_code=True)

lengths = []
for i, ex in enumerate(ds):
    if i >= N_SAMPLES:
        break
    prompt     = PROMPT_TEMPLATE.format(article=ex["article"])
    target     = ex["highlights"] + tok.eos_token
    prompt_len = len(tok(prompt,  truncation=False, add_special_tokens=True )["input_ids"])
    target_len = len(tok(target,  truncation=False, add_special_tokens=False)["input_ids"])
    lengths.append(prompt_len + target_len)   # raw (un-truncated) total length
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{N_SAMPLES}")

lengths = np.array(lengths)
p70  = np.percentile(lengths, 70)
p95  = np.percentile(lengths, 95)

print(f"\nMedian : {np.median(lengths):.0f} tokens")
print(f"p70    : {p70:.0f} tokens")
print(f"p95    : {p95:.0f} tokens")
print(f"Max    : {lengths.max()} tokens")
print(f"Seqs ≤ MAX_SEQ_LEN ({MAX_SEQ_LEN}): {(lengths <= MAX_SEQ_LEN).mean()*100:.1f}%")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

BG   = "#F8F9FA"
BLUE = "#4A90D9"

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# Bins up to 99th-percentile + a little headroom, one bin per 64 tokens
bin_max  = int(np.percentile(lengths, 99)) + 64
bins     = np.arange(0, bin_max + 64, 64)

counts, edges = np.histogram(lengths, bins=bins)
bin_centres   = (edges[:-1] + edges[1:]) / 2

# Horizontal bars
ax.barh(bin_centres, counts, height=60,
        color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.5)

# Percentile lines
for pct, val, ls, label in [
    (70, p70, "--", f"p70 = {p70:.0f} tok"),
    (95, p95, "-",  f"p95 = {p95:.0f} tok"),
]:
    ax.axhline(val, color="#E05C5C", linewidth=1.6, linestyle=ls,
               label=label, zorder=3)

# MAX_SEQ_LEN cap line
ax.axhline(MAX_SEQ_LEN, color="#F5A623", linewidth=1.4, linestyle=":",
           label=f"MAX_SEQ_LEN = {MAX_SEQ_LEN}", zorder=3)

# Shaded region above MAX_SEQ_LEN (truncated sequences)
ax.axhspan(MAX_SEQ_LEN, ax.get_ylim()[1] if ax.get_ylim()[1] > MAX_SEQ_LEN else bin_max,
           color="#F5A623", alpha=0.08)

ax.set_xlabel("Number of sequences", fontsize=11)
ax.set_ylabel("Sequence length (tokens, un-truncated)", fontsize=11)
ax.set_title(f"CNN/DailyMail token-length distribution\n"
             f"(first {N_SAMPLES:,} training examples, Qwen3-0.6B tokenizer)",
             fontsize=12, fontweight="bold")

ax.legend(fontsize=10, frameon=False)
ax.spines[["top", "right"]].set_visible(False)
ax.tick_params(labelsize=10)

# Annotation: % truncated
pct_trunc = (lengths > MAX_SEQ_LEN).mean() * 100
ax.text(counts.max() * 0.97, MAX_SEQ_LEN + 30,
        f"{pct_trunc:.1f}% truncated →",
        ha="right", va="bottom", fontsize=9, color="#F5A623")

plt.tight_layout()
out = "plots/seq_lengths.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"\nSaved → {out}")
