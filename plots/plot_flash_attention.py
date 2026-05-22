"""
Visualises Flash Attention 2 impact from metrics/flash.md.

Three configurations:
  baseline        — SDPA, no compile
  FA2             — Flash Attention 2, no compile
  FA2 + compile   — Flash Attention 2 + torch.compile

Two panels:
  Left  — Active tokens/s (higher is better)
  Right — Peak VRAM in GB (lower is better)
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Data — averaged across logged steps
# ---------------------------------------------------------------------------

CONFIGS = ["SDPA\n(baseline)", "FA2", "FA2 +\ntorch.compile"]

# active_tokens_per_sec
THROUGHPUT = [
    np.mean([4311, 4240, 4294]),          # baseline
    np.mean([6728, 6706, 6746, 6679]),    # FA2
    np.mean([11059, 11959, 11816]),       # FA2 + compile
]

# peak_vram_gb
VRAM = [
    np.mean([21.79, 21.79, 21.79]),       # baseline
    np.mean([21.39, 21.48, 21.40, 21.41]),# FA2
    np.mean([14.28, 14.22, 14.25]),       # FA2 + compile
]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG     = "#F8F9FA"
COLORS = ["#E05C5C", "#4A90D9", "#5CB85C"]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)

x = np.arange(len(CONFIGS))
BAR_W = 0.5

# ── Panel 1: throughput ─────────────────────────────────────────────────────

bars1 = ax1.bar(x, THROUGHPUT, width=BAR_W, color=COLORS, edgecolor="none", zorder=3)
ax1.set_ylabel("Active tokens / second", fontsize=11)
ax1.set_title("Throughput\n(higher is better)", fontsize=11, fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels(CONFIGS, fontsize=10)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax1.set_axisbelow(True)
ax1.tick_params(left=False)

# value labels
for bar, val in zip(bars1, THROUGHPUT):
    ax1.text(bar.get_x() + bar.get_width() / 2, val + 150,
             f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

# speedup annotations
base_tps = THROUGHPUT[0]
for i, val in enumerate(THROUGHPUT[1:], start=1):
    mult = val / base_tps
    ax1.text(x[i], val / 2, f"×{mult:.1f}",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color="white")

# ── Panel 2: VRAM ───────────────────────────────────────────────────────────

bars2 = ax2.bar(x, VRAM, width=BAR_W, color=COLORS, edgecolor="none", zorder=3)
ax2.set_ylabel("Peak VRAM (GB)", fontsize=11)
ax2.set_title("Peak VRAM\n(lower is better)", fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(CONFIGS, fontsize=10)
ax2.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax2.set_axisbelow(True)
ax2.tick_params(left=False)
ax2.set_ylim(0, max(VRAM) * 1.18)

# value labels
for bar, val in zip(bars2, VRAM):
    ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
             f"{val:.1f} GB", ha="center", va="bottom", fontsize=9, fontweight="bold")

# savings annotations
base_vram = VRAM[0]
for i, val in enumerate(VRAM[1:], start=1):
    saved = base_vram - val
    if saved > 0.3:
        ax2.text(x[i], val / 2, f"−{saved:.1f} GB",
                 ha="center", va="center", fontsize=10, fontweight="bold",
                 color="white")

plt.suptitle(
    "Flash Attention 2 vs SDPA — Qwen3-0.6B  (batch 4 × grad-accum 4, bf16, CNN/DailyMail)",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()

out = "plots/flash_attention.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
