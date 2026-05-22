"""
Visualises the optimizer memory profiles from metrics/optimizer.md.

Four configurations: {fp32, bf16} × {AdamW baseline, Adam8bit}

Two panels:
  Left  — Base memory (bef-fwd at start of each optimizer step, μ=1):
           persistent memory = model weights + optimizer states.
           Shows exactly how much 8-bit quantisation saves.
  Right — Peak VRAM (pk-bwd) across every micro-step of optimizer steps 1–2:
           shows the full training memory waterfall and where the savings appear.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Raw data — parsed from metrics/optimizer.md
# Format: list of rows, each row is one micro-step
#   (opt_step, micro_step, bef_fwd, aft_fwd, pk_fwd, aft_bwd, pk_bwd)
# ---------------------------------------------------------------------------

DATA = {
    "fp32 + Adam8bit": [
        # step 0
        (0,1, 2.221,19.015,21.297,4.458,23.579),
        (0,2, 4.458,20.978,23.224,4.458,25.470),
        (0,3, 4.458,22.090,24.518,4.458,26.946),
        (0,4, 4.458,22.360,24.823,4.458,27.286),
        # step 1
        (1,1, 4.267,22.169,24.632,6.488,27.096),
        (1,2, 6.488,24.391,26.854,6.488,29.317),
        (1,3, 6.488,23.335,25.654,6.488,27.972),
        (1,4, 6.488,24.943,27.479,6.488,30.015),
        # step 2
        (2,1, 4.267,23.253,25.861,6.488,28.469),
        (2,2, 6.488,23.159,25.441,6.488,27.723),
        (2,3, 6.488,24.673,27.172,6.488,29.672),
        (2,4, 6.488,18.848,20.551,6.488,22.253),
    ],
    "fp32 + AdamW": [
        (0,1, 2.221,19.015,21.297,4.458,23.579),
        (0,2, 4.458,20.978,23.224,4.458,25.470),
        (0,3, 4.458,22.090,24.518,4.458,26.946),
        (0,4, 4.458,22.360,24.823,4.458,27.286),
        (1,1, 6.678,24.580,27.044,8.903,29.507),
        (1,2, 8.903,26.805,29.268,8.903,31.732),
        (1,3, 8.903,25.750,28.069,8.903,30.387),
        (1,4, 8.903,27.356,29.892,8.903,32.427),
        (2,1, 6.678,25.664,28.272,8.899,30.880),
        (2,2, 8.899,25.543,27.825,8.899,30.107),
        (2,3, 8.899,27.084,29.583,8.899,32.083),
        (2,4, 8.899,21.255,22.957,8.899,24.660),
    ],
    "bf16 + Adam8bit": [
        (0,1, 1.111,11.749,15.172,2.251,16.313),
        (0,2, 2.251,12.815,16.183,2.251,17.306),
        (0,3, 2.251,13.457,17.099,2.251,18.313),
        (0,4, 2.251,13.622,17.317,2.251,18.548),
        (1,1, 3.136,14.500,18.195,4.325,19.426),
        (1,2, 4.325,15.691,19.386,4.325,20.618),
        (1,3, 4.325,15.045,18.522,4.325,19.681),
        (1,4, 4.325,16.032,19.835,4.325,21.103),
        (2,1, 3.136,15.183,19.095,4.303,20.399),
        (2,2, 4.303,14.887,18.310,4.303,19.451),
        (2,3, 4.303,15.898,19.648,4.303,20.897),
        (2,4, 4.303,12.163,14.717,4.303,15.569),
    ],
    "bf16 + AdamW": [
        (0,1, 1.111,11.749,15.172,2.251,16.313),
        (0,2, 2.251,12.815,16.183,2.251,17.306),
        (0,3, 2.251,13.457,17.099,2.251,18.313),
        (0,4, 2.251,13.622,17.317,2.251,18.548),
        (1,1, 3.347,14.710,18.405,4.523,19.637),
        (1,2, 4.523,15.890,19.585,4.523,20.816),
        (1,3, 4.523,15.234,18.712,4.523,19.871),
        (1,4, 4.523,16.230,20.034,4.523,21.301),
        (2,1, 3.347,15.394,19.306,4.509,20.610),
        (2,2, 4.509,15.079,18.502,4.509,19.643),
        (2,3, 4.509,16.104,19.853,4.509,21.103),
        (2,4, 4.509,12.368,14.922,4.509,15.774),
    ],
}

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG      = "#F8F9FA"
COLORS  = {
    "fp32 + AdamW":   "#E05C5C",
    "fp32 + Adam8bit":"#F5A623",
    "bf16 + AdamW":   "#4A90D9",
    "bf16 + Adam8bit":"#5CB85C",
}
LS      = {"AdamW": "-", "Adam8bit": "--"}

def col(name): return COLORS[name]
def ls(name):  return "--" if "Adam8bit" in name else "-"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def bef_fwd_series(rows):
    """bef-fwd at μ=1 for each optimizer step (persistent memory floor)."""
    return [(r[0], r[2]) for r in rows if r[1] == 1]

def pk_bwd_steps12(rows):
    """pk-bwd for all micro-steps of optimizer steps 1 and 2."""
    return [r[6] for r in rows if r[0] in (1, 2)]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.patch.set_facecolor(BG)
for ax in (ax1, ax2):
    ax.set_facecolor(BG)
    ax.spines[["top","right"]].set_visible(False)

# ── Panel 1: base memory (bef-fwd μ=1) across optimizer steps ──────────────

for name, rows in DATA.items():
    series = bef_fwd_series(rows)
    steps  = [s for s,_ in series]
    vals   = [v for _,v in series]
    ax1.plot(steps, vals, marker="o", linewidth=2, markersize=6,
             color=col(name), linestyle=ls(name), label=name)

# Annotate the saving at step 2
for pair in [("fp32 + AdamW","fp32 + Adam8bit"),
             ("bf16 + AdamW","bf16 + Adam8bit")]:
    base_val = bef_fwd_series(DATA[pair[0]])[-1][1]
    opt_val  = bef_fwd_series(DATA[pair[1]])[-1][1]
    saving   = base_val - opt_val
    mid_y    = (base_val + opt_val) / 2
    ax1.annotate(
        f"−{saving:.2f} GB",
        xy=(2, mid_y), xytext=(2.15, mid_y),
        fontsize=9, color="#555555", va="center",
        arrowprops=dict(arrowstyle="-", color="#AAAAAA", lw=0.8),
    )

ax1.set_xlabel("Optimizer step", fontsize=11)
ax1.set_ylabel("Base memory — bef-fwd, μ=1 (GB)", fontsize=11)
ax1.set_title("Persistent memory\n(weights + optimizer states)", fontsize=11, fontweight="bold")
ax1.set_xticks([0, 1, 2])
ax1.legend(fontsize=9, frameon=False, loc="upper left")
ax1.tick_params(labelsize=10)

# ── Panel 2: peak VRAM across micro-steps of steps 1–2 ─────────────────────

x = np.arange(8)   # 4 μ-steps × 2 opt-steps
step_labels = ["s1·μ1","s1·μ2","s1·μ3","s1·μ4",
                "s2·μ1","s2·μ2","s2·μ3","s2·μ4"]

for name, rows in DATA.items():
    vals = pk_bwd_steps12(rows)
    ax2.plot(x, vals, marker="o", linewidth=2, markersize=5,
             color=col(name), linestyle=ls(name), label=name)

# Shade the two optimizer steps
ax2.axvspan(-0.5, 3.5, alpha=0.04, color="#888888")
ax2.axvspan( 3.5, 7.5, alpha=0.08, color="#888888")

ax2.set_xlabel("Micro-step", fontsize=11)
ax2.set_ylabel("Peak VRAM — pk-bwd (GB)", fontsize=11)
ax2.set_title("Peak VRAM during backward\n(weights + states + activations + gradients)",
              fontsize=11, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(step_labels, fontsize=8.5)
ax2.legend(fontsize=9, frameon=False)
ax2.tick_params(labelsize=10)

# Re-add step labels now y-lim is set
for ax_x, label in [(1.5, "opt step 1"), (5.5, "opt step 2")]:
    ax2.text(ax_x, ax2.get_ylim()[0] + 0.3, label,
             ha="center", fontsize=8, color="#AAAAAA")

plt.suptitle("Optimizer memory comparison — Qwen3-0.6B  (batch 4 × grad-accum 4)",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()

out = "plots/optimizer_memory.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
