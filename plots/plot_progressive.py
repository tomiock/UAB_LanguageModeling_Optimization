"""
2D scatter: VRAM vs throughput, coloured by batch size.
Reads metrics/progressive.json.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path

METRICS_PATH = Path(__file__).parent.parent / "metrics" / "progressive.json"

with open(METRICS_PATH) as f:
    records = json.load(f)

vram  = [r["vram_avg"]  for r in records]
tps   = [r["tps_avg"]   for r in records]
bs    = [r["per_device_bs"] for r in records]
names = [r["name"]      for r in records]

# Display labels (cleaner than raw names)
LABELS = {
    "baseline":  "Baseline\n(sdpa)",
    "fa2":       "+FA2",
    "compile":   "+compile",
    "bs12":      "+bs×3",
    "bs12_sac":  "+bs×3\n+SAC",
    "bs16_sac":  "+bs×4\n+SAC",
}

# ---------------------------------------------------------------------------
# Colour by per_device_bs
# ---------------------------------------------------------------------------

unique_bs   = sorted(set(bs))
palette     = ["#4A90D9", "#E05C5C", "#5CB85C", "#F5A623"]
bs_to_color = {b: palette[i % len(palette)] for i, b in enumerate(unique_bs)}
colors      = [bs_to_color[b] for b in bs]

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

BG = "#F8F9FA"
fig, ax = plt.subplots(figsize=(10, 7))
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.spines[["top", "right"]].set_visible(False)

# Draw progression arrows
for i in range(len(records) - 1):
    ax.annotate(
        "",
        xy=(vram[i+1], tps[i+1]),
        xytext=(vram[i], tps[i]),
        arrowprops=dict(arrowstyle="-|>", color="#CCCCCC", lw=1.2),
        zorder=1,
    )

# Scatter
for i, (x, y, c, name) in enumerate(zip(vram, tps, colors, names)):
    ax.scatter(x, y, color=c, s=180, zorder=3, edgecolors="white", linewidths=1.5)

    label = LABELS.get(name, name)
    # Nudge labels to avoid overlap
    offsets = {
        "baseline":  ( 0.3, -700),
        "fa2":       ( 0.3,  300),
        "compile":   ( 0.3,  300),
        "bs12":      (-0.3, -800),
        "bs12_sac":  ( 0.3,  300),
        "bs16_sac":  (-0.3, -800),
    }
    dx, dy = offsets.get(name, (0.3, 300))
    ax.annotate(
        label,
        xy=(x, y),
        xytext=(x + dx, y + dy),
        fontsize=8.5,
        ha="left" if dx > 0 else "right",
        va="center",
        color="#333333",
        zorder=4,
    )

# Axis labels
ax.set_xlabel("Peak VRAM (GB)", fontsize=12)
ax.set_ylabel("Active tokens / second", fontsize=12)

ax.set_ylim(3200, 14500)

ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.xaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Legend for batch size
handles = [
    plt.scatter([], [], color=bs_to_color[b], s=100, edgecolors="white",
                linewidths=1.2, label=f"per_device_bs = {b}")
    for b in unique_bs
]
ax.legend(handles=handles, fontsize=10, frameon=False, loc="upper left")

ax.set_title(
    "Memory vs Throughput — progressive GPU optimisation\nQwen3-0.6B · bf16 · CNN/DailyMail",
    fontsize=12, fontweight="bold", pad=12,
)

plt.tight_layout()
out = Path(__file__).parent / "progressive.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
