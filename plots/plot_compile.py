"""
Plot throughput and memory metrics: torch.compile ON vs OFF.
Steady-state only — the first 'with' entry is skipped (compile warmup).
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Data  (steady-state averages)
# ---------------------------------------------------------------------------

with_rows = [
    {"throughput/tokens_per_sec": 4841, "throughput/samples_per_sec": 2.36, "memory/peak_vram_gb": 12.23},
    {"throughput/tokens_per_sec": 4844, "throughput/samples_per_sec": 2.37, "memory/peak_vram_gb": 12.23},
]

without_rows = [
    {"throughput/tokens_per_sec": 3878, "throughput/samples_per_sec": 1.89, "memory/peak_vram_gb": 18.88},
    {"throughput/tokens_per_sec": 3875, "throughput/samples_per_sec": 1.89, "memory/peak_vram_gb": 18.88},
    {"throughput/tokens_per_sec": 3877, "throughput/samples_per_sec": 1.89, "memory/peak_vram_gb": 18.88},
    {"throughput/tokens_per_sec": 3874, "throughput/samples_per_sec": 1.89, "memory/peak_vram_gb": 18.88},
]

def avg(rows, key):
    return np.mean([r[key] for r in rows])

metrics = [
    {
        "key":    "throughput/tokens_per_sec",
        "title":  "Tokens / sec",
        "unit":   "tok/s",
        "higher_is_better": True,
    },
    {
        "key":    "throughput/samples_per_sec",
        "title":  "Samples / sec",
        "unit":   "samp/s",
        "higher_is_better": True,
    },
    {
        "key":    "memory/peak_vram_gb",
        "title":  "Peak VRAM",
        "unit":   "GB",
        "higher_is_better": False,
    },
]

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

C_BASE    = "#6C757D"   # muted grey  — baseline (without compile)
C_COMPILE = "#2563EB"   # strong blue — torch.compile
C_GOOD    = "#16A34A"   # green  — improvement annotation
C_BAD     = "#DC2626"   # red    — regression annotation

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
fig.patch.set_facecolor("#F8F9FA")

BAR_W = 0.35
x = np.array([0])

for ax, m in zip(axes, metrics):
    v_base    = avg(without_rows, m["key"])
    v_compile = avg(with_rows,    m["key"])

    pct = (v_compile - v_base) / v_base * 100
    improved = (pct > 0) == m["higher_is_better"]
    ann_color = C_GOOD if improved else C_BAD
    sign = "+" if pct > 0 else ""

    bars_base    = ax.bar(x - BAR_W / 2, v_base,    BAR_W, color=C_BASE,    zorder=3)
    bars_compile = ax.bar(x + BAR_W / 2, v_compile, BAR_W, color=C_COMPILE, zorder=3)

    # Value labels inside bars
    for bar, val in [(bars_base[0], v_base), (bars_compile[0], v_compile)]:
        label = f"{val:,.0f}" if val > 10 else f"{val:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 0.5,
            label,
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white",
        )

    # Percentage badge above the compile bar
    badge_y = v_compile * 1.04
    ax.text(
        bars_compile[0].get_x() + bars_compile[0].get_width() / 2,
        badge_y,
        f"{sign}{pct:.1f}%",
        ha="center", va="bottom",
        fontsize=12, fontweight="bold", color=ann_color,
    )

    # Axes styling
    ax.set_title(m["title"], fontsize=14, fontweight="bold", pad=10)
    ax.set_ylabel(m["unit"], fontsize=11)
    ax.set_xticks([])
    ax.set_facecolor("#F8F9FA")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    top = max(v_base, v_compile)
    ax.set_ylim(0, top * 1.25)

# Shared legend
legend_handles = [
    mpatches.Patch(color=C_BASE,    label="Baseline"),
    mpatches.Patch(color=C_COMPILE, label="torch.compile"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=2,
    fontsize=12,
    frameon=False,
    bbox_to_anchor=(0.5, -0.04),
)

fig.suptitle("torch.compile  —  throughput & memory impact", fontsize=15, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("compile_metrics.png", dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
print("Saved → compile_metrics.png")
