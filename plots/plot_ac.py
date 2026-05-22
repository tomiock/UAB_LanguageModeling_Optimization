"""
Two plots comparing Activation Checkpointing (AC) variants:

  Plot 1 — Full AC on/off
    baseline : no AC          (steady-state: entry 2 only, entry 1 is warmup)
    variant  : full AC        (steady-state: avg entries 2–3, entry 1 is warmup)

  Plot 2 — Compiler Selective AC (SAC / Budgeted AC)
    baseline : torch.compile, no AC   (avg all 3 entries)
    variant  : torch.compile + SAC    (avg both entries)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

# --- Plot 1 -----------------------------------------------------------------
no_ac_rows = [
    # entry 1 skipped (warmup)
    {"throughput/tokens_per_sec": 3878, "throughput/samples_per_sec": 1.89, "memory/peak_vram_gb": 18.88},
]

full_ac_rows = [
    # entry 1 skipped (warmup)
    {"throughput/tokens_per_sec": 3414, "throughput/samples_per_sec": 1.67, "memory/peak_vram_gb": 10.56},
    {"throughput/tokens_per_sec": 3415, "throughput/samples_per_sec": 1.67, "memory/peak_vram_gb": 10.56},
]

# --- Plot 2 -----------------------------------------------------------------
compile_baseline_rows = [
    {"throughput/tokens_per_sec": 4854, "throughput/samples_per_sec": 2.37, "memory/peak_vram_gb": 12.23},
    {"throughput/tokens_per_sec": 4854, "throughput/samples_per_sec": 2.37, "memory/peak_vram_gb": 12.23},
    {"throughput/tokens_per_sec": 4853, "throughput/samples_per_sec": 2.37, "memory/peak_vram_gb": 12.23},
]

sac_rows = [
    {"throughput/tokens_per_sec": 4559, "throughput/samples_per_sec": 2.23, "memory/peak_vram_gb": 10.0},
    {"throughput/tokens_per_sec": 4559, "throughput/samples_per_sec": 2.23, "memory/peak_vram_gb": 10.0},
]

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

METRICS = [
    {"key": "throughput/tokens_per_sec",  "title": "Tokens / sec",  "unit": "tok/s",  "higher_is_better": True},
    {"key": "throughput/samples_per_sec", "title": "Samples / sec", "unit": "samp/s", "higher_is_better": True},
    {"key": "memory/peak_vram_gb",        "title": "Peak VRAM",     "unit": "GB",     "higher_is_better": False},
]

C_BASE = "#6C757D"
C_GOOD = "#16A34A"
C_BAD  = "#DC2626"
BG     = "#F8F9FA"
BAR_W  = 0.35


def avg(rows, key):
    return np.mean([r[key] for r in rows])


def _bar_label(ax, bar, val):
    label = f"{val:,.0f}" if val > 10 else f"{val:.2f}"
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() * 0.5,
        label,
        ha="center", va="center",
        fontsize=11, fontweight="bold", color="white",
    )


def _pct_badge(ax, bar, pct, label, y_offset, higher_is_better):
    improved  = (pct > 0) == higher_is_better
    color     = C_GOOD if improved else C_BAD
    sign      = "+" if pct > 0 else ""
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        y_offset,
        f"{label}{sign}{pct:.1f}%",
        ha="center", va="bottom",
        fontsize=10, fontweight="bold", color=color,
    )


def make_plot(baseline_rows, variant_rows, variant_color, baseline_label,
              variant_label, title, filename,
              extra_baseline=None):
    """
    extra_baseline: optional dict { "rows": list, "label": str, "color": str }
    When provided, every subplot gets a third bar and two percentage badges
    above the variant bar — one vs the primary baseline, one vs the extra baseline.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG)
    x = np.array([0])

    legend_handles = [
        mpatches.Patch(color=C_BASE,        label=baseline_label),
        mpatches.Patch(color=variant_color, label=variant_label),
    ]

    for ax, m in zip(axes, METRICS):
        v_base    = avg(baseline_rows, m["key"])
        v_variant = avg(variant_rows,  m["key"])

        if extra_baseline is not None:
            v_extra = avg(extra_baseline["rows"], m["key"])
            c_extra = extra_baseline["color"]

            W = 0.25
            gap = 0.05
            b_extra   = ax.bar(x - W - gap, v_extra,   W, color=c_extra,      zorder=3)
            b_base    = ax.bar(x,            v_base,    W, color=C_BASE,        zorder=3)
            b_variant = ax.bar(x + W + gap, v_variant, W, color=variant_color, zorder=3)

            for bar, val in [(b_extra[0], v_extra), (b_base[0], v_base), (b_variant[0], v_variant)]:
                _bar_label(ax, bar, val)

            top  = max(v_extra, v_base, v_variant)
            ylim = top * 1.35

            pct_vs_base  = (v_variant - v_base)  / v_base  * 100
            pct_vs_extra = (v_variant - v_extra) / v_extra * 100

            _pct_badge(ax, b_variant[0], pct_vs_base,  "vs compile: ", v_variant * 1.04, m["higher_is_better"])
            _pct_badge(ax, b_variant[0], pct_vs_extra, "vs base: ",   v_variant * 1.17, m["higher_is_better"])

        else:
            pct   = (v_variant - v_base) / v_base * 100
            b_base    = ax.bar(x - BAR_W / 2, v_base,    BAR_W, color=C_BASE,        zorder=3)
            b_variant = ax.bar(x + BAR_W / 2, v_variant, BAR_W, color=variant_color, zorder=3)

            for bar, val in [(b_base[0], v_base), (b_variant[0], v_variant)]:
                _bar_label(ax, bar, val)

            _pct_badge(ax, b_variant[0], pct, "", v_variant * 1.04, m["higher_is_better"])
            top  = max(v_base, v_variant)
            ylim = top * 1.25

        ax.set_title(m["title"], fontsize=14, fontweight="bold", pad=10)
        ax.set_ylabel(m["unit"], fontsize=11)
        ax.set_xticks([])
        ax.set_facecolor(BG)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ylim)

    if extra_baseline is not None:
        legend_handles.insert(0, mpatches.Patch(
            color=extra_baseline["color"],
            label=extra_baseline["label"],
        ))

    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=len(legend_handles), fontsize=12, frameon=False,
        bbox_to_anchor=(0.5, -0.04),
    )
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    out = OUT_DIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------

make_plot(
    baseline_rows   = no_ac_rows,
    variant_rows    = full_ac_rows,
    variant_color   = "#9333EA",          # purple — full AC
    baseline_label  = "Baseline (no AC)",
    variant_label   = "Full AC",
    title           = "Activation Checkpointing (full)  —  throughput & memory impact",
    filename        = "ac_full.png",
)

make_plot(
    baseline_rows   = compile_baseline_rows,
    variant_rows    = sac_rows,
    variant_color   = "#EA580C",          # orange — SAC
    baseline_label  = "torch.compile",
    variant_label   = "torch.compile + SAC (budget=0.5)",
    title           = "Selective AC (compiler)  —  throughput & memory impact",
    filename        = "ac_sac.png",
    extra_baseline={
        "rows":  no_ac_rows,
        "label": "Baseline (no AC)",
        "color": "#9CA3AF",               # lighter grey — distinguishable from compile baseline
    },
)
