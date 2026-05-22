"""
Visual explanation of the three batch padding strategies.
Uses illustrative hand-crafted token arrays — no model or dataset required.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ---------------------------------------------------------------------------
# Token types & colors
# ---------------------------------------------------------------------------

PROMPT, TARGET, PAD, EOS = 0, 1, 2, 3

COLORS = {
    PROMPT: "#4A90D9",   # blue
    TARGET: "#5CB85C",   # green
    PAD:    "#E2E2E2",   # light grey
    EOS:    "#F5A623",   # amber  — sequence separator in packing
}
LABELS = {
    PROMPT: "Prompt",
    TARGET: "Target (loss computed here)",
    PAD:    "Padding  (wasted compute)",
    EOS:    "EOS separator",
}

# ---------------------------------------------------------------------------
# Batch definitions
# 4 sequences of varying length: 7, 7, 8, 6 tokens
# MAX_LEN = 16  /  DYN_LEN = 8  (longest in batch)
# ---------------------------------------------------------------------------

P, T, _, E = PROMPT, TARGET, PAD, EOS

SEQ = [
    [P, P, P, P, P, T, T, T, T, T, T, T],          # 4 prompt + 3 target = 7
    [P, P, P, P, T, T, T, T, T, T ],          # 3 prompt + 4 target = 7
    [P, P, P, P, P, T, T, T, T, T, T, T, T],       # 5 prompt + 3 target = 8
    [P, P, T, T, T],             # 2 prompt + 4 target = 6
]
MAX_LEN = 16
DYN_LEN = max(len(s) for s in SEQ)  #8

def padded(seq, length):
    return seq + [PAD] * (length - len(seq))

fixed_batch   = [padded(s, MAX_LEN) for s in SEQ]
dynamic_batch = [padded(s, DYN_LEN) for s in SEQ]
packed_batch  = [
    # seq1 (7) + EOS + seq2 (7) + EOS  →  7+1+7+1 = 16
    [P, P, P, P, T, T, T, T, T, T, E, P, T,T,T, E],
    # seq3 (8) + EOS + seq4 (6) + EOS  →  8+1+6+1 = 16
    [P,P,T,T,T, E, P,P,T, E, P,P,T,T,T,E],
]

# ---------------------------------------------------------------------------
# Stats helper
# ---------------------------------------------------------------------------

def batch_stats(batch):
    total  = sum(len(r) for r in batch)
    active = sum(t != PAD for r in batch for t in r)
    return total, active, total - active, 100 * active / total

# ---------------------------------------------------------------------------
# Drawing helper
# ---------------------------------------------------------------------------

BG = "#F8F9FA"

def draw_batch(ax, batch, canvas_cols, label, not_computed_from=None):
    """
    Render a batch as a grid of rounded token cells.

    canvas_cols       : total columns rendered (≥ len(batch[0]))
    not_computed_from : shade columns from here to canvas_cols as
                        "not computed" (used for the dynamic panel)
    """
    n_rows     = len(batch)
    n_data_cols = len(batch[0])

    # ---- token cells -------------------------------------------------------
    for r, row in enumerate(batch):
        y = n_rows - 1 - r          # row 0 at top
        for c, tok in enumerate(row):
            # main cell
            ax.add_patch(mpatches.FancyBboxPatch(
                [c + 0.06, y + 0.06], 0.88, 0.88,
                boxstyle="round,pad=0.04",
                facecolor=COLORS[tok], edgecolor="white",
                linewidth=1.8, zorder=2,
            ))
            # diagonal hatch overlay on PAD cells to scream "wasted"
            if tok == PAD:
                ax.add_patch(mpatches.FancyBboxPatch(
                    [c + 0.06, y + 0.06], 0.88, 0.88,
                    boxstyle="round,pad=0.04",
                    fill=False, edgecolor="#BABABA",
                    linewidth=0, hatch="////", zorder=3,
                ))

    # ---- "not computed" shaded region  (dynamic panel only) ----------------
    if not_computed_from is not None:
        nc_cols = canvas_cols - not_computed_from
        ax.add_patch(mpatches.Rectangle(
            [not_computed_from, 0], nc_cols, n_rows,
            facecolor="#EFEFEF", edgecolor="#D5D5D5",
            linewidth=1, linestyle="--", zorder=1,
        ))
        ax.text(
            not_computed_from + nc_cols / 2, n_rows / 2,
            "not computed\n(saved)", ha="center", va="center",
            fontsize=9, color="#AAAAAA", fontstyle="italic",
        )

    # ---- sequence-boundary brackets in packing (above row 0 only) ----------
    # drawn as thin vertical dashed lines at EOS positions
    if any(E in row for row in batch):
        for r, row in enumerate(batch):
            y = n_rows - 1 - r
            for c, tok in enumerate(row):
                if tok == EOS and c < n_data_cols - 1:   # skip last EOS
                    ax.plot([c + 1, c + 1], [y + 0.08, y + 0.92],
                            color="#CCCCCC", linewidth=1.2,
                            linestyle="--", zorder=4)

    # ---- row labels (left) & per-row util (right) --------------------------
    for r, row in enumerate(batch):
        y = n_rows - 1 - r
        ax.text(-0.35, y + 0.5, f"seq {r + 1}",
                va="center", ha="right", fontsize=9, color="#555555")
        n_active = sum(t != PAD for t in row)
        ax.text(n_data_cols + 0.2, y + 0.5,
                f"{n_active} / {n_data_cols}",
                va="center", ha="left", fontsize=8.5, color="#666666")

    # ---- column indices (top) ----------------------------------------------
    for c in range(n_data_cols):
        ax.text(c + 0.5, n_rows + 0.08, str(c),
                ha="center", va="bottom", fontsize=7, color="#BBBBBB")

    # ---- panel label -------------------------------------------------------
    total, active, wasted, pct = batch_stats(batch)
    ops_str = f"{n_rows} seq × {n_data_cols} tok = {total} ops"
    stat_str = (
        f"{active} active ({pct:.0f}%)   "
        + (f"{wasted} wasted" if wasted else "0 wasted  ✓")
    )
    ax.text(-0.35, n_rows + 0.55, label,
            fontsize=13, fontweight="bold", va="bottom", color="#1A1A2E")
    ax.text(len(label) * 0.38, n_rows + 0.55, f"   {ops_str}   |   {stat_str}",
            fontsize=9, va="bottom", color="#555555")

    # ---- axes --------------------------------------------------------------
    ax.set_xlim(-1.5, canvas_cols + 1.8)
    ax.set_ylim(-0.15, n_rows + 0.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_facecolor(BG)

# ---------------------------------------------------------------------------
# Figure layout
# ---------------------------------------------------------------------------

fig = plt.figure(figsize=(15, 9.5))
fig.patch.set_facecolor(BG)

# Heights proportional to batch rows (fixed=4, dynamic=4, packing=2)
gs = GridSpec(3, 1, figure=fig,
              height_ratios=[4, 4, 2],
              hspace=0.55,
              top=0.93, bottom=0.08)

ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1])
ax3 = fig.add_subplot(gs[2])

draw_batch(ax1, fixed_batch,   MAX_LEN, "① Fixed padding")
draw_batch(ax2, dynamic_batch, MAX_LEN, "② Dynamic padding",
           not_computed_from=DYN_LEN)
draw_batch(ax3, packed_batch,  MAX_LEN, "③ Packing  (SFTTrainer)")

# ---------------------------------------------------------------------------
# Legend & title
# ---------------------------------------------------------------------------

fig.legend(
    handles=[
        mpatches.Patch(facecolor=COLORS[PROMPT], label=LABELS[PROMPT]),
        mpatches.Patch(facecolor=COLORS[TARGET], label=LABELS[TARGET]),
        mpatches.Patch(facecolor=COLORS[PAD], edgecolor="#BABABA",
                       hatch="////", label=LABELS[PAD]),
        mpatches.Patch(facecolor=COLORS[EOS],   label=LABELS[EOS]),
    ],
    loc="lower center", ncol=4, fontsize=10,
    frameon=False, bbox_to_anchor=(0.5, 0.0),
)

fig.suptitle("Batch padding strategies", fontsize=16, fontweight="bold")

out = "plots/padding_strategies.png"
fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
