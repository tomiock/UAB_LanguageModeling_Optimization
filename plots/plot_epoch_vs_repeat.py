"""
Validation loss: 1 epoch on full data  vs.  2 epochs on half the data.

Both runs have the same total optimizer steps (~1 795) and identical
hyperparameters.  The only difference is whether the model encounters
every training example once or half the examples twice.

Data pulled from W&B runs:
  epoch1   — adamw_fused_bs160_bf16_fa2_compile_epoch1_4gpu   (id: 0ekeoglt)
  half_2e  — adamw_fused_bs160_bf16_fa2_compile_half_2epochs_4gpu (id: zhdqj8mg)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ---------------------------------------------------------------------------
# Data  (pulled from W&B API)
# ---------------------------------------------------------------------------

# (optimizer_step, eval_loss, train_epoch_fraction)
EVAL = {
    "epoch1": [
        (500,  1.5512, 0.279),
        (1000, 1.5320, 0.557),
        (1500, 1.5293, 0.836),
        (1795, 1.5292, 1.000),
    ],
    "half_2e": [
        (500,  1.6237, 0.557),
        (1000, 1.6161, 1.114),
        (1500, 1.6161, 1.671),
        (1796, 1.6164, 2.000),
    ],
}

# (optimizer_step, train_loss)  — logged every 10 steps
TRAIN_EPOCH1 = [
    (10,2.3559),(20,2.1442),(30,1.9200),(40,1.8217),(50,1.7733),(60,1.7258),
    (70,1.7490),(80,1.7231),(90,1.7132),(100,1.7053),(110,1.6992),(120,1.6655),
    (130,1.6530),(140,1.6542),(150,1.6274),(160,1.6367),(170,1.6470),(180,1.6388),
    (190,1.6430),(200,1.6576),(210,1.6141),(220,1.6609),(230,1.6536),(240,1.6416),
    (250,1.6380),(260,1.6230),(270,1.6175),(280,1.6372),(290,1.6281),(300,1.6327),
    (310,1.5986),(320,1.6066),(330,1.6249),(340,1.6228),(350,1.6100),(360,1.5992),
    (370,1.6247),(380,1.6347),(390,1.5984),(400,1.6094),(410,1.6206),(420,1.6097),
    (430,1.6012),(440,1.5880),(450,1.5947),(460,1.5972),(470,1.5897),(480,1.6310),
    (490,1.6083),(500,1.6126),(510,1.6219),(520,1.5826),(530,1.6107),(540,1.5954),
    (550,1.5948),(560,1.5958),(570,1.6045),(580,1.6123),(590,1.6121),(600,1.5940),
    (610,1.5978),(620,1.6124),(630,1.6095),(640,1.6030),(650,1.6246),(660,1.5812),
    (670,1.5925),(680,1.6077),(690,1.6076),(700,1.6219),(710,1.6025),(720,1.5779),
    (730,1.5747),(740,1.5972),(750,1.5932),(760,1.5816),(770,1.5995),(780,1.5735),
    (790,1.5870),(800,1.6025),(810,1.5755),(820,1.5880),(830,1.5640),(840,1.6005),
    (850,1.5819),(860,1.5919),(870,1.5844),(880,1.5888),(890,1.5745),(900,1.6064),
    (910,1.6014),(920,1.5921),(930,1.5796),(940,1.5760),(950,1.5725),(960,1.6134),
    (970,1.5897),(980,1.5585),(990,1.5900),(1000,1.5918),(1010,1.5619),(1020,1.5741),
    (1030,1.5948),(1040,1.5837),(1050,1.5757),(1060,1.5777),(1070,1.5773),(1080,1.5781),
    (1090,1.5647),(1100,1.5766),(1110,1.5641),(1120,1.5592),(1130,1.5876),(1140,1.5867),
    (1150,1.5579),(1160,1.5826),(1170,1.5802),(1180,1.5876),(1190,1.5829),(1200,1.5580),
    (1210,1.5708),(1220,1.5783),(1230,1.5663),(1240,1.5953),(1250,1.5834),(1260,1.5656),
    (1270,1.5858),(1280,1.5802),(1290,1.5915),(1300,1.5918),(1310,1.6004),(1320,1.6138),
    (1330,1.5828),(1340,1.5758),(1350,1.5671),(1360,1.5520),(1370,1.5930),(1380,1.5587),
    (1390,1.5867),(1400,1.5797),(1410,1.5773),(1420,1.5932),(1430,1.5775),(1440,1.5726),
    (1450,1.5742),(1460,1.5774),(1470,1.5887),(1480,1.5714),(1490,1.5878),(1500,1.5796),
    (1510,1.5821),(1520,1.5822),(1530,1.5827),(1540,1.5808),(1550,1.5633),(1560,1.5767),
    (1570,1.5927),(1580,1.5693),(1590,1.5850),(1600,1.5882),(1610,1.5754),(1620,1.5773),
    (1630,1.5962),(1640,1.5828),(1650,1.5848),(1660,1.5862),(1670,1.5818),(1680,1.5898),
    (1690,1.5804),(1700,1.5976),(1710,1.5769),(1720,1.5757),(1730,1.5859),(1740,1.5741),
    (1750,1.5705),(1760,1.5766),(1770,1.5652),(1780,1.5756),(1790,1.5828),
]

TRAIN_HALF2E = [
    (10,2.4892),(20,2.2532),(30,2.0623),(40,1.9225),(50,1.8673),(60,1.8220),
    (70,1.8129),(80,1.7760),(90,1.7664),(100,1.7388),(110,1.7189),(120,1.7278),
    (130,1.7127),(140,1.7423),(150,1.6841),(160,1.7116),(170,1.6708),(180,1.6967),
    (190,1.6806),(200,1.6758),(210,1.7032),(220,1.6834),(230,1.6712),(240,1.6855),
    (250,1.6873),(260,1.6431),(270,1.6610),(280,1.6522),(290,1.6733),(300,1.6748),
    (310,1.6707),(320,1.6629),(330,1.6434),(340,1.6658),(350,1.6610),(360,1.6413),
    (370,1.6581),(380,1.6551),(390,1.6695),(400,1.6246),(410,1.6199),(420,1.6553),
    (430,1.6639),(440,1.6713),(450,1.6459),(460,1.6431),(470,1.6608),(480,1.6747),
    (490,1.6558),(500,1.6713),(510,1.6815),(520,1.6258),(530,1.6367),(540,1.6497),
    (550,1.6439),(560,1.6502),(570,1.6247),(580,1.6226),(590,1.6635),(600,1.6325),
    (610,1.6429),(620,1.6602),(630,1.6490),(640,1.6189),(650,1.6612),(660,1.6156),
    (670,1.6624),(680,1.6375),(690,1.6280),(700,1.6541),(710,1.6128),(720,1.6281),
    (730,1.6394),(740,1.6152),(750,1.6073),(760,1.6218),(770,1.5918),(780,1.6143),
    (790,1.6489),(800,1.6469),(810,1.6225),(820,1.6327),(830,1.6395),(840,1.6475),
    (850,1.6161),(860,1.6383),(870,1.6601),(880,1.6328),(890,1.6431),
    # ── epoch 1 ends near step 898 ──
    (900,1.5963),(910,1.5886),(920,1.5826),(930,1.5955),(940,1.6308),(950,1.6188),
    (960,1.5908),(970,1.5947),(980,1.5855),(990,1.6166),(1000,1.5836),(1010,1.5795),
    (1020,1.5920),(1030,1.6067),(1040,1.5930),(1050,1.6011),(1060,1.5777),(1070,1.5951),
    (1080,1.5644),(1090,1.5782),(1100,1.5833),(1110,1.5903),(1120,1.5896),(1130,1.5656),
    (1140,1.5715),(1150,1.5682),(1160,1.5870),(1170,1.6201),(1180,1.5632),(1190,1.5778),
    (1200,1.5982),(1210,1.6015),(1220,1.6023),(1230,1.5942),(1240,1.6235),(1250,1.6185),
    (1260,1.6053),(1270,1.5838),(1280,1.6156),(1290,1.5891),(1300,1.6006),(1310,1.6028),
    (1320,1.5964),(1330,1.5967),(1340,1.5952),(1350,1.6011),(1360,1.5996),(1370,1.5702),
    (1380,1.5756),(1390,1.5855),(1400,1.5917),(1410,1.5716),(1420,1.5907),(1430,1.6002),
    (1440,1.5890),(1450,1.5866),(1460,1.5900),(1470,1.6036),(1480,1.5978),(1490,1.5954),
    (1500,1.5968),(1510,1.5777),(1520,1.5745),(1530,1.6125),(1540,1.5982),(1550,1.6069),
    (1560,1.5834),(1570,1.6196),(1580,1.5757),(1590,1.5735),(1600,1.5894),(1610,1.6134),
    (1620,1.5875),(1630,1.5597),(1640,1.5610),(1650,1.6006),(1660,1.5758),(1670,1.5890),
    (1680,1.5580),(1690,1.6151),(1700,1.5958),(1710,1.5954),(1720,1.6105),(1730,1.5817),
    (1740,1.5680),(1750,1.6152),(1760,1.5839),(1770,1.5856),(1780,1.5833),(1790,1.5900),
]

EPOCH2_STEP = 898   # step where half_2e completes its first epoch (143 556 / 160)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

BG      = "#F8F9FA"
C_FULL  = "#2563EB"   # blue  — full epoch
C_HALF  = "#DC2626"   # red   — half × 2 epochs
ALPHA_TRAIN = 1
ALPHA_SMOOTH = 0.2

def smooth(series, w=25):
    """Simple centred rolling mean, no scipy needed."""
    vals = np.array([v for _, v in series], dtype=float)
    out  = np.convolve(vals, np.ones(w) / w, mode="same")
    # edges get partial windows — fix by using cumsum
    cumsum = np.cumsum(np.insert(vals, 0, 0))
    for i in range(len(vals)):
        lo = max(0, i - w // 2)
        hi = min(len(vals), i + w // 2 + 1)
        out[i] = vals[lo:hi].mean()
    return np.array([s for s, _ in series], dtype=float), out

# ---------------------------------------------------------------------------
# Figure: two stacked panels
# ---------------------------------------------------------------------------

fig, (ax_eval, ax_train) = plt.subplots(
    2, 1, figsize=(11, 8.5),
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0.42},
)
fig.patch.set_facecolor(BG)

for ax in (ax_eval, ax_train):
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("Optimizer step", fontsize=11)

# ── Panel 1: Eval loss ──────────────────────────────────────────────────────

for label, color, ls in [("epoch1", C_FULL, "-"), ("half_2e", C_HALF, "--")]:
    steps  = [s for s, l, _ in EVAL[label]]
    losses = [l for _, l, _ in EVAL[label]]
    ax_eval.plot(steps, losses, color=color, linewidth=2.2, linestyle=ls,
                 marker="o", markersize=7, zorder=3)

# Annotate final eval loss gap
final_epoch1 = EVAL["epoch1"][-1][1]
final_half2e = EVAL["half_2e"][-1][1]
last_step    = EVAL["epoch1"][-1][0]

ax_eval.annotate(
    "",
    xy=(last_step + 30, final_epoch1),
    xytext=(last_step + 30, final_half2e),
    arrowprops=dict(arrowstyle="<->", color="#888888", lw=1.5),
)
ax_eval.text(
    last_step + 60, (final_epoch1 + final_half2e) / 2,
    f"Δ {final_half2e - final_epoch1:.4f}",
    va="center", fontsize=9.5, color="#555555",
)

# Final loss labels on right
for loss, color in [(final_epoch1, C_FULL), (final_half2e, C_HALF)]:
    ax_eval.text(last_step + 180, loss, f"{loss:.4f}",
                 va="center", fontsize=9, color=color, fontweight="bold")

ax_eval.set_ylabel("Validation loss", fontsize=11)
ax_eval.set_title(
    "Validation loss — same compute budget, different data strategy",
    fontsize=12, fontweight="bold",
)
ax_eval.set_xlim(0, last_step + 350)
ax_eval.tick_params(labelsize=10)

# ── Panel 2: Train loss (raw + smoothed) ────────────────────────────────────

for series, color, ls in [
    (TRAIN_EPOCH1,  C_FULL,  "-"),
    (TRAIN_HALF2E,  C_HALF, "--"),
]:
    steps_raw  = np.array([s for s, _ in series])
    losses_raw = np.array([v for _, v in series])
    steps_sm, losses_sm = smooth(series, w=30)
    ax_train.plot(steps_raw,  losses_raw, color=color, alpha=ALPHA_TRAIN,  linewidth=1.0, linestyle=ls)
    ax_train.plot(steps_sm,   losses_sm,  color=color, alpha=ALPHA_SMOOTH, linewidth=2.0, linestyle=ls)

# Mark where epoch 2 starts for the half_2e run
ax_train.axvline(EPOCH2_STEP, color=C_HALF, linewidth=1.2, linestyle=":", alpha=0.7)
ax_train.text(
    EPOCH2_STEP + 18, ax_train.get_ylim()[1] if ax_train.get_ylim()[1] < 2.5 else 2.48,
    "epoch 2\nstarts", ha="left", va="top",
    fontsize=8.5, color=C_HALF, alpha=0.8,
    transform=ax_train.transData,
)

# Shade the second epoch region
ax_train.axvspan(EPOCH2_STEP, max(s for s, _ in TRAIN_HALF2E),
                 alpha=0.035, color=C_HALF, zorder=0)

ax_train.set_ylabel("Training loss (smoothed)", fontsize=11)
ax_train.set_title(
    "Training loss — repetition drives train loss down but val loss stays flat",
    fontsize=12, fontweight="bold",
)
ax_train.set_xlim(0, last_step + 350)
ax_train.tick_params(labelsize=10)

# ── Shared legend ────────────────────────────────────────────────────────────

legend_handles = [
    mlines.Line2D([], [], color=C_FULL, linewidth=2.2, linestyle="-",
                  marker="o", markersize=6,
                  label="1 epoch × 287 113 samples  (full dataset)"),
    mlines.Line2D([], [], color=C_HALF, linewidth=2.2, linestyle="--",
                  marker="o", markersize=6,
                  label="2 epochs × 143 556 samples  (half dataset, repeated)"),
]
fig.legend(
    handles=legend_handles,
    loc="lower center", ncol=2, fontsize=10,
    frameon=False, bbox_to_anchor=(0.5, -0.01),
)

# ── Footer note ──────────────────────────────────────────────────────────────

fig.text(
    0.5, -0.045,
    "Qwen3-0.6B-Base · CNN/DailyMail · effective batch 160 · 4× L40S · bf16 + FA2 + compile",
    ha="center", fontsize=9, color="#888888",
)

out = "plots/epoch_vs_repeat.png"
plt.savefig(out, dpi=180, bbox_inches="tight", facecolor=BG)
print(f"Saved → {out}")
