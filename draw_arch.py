"""
ResNetUDA figures
  arch.pdf/png      — architecture (v1 refined)
  training.pdf/png  — training stages with losses
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

# ── Palette ────────────────────────────────────────────────────────────────────
BB  = ("#D4E9F8", "#1E5F99", "#0F3560")   # backbone
FT  = ("#FDEBD0", "#C05810", "#7A3400")   # feature
PRO = ("#D4EDE2", "#197040", "#0B4020")   # projector
CLS = ("#F5D8CE", "#A03018", "#6A1A08")   # classifier
SE  = ("#E4DCF5", "#5040A0", "#2A2070")   # SE block
NEU = ("#EBEBEB", "#888888", "#444444")   # neutral

ARR   = "#555555"
LGRAY = "#CCCCCC"
DKGRAY= "#333333"

# ── Primitives ─────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, pal, title, sub=None, tfs=11, sfs=8.5, r=0.18, lw=1.6):
    fc, ec, tc = pal
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3))
    oy = 0.12 if sub else 0
    ax.text(x, y + oy, title, ha="center", va="center",
            fontsize=tfs, fontweight="bold", color=tc, zorder=4)
    if sub:
        ax.text(x, y - 0.21, sub, ha="center", va="center",
                fontsize=sfs, color=tc, alpha=0.72, zorder=4)

def av(ax, x, y0, y1, lw=1.4, c=ARR, ms=11):
    ax.annotate("", xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=ms), zorder=5)

def ah(ax, x0, x1, y, lw=1.4, c=ARR, ms=11):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=c, lw=lw,
                                mutation_scale=ms), zorder=5)

def seg(ax, x0, y0, x1, y1, lw=1.4, c=ARR, ls="-"):
    ax.plot([x0, x1], [y0, y1], color=c, lw=lw, ls=ls, zorder=4)

def badge(ax, x, y, txt, ec, fc="white", size=8.5):
    ax.text(x, y, txt, ha="center", va="center", fontsize=size,
            fontweight="bold", color=ec,
            bbox=dict(boxstyle="round,pad=0.28", facecolor=fc,
                      edgecolor=ec, linewidth=1.2, alpha=0.96), zorder=6)

def panel(ax, x, y, w, h, fc="#F6F9FC", ec="#C0D0E0", r=0.30, lw=1.0, a=0.65):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=a, zorder=1))


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Architecture  (refined v2: aligned panels, larger fonts)
# ══════════════════════════════════════════════════════════════════════════════
FW, FH = 11, 14
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")

CX = 3.80   # main-flow center x

# ── Title ──────────────────────────────────────────────────────────────────────
ax.text(FW/2, FH - 0.40, "ResNetUDA",
        ha="center", fontsize=24, fontweight="bold", color="#0F1A30")
ax.text(FW/2, FH - 0.90,
        "Unsupervised Domain Adaptation  ·  200-class Fine-grained Recognition",
        ha="center", fontsize=11, color="#888888")

# ── Y layout ───────────────────────────────────────────────────────────────────
IN_Y   = FH - 1.80;  IW = 2.50;  IH = 0.96
BB_Y   = FH - 4.00;  BW_BB = 4.60; BH_BB = 1.60
FT_Y   = FH - 6.60;  BW_FT = 2.20; BH_FT = 1.00
HD_Y   = FH - 9.00;  BW_HD = 2.70; BH_HD = 1.04
OUT_Y  = FH - 11.20; BW_OUT = 2.20; BH_OUT = 0.92
PROJ_X = CX - 1.75
CLS_X  = CX + 1.75

# shared panel geometry — both left and right panels start/end at the same Y
PANEL_Y = 1.10
PANEL_H = 11.60   # top at 12.70

# outer panel (left, main flow)
panel(ax, 0.50, PANEL_Y, 7.30, PANEL_H,
      fc="#F7FAFD", ec="#C0D2E6", r=0.38, lw=1.1)

# ── Input ──────────────────────────────────────────────────────────────────────
box(ax, CX, IN_Y, IW, IH, NEU,
    "Input  x", "224 × 224 × 3", tfs=13, sfs=10.5)
av(ax, CX, IN_Y - IH/2, BB_Y + BH_BB/2 + 0.10)

# ── Backbone ───────────────────────────────────────────────────────────────────
fc, ec, tc = BB
ax.add_patch(FancyBboxPatch(
    (CX - BW_BB/2, BB_Y - BH_BB/2), BW_BB, BH_BB,
    boxstyle="round,pad=0,rounding_size=0.28",
    facecolor=fc, edgecolor=ec, linewidth=2.2, zorder=3))

for frac in [0.30, 0.54, 0.76]:
    yy = BB_Y - BH_BB/2 + BH_BB * frac
    seg(ax, CX - BW_BB/2 + 0.34, yy, CX + BW_BB/2 - 0.34, yy,
        lw=0.7, c="#A4C8E8", ls="--")

ax.text(CX, BB_Y + 0.40, "Backbone  F",
        ha="center", fontsize=16, fontweight="bold", color=tc, zorder=4)
ax.text(CX, BB_Y - 0.04,
        "ResBlocks  ×  4 stages   ·   64 → 128 → 256 → 512 ch",
        ha="center", fontsize=11, color=tc, alpha=0.82, zorder=4)
ax.text(CX, BB_Y - 0.48,
        "GroupNorm  ·  SE attention  ·  skip connections",
        ha="center", fontsize=10, color=tc, alpha=0.70, zorder=4)

# SHOT dashed border
ax.add_patch(FancyBboxPatch(
    (CX - BW_BB/2 - 0.18, BB_Y - BH_BB/2 - 0.16),
    BW_BB + 0.36, BH_BB + 0.32,
    boxstyle="round,pad=0,rounding_size=0.32",
    fill=False, edgecolor="#CC6820", linewidth=1.5,
    linestyle=(0, (5, 3)), zorder=4))
ax.text(CX, BB_Y - BH_BB/2 - 0.36,
        "fine-tuned during SHOT  ·  Classifier C frozen",
        ha="center", fontsize=9.5, color="#B05A18", style="italic")

av(ax, CX, BB_Y - BH_BB/2 - 0.36 - 0.16, FT_Y + BH_FT/2 + 0.10)

# ── Feature z ──────────────────────────────────────────────────────────────────
box(ax, CX, FT_Y, BW_FT, BH_FT, FT,
    "z", "512-dim feature", tfs=17, sfs=11, r=0.22)

# ── Branch fork ────────────────────────────────────────────────────────────────
FORK_Y = FT_Y - BH_FT/2 - 0.55
seg(ax, CX, FT_Y - BH_FT/2, CX, FORK_Y)
seg(ax, PROJ_X, FORK_Y, CLS_X, FORK_Y)
av(ax, PROJ_X, FORK_Y, HD_Y + BH_HD/2 + 0.10, c=PRO[1])
av(ax, CLS_X,  FORK_Y, HD_Y + BH_HD/2 + 0.10, c=CLS[1])

# ── Contrastive branch ─────────────────────────────────────────────────────────
_bp_bot = OUT_Y - BH_OUT/2 - 0.56
_bp_top = HD_Y + BH_HD/2 + 0.50
panel(ax, PROJ_X - BW_HD/2 - 0.22, _bp_bot,
      BW_HD + 0.44, _bp_top - _bp_bot,
      fc="#EDF7F2", ec="#8BBCA8", r=0.22, lw=0.9)
ax.text(PROJ_X, HD_Y + BH_HD/2 + 0.34, "Contrastive Branch",
        ha="center", fontsize=11, fontweight="bold", color=PRO[1])
box(ax, PROJ_X, HD_Y, BW_HD, BH_HD, PRO,
    "Projector  P", "MLP  512 → 512 → 128", tfs=12, sfs=10)
av(ax, PROJ_X, HD_Y - BH_HD/2 - 0.10, OUT_Y + BH_OUT/2 + 0.10, c=PRO[1])
box(ax, PROJ_X, OUT_Y, BW_OUT, BH_OUT, PRO,
    "z̃  ∈  ℝ¹²⁸", "ℓ₂-normalized", tfs=13, sfs=10)
badge(ax, PROJ_X, OUT_Y - BH_OUT/2 - 0.38, "InfoNCE loss", PRO[1])

# ── Classification branch ──────────────────────────────────────────────────────
panel(ax, CLS_X - BW_HD/2 - 0.22, _bp_bot,
      BW_HD + 0.44, _bp_top - _bp_bot,
      fc="#FBF0EC", ec="#CCA090", r=0.22, lw=0.9)
ax.text(CLS_X, HD_Y + BH_HD/2 + 0.34, "Classification Branch",
        ha="center", fontsize=11, fontweight="bold", color=CLS[1])
box(ax, CLS_X, HD_Y, BW_HD, BH_HD, CLS,
    "Classifier  C", "Linear  512 → 200", tfs=12, sfs=10)
av(ax, CLS_X, HD_Y - BH_HD/2 - 0.10, OUT_Y + BH_OUT/2 + 0.10, c=CLS[1])
box(ax, CLS_X, OUT_Y, BW_OUT, BH_OUT, CLS,
    "ŷ  ∈  ℝ²⁰⁰", "200 classes", tfs=13, sfs=10)
badge(ax, CLS_X, OUT_Y - BH_OUT/2 - 0.38, "CE / IM loss", CLS[1])

# ── ResBlock panel (right) — same PANEL_Y / PANEL_H as left ───────────────────
RPX = 8.95; RPW = 2.50; RBH = 0.82
RPT = FH - 1.70

panel(ax, 7.20, PANEL_Y, 3.50, PANEL_H,
      fc="#F5F5F8", ec="#B8B8C8", r=0.30, lw=1.0, a=0.70)

ax.text(RPX, RPT, "ResBlock",
        ha="center", fontsize=15, fontweight="bold", color="#1A1A2E")
ax.text(RPX, RPT - 0.42, "× 2 per stage  ·  optional stride",
        ha="center", fontsize=10, color="#888888")

rb = [
    ("x",           None,                FT,  RPT - 0.90),
    ("Conv 3×3",    "GroupNorm · ReLU",  BB,  RPT - 2.00),
    ("Conv 3×3",    "GroupNorm",         BB,  RPT - 3.10),
    ("SE Block",    "channel attention", SE,  RPT - 4.20),
    ("Add + ReLU",  "⊕  residual",      NEU, RPT - 5.30),
    ("x′",          None,                FT,  RPT - 6.20),
]
rb_ys = [r[3] for r in rb]
for title, sub, pal, y in rb:
    box(ax, RPX, y, RPW, RBH, pal, title, sub, tfs=12, sfs=9.5, r=0.14)
for i in range(len(rb_ys) - 1):
    av(ax, RPX, rb_ys[i] - RBH/2, rb_ys[i+1] + RBH/2)

skx = RPX + RPW/2 + 0.38
st, sb = rb_ys[0] - RBH/2, rb_ys[-2] + RBH/2
seg(ax, RPX + RPW/2, st, skx, st, c=LGRAY)
seg(ax, skx, st, skx, sb, c=LGRAY)
ax.annotate("", xy=(RPX + RPW/2 + 0.07, sb), xytext=(skx, sb),
            arrowprops=dict(arrowstyle="-|>", color=LGRAY,
                            lw=1.1, mutation_scale=9), zorder=4)
ax.text(skx + 0.10, (st + sb)/2, "skip",
        ha="left", va="center", fontsize=9.5,
        color=LGRAY, style="italic")

ax.text(FW/2, 0.26,
        "CUB-200-2011  ↔  CUB-200-Paintings  ·  200-class UDA",
        ha="center", fontsize=9, color="#BBBBBB")

fig.savefig("arch.pdf", bbox_inches="tight", dpi=200)
fig.savefig("arch.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved: arch.pdf / arch.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Training stages with losses  (v2: wider, larger fonts, no overflow)
# ══════════════════════════════════════════════════════════════════════════════
TW, TH = 20, 12
fig, ax = plt.subplots(figsize=(TW, TH))
ax.set_xlim(0, TW); ax.set_ylim(0, TH); ax.axis("off")

ax.text(TW/2, TH - 0.44, "UDA Training Pipeline",
        ha="center", fontsize=22, fontweight="bold", color="#0F1A30")
ax.text(TW/2, TH - 0.94,
        "Each stage uses different data and loss functions",
        ha="center", fontsize=11.5, color="#888888")

# ── Stage layout ──────────────────────────────────────────────────────────────
N   = 4
SW  = 4.40     # stage panel width
SH  = 9.80     # stage panel height
GAP = 0.50
total_w = N * SW + (N-1) * GAP   # 19.10
SX0 = (TW - total_w) / 2         # 0.45
SY0 = 0.55
SCY = SY0 + SH / 2               # center y for inter-stage arrows

stage_titles = [
    "① Source Pre-train",
    "② Joint Warmup",
    "③ Pseudo-label Rounds",
    "④ SHOT Adaptation",
]

# mini model component sizes
mW = 1.30; mH = 0.62; mR = 0.14

def mini_box(ax, x, y, w, h, pal, label, frozen=False, tfs=12):
    fc, ec, tc = pal
    if frozen:
        fc = "#E8E8E8"; ec = "#AAAAAA"; tc = "#AAAAAA"
    ax.add_patch(FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle=f"round,pad=0,rounding_size={mR}",
        facecolor=fc, edgecolor=ec,
        linewidth=1.6 if not frozen else 0.9,
        linestyle="-" if not frozen else "--", zorder=4))
    ax.text(x, y, label, ha="center", va="center",
            fontsize=tfs, fontweight="bold" if not frozen else "normal",
            color=tc, zorder=5)

def data_tag(ax, x, y, label, color, sub=None):
    bw = 1.52
    bh = 0.88 if sub else 0.56
    ax.add_patch(FancyBboxPatch(
        (x - bw/2, y - bh/2), bw, bh,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=color + "28", edgecolor=color,
        linewidth=1.4, zorder=4))
    ax.text(x, y + (0.18 if sub else 0), label,
            ha="center", va="center", fontsize=12,
            fontweight="bold", color=color, zorder=5)
    if sub:
        ax.text(x, y - 0.18, sub,
                ha="center", va="center", fontsize=9.5,
                color=color, alpha=0.88, zorder=5)

def loss_box(ax, x, y, lines, ec):
    lh = 0.40 * len(lines) + 0.32
    ax.add_patch(FancyBboxPatch(
        (x - SW/2 + 0.22, y - lh/2), SW - 0.44, lh,
        boxstyle="round,pad=0,rounding_size=0.16",
        facecolor=ec + "15", edgecolor=ec,
        linewidth=1.6, zorder=4))
    for i, (txt, bold) in enumerate(lines):
        yy = y + (len(lines) - 1) * 0.20 - i * 0.40
        ax.text(x, yy, txt, ha="center", va="center",
                fontsize=11 if bold else 10,
                fontweight="bold" if bold else "normal",
                color="#111111" if bold else "#444444", zorder=5)

# ── Stage content ──────────────────────────────────────────────────────────────
stages = [
    # (panel_pal, data_items, frozen_C, frozen_P, use_P, loss_lines, note)
    (
        NEU,
        [("S", "#2260A0", "Labeled\nPaintings")],
        False, True, False,
        [("ℒ = ℒ_CE", True),
         ("CrossEntropy(ŷ, y)", False),
         ("y : ground-truth label", False)],
        "Source model training\n~68% src accuracy"
    ),
    (
        PRO,
        [("S", "#2260A0", "Labeled\nPaintings"),
         ("T", "#197040", "Unlabeled\nCUB Photos")],
        False, False, True,
        [("ℒ = ℒ_CE  +  λ · ℒ_NCE", True),
         ("ℒ_CE : CrossEntropy on source", False),
         ("ℒ_NCE : noise-based contrastive", False),
         ("λ = 0.05", False)],
        "Feature alignment\nlight augmentation"
    ),
    (
        FT,
        [("S", "#2260A0", "Labeled"),
         ("T̃", "#C05810", "Pseudo-labeled\n(conf ≥ 0.7)")],
        False, True, False,
        [("ℒ = ℒ_CE(S)  +  ℒ_CE(T̃)", True),
         ("ℒ_CE(S) : CrossEntropy on source", False),
         ("ℒ_CE(T̃) : CE on pseudo-labeled", False),
         ("threshold = 0.7,  7 × 10 ep", False)],
        "High-confidence selection\nbalanced supervision"
    ),
    (
        CLS,
        [("T", "#197040", "All Unlabeled\nCUB Photos")],
        True, True, False,
        [("ℒ = ℒ_IM  +  ℒ_PL", True),
         ("ℒ_IM = ℒ_ent − ℒ_div", False),
         ("ℒ_PL : prototype pseudo-labels", False),
         ("C frozen  ·  5 × 20 ep", False)],
        "Target-only adaptation\nno source labels needed"
    ),
]

for i, (stage_pal, data_items, freeze_C, freeze_P, use_P, loss_lines, note_txt) in enumerate(stages):
    sx  = SX0 + i * (SW + GAP)
    cx  = sx + SW / 2
    ec_col = stage_pal[1]

    # panel background
    ax.add_patch(FancyBboxPatch(
        (sx, SY0), SW, SH,
        boxstyle="round,pad=0,rounding_size=0.34",
        facecolor=stage_pal[0], edgecolor=ec_col,
        linewidth=2.0, alpha=0.28, zorder=1))
    ax.add_patch(FancyBboxPatch(
        (sx, SY0), SW, SH,
        boxstyle="round,pad=0,rounding_size=0.34",
        fill=False, edgecolor=ec_col,
        linewidth=2.0, zorder=2))

    # stage title + divider
    ax.text(cx, SY0 + SH - 0.44, stage_titles[i],
            ha="center", fontsize=13, fontweight="bold",
            color=ec_col, zorder=6)
    seg(ax, sx + 0.28, SY0 + SH - 0.72, sx + SW - 0.28, SY0 + SH - 0.72,
        lw=0.9, c=ec_col + "88")

    # ── section: Data ────────────────────────────────────────────────────────
    sec_y  = SY0 + SH - 1.02
    data_y = sec_y - 0.80
    ax.text(cx, sec_y - 0.12, "Data",
            ha="center", fontsize=10, color="#555555",
            fontweight="bold", style="italic")

    nd      = len(data_items)
    dx_step = 1.65 if nd > 1 else 0
    x_start = cx - (nd - 1) / 2 * dx_step
    for j, (tag, color, sub) in enumerate(data_items):
        data_tag(ax, x_start + j * dx_step, data_y, tag, color, sub)

    # ── section: Model ───────────────────────────────────────────────────────
    mod_top = data_y - 0.96
    ax.text(cx, mod_top, "Model",
            ha="center", fontsize=10, color="#555555",
            fontweight="bold", style="italic")

    F_y  = mod_top - 0.76
    FC_y = mod_top - 1.82
    out_y = FC_y - 0.82

    mini_box(ax, cx, F_y, mW, mH, BB, "F")

    if use_P:
        mini_box(ax, cx - 0.84, FC_y, mW, mH, PRO, "P")
        mini_box(ax, cx + 0.84, FC_y, mW, mH, CLS, "C", frozen=freeze_C)
        av(ax, cx - 0.44, F_y - mH/2, FC_y + mH/2, c=PRO[1], ms=10)
        av(ax, cx + 0.44, F_y - mH/2, FC_y + mH/2,
           c=CLS[1] if not freeze_C else LGRAY, ms=10)
        ow, oh = mW * 0.84, mH * 0.82
        mini_box(ax, cx - 0.84, out_y, ow, oh, PRO, "z̃", tfs=11)
        mini_box(ax, cx + 0.84, out_y, ow, oh,
                 CLS if not freeze_C else NEU, "ŷ", tfs=11, frozen=freeze_C)
        av(ax, cx - 0.84, FC_y - mH/2, out_y + oh/2, c=PRO[1], ms=8)
        av(ax, cx + 0.84, FC_y - mH/2, out_y + oh/2,
           c=CLS[1] if not freeze_C else LGRAY, ms=8)
    else:
        mini_box(ax, cx, FC_y, mW, mH, CLS, "C", frozen=freeze_C)
        av(ax, cx, F_y - mH/2, FC_y + mH/2,
           c=CLS[1] if not freeze_C else LGRAY, ms=10)
        ow, oh = mW * 0.84, mH * 0.82
        mini_box(ax, cx, out_y, ow, oh,
                 CLS if not freeze_C else NEU, "ŷ", tfs=11, frozen=freeze_C)
        av(ax, cx, FC_y - mH/2, out_y + oh/2,
           c=CLS[1] if not freeze_C else LGRAY, ms=8)

    # frozen legend
    if freeze_C or freeze_P:
        ax.text(cx, out_y - 0.52, "── frozen",
                ha="center", fontsize=9.5, color="#AAAAAA", style="italic")

    # ── section: Objective ───────────────────────────────────────────────────
    loss_top = out_y - (0.90 if freeze_C or freeze_P else 0.72)
    ax.text(cx, loss_top, "Objective",
            ha="center", fontsize=10, color="#555555",
            fontweight="bold", style="italic")

    lh = 0.40 * len(loss_lines) + 0.32
    loss_box(ax, cx, loss_top - 0.40 - lh / 2, loss_lines, ec_col)

    # note at bottom
    ax.text(cx, SY0 + 0.38, note_txt,
            ha="center", va="center", fontsize=9.5,
            color=ec_col, alpha=0.88, style="italic")

# ── arrows between stages ──────────────────────────────────────────────────────
for i in range(N - 1):
    ax_x = SX0 + (i + 1) * SW + i * GAP + GAP / 2
    ax.annotate("", xy=(ax_x + GAP/2 - 0.05, SCY),
                xytext=(ax_x - GAP/2 + 0.05, SCY),
                arrowprops=dict(arrowstyle="-|>", color="#888888",
                                lw=2.2, mutation_scale=16), zorder=6)

ax.text(TW/2, 0.24,
        "CUB-200-2011  ↔  CUB-200-Paintings  ·  200-class UDA",
        ha="center", fontsize=9.5, color="#BBBBBB")

fig.savefig("training.pdf", bbox_inches="tight", dpi=200)
fig.savefig("training.png", bbox_inches="tight", dpi=200)
plt.close(fig)
print("Saved: training.pdf / training.png")
