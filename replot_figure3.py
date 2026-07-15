"""
Regenerate pertB_direct_results.png from saved npz — no TF pass needed.
"""
import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

OUT_DIR = "."

ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "top1":   np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
}

print("Loading saved results ...")
d       = np.load(f"{OUT_DIR}/pertB_direct_data.npz", allow_pickle=True)
cc_lda  = d["cc_lda"]
ad_lda  = d["ad_lda"]
cc_fc_r = d["cc_fc_r"]

lda_mats = {pt: d[f"{pt}_lda"] for pt in ALPHA_GRIDS}
fcr_mats = {pt: d[f"{pt}_fcr"] for pt in ALPHA_GRIDS}
n_ad     = lda_mats["full_w"].shape[1]

print(f"  CC={cc_lda.mean():.3f}±{cc_lda.std():.3f}  "
      f"AD={ad_lda.mean():.3f}±{ad_lda.std():.3f}  n_AD={n_ad}")

PERT_LABELS = {
    "full_w": "Full-W  (all 121 sites)\nW_int = (1-α)·W_p + α·W_CC",
    "top5":   "Top-5 sites\n(largest ‖W_CC − W_p‖ cols)",
    "top1":   "Single site\n(largest ‖W_CC − W_p‖ col)",
}
COL_CC    = "#2196F3"
COL_AD    = "#E91E63"
COL_FC    = "#7B1FA2"
PERT_COLS = {"full_w": "#1B5E20", "top5": "#E65100", "top1": "#4A148C"}

y_all  = np.concatenate([m.flatten() for m in lda_mats.values()])
y_min  = float(np.nanmin(y_all)) - 0.3
y_max  = float(np.nanmax(y_all)) + 0.3
mid    = 0.5 * (cc_lda.mean() + ad_lda.mean())
gap    = ad_lda.mean() - cc_lda.mean()
norm_lda = lambda z: (ad_lda.mean() - z) / gap

fig = plt.figure(figsize=(20, 17), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                         height_ratios=[1.0, 0.85, 0.95],
                         hspace=0.45, wspace=0.30)

for ci, pert_type in enumerate(["full_w", "top5", "top1"]):
    alphas  = ALPHA_GRIDS[pert_type]
    lda_mat = lda_mats[pert_type]
    fcr_mat = fcr_mats[pert_type]
    col     = PERT_COLS[pert_type]

    # row 0 — LDA
    ax = fig.add_subplot(gs[0, ci])
    ax.axhspan(cc_lda.mean()-cc_lda.std(), cc_lda.mean()+cc_lda.std(),
               alpha=0.20, color=COL_CC, label="CC ±1σ")
    ax.axhline(cc_lda.mean(), color=COL_CC, lw=2, ls="--")
    ax.axhspan(ad_lda.mean()-ad_lda.std(), ad_lda.mean()+ad_lda.std(),
               alpha=0.12, color=COL_AD)
    ax.axhline(ad_lda.mean(), color=COL_AD, lw=1.5, ls=":", alpha=0.6)
    ax.axhline(mid, color="gray", lw=0.8, ls="-.", alpha=0.5)
    for pi in range(n_ad):
        ax.plot(alphas, lda_mat[:, pi], "-", lw=0.8, color=col, alpha=0.22)
    mt = lda_mat.mean(1); st = lda_mat.std(1)
    ax.fill_between(alphas, mt-st, mt+st, alpha=0.22, color=col)
    ax.plot(alphas, mt, "-o", ms=6, lw=2.5, color=col, zorder=5, label="AD mean ±1σ")
    ax.set_ylim(y_min, y_max)
    ax.set_title(PERT_LABELS[pert_type], fontsize=10)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("LDA score  (cond. B, K=25)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # row 1 — FC-r
    ax = fig.add_subplot(gs[1, ci])
    ax.axhspan(cc_fc_r.mean()-cc_fc_r.std(), cc_fc_r.mean()+cc_fc_r.std(),
               alpha=0.20, color=COL_CC, label="CC ±1σ")
    ax.axhline(cc_fc_r.mean(), color=COL_CC, lw=2, ls="--")
    for pi in range(n_ad):
        ax.plot(alphas, fcr_mat[:, pi], "-", lw=0.8, color=COL_FC, alpha=0.22)
    mf = fcr_mat.mean(1); sf = fcr_mat.std(1)
    ax.fill_between(alphas, mf-sf, mf+sf, alpha=0.22, color=COL_FC)
    ax.plot(alphas, mf, "-o", ms=6, lw=2.5, color=COL_FC, zorder=5, label="AD mean ±1σ")
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("FC-r (W·ᵀX vs CC template)", fontsize=9)
    ax.set_title("FC similarity to CC mean", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

# row 2 left (span 2): normalised comparison
ax_cmp = fig.add_subplot(gs[2, 0:2])
ax_cmp.axhspan(1 - cc_lda.std()/gap, 1 + cc_lda.std()/gap,
               alpha=0.20, color=COL_CC)
ax_cmp.axhline(1.0, color=COL_CC, lw=2, ls="--", label="CC mean")
ax_cmp.axhline(0.0, color=COL_AD, lw=1.5, ls=":", alpha=0.6, label="AD baseline")
ax_cmp.axhline(0.5, color="gray", lw=0.8, ls="-.", alpha=0.5, label="midpoint")
for pert_type in ["full_w", "top5", "top1"]:
    alphas = ALPHA_GRIDS[pert_type]
    mt = lda_mats[pert_type].mean(1)
    st = lda_mats[pert_type].std(1)
    col = PERT_COLS[pert_type]
    nm  = norm_lda(mt)
    nlo = norm_lda(mt + st)
    nhi = norm_lda(mt - st)
    ax_cmp.fill_between(alphas, nlo, nhi, alpha=0.18, color=col)
    ax_cmp.plot(alphas, nm, "-o", ms=6, lw=2.5, color=col,
                label=PERT_LABELS[pert_type].split("\n")[0])
ax_cmp.set_xlabel("alpha", fontsize=10)
ax_cmp.set_ylabel("Fraction of AD→CC gap closed\n(0 = AD baseline,  1 = CC mean)", fontsize=10)
ax_cmp.set_title("Comparison: perturbation efficiency across all types\n"
                 "(shared normalised y-axis)", fontsize=10)
ax_cmp.legend(fontsize=8, frameon=False, loc="upper left")
for sp in ["top","right"]: ax_cmp.spines[sp].set_visible(False)

# row 2 right: FC-r all types
ax_fc2 = fig.add_subplot(gs[2, 2])
ax_fc2.axhspan(cc_fc_r.mean()-cc_fc_r.std(), cc_fc_r.mean()+cc_fc_r.std(),
               alpha=0.20, color=COL_CC, label="CC ±1σ")
ax_fc2.axhline(cc_fc_r.mean(), color=COL_CC, lw=2, ls="--")
for pert_type in ["full_w", "top5", "top1"]:
    alphas = ALPHA_GRIDS[pert_type]
    mf     = fcr_mats[pert_type].mean(1)
    ax_fc2.plot(alphas, mf, "-o", ms=5, lw=2.0,
                color=PERT_COLS[pert_type],
                label=PERT_LABELS[pert_type].split("\n")[0])
ax_fc2.set_xlabel("alpha", fontsize=10)
ax_fc2.set_ylabel("FC-r vs CC template", fontsize=10)
ax_fc2.set_title("FC similarity comparison\n(mean over AD patients)", fontsize=10)
ax_fc2.legend(fontsize=7, frameon=False)
for sp in ["top","right"]: ax_fc2.spines[sp].set_visible(False)

fig.suptitle(
    "Direct perturbation — Condition B  (σ=0.05, K_LDA=25, sr=0.95)\n"
    "Rows 1–2: per-type individual patient trajectories  |  "
    "Row 3: all-type comparison on normalised scale",
    fontsize=11, fontweight="bold", y=1.002)

fig.savefig(f"{OUT_DIR}/pertB_direct_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_direct_results.png")
print("Done.")
