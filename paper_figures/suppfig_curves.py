"""
suppfig_curves.py
=================
Supporting Information figures — 3 separate panels saved together:
  S1: Full LDA learning curve — BAL-ACC and AUROC vs N  (with error bands)
  S2: K-sweep — LDA performance vs number of G-space PCs used (K_LDA)
  S3: LDA vs RF comparison at all N values (BAL-ACC and AUROC)

Reads:
  ../rf_results.npz       (all learning-curve + K-sweep data)
  ../g_space_cache.npz    (for cum_var annotation)

Output: suppfig_curves.png  (300 DPI)
"""
import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

# ── paper style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

LDA_COL  = "#1565C0"
RF1_COL  = "#2E7D32"   # RF-100 (unregularised)
RF2_COL  = "#7B1FA2"   # RF-reg (regularised)
K_OPT    = 15

# ── load data ──────────────────────────────────────────────────────────────────
lc = np.load("../rf_results.npz", allow_pickle=True)
N_grid       = lc["N_grid"]
lda_ba_m     = lc["LDA_bal_mean"];    lda_ba_s = lc["LDA_bal_std"]
lda_au_m     = lc["LDA_auc_mean"];    lda_au_s = lc["LDA_auc_std"]
rf1_ba_m     = lc["RF_100_bal_mean"]; rf1_ba_s = lc["RF_100_bal_std"]
rf1_au_m     = lc["RF_100_auc_mean"]; rf1_au_s = lc["RF_100_auc_std"]
rf2_ba_m     = lc["RF_reg_bal_mean"]; rf2_ba_s = lc["RF_reg_bal_std"]
rf2_au_m     = lc["RF_reg_auc_mean"]; rf2_au_s = lc["RF_reg_auc_std"]

K_sweep      = lc["K_sweep"]
sw_lda_ba_m  = lc["sweep_LDA_bal"];      sw_lda_ba_s = lc["sweep_LDA_bal_std"]
sw_lda_au_m  = lc["sweep_LDA_auc"];      sw_lda_au_s = lc["sweep_LDA_auc_std"]
sw_rf1_ba_m  = lc["sweep_RF_100_bal"];   sw_rf1_ba_s = lc["sweep_RF_100_bal_std"]
sw_rf1_au_m  = lc["sweep_RF_100_auc"];   sw_rf1_au_s = lc["sweep_RF_100_auc_std"]

cache = np.load("../g_space_cache.npz", allow_pickle=True)
# Recompute patient-space cumvar for annotation
G     = cache["G_pat_full"]
Gc    = G - G.mean(0)
C     = Gc.T @ Gc
evals = np.maximum(np.linalg.eigvalsh(C)[::-1], 0)
cum_p = np.cumsum(evals) / evals.sum() * 100

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14.0, 10.0), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig,
                        hspace=0.52, wspace=0.38)

# ── S1: LDA learning curve — BAL-ACC (span 2 cols) ─────────────────────────────
ax_s1a = fig.add_subplot(gs[0, 0])
ax_s1b = fig.add_subplot(gs[0, 1])

for ax, mean_v, std_v, ylabel, title, col in [
    (ax_s1a, lda_ba_m, lda_ba_s,
     "LOPO Balanced Accuracy",
     f"BAL-ACC vs N  (K = {K_OPT})", LDA_COL),
    (ax_s1b, lda_au_m, lda_au_s,
     "LOPO AUROC",
     f"AUROC vs N  (K = {K_OPT})", LDA_COL),
]:
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")
    ax.fill_between(N_grid, mean_v - std_v, mean_v + std_v,
                    alpha=0.20, color=col)
    ax.plot(N_grid, mean_v, "-o", ms=5, lw=2.0, color=col,
            label="LDA  (30 reps)")
    # annotate each point
    for x_, y_, s_ in zip(N_grid, mean_v, std_v):
        ax.text(x_, y_ + s_ + 0.005, f"{y_:.2f}",
                ha="center", va="bottom", fontsize=6.5, color=col)
    ax.set_xlabel("N patients per class")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(N_grid); ax.set_xticklabels(N_grid, rotation=45)
    ax.set_ylim(0.30, 0.90)
    ax.legend(frameon=False)

# ── S2: K-sweep — LDA vs RF ────────────────────────────────────────────────────
ax_s2a = fig.add_subplot(gs[1, 0])
ax_s2b = fig.add_subplot(gs[1, 1])

for ax, lda_m, lda_s, rf1_m, rf1_s, ylabel, title in [
    (ax_s2a, sw_lda_ba_m, sw_lda_ba_s, sw_rf1_ba_m, sw_rf1_ba_s,
     "LOPO Balanced Accuracy",
     f"BAL-ACC vs K  (N=40/class, 30 reps)"),
    (ax_s2b, sw_lda_au_m, sw_lda_au_s, sw_rf1_au_m, sw_rf1_au_s,
     "LOPO AUROC",
     f"AUROC vs K  (N=40/class, 30 reps)"),
]:
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")

    ax.fill_between(K_sweep, lda_m-lda_s, lda_m+lda_s, alpha=0.20, color=LDA_COL)
    ax.plot(K_sweep, lda_m, "-o", ms=5, lw=2.0, color=LDA_COL, label="LDA")

    ax.fill_between(K_sweep, rf1_m-rf1_s, rf1_m+rf1_s, alpha=0.15, color=RF1_COL)
    ax.plot(K_sweep, rf1_m, "-s", ms=5, lw=2.0, color=RF1_COL, label="RF-100")

    # Variance explained annotations
    K_max = G.shape[1]
    for k in K_sweep:
        if k <= K_max:
            ax.text(k, 0.315, f"{cum_p[k-1]:.0f}%",
                    ha="center", va="bottom", fontsize=6, color="#555",
                    rotation=90)

    # Mark optimal K
    ax.axvline(K_OPT, color="#E65100", ls=":", lw=1.5, alpha=0.85,
               label=f"K={K_OPT} (optimal)")

    ax.set_xlabel("K (number of G-space PCs used)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(K_sweep)
    ax.set_ylim(0.30, 0.88)
    ax.legend(frameon=False, fontsize=7.5)
    ax.text(0.02, 0.03, "% = variance explained",
            transform=ax.transAxes, fontsize=6, color="#555", va="bottom")

# ── S3: LDA vs RF comparison at all N values ───────────────────────────────────
ax_s3a = fig.add_subplot(gs[0, 2])
ax_s3b = fig.add_subplot(gs[1, 2])

for ax, lda_m, lda_s, rf1_m, rf1_s, rf2_m, rf2_s, ylabel, title in [
    (ax_s3a, lda_ba_m, lda_ba_s, rf1_ba_m, rf1_ba_s, rf2_ba_m, rf2_ba_s,
     "LOPO Balanced Accuracy",
     f"LDA vs RF — BAL-ACC  (K = {K_OPT})"),
    (ax_s3b, lda_au_m, lda_au_s, rf1_au_m, rf1_au_s, rf2_au_m, rf2_au_s,
     "LOPO AUROC",
     f"LDA vs RF — AUROC  (K = {K_OPT})"),
]:
    ax.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")

    ax.fill_between(N_grid, lda_m-lda_s, lda_m+lda_s, alpha=0.18, color=LDA_COL)
    ax.plot(N_grid, lda_m, "-o", ms=5, lw=2.2, color=LDA_COL, label="LDA")

    ax.fill_between(N_grid, rf1_m-rf1_s, rf1_m+rf1_s, alpha=0.15, color=RF1_COL)
    ax.plot(N_grid, rf1_m, "-s", ms=5, lw=1.8, color=RF1_COL, label="RF-100")

    ax.fill_between(N_grid, rf2_m-rf2_s, rf2_m+rf2_s, alpha=0.12, color=RF2_COL)
    ax.plot(N_grid, rf2_m, "-^", ms=5, lw=1.8, color=RF2_COL, label="RF-reg")

    ax.set_xlabel("N patients per class")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(N_grid); ax.set_xticklabels(N_grid, rotation=45)
    ax.set_ylim(0.30, 0.90)
    ax.legend(frameon=False, fontsize=7.5)

# ── panel labels ───────────────────────────────────────────────────────────────
panel_rows = [
    (ax_s1a, "S1a"), (ax_s1b, "S1b"),
    (ax_s2a, "S2a"), (ax_s2b, "S2b"),
    (ax_s3a, "S3a"), (ax_s3b, "S3b"),
]
for ax, lbl in panel_rows:
    ax.text(-0.14, 1.05, lbl, transform=ax.transAxes,
            fontsize=10, fontweight="bold", va="bottom", ha="left")

fig.suptitle(
    "Supporting Information — Classification performance details\n"
    "S1: LDA learning curve  |  S2: K-sweep (LDA vs RF, N=40)  |  "
    "S3: LDA vs Random Forest comparison  (30 reps, K=15)",
    fontsize=9, y=1.00)

out_path = "suppfig_curves.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_path}")
