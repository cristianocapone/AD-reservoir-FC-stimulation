"""
replot_figure3.py
=================
Re-render figure3_classification from the saved sweep (figure3_classification_data.npz)
and overlay the independent tangent-space-SVM FC benchmark (cc_vs_ad_cv.npz)
as a reference band.  No sweep re-run.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from sklearn.metrics import roc_curve, roc_auc_score

d = np.load("figure3_classification_data.npz", allow_pickle=True)
SIGMA_GRID = d["sigma_grid"]; K_GRID = d["k_grid"]
best_sigma = float(d["best_sigma"]); best_K = int(d["best_K"])
n_cc = int(d["n_cc"]); n_ad = int(d["n_ad"])
g_si = int(np.argmin(np.abs(SIGMA_GRID - best_sigma)))
g_ki = int(np.argmin(np.abs(K_GRID - best_K)))

COMBOS = [("G", "lda"), ("G", "rf"), ("FC", "lda"), ("FC", "rf")]
R = {c: {m: d[f"{c[0]}_{c[1]}_{m}"] for m in ["bal_m","bal_s","auc_m","auc_s"]}
     for c in COMBOS}

# ── tangent-space SVM benchmark: nested 5x5 patient-level CV (train-only ref) ──
cv = np.load("../tangent_fc_cv.npz", allow_pickle=True)
oof_y = cv["oof_y"]; oof_s = cv["oof_scores"]
b_auc_m = float(cv["fold_aucs"].mean()); b_auc_s = float(cv["fold_aucs"].std())
b_bal_m = float(cv["fold_bacs"].mean()); b_bal_s = float(cv["fold_bacs"].std())
bench_n_cc = int(cv["n_cc"]); bench_n_ad = int(cv["n_ad"])
print(f"benchmark (nested CV): AUROC={b_auc_m:.3f}+/-{b_auc_s:.3f}  "
      f"BAL={b_bal_m:.3f}+/-{b_bal_s:.3f}  (n={bench_n_cc} CC + {bench_n_ad} AD sessions)")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

G_COL = "#7B1FA2"; FC_COL = "#00838F"; BM_COL = "#455A64"
STYLE = {
    ("G","lda"):  dict(color=G_COL,  ls="-",  marker="o", label="G-space · LDA"),
    ("G","rf"):   dict(color=G_COL,  ls="--", marker="^", label="G-space · RF"),
    ("FC","lda"): dict(color=FC_COL, ls="-",  marker="s", label="FC-lag · LDA"),
    ("FC","rf"):  dict(color=FC_COL, ls="--", marker="D", label="FC-lag · RF"),
}
YL = (0.42, 0.82)

# benchmark ROC (nested-CV out-of-fold) and 2-D CC/AD embedding (from same npz)
fpr, tpr, _ = roc_curve(oof_y, oof_s); oof_auc = roc_auc_score(oof_y, oof_s)
pcs = cv["pcs"]; yemb = cv["y"]
CC_COL = "#1565C0"; AD_COL = "#C62828"

fig = plt.figure(figsize=(16.5, 9.0), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.30,
                        top=0.89, bottom=0.08, left=0.06, right=0.98)

def _tag(ax, t):
    ax.text(-0.15, 1.05, t, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")

def _ellipse(ax, P, color):
    mu = P.mean(0); cov = np.cov(P.T)
    vals, vecs = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(*vecs[:, 1][::-1]))
    for nsd in (1, 2):
        w, h = 2 * nsd * np.sqrt(vals)
        ax.add_patch(Ellipse(mu, w, h, angle=ang, fc="none", ec=color,
                             lw=1.4, ls="--", alpha=0.8))
    ax.scatter(*mu, c=color, s=90, marker="X", edgecolors="k",
               linewidths=0.8, zorder=5)

def _curve(ax, x, m, s, st):
    ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.10)
    ax.plot(x, m, st["ls"], marker=st["marker"], ms=4.5, lw=1.8,
            color=st["color"], label=st["label"])

def _bench(ax, mval, sval):
    ax.axhspan(mval - sval, mval + sval, color=BM_COL, alpha=0.12, zorder=0)
    ax.axhline(mval, color=BM_COL, ls=":", lw=1.6,
               label=f"tangent-SVM (FC): {mval:.2f}")

# A: bal-acc vs K
ax = fig.add_subplot(gs[0, 0])
for c in COMBOS: _curve(ax, K_GRID, R[c]["bal_m"][g_si], R[c]["bal_s"][g_si], STYLE[c])
_bench(ax, b_bal_m, b_bal_s)
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("K  (# SVD components)"); ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Accuracy vs. # SVD components  (σ = {best_sigma:g})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7); _tag(ax, "A")

# B: AUROC vs K
ax = fig.add_subplot(gs[0, 1])
for c in COMBOS: _curve(ax, K_GRID, R[c]["auc_m"][g_si], R[c]["auc_s"][g_si], STYLE[c])
_bench(ax, b_auc_m, b_auc_s)
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("K  (# SVD components)"); ax.set_ylabel("AUROC")
ax.set_title(f"AUROC vs. # SVD components  (σ = {best_sigma:g})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7); _tag(ax, "B")

# C: bal-acc vs sigma
ax = fig.add_subplot(gs[1, 0])
for c in COMBOS: _curve(ax, SIGMA_GRID, R[c]["bal_m"][:, g_ki], R[c]["bal_s"][:, g_ki], STYLE[c])
_bench(ax, b_bal_m, b_bal_s)
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6); ax.set_xscale("log")
ax.set_xlabel("W-fit regularisation noise  σ"); ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Accuracy vs. noise  (K = {best_K})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7); _tag(ax, "C")

# D: AUROC vs sigma
ax = fig.add_subplot(gs[1, 1])
for c in COMBOS: _curve(ax, SIGMA_GRID, R[c]["auc_m"][:, g_ki], R[c]["auc_s"][:, g_ki], STYLE[c])
_bench(ax, b_auc_m, b_auc_s)
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6); ax.set_xscale("log")
ax.set_xlabel("W-fit regularisation noise  σ"); ax.set_ylabel("AUROC")
ax.set_title(f"AUROC vs. noise  (K = {best_K})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7); _tag(ax, "D")

# E: ROC curve — nested-CV tangent-space FC benchmark (out-of-fold)
ax = fig.add_subplot(gs[0, 2])
ax.plot(fpr, tpr, color=BM_COL, lw=2.4,
        label=f"tangent-SVM FC (AUC={oof_auc:.2f})")
ax.fill_between(fpr, tpr, alpha=0.12, color=BM_COL)
ax.plot([0, 1], [0, 1], "k:", lw=1, alpha=0.5)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("Tangent-space FC benchmark --- ROC\n(nested 5$\\times$5 patient-level CV)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(frameon=False, fontsize=8, loc="lower right"); _tag(ax, "E")

# F: 2-D embedding of the FC features — CC vs AD clouds
ax = fig.add_subplot(gs[1, 2])
for lab, col, name in [(0, CC_COL, "CC"), (1, AD_COL, "AD")]:
    P = pcs[yemb == lab]
    ax.scatter(P[:, 0], P[:, 1], s=10, c=col, alpha=0.35, edgecolors="none",
               label=f"{name} (n={int((yemb==lab).sum())})")
    _ellipse(ax, P, col)
ax.set_xlabel("FC tangent-space PC 1"); ax.set_ylabel("FC tangent-space PC 2")
ax.set_title("CC vs AD point clouds\n(2-D PCA of FC features)")
ax.legend(frameon=False, fontsize=8, loc="best"); _tag(ax, "F")

fig.suptitle(
    "CC vs AD classification: G-space geometry vs. lagged-FC of reconstruction, "
    "with a nested-CV tangent-space FC-SVM benchmark\n"
    f"(reservoir read-outs: per-patient W, LDA & RF, repeated 5-fold 80/20 CV, "
    f"N={n_cc} CC + {n_ad} AD; dotted band = tangent-FC benchmark)",
    fontsize=10.5, fontweight="bold", y=0.975)

for ext in ("png", "pdf"):
    fig.savefig(f"figure3_classification.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figure3_classification.{ext}")
plt.close(fig)
