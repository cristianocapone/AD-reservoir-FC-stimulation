"""
plot_learning_curve.py
======================
Replot learning curve from saved npz, adding:
  - Reference line: leaky G-space estimate (0.683 BAL-ACC, 0.672 AUROC)
  - Dotted annotation explaining the bias
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

d = np.load("learning_curve_lopo_data.npz")
N_arr   = d["N_grid"]
ba_mean = d["bal_mean"]
ba_std  = d["bal_std"]
au_mean = d["auc_mean"]
au_std  = d["auc_std"]
k_eff   = d["k_eff"]
K_LDA   = 25

# Reference values from sigma_klda_lopo.py (leaky G-space, all 76 patients)
REF_BA  = 0.6833
REF_AUC = 0.6722

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

for ax, mean_v, std_v, ref_val, ylabel, title, col in [
        (axes[0], ba_mean, ba_std, REF_BA,
         "LOPO Balanced Accuracy", "BAL-ACC vs N patients / class", "#1565C0"),
        (axes[1], au_mean, au_std, REF_AUC,
         "LOPO AUROC", "AUROC vs N patients / class", "#1565C0")]:

    ax.axhline(0.50, color="gray", ls="--", lw=1.0, alpha=0.6, label="chance (0.50)")

    # Leaky G-space reference
    ax.axhline(ref_val, color="#C62828", ls="-.", lw=1.5, alpha=0.85,
               label=f"leaky G-space (fixed PCA, all 76 pts): {ref_val:.3f}")

    # Orange shading where K_LDA was capped
    capped = k_eff < K_LDA
    if capped.any():
        x_cap = N_arr[capped]
        ax.axvspan(x_cap.min()-0.5, x_cap.max()+0.5,
                   alpha=0.09, color="orange",
                   label=f"K_LDA capped (< {K_LDA})")

    # Learning curve
    ax.fill_between(N_arr, mean_v - std_v, mean_v + std_v,
                    alpha=0.25, color=col)
    ax.plot(N_arr, mean_v, "-o", ms=7, lw=2.5, color=col,
            label=f"proper per-fold SVD  (mean ± std, 30 reps/N)")

    # Annotate k_eff
    for xi, (N, k, m) in enumerate(zip(N_arr, k_eff, mean_v)):
        ax.text(N, m + std_v[xi] + 0.006,
                f"k={k}", ha="center", va="bottom", fontsize=7, color="#555")

    # Arrow showing the gap at N=35
    ax.annotate(
        f"~{ref_val - mean_v[-1]:.2f} bias\n(G-space leak)",
        xy=(N_arr[-1], mean_v[-1] + 0.008),
        xytext=(N_arr[-1] - 5, ref_val - 0.02),
        fontsize=8, color="#C62828",
        arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2),
        ha="center",
    )

    ax.set_xlabel("N patients per class", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(N_arr)
    ax.set_ylim(0.30, 0.90)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

fig.suptitle(
    "Learning curve — LOPO CV with proper per-fold SVD (kernel PCA)\n"
    "Condition B: σ=0.05, K_LDA=25  |  Red line = leaky-G-space estimate (fixed PCA on all 76 pts)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig("learning_curve_lopo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved learning_curve_lopo.png")
