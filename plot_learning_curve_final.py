"""
Plot final learning curve comparing old (wrong scale) vs new (per-fold SVD, normalised).
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

# ── load data ──────────────────────────────────────────────────────────────────
dn = np.load("learning_curve_direct_data.npz")   # new (correct)
do = np.load("learning_curve_lopo_data.npz")      # old (wrong scale)

N_new = dn["N_grid"]
ba_new = dn["bal_mean"]; ba_std_new = dn["bal_std"]
au_new = dn["auc_mean"]; au_std_new = dn["auc_std"]
k_eff  = dn["k_eff"]

N_old  = do["N_grid"]
ba_old = do["bal_mean"]; ba_std_old = do["bal_std"]
au_old = do["auc_mean"]; au_std_old = do["auc_std"]

K_LDA = 25
REF_BA  = 0.6833   # leaky G-space (fixed PCA, all 76 patients)
REF_AUC = 0.6722

# ── linear extrapolation from N>=15 ──────────────────────────────────────────
mask = N_new >= 15
slope_ba, icept_ba = np.polyfit(N_new[mask].astype(float), ba_new[mask], 1)
slope_au, icept_au = np.polyfit(N_new[mask].astype(float), au_new[mask], 1)
N_ext = np.array([35, 50, 75, 100])

# ── plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")

for ax, ba_n, ba_s_n, ba_o, ba_s_o, slope, icept, ref, ylabel, title in [
    (axes[0],
     ba_new, ba_std_new, ba_old, ba_std_old,
     slope_ba, icept_ba, REF_BA,
     "LOPO Balanced Accuracy", "BAL-ACC vs N patients / class"),
    (axes[1],
     au_new, au_std_new, au_old, au_std_old,
     slope_au, icept_au, REF_AUC,
     "LOPO AUROC", "AUROC vs N patients / class"),
]:
    ax.axhline(0.50,  color="gray",    ls="--", lw=1.0, alpha=0.6, label="chance (0.50)")
    ax.axhline(ref,   color="#C62828", ls="-.", lw=1.5, alpha=0.85,
               label=f"leaky G-space (fixed PCA, all 76 pts): {ref:.3f}")

    # old (wrong scale) — faint
    ax.fill_between(N_old, ba_o - ba_s_o, ba_o + ba_s_o,
                    alpha=0.10, color="#888")
    ax.plot(N_old, ba_o, "--s", ms=5, lw=1.5, color="#888", alpha=0.6,
            label="old (aggregated scores, wrong scale)")

    # orange shading where K_LDA capped
    capped = k_eff < K_LDA
    if capped.any():
        ax.axvspan(N_new[capped].min()-0.5, N_new[capped].max()+0.5,
                   alpha=0.10, color="orange",
                   label=f"K_LDA capped (< {K_LDA})")

    # new (correct)
    ax.fill_between(N_new, ba_n - ba_s_n, ba_n + ba_s_n,
                    alpha=0.25, color="#1565C0")
    ax.plot(N_new, ba_n, "-o", ms=7, lw=2.5, color="#1565C0",
            label="new (per-fold SVD→test, normalised score)")

    # linear extrapolation
    extrap = icept + slope * N_ext
    ax.plot(N_ext, extrap, ":", lw=1.8, color="#1565C0", alpha=0.60)
    ax.plot(N_ext[1:], extrap[1:], "o", ms=5, color="#1565C0", alpha=0.45)
    ax.text(N_ext[-1] + 1, extrap[-1],
            f"~{extrap[-1]:.2f}\n@N=100",
            va="center", fontsize=8, color="#1565C0")

    # k_eff annotations
    for xi, (N, k, m) in enumerate(zip(N_new, k_eff, ba_n)):
        ax.text(N, m + ba_s_n[xi] + 0.007,
                f"k={k}", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xlabel("N patients per class", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(list(N_new) + [50, 75, 100])
    ax.set_xlim(2, 110)
    ax.set_ylim(0.15, 0.90)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    "Learning curve — condition B  (σ=0.05, K_LDA=25)\n"
    "Per-fold SVD applied to test + score normalised by class gap\n"
    "Dotted: linear extrapolation from N≥15  |  Grey: previous (wrong-scale) estimate",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig("learning_curve_direct.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved learning_curve_direct.png")
