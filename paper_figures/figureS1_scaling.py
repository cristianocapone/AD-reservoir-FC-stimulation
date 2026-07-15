"""
figureS1_scaling.py
===================
Supplementary Figure S1 — classification accuracy vs cohort size, with an
extrapolation to larger cohorts.

Reads the learning-curve sweep (rf_results.npz): G-space LDA balanced accuracy
and AUROC as a function of the number of patients per group used for fitting.
A saturating power law  m(N) = A - B * N^{-C}  is fit to each metric and
extrapolated, giving the asymptotic ("reachable") accuracy and predictions at
N = 100 and N = 200.

Saves: figureS1_scaling.{png,pdf}
"""
import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

d = np.load("../rf_results.npz", allow_pickle=True)
N   = d["N_grid"].astype(float)
metrics = {
    "AUROC":             (d["LDA_auc_mean"], d["LDA_auc_std"]),
    "Balanced accuracy": (d["LDA_bal_mean"], d["LDA_bal_std"]),
}

def sat(N, A, B, C):           # saturating power law
    return A - B * np.power(N, -C)

def fit_extrap(N, m, s):
    s = np.clip(s, 1e-3, None)
    p0 = [0.8, 1.0, 0.5]
    bounds = ([0.5, 0.0, 0.05], [1.0, 50.0, 3.0])
    popt, pcov = curve_fit(sat, N, m, p0=p0, sigma=s, absolute_sigma=True,
                           bounds=bounds, maxfev=20000)
    perr = np.sqrt(np.diag(pcov))
    return popt, perr

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor="white")
fig.subplots_adjust(wspace=0.28, top=0.84, bottom=0.16, left=0.08, right=0.97)

Nfine = np.linspace(N.min(), 220, 400)
COL   = "#1565C0"

rng = np.random.default_rng(0)
for ax, (name, (m, s)) in zip(axes, metrics.items()):
    popt, pcov = None, None
    s_ = np.clip(s, 1e-3, None)
    popt, pcov = curve_fit(sat, N, m, p0=[0.8, 1.0, 0.5], sigma=s_,
                           absolute_sigma=True,
                           bounds=([0.5, 0.0, 0.05], [1.0, 50.0, 3.0]),
                           maxfev=20000)
    A, B, C = popt
    # extrapolation uncertainty band by Monte-Carlo sampling of the fit
    samp = rng.multivariate_normal(popt, pcov, size=400)
    samp = samp[(samp[:, 0] >= 0.5) & (samp[:, 0] <= 1.0)]
    curves = np.array([sat(Nfine, *p) for p in samp])
    lo, hi = np.nanpercentile(curves, [16, 84], axis=0)

    # extrapolation region shading
    ax.axvspan(N.max(), 220, color="#FFF3E0", zorder=0)
    ax.fill_between(Nfine, lo, hi, color=COL, alpha=0.15, zorder=1)
    ax.plot(Nfine, sat(Nfine, A, B, C), "-", color=COL, lw=2, zorder=2,
            label=r"fit $A-B\,N^{-C}$ (68\% band)")
    ax.errorbar(N, m, yerr=s, fmt="o", ms=5, color=COL, capsize=3,
                zorder=3, label="observed (G-space LDA)")
    ax.axvline(N.max(), color="gray", ls=":", lw=1, alpha=0.7)
    ax.text(N.max()-3, 0.47, "current\ncohort", fontsize=6.5, color="gray",
            va="bottom", ha="right")

    for Np in (100, 200):
        yp = sat(float(Np), A, B, C)
        ax.scatter([Np], [yp], color="#E65100", s=48, zorder=4,
                   edgecolors="k", lw=0.6)
        ax.annotate(f"$N{{=}}{Np}$: {yp:.2f}", (Np, yp),
                    textcoords="offset points", xytext=(5, -11),
                    fontsize=7.5, color="#E65100", fontweight="bold")
    ax.axhline(0.5, color="gray", ls="-", lw=0.8, alpha=0.3)
    ax.set_xlabel("Patients per group used for fitting,  $N$")
    ax.set_ylabel(name)
    ax.set_title(f"{name} vs cohort size")
    ax.set_xlim(0, 220); ax.set_ylim(0.45, 0.9)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    print(f"{name}: A={A:.3f} B={B:.3f} C={C:.3f}  "
          f"-> N=100:{sat(100,A,B,C):.3f}  N=200:{sat(200,A,B,C):.3f}")

fig.suptitle("Classification accuracy scales with cohort size: extrapolation to "
             "larger cohorts\n(G-space LDA, leave-one-patient-out; saturating "
             "power-law fit; shaded region = extrapolation beyond current cohort)",
             fontsize=10.5, fontweight="bold", y=0.99)

for ext in ("png", "pdf"):
    fig.savefig(f"figureS1_scaling.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figureS1_scaling.{ext}")
plt.close(fig)
