"""
figure4_stimulation.py
======================
Figure 4 — In-silico stimulation: shifting AD readout matrices toward CC.

Layout (3 rows x 3 cols):
  A  G-space baseline distributions   B  G-space full-W dose-response       C  G-space focal dose-response
  D  FC-lag  baseline distributions   E  FC-lag  full-W dose-response        F  FC-lag  focal dose-response
  G  G-space reclassification         H  FC-lag  reclassification             I  Summary at physio alpha

Reads: ../pert_sites_data.npz
Output: figure4_stimulation.png  (300 DPI)
"""
import sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore")

# ── paper style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

CC_COL  = "#1565C0"
AD_COL  = "#C62828"
FW_COL  = "#2E7D32"
T5_COL  = "#E65100"
T1_COL  = "#6A1B9A"
GEO_COL = "#00838F"
PHYSIO  = 4.0

# ── load data ──────────────────────────────────────────────────────────────────
d = np.load("../pert_sites_data.npz", allow_pickle=True)

cc_g  = d["cc_g"];   ad_g  = d["ad_g"];   thr_g  = float(d["thr_g"])
cc_fl = d["cc_fl"];  ad_fl = d["ad_fl"];  thr_fl = float(d["thr_fl"])

ALPHAS_FW  = d["alphas_fw"]    # 0 → 2.0  (full correction)
ALPHAS_T5  = d["alphas_t5"]    # 0 → 20   (focal top-5)
ALPHAS_T1  = d["alphas_t1"]    # 0 → 50   (focal top-1)
ALPHAS_GEO = d["alphas_geo"]   # 0 → 20   (top-1 + 10 geo-nbrs)

n_ad = ad_g.shape[0];  n_cc = cc_g.shape[0]

FOCAL = [
    ("top5",     ALPHAS_T5,  T5_COL,  "Top-5 sites"),
    ("top1",     ALPHAS_T1,  T1_COL,  "Top-1 site"),
    ("top1_geo", ALPHAS_GEO, GEO_COL, "Top-1 + 10 geo-nbrs"),
]
ALL_STRATS = [("full_w", ALPHAS_FW, FW_COL, "Full-W (121 sites)")] + FOCAL

def load(strat, alphas, sp):
    return np.array([d[f"{strat}_{ai}_{sp}"] for ai in range(len(alphas))])

def violin_pair(ax, vals_cc, vals_ad, col_cc, col_ad, thr, thr_lbl, ylabel, title):
    rng = np.random.default_rng(42)
    for xi, (vals, col) in enumerate([(vals_cc, col_cc), (vals_ad, col_ad)]):
        p = ax.violinplot([vals], positions=[xi], widths=0.5,
                           showmedians=True, showextrema=False)
        p["bodies"][0].set_facecolor(col); p["bodies"][0].set_alpha(0.45)
        p["cmedians"].set_color("k"); p["cmedians"].set_linewidth(1.2)
        jit = rng.uniform(-0.09, 0.09, len(vals))
        ax.scatter(xi + jit, vals, s=14, c=col, alpha=0.6,
                   edgecolors="white", linewidths=0.3, zorder=3)
    ax.axhline(thr, color="gray", ls="--", lw=1.2, label=thr_lbl)
    stat, pval = mannwhitneyu(vals_cc, vals_ad, alternative="two-sided")
    ax.text(0.97, 0.03, f"p={pval:.2e}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"CC\n(n={n_cc})", f"AD\n(n={n_ad})"])
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=7)

def dose_panel(ax, strats, space, cc_vals, thr, physio_vline=True, xlim=None):
    ax.axhspan(cc_vals.mean()-cc_vals.std(), cc_vals.mean()+cc_vals.std(),
               alpha=0.10, color=CC_COL)
    ax.axhline(cc_vals.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean ±1σ")
    ax.axhline(thr, color="gray", lw=1, ls="-.", label="Boundary")
    for strat, alphas, col, lbl in strats:
        mat = load(strat, alphas, space)
        m = mat.mean(1); s = mat.std(1)
        ax.fill_between(alphas, m-s, m+s, alpha=0.13, color=col)
        ax.plot(alphas, m, "-o", ms=4, lw=2, color=col, label=lbl)
    if physio_vline:
        ax.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6,
                   label=f"Physio (α={PHYSIO:.0f})")
    if xlim is not None:
        ax.set_xlim(xlim)
    ax.legend(frameon=False, fontsize=7)

# ── figure layout ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 7.2), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40,
                        top=0.965, bottom=0.09, left=0.06, right=0.98)

# ── Row 1: G-space ─────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
violin_pair(ax_a, cc_g, ad_g, CC_COL, AD_COL,
            thr_g, f"Boundary ({thr_g:.2f})",
            "G-space LDA score", "Baseline G-space LDA")

ax_b = fig.add_subplot(gs[0, 1])
dose_panel(ax_b, [("full_w", ALPHAS_FW, FW_COL, "Full-W (121 sites)")],
           "g", cc_g, thr_g, physio_vline=False)
ax_b.set_xlabel("Perturbation strength  α")
ax_b.set_ylabel("G-space LDA score  (mean ± 1σ)")
ax_b.set_title("G-space LDA — full-W (all 121 sites)")

ax_c = fig.add_subplot(gs[0, 2])
dose_panel(ax_c, FOCAL, "g", cc_g, thr_g)
ax_c.set_xlim(-0.5, 20.5)
ax_c.set_xlabel("Perturbation strength  α")
ax_c.set_ylabel("G-space LDA score  (mean ± 1σ)")
ax_c.set_title("G-space LDA — focal stimulation")

# ── Row 2: FC-lag ─────────────────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
violin_pair(ax_d, cc_fl, ad_fl, CC_COL, AD_COL,
            thr_fl, f"Boundary ({thr_fl:.2f})",
            "FC-lag LDA score", "Baseline FC-lag LDA")

ax_e = fig.add_subplot(gs[1, 1])
dose_panel(ax_e, [("full_w", ALPHAS_FW, FW_COL, "Full-W (121 sites)")],
           "fl", cc_fl, thr_fl, physio_vline=False)
ax_e.set_xlabel("Perturbation strength  α")
ax_e.set_ylabel("FC-lag LDA score  (mean ± 1σ)")
ax_e.set_title("FC-lag LDA — full-W (all 121 sites)")

ax_f = fig.add_subplot(gs[1, 2])
dose_panel(ax_f, FOCAL, "fl", cc_fl, thr_fl)
ax_f.set_xlim(-0.5, 20.5)
ax_f.set_xlabel("Perturbation strength  α")
ax_f.set_ylabel("FC-lag LDA score  (mean ± 1σ)")
ax_f.set_title("FC-lag LDA — focal stimulation")

# ── panel labels ───────────────────────────────────────────────────────────────
for ax, lbl in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                   ["A", "B", "C", "D", "E", "F"]):
    ax.text(-0.13, 1.05, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")

for ext in ("png", "pdf"):
    out = f"figure4_stimulation.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
plt.close()
