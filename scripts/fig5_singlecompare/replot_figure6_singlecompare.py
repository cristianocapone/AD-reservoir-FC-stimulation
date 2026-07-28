"""
replot_figure6_singlecompare.py
================================
Re-render figure6_singlecompare from cached sweeps, no simulation re-run.

Top row (A-C): single-site dose-response, from pert_single_compare_data.npz
(unchanged from the original figure6_singlecompare.py).

Bottom row (D-F): replaces the old 3-condition offline/oracle closed-loop
comparison (open-loop / CL-amplitude / CL-amplitude+site, all reclassifying
100% -> uninformative panel D) with the causally-valid REAL-TIME closed-loop
controller from pert_online_cl.py: a fixed personalised (LDA-resonant) site,
amplitude titrated online from a sliding-window FC-lag biomarker estimate
(no oracle access to the future trajectory). Data: pert_online_cl_data.npz.
  D: paired per-patient stimulation amplitude (dose), open-loop vs online CL.
  E: paired per-patient FC distance from baseline, among patients reclassified
     under BOTH conditions (fair cost comparison).
  F: per-patient scatter, open-loop distance vs online-CL distance.
Reclassification rate (100% open-loop vs 97.5% online CL, i.e. 39/40) is
reported as text rather than as its own bar panel, since near-ceiling rates
are not the informative axis; the dose/distance panels are.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sc = np.load("../pert_single_compare_data.npz", allow_pickle=True)
ALPHAS = sc["alphas"]; AMPS = sc["amps"]
recl_dw = sc["recl_dw"]; recl_os = sc["recl_os"]; recl_lp = sc["recl_lp"]
ddw = sc["ddw"]; dos = sc["dos"]; dlp = sc["dlp"]

ocl = np.load("../pert_online_cl_data.npz", allow_pickle=True)
ol_rec = ocl["ol_rec"]; ol_dist = ocl["ol_dist"]
ocl_rec = ocl["ocl_rec"]; ocl_dist = ocl["ocl_dist"]; ocl_amp = ocl["ocl_amp"]
a_start = float(ocl["a_start"]); n_ad = len(ol_rec)
ol_amp = np.full(n_ad, a_start)

print(f"open-loop:  reclass={ol_rec.mean()*100:.1f}%  "
      f"dist={np.nanmean(ol_dist[ol_rec>0]):.3f}  amp={a_start:.2f}")
print(f"online CL:  reclass={ocl_rec.mean()*100:.1f}%  "
      f"dist={np.nanmean(ocl_dist[ocl_rec>0]):.3f}  amp={np.nanmean(ocl_amp[ocl_rec>0]):.2f}")

# closed-loop operating point overlay for panels A-C (relative to AMPS.max()=20)
CL_DOSE = float(np.nanmean(ocl_amp[ocl_rec > 0])) / AMPS.max()
CL_DIST = float(np.nanmean(ocl_dist[ocl_rec > 0]))
CL_RECL = float(ocl_rec.mean() * 100)
OL_DOSE = a_start / AMPS.max()
OL_DIST = float(np.nanmean(ol_dist[ol_rec > 0]))

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
    "figure.dpi": 300, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
CDW = "#1A237E"; COS = "#E65100"; CLA = "#2E7D32"
COL_OL = "#455A64"; COL_OCL = "#E65100"
dwn = ALPHAS / ALPHAS.max(); osn = AMPS / AMPS.max()
fig = plt.figure(figsize=(13.5, 8.6), facecolor="white")
gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.30, hspace=0.42, left=0.07, right=0.985, top=0.95, bottom=0.07)
def tag(ax, s): ax.text(-0.17, 1.05, s, transform=ax.transAxes, fontsize=13, fontweight="bold")
L_DW = "theoretical $\\Delta W$ (top-1, $\\alpha\\!\\to\\!50$)"
L_OS = "resonant osc ($\\Delta W$ top-1 site)"
L_LP = "resonant osc (LDA-resonant site)"

# A: reclassification vs dose
ax = fig.add_subplot(gs[0, 0])
ax.plot(dwn, recl_dw, "-o", ms=4, color=CDW, lw=2, label=L_DW)
ax.plot(osn, recl_os, "-s", ms=4, color=COS, lw=2, label=L_OS)
ax.plot(osn, recl_lp, "-^", ms=4, color=CLA, lw=2, label=L_LP)
ax.scatter(OL_DOSE, 100, marker="X", s=130, c=COL_OL, edgecolors="k", lw=0.6, zorder=6)
ax.scatter(CL_DOSE, CL_RECL, marker="P", s=130, c=COL_OCL, edgecolors="k", lw=0.6, zorder=6)
ax.set_xlabel("relative dose"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification vs dose"); ax.set_ylim(-2, 107); ax.legend(frameon=False); tag(ax, "A")

# B: distance vs dose
ax = fig.add_subplot(gs[0, 1])
ax.plot(dwn, ddw, "-o", ms=4, color=CDW, lw=2, label=L_DW)
ax.plot(osn, dos, "-s", ms=4, color=COS, lw=2, label=L_OS)
ax.plot(osn, dlp, "-^", ms=4, color=CLA, lw=2, label=L_LP)
ax.scatter(OL_DOSE, OL_DIST, marker="X", s=130, c=COL_OL, edgecolors="k", lw=0.6, zorder=6)
ax.scatter(CL_DOSE, CL_DIST, marker="P", s=130, c=COL_OCL, edgecolors="k", lw=0.6, zorder=6)
ax.set_xlabel("relative dose"); ax.set_ylabel("distance from original FC ($1-$corr)")
ax.set_title("Distance from unstimulated FC vs dose"); ax.legend(frameon=False); tag(ax, "B")

# C: efficacy vs perturbation cost
ax = fig.add_subplot(gs[0, 2])
ax.plot(ddw, recl_dw, "-o", ms=4, color=CDW, lw=2, label=L_DW)
ax.plot(dos, recl_os, "-s", ms=4, color=COS, lw=2, label=L_OS)
ax.plot(dlp, recl_lp, "-^", ms=4, color=CLA, lw=2, label=L_LP)
ax.scatter(OL_DIST, 100, marker="X", s=150, c=COL_OL, edgecolors="k", lw=0.6, zorder=6, label="open-loop (fixed $A$)")
ax.scatter(CL_DIST, CL_RECL, marker="P", s=150, c=COL_OCL, edgecolors="k", lw=0.6, zorder=6, label="online CL (dose-min)")
ax.set_xlabel("distance from original FC ($1-$corr)"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Efficacy vs perturbation cost"); ax.set_ylim(-2, 107)
ax.legend(frameon=False, fontsize=6.6, loc="lower left"); tag(ax, "C")

NAMES = ["Open-loop\n(fixed $A=6$)", "Online CL\n(dose-min., real-time)"]
rng_j = np.random.default_rng(7)

def paired_panel(ax, vals_ol, vals_ocl, ylabel, title, lbl):
    n = len(vals_ol)
    jit = rng_j.uniform(-0.06, 0.06, n)
    for i in range(n):
        col = "#BDBDBD" if vals_ocl[i] <= vals_ol[i] else "#EF9A9A"
        ax.plot([0 + jit[i], 1 + jit[i]], [vals_ol[i], vals_ocl[i]], lw=0.6, color=col, alpha=0.6, zorder=1)
    ax.scatter(0 + jit, vals_ol, s=18, color=COL_OL, alpha=0.8, zorder=3, edgecolors="none")
    ax.scatter(1 + jit, vals_ocl, s=18, color=COL_OCL, alpha=0.8, zorder=3, edgecolors="none")
    for xi, (vals, col) in enumerate([(vals_ol, COL_OL), (vals_ocl, COL_OCL)]):
        m, se = np.nanmean(vals), np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))
        ax.plot([xi - 0.18, xi + 0.18], [m, m], lw=2.5, color=col, zorder=4)
        ax.errorbar(xi, m, yerr=se, fmt="none", ecolor=col, capsize=4, lw=1.5, zorder=4)
    pct = (np.nanmean(vals_ocl) - np.nanmean(vals_ol)) / np.nanmean(vals_ol) * 100
    sign = "+" if pct > 0 else ""
    ax.text(0.5, 0.97, f"{sign}{pct:.0f}%", transform=ax.transAxes, ha="center", va="top",
            fontsize=9, fontweight="bold", color=COL_OCL if pct < 0 else "#C62828")
    ax.set_xticks([0, 1]); ax.set_xticklabels(NAMES)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.text(-0.16, 1.04, lbl, transform=ax.transAxes, fontsize=13, fontweight="bold")

# D: paired per-patient dose (amplitude)
paired_panel(fig.add_subplot(gs[1, 0]), ol_amp, ocl_amp,
             "Mean stimulation amplitude", "Real-time closed-loop: dose", "D")

# E: paired per-patient FC distance (reclassified under both)
m_both = (ol_rec > 0) & (ocl_rec > 0) & ~np.isnan(ol_dist) & ~np.isnan(ocl_dist)
paired_panel(fig.add_subplot(gs[1, 1]), ol_dist[m_both], ocl_dist[m_both],
             "distance from original FC ($1-$corr)",
             "Real-time closed-loop: perturbation cost\n(reclassified under both)", "E")

# F: per-patient scatter, open-loop vs online-CL distance
ax = fig.add_subplot(gs[1, 2])
ax.scatter(ol_dist[m_both], ocl_dist[m_both], s=24, color=COL_OCL, alpha=0.75, edgecolors="none")
lim = [0, max(np.nanmax(ol_dist[m_both]), np.nanmax(ocl_dist[m_both])) * 1.05]
ax.plot(lim, lim, "k--", lw=1, alpha=0.6)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.text(0.62, 0.16, "online CL\nbetter", transform=ax.transAxes, fontsize=8, color=COL_OCL, ha="center")
ax.set_xlabel("open-loop distance (fixed $A$)"); ax.set_ylabel("online-CL distance (real-time, dose-min.)")
ax.set_title(f"Per-patient improvement\n({int(m_both.sum())}/{n_ad} reclassified under both)")
tag(ax, "F")

for ext in ("png", "pdf"):
    fig.savefig(f"figure6_singlecompare.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure6_singlecompare.{ext}")
plt.close(fig)
print("Done.")
