"""
replot_oscfreq.py — Figure 6 from pert_oscfreq_data.npz (amplitudes up to 10).
Panels: A freq tuning, B dose-response (to A=10), C brain (large),
        D LDA-score distributions (CC vs AD vs stimulated), E cured-patient counts.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread

d = np.load("pert_oscfreq_data.npz", allow_pickle=True)
FREQS = d["freqs"]; AMPS = d["amps"]; K_SITES = list(d["k_sites"])
FL = d["FL"]                                  # (nK, nF, nA, n_ad)
thr = float(d["thr_fl"]); fl_base = d["fl_base"]; cc_fl = d["cc_fl"]
f_eig = float(d["f_eig"]); f_fft = float(d["f_fft"])
n_ad = FL.shape[-1]
ki1 = K_SITES.index(1); ki5 = K_SITES.index(5)
fi_eig = int(np.argmin(np.abs(FREQS - f_eig)))
fi_off = int(np.argmax(FREQS))
ai4 = int(np.argmin(np.abs(AMPS - 4)))
aimax = len(AMPS) - 1

def sem(a, ax): return a.std(ax) / np.sqrt(a.shape[ax])
def cured(ki, fi):           # per-amplitude count of AD with score < threshold
    return (FL[ki, fi] < thr).sum(1)

print(f"thr={thr:.3f}  AMPS={AMPS}")
for name, ki, fi in [("top-1 resonant", ki1, fi_eig), ("top-5 resonant", ki5, fi_eig),
                     ("top-1 off-res", ki1, fi_off), ("top-5 off-res", ki5, fi_off)]:
    c = cured(ki, fi)
    print(f"  {name:16s} cured(of {n_ad}) vs amp: {list(c)}")

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})
K_COL = {1: "#6A1B9A", 5: "#2E7D32"}
CC_COL = "#1565C0"; AD_COL = "#C62828"

fig = plt.figure(figsize=(11.5, 8.4), facecolor="white")
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.30,
                       left=0.08, right=0.975, top=0.90, bottom=0.08)

def _tag(ax, t, x=-0.16):
    ax.text(x, 1.05, t, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")

# ── A: frequency tuning at moderate amplitude (A=4) ───────────────────────────
ax = fig.add_subplot(gs[0, 0])
for k, ki in [(1, ki1), (5, ki5)]:
    m = FL[ki, :, ai4].mean(1); e = sem(FL[ki, :, ai4], 1)
    ax.fill_between(FREQS, m-e, m+e, color=K_COL[k], alpha=0.18)
    ax.plot(FREQS, m, "-o", ms=4, color=K_COL[k], lw=2, label=f"k={k}")
ax.axhline(thr, color="gray", ls="-.", lw=1, label="boundary")
ax.axvline(f_eig, color="#C62828", ls="--", lw=1.5, label=f"$f_\\mathrm{{eig}}$={f_eig:.3f}")
ax.axvline(f_fft, color="#E65100", ls=":", lw=1.5, label=f"$f_\\mathrm{{FFT}}$={f_fft:.3f}")
ax.set_xlabel("frequency (cycles/step)"); ax.set_ylabel("FC-lag LDA score")
ax.set_title(f"Frequency tuning  ($A={AMPS[ai4]:.0f}$)")
ax.legend(frameon=False, fontsize=7); _tag(ax, "A")

# ── B: dose-response up to A=10, resonant vs off-resonance, k=1 & k=5 ──────────
ax = fig.add_subplot(gs[0, 1])
ax.axhline(thr, color="gray", ls="-.", lw=1, label="boundary")
ax.axhline(cc_fl.mean(), color=CC_COL, ls="--", lw=1.2, alpha=0.8, label="CC mean")
for k, ki in [(1, ki1), (5, ki5)]:
    m = FL[ki, fi_eig].mean(1); e = sem(FL[ki, fi_eig], 1)
    ax.fill_between(AMPS, m-e, m+e, color=K_COL[k], alpha=0.15)
    ax.plot(AMPS, m, "-o", ms=4, color=K_COL[k], lw=2, label=f"k={k} resonant")
    m = FL[ki, fi_off].mean(1)
    ax.plot(AMPS, m, "--s", ms=3.5, color=K_COL[k], lw=1.6, alpha=0.7,
            label=f"k={k} off-res.")
ax.set_xlabel("stimulation amplitude  $A$"); ax.set_ylabel("FC-lag LDA score")
ax.set_title("Dose-response to high amplitude\n(resonant vs off-resonant)")
ax.legend(frameon=False, fontsize=6.5, ncol=2); _tag(ax, "B")

# ── C: LDA-score distributions (CC vs AD vs stimulated) ───────────────────────
ax = fig.add_subplot(gs[1, 0])
data = [cc_fl, fl_base, FL[ki5, fi_eig, ai4], FL[ki5, fi_eig, aimax]]
labs = [f"CC\nbaseline", f"AD\nbaseline",
        f"AD\nstim $A={AMPS[ai4]:.0f}$", f"AD\nstim $A={AMPS[aimax]:.0f}$"]
cols = [CC_COL, AD_COL, "#FB8C00", "#2E7D32"]
vp = ax.violinplot(data, positions=range(4), showmedians=True, widths=0.7)
for b, c in zip(vp["bodies"], cols):
    b.set_facecolor(c); b.set_alpha(0.5)
for kk in ["cmedians", "cbars", "cmins", "cmaxes"]:
    vp[kk].set_color("k"); vp[kk].set_linewidth(1.0)
ax.axhline(thr, color="gray", ls="-.", lw=1.2, label="decision boundary")
rng = np.random.default_rng(0)
for xi, (arr, c) in enumerate(zip(data, cols)):
    ax.scatter(xi + rng.uniform(-0.08, 0.08, len(arr)), arr, s=6, c=c,
               alpha=0.5, edgecolors="none", zorder=3)
# annotate cured % for the two stimulated AD conditions
for xi, arr in [(2, data[2]), (3, data[3])]:
    pct = (arr < thr).mean() * 100
    ax.text(xi, ax.get_ylim()[1]*0.92, f"{pct:.0f}%\ncured", ha="center",
            fontsize=7, color="#333")
ax.set_xticks(range(4)); ax.set_xticklabels(labs, fontsize=7.5)
ax.set_ylabel("FC-lag LDA score")
ax.set_title("Score distributions\n(top-5 resonant stimulation)")
ax.legend(frameon=False, fontsize=7, loc="lower left"); _tag(ax, "C")

# ── D: cured-patient counts vs amplitude ──────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
for k, ki in [(1, ki1), (5, ki5)]:
    ax.plot(AMPS, cured(ki, fi_eig), "-o", ms=4, color=K_COL[k], lw=2,
            label=f"k={k} resonant")
    ax.plot(AMPS, cured(ki, fi_off), "--s", ms=3.5, color=K_COL[k], lw=1.6,
            alpha=0.7, label=f"k={k} off-res.")
ax.axhline((fl_base < thr).sum(), color=AD_COL, ls=":", lw=1.2,
           label=f"baseline ({int((fl_base<thr).sum())})")
ax.set_xlabel("stimulation amplitude  $A$")
ax.set_ylabel(f"# AD patients cured (of {n_ad})")
ax.set_title("Patients reclassified as CC\n(score $<$ boundary)")
ax.legend(frameon=False, fontsize=6.5, ncol=2); _tag(ax, "D")

fig.suptitle("Oscillatory stimulation: frequency prescription, high-amplitude "
             "dose-response, and patients reclassified",
             fontsize=11.5, fontweight="bold", y=0.985)

for ext in ("png", "pdf"):
    fig.savefig(f"figureS8_oscfreq.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figureS8_oscfreq.{ext}")
plt.close(fig)
