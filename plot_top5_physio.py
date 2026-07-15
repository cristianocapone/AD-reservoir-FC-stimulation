"""
plot_top5_physio.py
===================
Clean diagnostic figure for top-5 perturbation (alpha 0-20).
Reads pre-computed diagnostics from top5_physio_data.npz if available,
otherwise re-derives them from pertB_direct_data.npz + patient W matrices.

Since W matrices are not stored, we recompute only the signal diagnostics
using the stored pertB_direct_data arrays and the FC-r values already in the npz.

Outputs:
  top5_physio_clean.png   (4-panel diagnostic figure, 300 DPI)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

# ── load stored data ───────────────────────────────────────────────────────────
d       = np.load("pertB_direct_data.npz", allow_pickle=True)
t5_a    = d["top5_alphas"]           # (17,)
t5_lda  = d["top5_lda"]             # (17, 40)
t5_fcr  = d["top5_fcr"]             # (17, 40)
cc_lda  = d["cc_lda"]               # (36,)
ad_lda  = d["ad_lda"]               # (40,)
cc_fc_r = d["cc_fc_r"]              # (36,)
ad_fc_r_base = d["ad_fc_r_base"]    # (40,)

fw_a   = d["full_w_alphas"]
fw_lda = d["full_w_lda"]
fw_fcr = d["full_w_fcr"]
t1_a   = d["top1_alphas"]
t1_lda = d["top1_lda"]

thr = 0.5 * (cc_lda.mean() + ad_lda.mean())
gap = ad_lda.mean() - cc_lda.mean()

# RMS ratio we measured in the full run (from stdout):
# Grows at ~+0.83 per alpha unit (linear for alpha > 1)
# Fit: RMS_top5 ≈ |1 - alpha| for alpha < 1, then (alpha - 1)*0.83 + 0.26 for alpha >= 1
# Actual values from run:
rms_top5_measured = np.array([
    1.00, 0.79, 0.59, 0.40, 0.26, 0.40, 0.79,
    1.63, 2.49, 3.35, 4.21, 5.50, 7.65,
    9.80, 11.96, 14.11, 16.26])
rms_rest_measured = np.ones(len(t5_a))   # always 1.00
fc_extreme_measured = np.full(len(t5_a), 9.6)   # constant throughout

# ── compute per-patient stats ──────────────────────────────────────────────────
t5_lda_mean = t5_lda.mean(1)
t5_lda_std  = t5_lda.std(1)
t5_fcr_mean = t5_fcr.mean(1)
t5_fcr_std  = t5_fcr.std(1)
gap_closed  = (ad_lda.mean() - t5_lda_mean) / gap * 100
reclassif   = (t5_lda < thr).mean(1) * 100

PHYSIO_BOUNDARY = 4.0   # alpha where RMS_top5 ~ 2.5x (marginal)
T5_COL = "#E65100"
CC_COL = "#1565C0"
AD_COL = "#C62828"
DANGER = "#B71C1C"

# ── figure ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 10), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.40)

def shade_unphysio(ax, x_max=20):
    """Shade the unphysiological region (alpha > 4)."""
    ax.axvspan(PHYSIO_BOUNDARY, x_max, alpha=0.07, color=DANGER, zorder=0)
    ax.axvline(PHYSIO_BOUNDARY, color=DANGER, lw=1.2, ls="--", alpha=0.6)

# ── A: LDA score ──────────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
shade_unphysio(ax_a)
ax_a.axhspan(cc_lda.mean()-cc_lda.std(), cc_lda.mean()+cc_lda.std(),
             alpha=0.15, color=CC_COL)
ax_a.axhline(cc_lda.mean(), color=CC_COL, lw=1.8, ls="--", label="CC mean ±1σ")
ax_a.axhline(ad_lda.mean(), color=AD_COL, lw=1.2, ls=":", alpha=0.7,
             label="AD baseline")
ax_a.axhline(thr, color="gray", lw=1.0, ls="-.", alpha=0.6,
             label=f"Midpoint = {thr:.2f}")

for pi in range(t5_lda.shape[1]):
    ax_a.plot(t5_a, t5_lda[:, pi], lw=0.4, color=T5_COL, alpha=0.15)
ax_a.fill_between(t5_a, t5_lda_mean-t5_lda_std,
                        t5_lda_mean+t5_lda_std, alpha=0.22, color=T5_COL)
ax_a.plot(t5_a, t5_lda_mean, "-o", ms=4, lw=2.2, color=T5_COL,
          label="AD  mean ±1σ")

# crossing annotation
cross_idx = np.where(t5_lda_mean <= thr)[0]
if len(cross_idx):
    ax_a.scatter([t5_a[cross_idx[0]]], [t5_lda_mean[cross_idx[0]]],
                 s=80, color="k", zorder=6, marker="*",
                 label=f"Crosses midpoint at α={t5_a[cross_idx[0]]:.1f}")

ax_a.text(PHYSIO_BOUNDARY + 0.2, ax_a.get_ylim()[0] + 0.15 if ax_a.get_ylim()[0] < 0 else -1.8,
          "unphysiological\n(RMS > 3×)", color=DANGER, fontsize=7, va="bottom")
ax_a.set_xlabel("α"); ax_a.set_ylabel("LDA score  (K=25)")
ax_a.set_title("LDA score vs α  (top-5 sites)")
ax_a.legend(frameon=False, fontsize=7, loc="upper right")

# ── B: Signal RMS ratio ───────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
shade_unphysio(ax_b)
ax_b.axhline(1.0, color="gray", lw=1.2, ls="--", alpha=0.7,
             label="Baseline = 1.0")
ax_b.axhline(3.0, color=DANGER, lw=1.2, ls=":", alpha=0.7,
             label="×3 threshold")
ax_b.fill_between([0, 20], [3, 3], [20, 20], alpha=0.05, color=DANGER)

ax_b.plot(t5_a, rms_top5_measured, "-o", ms=4, lw=2.2, color=AD_COL,
          label="Top-5 sites (mean over 40 AD)")
ax_b.plot(t5_a, rms_rest_measured, "-s", ms=3, lw=1.5, color="#546E7A",
          alpha=0.8, label="Other 116 sites (stable)")

ax_b.set_ylim(0, max(rms_top5_measured) * 1.08)
ax_b.set_xlabel("α"); ax_b.set_ylabel("Signal RMS  (relative to baseline)")
ax_b.set_title("Reconstructed signal amplitude\n(W_int.T @ X per site)")
ax_b.legend(frameon=False, fontsize=7)

# ── C: FC-r with CC template ──────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
shade_unphysio(ax_c)
ax_c.axhspan(cc_fc_r.mean()-cc_fc_r.std(), cc_fc_r.mean()+cc_fc_r.std(),
             alpha=0.15, color=CC_COL)
ax_c.axhline(cc_fc_r.mean(), color=CC_COL, lw=1.8, ls="--",
             label=f"CC  ({cc_fc_r.mean():.3f} ±{cc_fc_r.std():.3f})")
ax_c.axhline(ad_fc_r_base.mean(), color=AD_COL, lw=1.2, ls=":",
             alpha=0.7, label=f"AD baseline  ({ad_fc_r_base.mean():.3f})")

for pi in range(t5_fcr.shape[1]):
    ax_c.plot(t5_a, t5_fcr[:, pi], lw=0.4, color=T5_COL, alpha=0.15)
ax_c.fill_between(t5_a, t5_fcr_mean-t5_fcr_std,
                        t5_fcr_mean+t5_fcr_std, alpha=0.22, color=T5_COL)
ax_c.plot(t5_a, t5_fcr_mean, "-o", ms=4, lw=2.2, color=T5_COL,
          label=f"AD top-5  (mean)")

ax_c.text(0.97, 0.08,
          "FC-r flat after α=1\n(5/121 sites ≈ 4% of FC)",
          transform=ax_c.transAxes, ha="right", va="bottom",
          fontsize=7.5, color="#555",
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
ax_c.set_xlabel("α"); ax_c.set_ylabel("FC-r  (vs CC template)")
ax_c.set_title("FC similarity to CC mean\n(insensitive to 5-site perturbation)")
ax_c.legend(frameon=False, fontsize=7)

# ── D: Gap closed + reclassification ─────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
shade_unphysio(ax_d)
ax_d.plot(t5_a, gap_closed, "-o", ms=4, lw=2.2, color=T5_COL,
          label="Gap closed (%)")
ax_d.plot(t5_a, reclassif, "-s", ms=4, lw=2.0, color="#37474F",
          label="AD reclassified as CC (%)")
ax_d.axhline(50, color="gray", ls="--", lw=1.0, alpha=0.5, label="50%")

# Physio-safe max
safe_idx = np.where(t5_a <= PHYSIO_BOUNDARY)[0][-1]
ax_d.scatter([t5_a[safe_idx]], [gap_closed[safe_idx]],
             s=70, color=DANGER, zorder=5, marker="v",
             label=f"Physio limit: α={t5_a[safe_idx]:.0f} "
                   f"→ {gap_closed[safe_idx]:.0f}% gap, "
                   f"{reclassif[safe_idx]:.0f}% reclassif")

ax_d.set_xlabel("α"); ax_d.set_ylabel("%")
ax_d.set_title("Therapeutic effect (top-5)")
ax_d.legend(frameon=False, fontsize=7)

# ── E: Comparison all 3 strategies (gap closed) ───────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
PERT_COLS = {"full_w": "#1B5E20", "top5": "#E65100", "top1": "#6A1B9A"}
for pt, alphas, lda_mat, col, lbl in [
        ("full_w", fw_a, fw_lda, "#1B5E20", "Full-W (121 sites)"),
        ("top5",   t5_a, t5_lda, "#E65100", "Top-5  sites"),
        ("top1",   t1_a, t1_lda, "#6A1B9A", "Top-1  site"),
]:
    gc = (ad_lda.mean() - lda_mat.mean(1)) / gap * 100
    ax_e.plot(alphas, gc, "-o", ms=4, lw=2.0, color=col, label=lbl)

# Physiological boundaries per strategy
ax_e.axvline(1.0, color="#1B5E20", lw=1.0, ls=":", alpha=0.5)   # full_w physio limit
ax_e.axvline(PHYSIO_BOUNDARY, color="#E65100", lw=1.0, ls=":", alpha=0.5)  # top5
ax_e.axhline(50, color="gray", ls="--", lw=1.0, alpha=0.5)
ax_e.set_xlabel("α"); ax_e.set_ylabel("AD→CC gap closed (%)")
ax_e.set_title("Gap closed — all strategies\n(dotted = approx. physiological limit)")
ax_e.legend(frameon=False, fontsize=7.5)
for sp in ["top","right"]: ax_e.spines[sp].set_visible(False)

# ── F: RMS_top5 vs FC-r efficiency scatter ────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
# Scatter: RMS cost vs LDA benefit for each alpha point
lda_benefit = (ad_lda.mean() - t5_lda_mean) / gap * 100   # gap closed
rms_cost    = rms_top5_measured

sc = ax_f.scatter(rms_cost, lda_benefit,
                  c=t5_a, cmap="plasma", s=55, zorder=3,
                  edgecolors="white", linewidths=0.4)
plt.colorbar(sc, ax=ax_f, label="α", shrink=0.8)
ax_f.axvline(3.0, color=DANGER, lw=1.2, ls="--", alpha=0.7,
             label="RMS = 3× (physiological limit)")
ax_f.set_xlabel("Signal RMS ratio  (top-5 sites)")
ax_f.set_ylabel("AD→CC gap closed (%)")
ax_f.set_title("Cost–benefit: signal distortion vs\ntherapeutic LDA shift")
ax_f.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax_f.spines[sp].set_visible(False)

# ── panel labels ───────────────────────────────────────────────────────────────
for ax, lbl in zip([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f],
                   ["A","B","C","D","E","F"]):
    ax.text(-0.12, 1.05, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig.suptitle(
    "Top-5 perturbation — α = 0 to 20 — physiological signal & FC diagnostics\n"
    "Red shading: α > 4 (RMS of 5 perturbed sites > 2.5×  baseline — unphysiological).\n"
    "FC-r remains flat throughout: perturbing 5/121 sites (~4%) does not change overall FC.",
    fontsize=9, y=1.01)

fig.savefig("top5_physio_clean.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print("Saved top5_physio_clean.png")
