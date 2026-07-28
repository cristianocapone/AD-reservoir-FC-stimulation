"""
figure1_data.py
===============
Figure 1 — Dataset overview.
Extracted from summary_figures.ipynb (cells 1-4 + Fig1 cell).

Panels (2 rows x 4 cols):
  A  Acquisition table         B  Sample BOLD timeseries (2 cols)   C  Mean FC violin
  D  FC matrix — CC            E  FC matrix — AD                     F  Population PCA scree
  G  FC similarity to CC mean

Saves: figure1_data.png  figure1_data.pdf  (300 DPI)
"""
import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings; warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
TR          = 3.0
trial_dur   = 139
N_PC_MODEL  = 50
TIMES_SKIP  = 10
TS_ROOT     = "../timeseries"
OUT_DIR     = "."

CC_COL = "#2196F3"
AD_COL = "#E91E63"

matplotlib.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "lines.linewidth": 1.8,
    "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.dpi": 300,
})

def _tag(ax, ltr, x=-0.16, y=1.09):
    ax.text(x, y, ltr, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# ── data loading ──────────────────────────────────────────────────────────────
print("Loading signals ...")
rng = np.random.default_rng(RNG_SEED)
collected_signals, identifiers = [], []

for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    if not os.path.isdir(folder):
        print(f"WARNING: {folder} not found"); continue
    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)),
                                replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= trial_dur:
            collected_signals.append(arr)
            identifiers.append([label, fname.split("_ses-")[0]])

identifiers  = np.array(identifiers, dtype=object)
state_labels = np.array([0 if r[0] == "CC" else 1 for r in identifiers])
ctrl_indices = np.where(state_labels == 0)[0]
ad_indices   = np.where(state_labels == 1)[0]
sigs = [s.T for s in collected_signals]
print(f"  {len(sigs)} sessions  (CC={len(ctrl_indices)}, AD={len(ad_indices)})")

# ── population PCA ────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in sigs], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
order        = np.argsort(evals)[::-1]
evecs        = evecs[:, order]; evals_sorted = evals[order]
expl_var     = evals_sorted / evals_sorted.sum()
ev50         = evecs[:, :N_PC_MODEL]

# ── per-session FC ─────────────────────────────────────────────────────────────
print("Computing FC matrices ...")
FC_collected = []
for sig in sigs:
    proj = (sig.T @ ev50 @ ev50.T).T
    fc   = np.nan_to_num(np.corrcoef(proj))
    FC_collected.append(fc)

fc_ctrl_mean     = np.mean([FC_collected[i] for i in ctrl_indices], axis=0)
fc_ctrl_flat_vec = fc_ctrl_mean[np.triu_indices(N_SITES, k=1)]

# ── atlas ordering ────────────────────────────────────────────────────────────
print("Loading atlas ...")
try:
    from nilearn import datasets as nl_datasets
    schaefer   = nl_datasets.fetch_atlas_schaefer_2018(n_rois=100,
                                                        yeo_networks=7,
                                                        resolution_mm=2)
    atlas_lbls = [l.decode() if isinstance(l, bytes) else l
                  for l in schaefer.labels]
    NET_ORDER  = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
    NET_COLORS_MAP = dict(Vis="#1f77b4", SomMot="#ff7f0e", DorsAttn="#2ca02c",
                          SalVentAttn="#d62728", Limbic="#9467bd",
                          Cont="#8c564b", Default="#e377c2")
    def _get_net(lbl):
        for n in NET_ORDER:
            if n in lbl: return n
        return "Subcortical"
    net_assign  = [_get_net(l) for l in atlas_lbls]
    sorted_idx  = sorted(range(100),
                         key=lambda i: (NET_ORDER.index(net_assign[i])
                                        if net_assign[i] in NET_ORDER else 7, i))
    sorted_nets = [net_assign[i] for i in sorted_idx]
    net_bounds  = [i - 0.5 for i in range(1, 100)
                   if sorted_nets[i] != sorted_nets[i-1]]
    have_atlas  = True
    print("  Atlas loaded.")
except Exception as e:
    print(f"  Atlas unavailable ({e}), using identity ordering")
    sorted_idx = list(range(100)); net_bounds = []
    NET_COLORS_MAP = {}; have_atlas = False

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering figure ...")
PSIZ = 3.5
fig1 = plt.figure(figsize=(PSIZ*4 + 0.8, PSIZ*2 + 0.8), facecolor="white")
gs1  = gridspec.GridSpec(2, 4, figure=fig1,
                         hspace=0.48, wspace=0.40,
                         top=0.94, bottom=0.06, left=0.07, right=0.97)

# ── A: Acquisition table ──────────────────────────────────────────────────────
ax_a = fig1.add_subplot(gs1[0, 0]); ax_a.axis("off")
rows_tbl = [
    ["TR",        "3.0 s"],
    ["Volumes",   "140 (~7 min)"],
    ["Atlas",     "Schaefer-100\n+ HO-21 sub-ctx"],
    ["Parcels",   "121"],
    ["Confounds", "24-HMP + WM + CSF"],
    ["Bandpass",  "0.01-0.10 Hz"],
    ["CC (CN)",   f"{len(ctrl_indices)} sessions"],
    ["AD",        f"{len(ad_indices)} sessions"],
]
tbl = ax_a.table(cellText=rows_tbl, colLabels=["Parameter", "Value"],
                 cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0: cell.set_facecolor("#e0e0e0")
_tag(ax_a, "A"); ax_a.set_title("Acquisition setup", pad=4)

# ── B: Sample BOLD timeseries (spans 2 cols) ──────────────────────────────────
ax_b = fig1.add_subplot(gs1[0, 1:3])
t_ax = np.arange(trial_dur) * TR
parcels_show = [48, 0, 37, 103]
pnames       = ["pCunPCC (Default)", "LH Vis-1", "Default Temp", "L-Thalamus"]
ex_cc = sigs[ctrl_indices[0]]; ex_ad = sigs[ad_indices[0]]
for ri, (pi, pn) in enumerate(zip(parcels_show, pnames)):
    off = ri * 5
    ax_b.plot(t_ax, ex_cc[pi, :trial_dur] + off, lw=1.0, color=CC_COL,
              alpha=0.9, label="CC" if ri == 0 else "")
    ax_b.plot(t_ax, ex_ad[pi, :trial_dur] + off, lw=1.0, color=AD_COL,
              alpha=0.9, label="AD" if ri == 0 else "")
    ax_b.text(-10, off, pn, fontsize=7, va="center", ha="right")
ax_b.set_xlim(-18, t_ax[-1]+5); ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("BOLD + offset"); ax_b.set_yticks([])
ax_b.legend(fontsize=8, loc="upper right")
_tag(ax_b, "B"); _clean(ax_b); ax_b.set_title("Sample BOLD timeseries", pad=4)

# ── C: Mean FC violin ─────────────────────────────────────────────────────────
ax_c = fig1.add_subplot(gs1[0, 3])
fmc_cc = [FC_collected[i][np.triu_indices(N_SITES, k=1)].mean()
          for i in ctrl_indices]
fmc_ad = [FC_collected[i][np.triu_indices(N_SITES, k=1)].mean()
          for i in ad_indices]
vp = ax_c.violinplot([fmc_cc, fmc_ad], positions=[0, 1], showmedians=True)
vp["bodies"][0].set_facecolor(CC_COL); vp["bodies"][1].set_facecolor(AD_COL)
for kv in ["cbars", "cmins", "cmaxes", "cmedians"]:
    vp[kv].set_color("k"); vp[kv].set_linewidth(1.2)
ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(["CC", "AD"])
ax_c.set_ylabel("Mean FC (Pearson r)")
_tag(ax_c, "C"); _clean(ax_c); ax_c.set_title("Mean FC per session", pad=4)

# ── D / E: FC matrices ────────────────────────────────────────────────────────
for col_i, (gi, gname, gcol) in enumerate([
        (ctrl_indices[0], "CC", CC_COL),
        (ad_indices[0],   "AD", AD_COL)]):
    ax = fig1.add_subplot(gs1[1, col_i])
    fc_s = FC_collected[gi][:100, :100][np.ix_(sorted_idx, sorted_idx)]
    im   = ax.imshow(fc_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03, fraction=0.046)
    for b in net_bounds:
        ax.axhline(b, color="white", lw=0.5)
        ax.axvline(b, color="white", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Brain region"); ax.set_ylabel("Brain region")
    _tag(ax, "DE"[col_i]); ax.set_title(f"FC matrix — {gname}", color=gcol, pad=4)

# ── F: Population PCA scree ───────────────────────────────────────────────────
ax_f = fig1.add_subplot(gs1[1, 2])
ax_f.bar(range(1, 21), expl_var[:20]*100,
         color="#455A64", edgecolor="white", lw=0.5)
ax_f.set_xlabel("Principal component")
ax_f.set_ylabel("Explained variance (%)")
ax_f.set_xticks([1, 5, 10, 15, 20])
_tag(ax_f, "F"); _clean(ax_f); ax_f.set_title("Population PCA scree", pad=4)

# ── G: FC similarity to CC mean ───────────────────────────────────────────────
ax_g = fig1.add_subplot(gs1[1, 3])
iu = np.triu_indices(N_SITES, k=1)
cc_corr_cc = [np.corrcoef(FC_collected[i][iu], fc_ctrl_flat_vec)[0, 1]
              for i in ctrl_indices]
cc_corr_ad = [np.corrcoef(FC_collected[i][iu], fc_ctrl_flat_vec)[0, 1]
              for i in ad_indices]
allv  = cc_corr_cc + cc_corr_ad
bins_g = np.linspace(min(allv)-0.02, max(allv)+0.02, 26)
ax_g.hist(cc_corr_cc, bins=bins_g, alpha=0.65, color=CC_COL,
          label=f"CC  med={np.median(cc_corr_cc):.3f}")
ax_g.hist(cc_corr_ad, bins=bins_g, alpha=0.65, color=AD_COL,
          label=f"AD  med={np.median(cc_corr_ad):.3f}")
ax_g.axvline(np.median(cc_corr_cc), color=CC_COL, lw=1.5, ls="--")
ax_g.axvline(np.median(cc_corr_ad), color=AD_COL, lw=1.5, ls="--")
ax_g.set_xlabel("FC corr. to CC mean"); ax_g.set_ylabel("Count")
ax_g.legend(frameon=False)
_tag(ax_g, "G"); _clean(ax_g); ax_g.set_title("FC similarity to CC mean", pad=4)

# ── network legend ────────────────────────────────────────────────────────────
if have_atlas:
    handles = [Patch(color=c, label=n) for n, c in NET_COLORS_MAP.items()]
    fig1.legend(handles=handles, loc="lower center", ncol=7, fontsize=7,
                bbox_to_anchor=(0.5, -0.01),
                title="Schaefer-100 networks", title_fontsize=8)

# ── save PNG + PDF ────────────────────────────────────────────────────────────
for ext in ("png", "pdf"):
    out = f"figure1_data.{ext}"
    fig1.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
plt.close()
