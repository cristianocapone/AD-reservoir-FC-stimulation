
#!/usr/bin/env python3
"""
Resting-state fMRI analysis — AD vs MCI
Figures produced:
  analysis/fig1_bold_samples.png      – sample BOLD timeseries (6 parcels × 4 sessions)
  analysis/fig2_pca_timeseries.png    – PC1/PC2 of timeseries + parcel loadings
  analysis/fig3_fc_examples.png       – 6 example FC matrices (3 AD, 3 MCI)
  analysis/fig4_fc_group_means.png    – mean AD / mean MCI / difference FC
  analysis/fig5_latent_space.png      – 2-D PCA + t-SNE of FC vectors, AD vs MCI
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
TS_DIR   = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries")
OUT_DIR  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\analysis")
OUT_DIR.mkdir(exist_ok=True)

# ── Colours ───────────────────────────────────────────────────────────────────
COL = {"AD": "#d62728", "MCI": "#1f77b4"}

# ── Parcel labels & network assignment ───────────────────────────────────────
def load_labels():
    lines = (TS_DIR / "parcel_labels.txt").read_text().splitlines()
    labels = [l.split("  ", 1)[1].strip() for l in lines if l.strip()]
    labels = [l for l in labels if l != "Background"]
    return labels

NET_ORDER  = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_COLORS = {"Vis": "#1f77b4", "SomMot": "#ff7f0e", "DorsAttn": "#2ca02c",
              "SalVentAttn": "#d62728", "Limbic": "#9467bd",
              "Cont": "#8c564b", "Default": "#e377c2",
              "Subcortical": "#7f7f7f"}

def get_net(lbl):
    for n in NET_ORDER:
        if n in lbl:
            return n
    return "Subcortical"

# ── Load data ─────────────────────────────────────────────────────────────────
def load_group(group, require_n=None):
    files = sorted((TS_DIR / group).glob("*.npy"))
    data, names = [], []
    for f in files:
        ts = np.load(f)   # N × T
        if ts.ndim != 2 or ts.shape[0] < 100:
            continue
        if require_n is not None and ts.shape[0] != require_n:
            continue
        data.append(ts)
        names.append(f.stem)
    return data, names

REQUIRE_N = 114   # keep only sessions with this exact parcel count

print("Loading AD timeseries …")
ad_ts, ad_names = load_group("AD", require_n=REQUIRE_N)
print(f"  {len(ad_ts)} sessions (N={REQUIRE_N})")
print("Loading MCI timeseries …")
mci_ts, mci_names = load_group("MCI", require_n=REQUIRE_N)
print(f"  {len(mci_ts)} sessions (N={REQUIRE_N})")

all_labels_full = load_labels()
N = ad_ts[0].shape[0]               # actual number of parcels (114)
parcel_labels = all_labels_full[:N]  # trim to match

net_assign = [get_net(l) for l in parcel_labels]

# Network sort order for FC plots
sort_key  = lambda i: (NET_ORDER.index(net_assign[i])
                        if net_assign[i] in NET_ORDER else len(NET_ORDER), i)
sorted_idx = sorted(range(N), key=sort_key)

# ── FC computation ────────────────────────────────────────────────────────────
def fc_matrix(ts):
    """114×114 Pearson r from N×T matrix."""
    return np.corrcoef(ts)

print("Computing FC matrices …")
ad_fc  = [fc_matrix(ts) for ts in ad_ts]
mci_fc = [fc_matrix(ts) for ts in mci_ts]

# Upper-triangle vectors for embedding
def triu_vec(fc):
    idx = np.triu_indices(fc.shape[0], k=1)
    return fc[idx]

ad_vecs  = np.array([triu_vec(fc) for fc in ad_fc])   # (n_AD, D)
mci_vecs = np.array([triu_vec(fc) for fc in mci_fc])  # (n_MCI, D)
all_vecs = np.vstack([ad_vecs, mci_vecs])
all_grp  = (["AD"] * len(ad_vecs)) + (["MCI"] * len(mci_vecs))
all_cols = [COL[g] for g in all_grp]

# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 – Sample BOLD timeseries
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 1: BOLD samples …")

# Pick 6 parcels spanning networks
sel_nets = ["Vis", "SomMot", "DorsAttn", "Default", "Hippocampus/Subcortical", "Limbic"]
sel_parcels = []
for net in ["Vis", "SomMot", "DorsAttn", "Default", "Subcortical", "Limbic"]:
    idxs = [i for i, n in enumerate(net_assign) if n == net]
    if idxs:
        sel_parcels.append(idxs[0])
while len(sel_parcels) < 6:
    sel_parcels.append(len(sel_parcels))

# 4 sessions: 2 AD, 2 MCI
sample_sessions = [("AD", 0, ad_ts[0], ad_names[0]),
                   ("AD", 1, ad_ts[len(ad_ts)//3], ad_names[len(ad_ts)//3]),
                   ("MCI", 0, mci_ts[0], mci_names[0]),
                   ("MCI", 1, mci_ts[len(mci_ts)//3], mci_names[len(mci_ts)//3])]

fig, axes = plt.subplots(4, 6, figsize=(20, 10), sharey="row")
fig.suptitle("Sample BOLD timeseries — 6 parcels × 4 sessions", fontsize=13, fontweight="bold")

for row, (grp, _, ts, name) in enumerate(sample_sessions):
    T = ts.shape[1]
    t = np.arange(T) * 3.0   # TR=3s
    for col, pi in enumerate(sel_parcels):
        ax = axes[row, col]
        ax.plot(t, ts[pi], lw=0.7, color=COL[grp], alpha=0.9)
        ax.axhline(0, color="k", lw=0.4, ls="--")
        ax.set_xlim(0, t[-1])
        if row == 0:
            net = net_assign[pi]
            short = parcel_labels[pi].replace("7Networks_", "").replace("LH_", "L-").replace("RH_", "R-")
            ax.set_title(f"{short[:22]}", fontsize=7)
        if col == 0:
            ax.set_ylabel(f"{grp}\n{name[:18]}\n(z)", fontsize=6)
        if row == 3:
            ax.set_xlabel("Time (s)", fontsize=6)
        ax.tick_params(labelsize=5)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig1_bold_samples.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig1_bold_samples.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 – PCA of timeseries (per session)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 2: PCA of timeseries …")

fig = plt.figure(figsize=(20, 12))
fig.suptitle("PCA of BOLD timeseries — PC1 & PC2 timecourses + parcel loadings",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.35)

# 3 AD + 3 MCI sessions
pca_sessions = [(ad_ts[i], ad_names[i], "AD") for i in [0, len(ad_ts)//3, 2*len(ad_ts)//3]] + \
               [(mci_ts[i], mci_names[i], "MCI") for i in [0, len(mci_ts)//3, 2*len(mci_ts)//3]]

for si, (ts, name, grp) in enumerate(pca_sessions):
    row, col = divmod(si, 3)
    T = ts.shape[1]
    t = np.arange(T) * 3.0

    pca = PCA(n_components=2)
    scores = pca.fit_transform(ts.T)   # T × 2
    ev = pca.explained_variance_ratio_

    ax = fig.add_subplot(gs[row * 2, col])
    ax.plot(t, scores[:, 0], color=COL[grp], lw=0.8, label=f"PC1 ({ev[0]*100:.1f}%)")
    ax.plot(t, scores[:, 1], color=COL[grp], lw=0.8, alpha=0.5,
            ls="--", label=f"PC2 ({ev[1]*100:.1f}%)")
    ax.set_title(f"{grp}: {name[:22]}", fontsize=7, color=COL[grp])
    ax.legend(fontsize=6, loc="upper right")
    ax.set_xlabel("Time (s)", fontsize=6)
    ax.tick_params(labelsize=5)

    # Parcel loadings bar (PC1)
    ax2 = fig.add_subplot(gs[row * 2 + 1, col])
    loadings = pca.components_[0]  # N
    bar_cols = [NET_COLORS.get(net_assign[i], "gray") for i in range(N)]
    ax2.bar(range(N), loadings, color=bar_cols, width=1.0, edgecolor="none")
    ax2.axhline(0, color="k", lw=0.4)
    ax2.set_xlabel("Parcel index", fontsize=6)
    ax2.set_ylabel("PC1 loading", fontsize=6)
    ax2.tick_params(labelsize=5)

# Legend for networks
from matplotlib.patches import Patch
handles = [Patch(color=c, label=n) for n, c in NET_COLORS.items()]
fig.legend(handles=handles, loc="lower right", fontsize=7, ncol=2,
           title="Network", title_fontsize=7)

fig.savefig(OUT_DIR / "fig2_pca_timeseries.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig2_pca_timeseries.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 – Example FC matrices (3 AD + 3 MCI)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 3: Example FC matrices …")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Functional Connectivity matrices — Schaefer-100 + subcortical\n(sorted by network)",
             fontsize=13, fontweight="bold")

ex_sessions = [(ad_fc[i], ad_names[i], "AD")
               for i in [0, len(ad_fc)//3, 2*len(ad_fc)//3]] + \
              [(mci_fc[i], mci_names[i], "MCI")
               for i in [0, len(mci_fc)//3, 2*len(mci_fc)//3]]

sorted_nets = [net_assign[i] for i in sorted_idx]

def net_boundaries(sorted_nets):
    bounds = []
    prev = sorted_nets[0]
    for i, n in enumerate(sorted_nets):
        if n != prev:
            bounds.append(i - 0.5)
            prev = n
    return bounds

bounds = net_boundaries(sorted_nets)

for ax, (fc, name, grp) in zip(axes.flat, ex_sessions):
    fc_s = fc[np.ix_(sorted_idx, sorted_idx)]
    im = ax.imshow(fc_s, vmin=-0.8, vmax=0.8, cmap="RdBu_r",
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.75, label="r")
    for b in bounds:
        ax.axhline(b, color="white", lw=0.6)
        ax.axvline(b, color="white", lw=0.6)
    ax.set_title(f"{grp}: {name[:30]}", fontsize=8, color=COL[grp])
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(OUT_DIR / "fig3_fc_examples.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig3_fc_examples.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 – Group-mean FC matrices + difference
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 4: Group-mean FC …")

mean_ad_fc  = np.mean(ad_fc, axis=0)
mean_mci_fc = np.mean(mci_fc, axis=0)
diff_fc     = mean_mci_fc - mean_ad_fc

mean_ad_s   = mean_ad_fc[np.ix_(sorted_idx, sorted_idx)]
mean_mci_s  = mean_mci_fc[np.ix_(sorted_idx, sorted_idx)]
diff_s      = diff_fc[np.ix_(sorted_idx, sorted_idx)]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Group-mean FC matrices (network-sorted)", fontsize=13, fontweight="bold")

panels = [(mean_ad_s,  f"Mean AD  (n={len(ad_fc)})",  "RdBu_r", -0.6, 0.6),
          (mean_mci_s, f"Mean MCI (n={len(mci_fc)})", "RdBu_r", -0.6, 0.6),
          (diff_s,     "MCI − AD",                    "PuOr",   -0.15, 0.15)]

for ax, (mat, title, cmap, vmin, vmax) in zip(axes, panels):
    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap,
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
    for b in bounds:
        ax.axhline(b, color="white", lw=0.5)
        ax.axvline(b, color="white", lw=0.5)
    # network labels on diagonal
    prev_b = 0
    for b_end in bounds + [N]:
        mid = (prev_b + b_end) / 2
        net = sorted_nets[int(mid)]
        ax.text(mid, mid, net[:6], ha="center", va="center", fontsize=5.5,
                fontweight="bold", color="white",
                bbox=dict(fc=NET_COLORS.get(net, "gray"), ec="none", alpha=0.7, pad=1))
        prev_b = b_end
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(OUT_DIR / "fig4_fc_group_means.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig4_fc_group_means.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 – 2-D latent space (PCA + t-SNE)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 5: Latent space …")

scaler = StandardScaler()
X = scaler.fit_transform(all_vecs)
labels_arr = np.array(all_grp)

# PCA
pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X)

# t-SNE
print("  Running t-SNE (may take ~1 min) …")
tsne = TSNE(n_components=2, perplexity=min(30, len(X)//4),
            random_state=42, max_iter=1000)
X_tsne = tsne.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("2-D Latent Space of FC matrices — AD vs MCI",
             fontsize=13, fontweight="bold")

for ax, (Xemb, title) in zip(axes, [
        (X_pca,  f"PCA  (PC1={pca2.explained_variance_ratio_[0]*100:.1f}%, "
                 f"PC2={pca2.explained_variance_ratio_[1]*100:.1f}%)"),
        (X_tsne, "t-SNE")]):
    for grp in ["AD", "MCI"]:
        mask = labels_arr == grp
        ax.scatter(Xemb[mask, 0], Xemb[mask, 1],
                   c=COL[grp], label=f"{grp} (n={mask.sum()})",
                   alpha=0.7, s=35, edgecolors="white", linewidths=0.3)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_xlabel("Dim 1"); ax.set_ylabel("Dim 2")
    ax.grid(True, alpha=0.2)

# Inset: PCA scree
ax_inset = axes[0].inset_axes([0.67, 0.67, 0.31, 0.3])
pca_full = PCA(n_components=20).fit(X)
ax_inset.bar(range(1, 21), pca_full.explained_variance_ratio_ * 100,
             color="#aec7e8", edgecolor="k", linewidth=0.4)
ax_inset.set_xlabel("PC", fontsize=6); ax_inset.set_ylabel("Var %", fontsize=6)
ax_inset.tick_params(labelsize=5)
ax_inset.set_title("Scree", fontsize=6)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig5_latent_space.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig5_latent_space.png")

# ─────────────────────────────────────────────────────────────────────────────
# Summary stats
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
idx_u = np.triu_indices(N, k=1)
ad_mean_fc  = np.mean([fc[idx_u] for fc in ad_fc])
mci_mean_fc = np.mean([fc[idx_u] for fc in mci_fc])
ad_std_fc   = np.std([fc[idx_u].mean() for fc in ad_fc])
mci_std_fc  = np.std([fc[idx_u].mean() for fc in mci_fc])
print(f"Mean FC (off-diag)  AD : {ad_mean_fc:.4f}  (session SD={ad_std_fc:.4f})")
print(f"Mean FC (off-diag)  MCI: {mci_mean_fc:.4f}  (session SD={mci_std_fc:.4f})")
print(f"PCA var explained PC1+PC2: {pca2.explained_variance_ratio_[:2].sum()*100:.1f}%")
print(f"Output: {OUT_DIR}")
print("=" * 55)
