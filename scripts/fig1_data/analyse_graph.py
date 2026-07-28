#!/usr/bin/env python3
"""
Graph-theoretic analysis of FC networks — AD vs MCI.

Metrics computed per session (on proportionally thresholded, weighted graph):
  - Node strength            (N-dim)  : sum of weights per node
  - Weighted clustering coef (N-dim)  : Onnela et al. formula
  - Global efficiency        (scalar) : 1 / mean shortest path (binarised)
  - Local efficiency         (N-dim)  : efficiency within each node's neighbourhood
  - Betweenness centrality   (N-dim)  : fraction of shortest paths through node
  - Small-worldness sigma    (scalar) : (C/C_rand) / (L/L_rand)

Figures:
  analysis/fig10_graph_group.png      -- mean strength/clustering/efficiency per network
  analysis/fig11_hub_maps.png         -- hub score brain maps (strength x betweenness)
  analysis/fig12_graph_classify.png   -- classification using graph features vs FC
  analysis/fig13_smallworld.png       -- small-worldness distribution AD vs MCI
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
TS_DIR    = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries")
OUT_DIR   = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\analysis")
OUT_DIR.mkdir(exist_ok=True)

REQUIRE_N    = 114
DENSITY      = 0.20   # keep top 20% of edges (proportional threshold)
N_RAND       = 20     # random networks for small-worldness null

COL = {"AD": "#d62728", "MCI": "#1f77b4"}
NET_ORDER  = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default","Subcortical"]
NET_COLORS = {"Vis":"#1f77b4","SomMot":"#ff7f0e","DorsAttn":"#2ca02c",
              "SalVentAttn":"#d62728","Limbic":"#9467bd","Cont":"#8c564b",
              "Default":"#e377c2","Subcortical":"#7f7f7f"}

# ── Labels ────────────────────────────────────────────────────────────────────
def load_labels():
    lines = (TS_DIR / "parcel_labels.txt").read_text().splitlines()
    lbl = [l.split("  ",1)[1].strip() for l in lines if l.strip()]
    return [l for l in lbl if l != "Background"][:REQUIRE_N]

def get_net(lbl):
    for n in NET_ORDER[:-1]:
        if n in lbl: return n
    return "Subcortical"

parcel_labels = load_labels()
net_assign    = [get_net(l) for l in parcel_labels]
N = len(parcel_labels)

# ── Load FC ───────────────────────────────────────────────────────────────────
def load_group(group):
    data, names = [], []
    for f in sorted((TS_DIR / group).glob("*.npy")):
        ts = np.load(f)
        if ts.shape[0] != REQUIRE_N: continue
        fc = np.corrcoef(ts)
        data.append(fc); names.append(f.stem)
    return data, names

print("Loading FC matrices ...")
ad_fc,  ad_names  = load_group("AD")
mci_fc, mci_names = load_group("MCI")
print(f"  AD={len(ad_fc)}  MCI={len(mci_fc)}")

# ── Graph construction ────────────────────────────────────────────────────────
def threshold_fc(fc, density=DENSITY):
    """Keep top-density fraction of positive edges; set diagonal to 0."""
    W = fc.copy(); np.fill_diagonal(W, 0)
    W[W < 0] = 0
    thresh = np.percentile(W[W > 0], 100 * (1 - density))
    W[W < thresh] = 0
    return W

# ── Graph metrics (numpy/scipy, fast) ────────────────────────────────────────
def node_strength(W):
    return W.sum(axis=1)

def weighted_clustering(W):
    """Onnela et al. (2005) weighted clustering coefficient."""
    W_norm = W / W.max()
    W_cube = np.cbrt(W_norm)
    deg    = (W > 0).sum(axis=1).astype(float)
    tri    = np.diag(W_cube @ W_cube @ W_cube)
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = np.where(deg > 1, tri / (deg * (deg - 1)), 0.0)
    return cc

def global_efficiency(W):
    """Global efficiency on binarised graph (faster than weighted)."""
    B = (W > 0).astype(float)
    G = nx.from_numpy_array(B)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    paths = dict(nx.all_pairs_shortest_path_length(G))
    n = G.number_of_nodes()
    if n < 2: return 0.0
    total = sum(1/d for u in paths for v, d in paths[u].items() if d > 0)
    return total / (n * (n - 1))

def local_efficiency(W):
    """Local efficiency: efficiency of each node's neighbourhood subgraph."""
    B   = (W > 0).astype(float)
    eff = np.zeros(N)
    for i in range(N):
        nbrs = np.where(B[i] > 0)[0]
        if len(nbrs) < 2:
            continue
        sub = B[np.ix_(nbrs, nbrs)]
        G_sub = nx.from_numpy_array(sub)
        n_sub = len(nbrs)
        if not nx.is_connected(G_sub):
            G_sub = G_sub.subgraph(max(nx.connected_components(G_sub), key=len)).copy()
            n_sub = G_sub.number_of_nodes()
        if n_sub < 2: continue
        paths = dict(nx.all_pairs_shortest_path_length(G_sub))
        total = sum(1/d for u in paths for v, d in paths[u].items() if d > 0)
        eff[i] = total / (n_sub * (n_sub - 1))
    return eff

def betweenness(W):
    """Betweenness centrality (normalised) on binarised graph."""
    B = (W > 0).astype(float)
    G = nx.from_numpy_array(B)
    bc = nx.betweenness_centrality(G, normalized=True)
    return np.array([bc[i] for i in range(N)])

def small_worldness(W, n_rand=N_RAND):
    """
    sigma = (C/C_rand) / (L/L_rand)
    C_rand, L_rand estimated from degree-preserving random networks.
    """
    B = (W > 0).astype(int)
    G = nx.from_numpy_array(B.astype(float))
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()

    C = nx.average_clustering(G)
    if nx.is_connected(G) and G.number_of_nodes() > 1:
        L = nx.average_shortest_path_length(G)
    else:
        return np.nan

    C_rands, L_rands = [], []
    for _ in range(n_rand):
        Gr = nx.random_reference(G, niter=5, seed=None)
        C_rands.append(nx.average_clustering(Gr))
        if nx.is_connected(Gr):
            L_rands.append(nx.average_shortest_path_length(Gr))

    if not C_rands or not L_rands: return np.nan
    C_rand = np.mean(C_rands); L_rand = np.mean(L_rands)
    if C_rand == 0 or L_rand == 0: return np.nan
    return (C / C_rand) / (L / L_rand)

# ── Compute metrics for all sessions ─────────────────────────────────────────
def compute_metrics(fc_list, names, group):
    records = []
    for i, (fc, name) in enumerate(zip(fc_list, names)):
        if (i+1) % 20 == 0:
            print(f"  [{group}] {i+1}/{len(fc_list)} ...")
        W = threshold_fc(fc)
        rec = {
            "name":    name,
            "group":   group,
            "strength":  node_strength(W),
            "clustering": weighted_clustering(W),
            "local_eff":  local_efficiency(W),
            "betweenness": betweenness(W),
        }
        records.append(rec)
    return records

print("\nComputing graph metrics (this takes a few minutes) ...")
print("  AD ...")
ad_recs  = compute_metrics(ad_fc,  ad_names,  "AD")
print("  MCI ...")
mci_recs = compute_metrics(mci_fc, mci_names, "MCI")
all_recs = ad_recs + mci_recs
groups   = np.array([r["group"] for r in all_recs])

# Per-network mean for each metric
def net_mean(arr):
    return {net: arr[[i for i,n in enumerate(net_assign) if n==net]].mean()
            for net in NET_ORDER}

# ── Small-worldness (slower — subset only) ───────────────────────────────────
print("\nComputing small-worldness (sample of 30 sessions) ...")
sample_idx = np.random.default_rng(42).choice(len(all_recs), size=min(30, len(all_recs)), replace=False)
sw_vals, sw_groups = [], []
for idx in sample_idx:
    r = all_recs[idx]
    grp = r["group"]
    fc  = (ad_fc + mci_fc)[idx]
    W   = threshold_fc(fc)
    sw  = small_worldness(W, n_rand=10)
    sw_vals.append(sw); sw_groups.append(grp)
    if not np.isnan(sw):
        print(f"  {r['name'][:30]:30s} [{grp}]  sigma={sw:.3f}")

sw_vals   = np.array(sw_vals, dtype=float)
sw_groups = np.array(sw_groups)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 – Group mean per network for strength / clustering / local_eff
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 10 ...")
metrics_plot = ["strength", "clustering", "local_eff", "betweenness"]
metric_labels = ["Node Strength", "Clustering Coef.", "Local Efficiency", "Betweenness Centrality"]

fig, axes = plt.subplots(len(metrics_plot), 1, figsize=(14, 4*len(metrics_plot)))
fig.suptitle("Graph metrics per network — AD vs MCI (mean +/- SE)",
             fontsize=13, fontweight="bold")

for ax, metric, mlabel in zip(axes, metrics_plot, metric_labels):
    x = np.arange(len(NET_ORDER))
    w = 0.35
    for gi, grp in enumerate(["AD","MCI"]):
        mask = groups == grp
        recs = [r for r in all_recs if r["group"] == grp]
        net_vals = np.array([[r[metric][[i for i,n in enumerate(net_assign) if n==net]].mean()
                              for net in NET_ORDER] for r in recs])
        mu  = net_vals.mean(axis=0)
        se  = net_vals.std(axis=0) / np.sqrt(len(recs))
        bars = ax.bar(x + gi*w - w/2, mu, w, color=COL[grp],
                      label=f"{grp} (n={len(recs)})", alpha=0.85)
        ax.errorbar(x + gi*w - w/2, mu, yerr=se, fmt="none",
                    color="k", capsize=3, lw=1.0)

    # t-test per network
    for ni, net in enumerate(NET_ORDER):
        ad_v  = np.array([r[metric][[i for i,n in enumerate(net_assign) if n==net]].mean()
                          for r in ad_recs])
        mci_v = np.array([r[metric][[i for i,n in enumerate(net_assign) if n==net]].mean()
                          for r in mci_recs])
        _, p = stats.ttest_ind(ad_v, mci_v)
        if p < 0.05:
            ymax = max(ad_v.mean(), mci_v.mean())
            ax.text(ni, ymax * 1.05, "*" if p < 0.05 else "",
                    ha="center", fontsize=11, color="k")

    ax.set_xticks(x); ax.set_xticklabels(NET_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(mlabel, fontsize=9); ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig10_graph_group.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig10_graph_group.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 11 – Hub maps: strength × betweenness, difference AD vs MCI
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 11 ...")
def group_mean_metric(recs, metric):
    return np.mean([r[metric] for r in recs], axis=0)

ad_str  = group_mean_metric(ad_recs,  "strength")
mci_str = group_mean_metric(mci_recs, "strength")
ad_btw  = group_mean_metric(ad_recs,  "betweenness")
mci_btw = group_mean_metric(mci_recs, "betweenness")

# Hub score = z(strength) * z(betweenness)
def hub_score(strength, btw):
    zs = (strength - strength.mean()) / strength.std()
    zb = (btw     - btw.mean())      / btw.std()
    return zs * zb

ad_hub  = hub_score(ad_str,  ad_btw)
mci_hub = hub_score(mci_str, mci_btw)
diff_hub = mci_hub - ad_hub

sorted_idx = sorted(range(N), key=lambda i: (
    NET_ORDER.index(net_assign[i]) if net_assign[i] in NET_ORDER else len(NET_ORDER), i))
net_colors_arr = [NET_COLORS.get(net_assign[i], "gray") for i in range(N)]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Hub scores (z-strength x z-betweenness) per parcel", fontsize=12, fontweight="bold")

for ax, vals, title in [(axes[0], ad_hub,  f"AD (n={len(ad_recs)})"),
                         (axes[1], mci_hub, f"MCI (n={len(mci_recs)})"),
                         (axes[2], diff_hub, "MCI - AD")]:
    cols = [NET_COLORS.get(net_assign[i], "gray") for i in range(N)]
    sort_v = vals[sorted_idx]
    sort_c = [cols[i] for i in sorted_idx]
    ax.bar(range(N), sort_v, color=sort_c, edgecolor="none", width=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Parcel (network sorted)", fontsize=8)
    ax.set_ylabel("Hub score", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=7)

from matplotlib.patches import Patch
handles = [Patch(color=c, label=n) for n, c in NET_COLORS.items()]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
           title="Network", title_fontsize=7, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout()
fig.savefig(OUT_DIR / "fig11_hub_maps.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig11_hub_maps.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 12 – Classification: graph features vs raw FC
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 12 ...")

# Build feature matrix: per-network mean of each metric
def build_graph_features(recs):
    rows = []
    for r in recs:
        row = []
        for metric in ["strength","clustering","local_eff","betweenness"]:
            for net in NET_ORDER:
                idx = [i for i,n in enumerate(net_assign) if n==net]
                row.append(r[metric][idx].mean())
        rows.append(row)
    return np.array(rows)

X_graph = build_graph_features(all_recs)
y        = (groups == "AD").astype(int)
cv5      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scaler_g = StandardScaler()
X_g_sc   = scaler_g.fit_transform(X_graph)
score_graph = cross_val_score(LogisticRegression(max_iter=1000), X_g_sc, y,
                               cv=cv5, scoring="balanced_accuracy")

# FC triu for comparison
all_fc = ad_fc + mci_fc
X_fc   = np.array([fc[np.triu_indices(N,k=1)] for fc in all_fc])
scaler_fc = StandardScaler()
X_fc_sc   = scaler_fc.fit_transform(X_fc)
score_fc  = cross_val_score(LinearSVC(C=0.01, max_iter=5000), X_fc_sc, y,
                             cv=cv5, scoring="balanced_accuracy")

# Combined
X_comb  = np.hstack([X_g_sc, X_fc_sc])
score_comb = cross_val_score(LinearSVC(C=0.01, max_iter=5000), X_comb, y,
                              cv=cv5, scoring="balanced_accuracy")

fig, ax = plt.subplots(figsize=(8, 5))
fig.suptitle("Classification balanced accuracy (5-fold CV)\nAD vs MCI",
             fontsize=12, fontweight="bold")

methods = ["FC only\n(LinearSVC)", "Graph features\n(LogReg)", "FC + Graph\n(LinearSVC)"]
scores  = [score_fc, score_graph, score_comb]
colors  = ["#aec7e8", "#ffbb78", "#98df8a"]
for xi, (method, sc, col) in enumerate(zip(methods, scores, colors)):
    ax.bar(xi, sc.mean(), color=col, edgecolor="k", lw=0.8, width=0.5,
           label=f"{sc.mean():.3f} +/- {sc.std():.3f}")
    ax.errorbar(xi, sc.mean(), yerr=sc.std(), fmt="none",
                color="k", capsize=6, lw=1.5)
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="Chance")
ax.set_xticks(range(len(methods))); ax.set_xticklabels(methods, fontsize=10)
ax.set_ylabel("Balanced accuracy", fontsize=10)
ax.set_ylim(0.4, 0.85)
ax.legend(fontsize=8, title="Mean +/- SD", loc="upper left")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig.savefig(OUT_DIR / "fig12_graph_classify.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig12_graph_classify.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 13 – Small-worldness distribution
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 13 ...")
fig, ax = plt.subplots(figsize=(7, 5))
fig.suptitle("Small-worldness (sigma) distribution — AD vs MCI",
             fontsize=12, fontweight="bold")

for grp in ["AD","MCI"]:
    mask = (sw_groups == grp) & ~np.isnan(sw_vals)
    v    = sw_vals[mask]
    if len(v) == 0: continue
    ax.hist(v, bins=10, alpha=0.6, color=COL[grp], label=f"{grp} (n={len(v)}, mean={v.mean():.2f})")
    ax.axvline(v.mean(), color=COL[grp], lw=2, ls="--")

ax.axvline(1.0, color="k", lw=1, ls=":", label="sigma=1 (random)")
ax.set_xlabel("Small-worldness sigma", fontsize=10)
ax.set_ylabel("Count", fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

ad_sw  = sw_vals[(sw_groups=="AD")  & ~np.isnan(sw_vals)]
mci_sw = sw_vals[(sw_groups=="MCI") & ~np.isnan(sw_vals)]
if len(ad_sw) >= 2 and len(mci_sw) >= 2:
    _, p = stats.ttest_ind(ad_sw, mci_sw)
    ax.set_title(f"t-test p={p:.3f}", fontsize=10)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig13_smallworld.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig13_smallworld.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("GRAPH ANALYSIS SUMMARY")
print("="*60)
print(f"  FC only          : {score_fc.mean():.3f} +/- {score_fc.std():.3f}")
print(f"  Graph features   : {score_graph.mean():.3f} +/- {score_graph.std():.3f}")
print(f"  FC + Graph       : {score_comb.mean():.3f} +/- {score_comb.std():.3f}")
print(f"  Small-worldness  : AD={ad_sw.mean():.2f}  MCI={mci_sw.mean():.2f}" if len(ad_sw)>0 and len(mci_sw)>0 else "")
print("="*60)
