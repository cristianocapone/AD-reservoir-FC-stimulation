#!/usr/bin/env python3
"""
Longitudinal FC trajectory analysis — AD vs MCI
Each subject's FC is regressed against time; slope = rate of change.

Clean subjects (unambiguous 1 individual per output label):
  AD : sub-002 (4 ses), sub-013 (10 ses), sub-031 (4 ses), sub-100 (9 ses)
  MCI: all subjects with ≥3 sessions

Figures:
  analysis/fig6_trajectories.png      – FC summary metric over time per subject
  analysis/fig7_slope_comparison.png  – per-subject slope features, AD vs MCI
  analysis/fig8_edge_slopes.png       – FC edge-level slopes (mean per group)
  analysis/fig9_longitudinal_latent.png – 2-D PCA of slope vectors
"""

from pathlib import Path
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
TS_DIR  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries")
OUT_DIR = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\analysis")
OUT_DIR.mkdir(exist_ok=True)

COL     = {"AD": "#d62728", "MCI": "#1f77b4"}
REQUIRE_N = 114

# Clean AD subjects (1 individual per fmriprep output label)
CLEAN_AD = {"sub-002", "sub-013", "sub-031", "sub-100"}

# ── Network definitions ───────────────────────────────────────────────────────
NET_ORDER = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default","Subcortical"]

def get_net(lbl):
    for n in NET_ORDER[:-1]:
        if n in lbl:
            return n
    return "Subcortical"

def load_labels():
    lines = (TS_DIR / "parcel_labels.txt").read_text().splitlines()
    labels = [l.split("  ", 1)[1].strip() for l in lines if l.strip()]
    return [l for l in labels if l != "Background"][:REQUIRE_N]

parcel_labels = load_labels()
net_assign    = [get_net(l) for l in parcel_labels]
N = len(parcel_labels)

# ── Parse filenames ───────────────────────────────────────────────────────────
DATE_RE = re.compile(r"ses-(\d{8})")
SUB_RE  = re.compile(r"^(sub-[^_]+)")

def parse_file(f: Path):
    m_date = DATE_RE.search(f.stem)
    m_sub  = SUB_RE.match(f.stem)
    if not m_date or not m_sub:
        return None
    date = datetime.strptime(m_date.group(1), "%Y%m%d")
    return m_sub.group(1), date

# ── Load and group ────────────────────────────────────────────────────────────
def load_group_longitudinal(group, clean_set=None):
    """Return dict: subject_id -> sorted list of (date, ts array)."""
    subj_dict = {}
    for f in sorted((TS_DIR / group).glob("*.npy")):
        parsed = parse_file(f)
        if parsed is None:
            continue
        sub, date = parsed
        if clean_set is not None and sub not in clean_set:
            continue
        ts = np.load(f)
        if ts.shape[0] != REQUIRE_N:
            continue
        subj_dict.setdefault(sub, []).append((date, ts))
    # sort each subject's sessions by date
    for sub in subj_dict:
        subj_dict[sub].sort(key=lambda x: x[0])
    return subj_dict

print("Loading data …")
ad_long  = load_group_longitudinal("AD",  clean_set=CLEAN_AD)
mci_long = load_group_longitudinal("MCI")
# Keep only MCI subjects with ≥3 sessions
mci_long = {s: v for s, v in mci_long.items() if len(v) >= 3}

print(f"  AD  clean subjects: {len(ad_long)}  "
      f"({sum(len(v) for v in ad_long.values())} sessions)")
print(f"  MCI subjects >=3 ses: {len(mci_long)}  "
      f"({sum(len(v) for v in mci_long.values())} sessions)")

# ── FC + summary metrics per session ─────────────────────────────────────────
def fc_matrix(ts):
    return np.corrcoef(ts)          # N×N

def triu_vec(fc):
    idx = np.triu_indices(N, k=1)
    return fc[idx]

def within_net_fc(fc):
    """Mean FC within each network."""
    vals = {}
    for net in NET_ORDER:
        idx = [i for i, n in enumerate(net_assign) if n == net]
        if len(idx) < 2:
            continue
        pairs = [(i, j) for ii, i in enumerate(idx)
                          for j in idx[ii+1:]]
        vals[net] = np.mean([fc[i, j] for i, j in pairs])
    return vals

def subject_trajectory(sessions):
    """
    sessions: list of (date, ts)
    Returns:
      t_months  : array of time in months from first scan
      fc_list   : list of N×N FC matrices
      global_fc : mean off-diag FC per session
      net_fc    : dict net -> array of mean FC per session
    """
    dates   = [s[0] for s in sessions]
    t0      = dates[0]
    t_months = np.array([(d - t0).days / 30.44 for d in dates])
    fc_list  = [fc_matrix(s[1]) for s in sessions]
    global_fc = np.array([triu_vec(fc).mean() for fc in fc_list])
    net_vals  = {net: [] for net in NET_ORDER}
    for fc in fc_list:
        for net, val in within_net_fc(fc).items():
            net_vals[net].append(val)
    net_fc = {net: np.array(v) for net, v in net_vals.items() if v}
    return t_months, fc_list, global_fc, net_fc

# ── Compute slopes ────────────────────────────────────────────────────────────
print("Computing trajectories and slopes …")

def lin_slope(t, y):
    if len(t) < 2:
        return np.nan
    slope, *_ = np.polyfit(t, y, 1)
    return slope

subjects, groups, global_slopes, net_slopes, edge_slopes = [], [], [], [], []

for group, long_dict in [("AD", ad_long), ("MCI", mci_long)]:
    for sub, sessions in sorted(long_dict.items()):
        t, fc_list, gfc, nfc = subject_trajectory(sessions)

        g_slope   = lin_slope(t, gfc)
        n_slopes  = {net: lin_slope(t, nfc[net]) for net in nfc}

        # Edge-level slopes
        edge_mat = np.stack([triu_vec(fc) for fc in fc_list], axis=0)  # T×D
        if len(t) >= 2:
            e_slopes = np.array([lin_slope(t, edge_mat[:, d])
                                 for d in range(edge_mat.shape[1])])
        else:
            e_slopes = np.zeros(edge_mat.shape[1])

        subjects.append(sub)
        groups.append(group)
        global_slopes.append(g_slope)
        net_slopes.append(n_slopes)
        edge_slopes.append(e_slopes)

        print(f"  [{group}] {sub:20s}  {len(sessions)} ses  "
              f"global_slope={g_slope:+.5f}/mo  "
              f"DMN={n_slopes.get('Default', np.nan):+.5f}/mo")

groups_arr = np.array(groups)

# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 – Trajectory plots
# ─────────────────────────────────────────────────────────────────────────────
print("\nFigure 6: Trajectories …")
all_subjects = list(ad_long.items()) + list(mci_long.items())
n_subs = len(all_subjects)
ncols  = 4
nrows  = int(np.ceil(n_subs / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3.2), sharey=False)
fig.suptitle("Longitudinal FC trajectories — global mean FC over time",
             fontsize=13, fontweight="bold")
axes_list = axes.flatten().tolist()

for ax_idx, ((sub, sessions), grp) in enumerate(
        zip(all_subjects, ["AD"]*len(ad_long) + ["MCI"]*len(mci_long))):
    ax = axes_list[ax_idx]
    t, fc_list, gfc, nfc = subject_trajectory(sessions)
    col = COL[grp]

    # Global FC
    ax.plot(t, gfc, "o-", color=col, lw=1.5, ms=5, label="Global FC")

    # Overlay a few networks
    for net, ls in [("Default", "--"), ("Vis", ":")]:
        if net in nfc:
            ax.plot(t, nfc[net], ls, color=col, lw=0.9, alpha=0.6, label=net)

    # Regression line
    if len(t) >= 2:
        slope, intercept = np.polyfit(t, gfc, 1)
        t_fit = np.linspace(t[0], t[-1], 50)
        ax.plot(t_fit, slope * t_fit + intercept, "-", color="k",
                lw=1.0, alpha=0.5)

    ax.set_title(f"{grp}: {sub[4:]}\n"
                 f"slope={lin_slope(t,gfc):+.5f}/mo  n={len(sessions)}",
                 fontsize=7, color=col)
    ax.set_xlabel("Months since baseline", fontsize=6)
    ax.set_ylabel("Mean FC (r)", fontsize=6)
    ax.tick_params(labelsize=5)
    if ax_idx == 0:
        ax.legend(fontsize=5)

# Hide unused axes
for ax in axes_list[n_subs:]:
    ax.set_visible(False)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig6_trajectories.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig6_trajectories.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 – Slope comparison per network
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 7: Slope comparison …")

nets_present = [n for n in NET_ORDER if any(n in ns for ns in net_slopes)]
n_nets = len(nets_present) + 1  # +1 for global

fig, axes = plt.subplots(1, n_nets, figsize=(3 * n_nets, 5))
fig.suptitle("FC slope (Δr/month) by network — AD vs MCI",
             fontsize=12, fontweight="bold")

all_net_data = {"Global": global_slopes}
for net in nets_present:
    all_net_data[net] = [ns.get(net, np.nan) for ns in net_slopes]

for ax, (net_name, vals) in zip(axes, all_net_data.items()):
    vals_arr = np.array(vals, dtype=float)
    for gi, grp in enumerate(["AD", "MCI"]):
        mask = groups_arr == grp
        v    = vals_arr[mask]
        v    = v[~np.isnan(v)]
        x    = gi + np.random.RandomState(42).uniform(-0.15, 0.15, len(v))
        ax.scatter(x, v, color=COL[grp], alpha=0.8, s=40, zorder=3)
        ax.plot([gi - 0.2, gi + 0.2], [v.mean(), v.mean()],
                color=COL[grp], lw=2.5, zorder=4)
        ax.errorbar(gi, v.mean(), yerr=v.std() / np.sqrt(len(v)),
                    fmt="none", color=COL[grp], capsize=4, lw=1.5)

    # t-test
    ad_v  = vals_arr[groups_arr == "AD"]
    mci_v = vals_arr[groups_arr == "MCI"]
    ad_v  = ad_v[~np.isnan(ad_v)]
    mci_v = mci_v[~np.isnan(mci_v)]
    if len(ad_v) >= 2 and len(mci_v) >= 2:
        t, p = stats.ttest_ind(ad_v, mci_v)
        ax.set_title(f"{net_name}\np={p:.3f}", fontsize=8)
    else:
        ax.set_title(net_name, fontsize=8)

    ax.axhline(0, color="k", lw=0.6, ls="--")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["AD", "MCI"], fontsize=9)
    ax.set_ylabel("Slope (Δr/month)", fontsize=7)
    ax.tick_params(labelsize=6)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig7_slope_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig7_slope_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 – Edge-level slope maps (mean per group)
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 8: Edge-level slope matrices …")

edge_arr = np.array(edge_slopes)   # n_subjects × D
ad_mask  = groups_arr == "AD"
mci_mask = groups_arr == "MCI"

mean_ad_slope  = edge_arr[ad_mask].mean(axis=0)
mean_mci_slope = edge_arr[mci_mask].mean(axis=0)
diff_slope     = mean_mci_slope - mean_ad_slope

def vec_to_mat(vec):
    mat = np.zeros((N, N))
    idx = np.triu_indices(N, k=1)
    mat[idx] = vec
    mat = mat + mat.T
    return mat

sorted_idx = sorted(range(N), key=lambda i: (
    NET_ORDER.index(net_assign[i]) if net_assign[i] in NET_ORDER else len(NET_ORDER), i))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Edge-level FC slope (Δr/month) — network sorted",
             fontsize=12, fontweight="bold")

panels = [
    (vec_to_mat(mean_ad_slope),  f"Mean slope — AD (n={ad_mask.sum()})"),
    (vec_to_mat(mean_mci_slope), f"Mean slope — MCI (n={mci_mask.sum()})"),
    (vec_to_mat(diff_slope),     "Difference (MCI − AD)"),
]

for ax, (mat, title) in zip(axes, panels):
    mat_s = mat[np.ix_(sorted_idx, sorted_idx)]
    vlim  = np.percentile(np.abs(mat_s), 97)
    im = ax.imshow(mat_s, vmin=-vlim, vmax=vlim, cmap="RdBu_r",
                   aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.8, label="Δr/month")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

plt.tight_layout()
fig.savefig(OUT_DIR / "fig8_edge_slopes.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig8_edge_slopes.png")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 – 2-D PCA of slope vectors + LOO classification
# ─────────────────────────────────────────────────────────────────────────────
print("Figure 9: Latent space of slopes …")

X_slopes = edge_arr
y_slopes  = (groups_arr == "AD").astype(int)

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X_slopes)

pca2 = PCA(n_components=2)
X_pca = pca2.fit_transform(X_sc)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("2-D PCA of FC slope vectors — AD vs MCI",
             fontsize=12, fontweight="bold")

ax = axes[0]
for grp in ["AD", "MCI"]:
    mask = groups_arr == grp
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               c=COL[grp], s=80, alpha=0.85,
               edgecolors="white", linewidths=0.5,
               label=f"{grp} (n={mask.sum()})")
    for i in np.where(mask)[0]:
        ax.annotate(subjects[i][4:], (X_pca[i, 0], X_pca[i, 1]),
                    fontsize=6, alpha=0.7)
ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]*100:.1f}%)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]*100:.1f}%)")
ax.legend(fontsize=9)
ax.set_title("Subject slope vectors in 2-D", fontsize=10)
ax.grid(True, alpha=0.2)

# LOO classification on global + network slopes
feature_names = ["Global"] + nets_present
X_feat = np.column_stack([
    global_slopes,
    *[[ns.get(net, 0) for ns in net_slopes] for net in nets_present]
])
X_feat_sc = StandardScaler().fit_transform(X_feat)

loo  = LeaveOneOut()
lr   = LogisticRegression(max_iter=1000)
preds, trues = [], []
for train_idx, test_idx in loo.split(X_feat_sc):
    lr.fit(X_feat_sc[train_idx], y_slopes[train_idx])
    preds.append(lr.predict(X_feat_sc[test_idx])[0])
    trues.append(y_slopes[test_idx][0])

preds = np.array(preds); trues = np.array(trues)
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
bac = balanced_accuracy_score(trues, preds)
cm  = confusion_matrix(trues, preds)

ax2 = axes[1]
im  = ax2.imshow(cm, cmap="Blues")
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(["AD", "MCI"]); ax2.set_yticklabels(["AD", "MCI"])
ax2.set_xlabel("Predicted"); ax2.set_ylabel("True")
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(cm[i, j]), ha="center", va="center",
                 fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")
ax2.set_title(f"LOO-CV Confusion Matrix\n"
              f"Balanced accuracy = {bac:.3f}  (chance=0.500)\n"
              f"Features: global + per-network slopes", fontsize=9)

plt.tight_layout()
fig.savefig(OUT_DIR / "fig9_longitudinal_latent.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved fig9_longitudinal_latent.png")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("LONGITUDINAL ANALYSIS SUMMARY")
print("=" * 60)
ad_gs   = np.array(global_slopes)[ad_mask]
mci_gs  = np.array(global_slopes)[mci_mask]
t_stat, p_val = stats.ttest_ind(ad_gs, mci_gs)
print(f"  Global FC slope  AD : {ad_gs.mean():+.5f} +/- {ad_gs.std():.5f} dr/month")
print(f"  Global FC slope  MCI: {mci_gs.mean():+.5f} +/- {mci_gs.std():.5f} dr/month")
print(f"  t-test p-value       : {p_val:.4f}")
print(f"  LOO balanced accuracy: {bac:.3f}  (n={len(subjects)} subjects)")
print("=" * 60)
