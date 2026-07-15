"""
pertB_comparison_figure.py
==========================
Builds a 3-panel comparison figure:
  Row 1 — LDA score vs alpha for all 3 perturbation types (shared y-axis)
  Row 2 — Top-5 and Top-1 site distributions across AD subjects

Requires the condition-B W matrices, so we re-run the TF pass + W fitting.
LDA trajectories are loaded from pertB_direct_data.npz.
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── hyper-parameters ───────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
SIGMA       = 0.05
SR          = 0.95
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "top1":   np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
}

# ── load pre-computed LDA trajectories ────────────────────────────────────────
print("Loading LDA results from pertB_direct_data.npz ...")
d       = np.load(f"{OUT_DIR}/pertB_direct_data.npz", allow_pickle=True)
cc_lda  = d["cc_lda"]
ad_lda  = d["ad_lda"]
Z_base  = d["Z_base"]
cc_fc_r = d["cc_fc_r"]

lda_curves = {pt: (d[f"{pt}_alphas"], d[f"{pt}_lda"]) for pt in ALPHA_GRIDS}
fcr_curves = {pt: (d[f"{pt}_alphas"], d[f"{pt}_fcr"]) for pt in ALPHA_GRIDS}

# ── re-run data loading + TF + W fitting to get W matrices ────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, pid_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            pid_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

pid_raw    = np.array(pid_raw)
labels_raw = np.array(labels_raw)
N_subj     = len(signals)
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_pids = [unique_pids[i] for i in np.where(patient_labels == 0)[0]]
ad_pids = [unique_pids[i] for i in np.where(patient_labels == 1)[0]]
print(f"  {N_patients} patients ({len(cc_pids)} CC, {len(ad_pids)} AD)")

print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

print("TF pass ...")
sess_X, sess_Y = {}, {}
for idx in trange(N_subj, desc="  TF"):
    s     = signals[idx]; T_s = s.shape[1]
    tgt   = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    Xraw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        Xraw.append(res.X.copy())
    Xf          = np.array(Xraw)[TIMES_SKIP:]
    sess_X[idx] = Xf
    sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T

first_idx   = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single = {pid: sess_X[first_idx[pid]] for pid in unique_pids}
patY_single = {pid: sess_Y[first_idx[pid]] for pid in unique_pids}

print("Fitting W ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc   # (N_hidden, N_sites)

W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

# ── compute per-patient column norms of ΔW ────────────────────────────────────
print("\nComputing ΔW column norms ...")
# site_norms[pid] = ||W_cc - W_p||_col  shape (N_sites,)
site_norms = {}
for pid in ad_pids:
    dW = W_cc_mean - pat_W[pid]
    site_norms[pid] = np.linalg.norm(dW, axis=0)   # (N_sites,)

# Mean norm across all AD patients (shows overall AD-CC difference per site)
mean_norm  = np.mean([site_norms[pid] for pid in ad_pids], axis=0)
std_norm   = np.std( [site_norms[pid] for pid in ad_pids], axis=0)

# Top-5 and top-1 per patient
top5_indices = {pid: np.argsort(site_norms[pid])[::-1][:5]  for pid in ad_pids}
top1_indices = {pid: np.argsort(site_norms[pid])[::-1][0]   for pid in ad_pids}

# Frequency: how many patients put each site in their top-5 / top-1?
top5_count = np.zeros(N_SITES, dtype=int)
top1_count = np.zeros(N_SITES, dtype=int)
for pid in ad_pids:
    for s in top5_indices[pid]:
        top5_count[s] += 1
    top1_count[top1_indices[pid]] += 1

# Sort sites by top5 frequency (for ranked display)
top5_rank_order = np.argsort(top5_count)[::-1]
top1_rank_order = np.argsort(top1_count)[::-1]

# Per-patient top5 membership matrix (N_ad × N_sites) — 1 if in top5 for that patient
top5_matrix = np.zeros((len(ad_pids), N_SITES), dtype=float)
for pi, pid in enumerate(ad_pids):
    top5_matrix[pi, top5_indices[pid]] = 1.0

# Rank of each site per patient (1=highest norm, 121=lowest)
rank_matrix = np.zeros((len(ad_pids), N_SITES), dtype=float)
for pi, pid in enumerate(ad_pids):
    order = np.argsort(site_norms[pid])[::-1]   # site index sorted by norm desc
    ranks = np.empty_like(order); ranks[order] = np.arange(1, N_SITES+1)
    rank_matrix[pi] = ranks

print(f"  Top-1 site overall: site {top1_rank_order[0]} "
      f"(chosen by {top1_count[top1_rank_order[0]]}/{len(ad_pids)} patients)")
print(f"  Top-5 most frequent sites: {top5_rank_order[:5].tolist()} "
      f"(counts: {top5_count[top5_rank_order[:5]].tolist()})")

# ── figure ─────────────────────────────────────────────────────────────────────
print("\nBuilding comparison figure ...")

COL_CC    = "#2196F3"
COL_AD    = "#E91E63"
COL_FW    = "#1B5E20"   # full_w
COL_T5    = "#E65100"   # top5
COL_T1    = "#4A148C"   # top1
ALPHA_IND = 0.20        # individual line opacity

PERT_COLORS = {"full_w": COL_FW, "top5": COL_T5, "top1": COL_T1}
PERT_LABELS_SHORT = {
    "full_w": "Full-W (all 121 sites)",
    "top5":   "Top-5 sites",
    "top1":   "Top-1 site",
}

fig = plt.figure(figsize=(22, 18), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                         height_ratios=[1.1, 1.1, 1.0],
                         hspace=0.42, wspace=0.32)

# ─────────────────────────────────────────────────────────────────────────────
# Row 0-1, col 0: combined LDA comparison (all 3 types, mean ± std)
# ─────────────────────────────────────────────────────────────────────────────
ax_cmp = fig.add_subplot(gs[0:2, 0])

# CC band
ax_cmp.axhspan(cc_lda.mean() - cc_lda.std(), cc_lda.mean() + cc_lda.std(),
               alpha=0.18, color=COL_CC)
ax_cmp.axhline(cc_lda.mean(), color=COL_CC, lw=2, ls="--", label="CC mean ±1σ")

# AD baseline band
ax_cmp.axhspan(ad_lda.mean() - ad_lda.std(), ad_lda.mean() + ad_lda.std(),
               alpha=0.12, color=COL_AD)
ax_cmp.axhline(ad_lda.mean(), color=COL_AD, lw=1.5, ls=":", alpha=0.7, label="AD baseline")

# Midpoint reference
mid = 0.5*(cc_lda.mean() + ad_lda.mean())
ax_cmp.axhline(mid, color="gray", lw=0.8, ls="-.", alpha=0.5, label="midpoint")

# Trajectories — normalize x to "fraction of W changed"
# full_w: changes all 121 sites by α  → effective fraction = α (0→2)
# top5:   changes 5 sites by α        → we plot as-is for clarity
# top1:   changes 1 site by α         → we plot as-is
for pt, col in PERT_COLORS.items():
    alphas, lda_mat = lda_curves[pt]
    mean_t = lda_mat.mean(1)
    std_t  = lda_mat.std(1)
    # scale x so max α maps to [0, 1] for top5/top1 to see them on same axis
    ax_cmp.fill_between(alphas, mean_t - std_t, mean_t + std_t,
                        alpha=0.20, color=col)
    ax_cmp.plot(alphas, mean_t, "-o", ms=5, lw=2.2, color=col,
                label=f"{PERT_LABELS_SHORT[pt]}")

ax_cmp.set_xlabel("alpha", fontsize=10)
ax_cmp.set_ylabel("Mean LDA score (Cond. B, K=25)", fontsize=10)
ax_cmp.set_title("LDA score vs perturbation strength\n(mean ± 1σ over 40 AD patients)",
                 fontsize=10)
ax_cmp.legend(fontsize=8, frameon=False, loc="upper right")
for sp in ["top","right"]: ax_cmp.spines[sp].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# Row 0, col 1-2: per-type LDA with individual patient lines (shared y)
# ─────────────────────────────────────────────────────────────────────────────
y_min = min(cc_lda.mean() - 1.5*cc_lda.std(), ad_lda.mean() - 1.5*ad_lda.std()) - 0.5
y_max = ad_lda.mean() + 1.5*ad_lda.std() + 0.5

for ci, pt in enumerate(["top5", "top1"]):
    ax = fig.add_subplot(gs[0, ci+1])
    col = PERT_COLORS[pt]
    alphas, lda_mat = lda_curves[pt]

    ax.axhspan(cc_lda.mean() - cc_lda.std(), cc_lda.mean() + cc_lda.std(),
               alpha=0.18, color=COL_CC)
    ax.axhline(cc_lda.mean(), color=COL_CC, lw=1.5, ls="--")
    ax.axhline(ad_lda.mean(), color=COL_AD, lw=1.5, ls=":", alpha=0.6)
    ax.axhline(mid, color="gray", lw=0.8, ls="-.", alpha=0.5)

    for pi in range(lda_mat.shape[1]):
        ax.plot(alphas, lda_mat[:, pi], "-", lw=0.7, color=col, alpha=ALPHA_IND)
    mean_t = lda_mat.mean(1); std_t = lda_mat.std(1)
    ax.fill_between(alphas, mean_t-std_t, mean_t+std_t, alpha=0.25, color=col)
    ax.plot(alphas, mean_t, "-o", ms=5, lw=2.2, color=col, label="AD mean ±1σ")

    ax.set_ylim(y_min, y_max)
    ax.set_title(f"{PERT_LABELS_SHORT[pt]}", fontsize=10)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("LDA score", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

# also show full_w with individuals (reuse gs[1, 1])
ax_fw = fig.add_subplot(gs[1, 1])
col   = PERT_COLORS["full_w"]
alphas, lda_mat = lda_curves["full_w"]
ax_fw.axhspan(cc_lda.mean() - cc_lda.std(), cc_lda.mean() + cc_lda.std(),
              alpha=0.18, color=COL_CC)
ax_fw.axhline(cc_lda.mean(), color=COL_CC, lw=1.5, ls="--")
ax_fw.axhline(ad_lda.mean(), color=COL_AD, lw=1.5, ls=":", alpha=0.6)
ax_fw.axhline(mid, color="gray", lw=0.8, ls="-.", alpha=0.5)
for pi in range(lda_mat.shape[1]):
    ax_fw.plot(alphas, lda_mat[:, pi], "-", lw=0.7, color=col, alpha=ALPHA_IND)
mean_t = lda_mat.mean(1); std_t = lda_mat.std(1)
ax_fw.fill_between(alphas, mean_t-std_t, mean_t+std_t, alpha=0.25, color=col)
ax_fw.plot(alphas, mean_t, "-o", ms=5, lw=2.2, color=col, label="AD mean ±1σ")
ax_fw.set_ylim(y_min, y_max)
ax_fw.set_title("Full-W (all 121 sites) — individual patients", fontsize=10)
ax_fw.set_xlabel("alpha", fontsize=9); ax_fw.set_ylabel("LDA score", fontsize=9)
ax_fw.legend(fontsize=7, frameon=False)
for sp in ["top","right"]: ax_fw.spines[sp].set_visible(False)

# FC-r comparison (gs[1, 2])
ax_fc = fig.add_subplot(gs[1, 2])
ax_fc.axhspan(cc_fc_r.mean() - cc_fc_r.std(), cc_fc_r.mean() + cc_fc_r.std(),
              alpha=0.18, color=COL_CC, label="CC ±1σ")
ax_fc.axhline(cc_fc_r.mean(), color=COL_CC, lw=1.5, ls="--")
for pt, col in PERT_COLORS.items():
    alphas, fcr_mat = fcr_curves[pt]
    ax_fc.plot(alphas, fcr_mat.mean(1), "-o", ms=4, lw=1.8, color=col,
               label=PERT_LABELS_SHORT[pt])
ax_fc.set_xlabel("alpha", fontsize=9)
ax_fc.set_ylabel("FC-r (W_int.T @ X) vs CC", fontsize=9)
ax_fc.set_title("FC similarity to CC template", fontsize=9)
ax_fc.legend(fontsize=7, frameon=False, loc="lower left")
for sp in ["top","right"]: ax_fc.spines[sp].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# Row 2, col 0-1: Top-5 site frequency over subjects
# ─────────────────────────────────────────────────────────────────────────────
ax_t5 = fig.add_subplot(gs[2, 0:2])

# Mean ΔW norm (background bars, light gray)
sort_sites = np.argsort(mean_norm)[::-1]   # sites sorted by mean ΔW norm
x_pos      = np.arange(N_SITES)

# bar chart: top5_count per site, sorted by count descending
counts_sorted  = top5_count[sort_sites]
norms_sorted   = mean_norm[sort_sites]

# color by whether the count is above a threshold
bar_cols = [COL_T5 if c >= 3 else "#FFCCBC" for c in counts_sorted]
bars = ax_t5.bar(x_pos, counts_sorted, color=bar_cols, edgecolor="none", width=0.85)

# overlay normalised mean ΔW norm as a line (right y-axis)
ax_t5b = ax_t5.twinx()
ax_t5b.plot(x_pos, norms_sorted / norms_sorted.max(), "-", lw=1.2,
            color="#37474F", alpha=0.6, label="Mean ‖ΔW‖ (normalised)")
ax_t5b.set_ylabel("Mean ‖ΔW‖ (normalised)", fontsize=8, color="#37474F")
ax_t5b.tick_params(axis="y", labelcolor="#37474F", labelsize=7)

# annotate top sites
n_label = min(10, N_SITES)
for rank, si in enumerate(sort_sites[:n_label]):
    ax_t5.text(rank, counts_sorted[rank] + 0.3, f"{si}",
               ha="center", va="bottom", fontsize=6.5, color="black",
               rotation=90)

ax_t5.set_xlim(-0.5, N_SITES - 0.5)
ax_t5.set_xticks([])
ax_t5.set_ylabel("# AD patients (site in their top-5)", fontsize=9)
ax_t5.set_xlabel("Sites (sorted by ‖ΔW‖ rank)", fontsize=9)
ax_t5.set_title("Top-5 site selection: frequency across AD patients\n"
                "(orange = site in top-5 for ≥3 patients; site index labelled for top-10)",
                fontsize=9)
ax_t5.axhline(len(ad_pids), color="gray", ls="--", lw=0.8, alpha=0.5)
for sp in ["top","right"]: ax_t5.spines[sp].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
# Row 2, col 2: Top-1 site frequency + per-patient heatmap strip
# ─────────────────────────────────────────────────────────────────────────────
ax_t1 = fig.add_subplot(gs[2, 2])

# Show only sites that appear at least once as top-1
nonzero_mask = top1_count > 0
n_nonzero    = nonzero_mask.sum()
nonzero_sites  = np.where(nonzero_mask)[0]
order_t1       = nonzero_sites[np.argsort(top1_count[nonzero_sites])[::-1]]
counts_t1      = top1_count[order_t1]
x_t1           = np.arange(len(order_t1))

bar_cols_t1 = [COL_T1 if c >= 3 else "#CE93D8" for c in counts_t1]
ax_t1.bar(x_t1, counts_t1, color=bar_cols_t1, edgecolor="none", width=0.8)

for xi, (si, cnt) in enumerate(zip(order_t1, counts_t1)):
    ax_t1.text(xi, cnt + 0.15, f"{si}", ha="center", va="bottom",
               fontsize=7.5, fontweight="bold" if cnt >= 3 else "normal")

ax_t1.set_xticks(x_t1)
ax_t1.set_xticklabels([f"site {s}" for s in order_t1],
                       rotation=55, ha="right", fontsize=7)
ax_t1.set_ylabel("# AD patients (site is their top-1)", fontsize=9)
ax_t1.set_title(f"Top-1 site selection\n({n_nonzero} distinct sites across {len(ad_pids)} AD patients)",
                fontsize=9)
for sp in ["top","right"]: ax_t1.spines[sp].set_visible(False)

# ─────────────────────────────────────────────────────────────────────────────
fig.suptitle(
    "Perturbation experiment — Condition B (σ=0.05, K_LDA=25)\n"
    "Direct G-score of W_int = (1-α)·W_patient + α·W_CC_mean",
    fontsize=12, fontweight="bold", y=0.99)

fig.savefig(f"{OUT_DIR}/pertB_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Additional: per-patient top-5 site heatmap (who picks which sites?)
# ─────────────────────────────────────────────────────────────────────────────
fig2, ax = plt.subplots(figsize=(16, 8), facecolor="white")

# Sort patients by their baseline LDA score
base_z  = np.array([Z_base[list(unique_pids).index(pid)] for pid in ad_pids])
pat_ord = np.argsort(base_z)[::-1]   # most AD-like first

# Sort sites by top5 frequency (most common left)
site_ord = np.argsort(top5_count)[::-1]

# Build display matrix: rows = patients (sorted), cols = sites (sorted)
# Value = 1/rank  (1 if top-1, 1/2 if top-2, …, 0 if not in top-5)
display_mat = np.zeros((len(ad_pids), N_SITES))
for pi_sorted, pi_orig in enumerate(pat_ord):
    pid  = ad_pids[pi_orig]
    nrm  = site_norms[pid]
    top5 = np.argsort(nrm)[::-1][:5]
    for rank, site in enumerate(top5):
        display_mat[pi_sorted, site] = 1.0 / (rank + 1)

display_sorted = display_mat[:, site_ord]

im = ax.imshow(display_sorted, aspect="auto", origin="upper",
               cmap="YlOrRd", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, shrink=0.6, label="1/rank  (1=top-1, 0.2=top-5, 0=not selected)")

# x-tick labels: site index for the most common sites
xtick_step = max(1, N_SITES // 20)
ax.set_xticks(range(0, N_SITES, xtick_step))
ax.set_xticklabels([str(site_ord[i]) for i in range(0, N_SITES, xtick_step)],
                   rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Site index  (sorted by top-5 frequency →)", fontsize=10)
ax.set_ylabel("AD patient  (sorted by baseline LDA ↓ most AD-like)", fontsize=9)
ax.set_title("Per-patient top-5 site selection\n"
             "Rows = AD patients, Columns = sites (by frequency); "
             "colour = inverse rank (bright = most perturbed for that patient)",
             fontsize=10)

# vertical line after top-10 most common sites
ax.axvline(9.5, color="white", lw=1.5, ls="--", alpha=0.8)
ax.text(10.5, -1.5, "← top-10 sites", fontsize=8, color="white",
        va="bottom", ha="left")

fig2.tight_layout()
fig2.savefig(f"{OUT_DIR}/pertB_top5_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_top5_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# Print summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Top-5 most common sites in top-5 ──")
for k in range(min(10, N_SITES)):
    s = top5_rank_order[k]
    print(f"  Site {s:3d}: selected by {top5_count[s]:2d}/{len(ad_pids)} patients  "
          f"(mean ΔW norm = {mean_norm[s]:.4f} ± {std_norm[s]:.4f})")

print("\n── Top-5 most common top-1 sites ──")
for k in range(min(10, N_SITES)):
    s = top1_rank_order[k]
    if top1_count[s] == 0: break
    print(f"  Site {s:3d}: top-1 for {top1_count[s]:2d}/{len(ad_pids)} patients  "
          f"(mean ΔW norm = {mean_norm[s]:.4f} ± {std_norm[s]:.4f})")

print("\nDone.")
