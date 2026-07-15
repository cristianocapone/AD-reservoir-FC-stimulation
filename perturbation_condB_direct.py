"""
perturbation_condB_direct.py
============================
Perturbation experiment — direct G-score of W_int (no simulation).

For each AD patient and (perturbation_type, alpha):
  W_int = perturbed W  →  project into condition-B G-space  →  LDA score

No simulation, no re-fit: the G-score reflects where the interpolated
connectivity sits in the learned population manifold.

Condition B setup:
  single-session W per patient, sigma=0.05, K_LDA=25, sr=0.95

Perturbation types:
  full_w : W_int = (1-α)·Wp + α·Wcc_mean               α ∈ [0, 2]
  top5   : interpolate top-5 columns (by ‖ΔW‖_col)       α ∈ [0, 5]
  top1   : interpolate top-1 column                        α ∈ [0, 10]

Outputs:
  pertB_direct_results.png        3×2 panel: LDA score + FC-r vs alpha
  pertB_direct_heatmap.png        per-patient LDA heat-map
  pertB_direct_data.npz
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
from sklearn.metrics import roc_auc_score
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
K_LDA       = 25
SR          = 0.95
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "top1":   np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
}

# ── load data ──────────────────────────────────────────────────────────────────
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
print(f"  {N_patients} patients ({len(cc_pids)} CC, {len(ad_pids)} AD), {N_subj} sessions")

# ── population PCA ─────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── reservoir ─────────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

# ── TF pass ────────────────────────────────────────────────────────────────────
print("TF pass (all sessions) ...")
sess_X, sess_Y, sess_tgt = {}, {}, {}
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
    sess_tgt[idx] = tgt
print("  TF done.\n")

# ── condition B setup ──────────────────────────────────────────────────────────
print("Condition B — single-session setup ...")
rng_w = np.random.default_rng(RNG_SEED + 1)

first_idx      = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single    = {pid: sess_X[first_idx[pid]]   for pid in unique_pids}
patY_single    = {pid: sess_Y[first_idx[pid]]   for pid in unique_pids}
pat_tgt_single = {pid: sess_tgt[first_idx[pid]] for pid in unique_pids}

# Pre-compute per-patient SVD of X (used for W→G projection)
print("  Pre-computing X-space SVDs ...")
pat_Vtk = {}
for pid in tqdm(unique_pids, desc="  SVD", leave=False):
    Xca = patX_single[pid].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    pat_Vtk[pid] = Vtx[:kk]

def project_W(W, pid):
    """W (N_hidden, N_sites) → flat projected vector in patient's X-subspace."""
    Vt_k = pat_Vtk[pid]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

# Fit W per patient
print("  Fitting W (sigma=0.05) ...")
pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

# ── G-space + LDA ──────────────────────────────────────────────────────────────
print("  Building G-space + LDA ...")
Wproj_B = np.array([project_W(pat_W[pid], pid) for pid in unique_pids])
Wmean_B = Wproj_B.mean(0)
Wcent_B = Wproj_B - Wmean_B
_, _, Vsvd_B = np.linalg.svd(Wcent_B, full_matrices=False)
Meff_B  = N_patients - 1
G_B     = Wcent_B @ Vsvd_B[:Meff_B].T

class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_   = w
        self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i,n,replace=False),
                          rng2.choice(c1i,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

Xlda, ylda = _balance(G_B[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_B = _LDA().fit(Xlda, ylda)
Z_base = lda_B.transform(G_B[:, :K_LDA])
if Z_base[patient_labels==0].mean() > Z_base[patient_labels==1].mean():
    lda_B.w_ *= -1; lda_B.thr_ *= -1
Z_base = lda_B.transform(G_B[:, :K_LDA])

cc_lda = Z_base[patient_labels==0]
ad_lda = Z_base[patient_labels==1]
auc_base = roc_auc_score(patient_labels, Z_base)
print(f"  Baseline: CC={cc_lda.mean():.3f}±{cc_lda.std():.3f}  "
      f"AD={ad_lda.mean():.3f}±{ad_lda.std():.3f}  AUROC={auc_base:.4f}")

pid_to_z0 = {pid: float(Z_base[i]) for i, pid in enumerate(unique_pids)}

# ── CC mean W + FC template ────────────────────────────────────────────────────
print("\nCC mean W and FC template ...")
W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

fc_cc_list = []
for pid in cc_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc_cc_list.append(np.nan_to_num(np.corrcoef(tgt)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)

# Baseline FC-r for AD patients
pat_fc_r_base = {}
for pid in ad_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc  = np.nan_to_num(np.corrcoef(tgt)).flatten()
    pat_fc_r_base[pid] = float(np.corrcoef(fc, FC_cc_mean)[0, 1])

# Also: CC patients' FC-r with CC template (for reference band)
cc_fc_r = []
for pid in cc_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc  = np.nan_to_num(np.corrcoef(tgt)).flatten()
    cc_fc_r.append(float(np.corrcoef(fc, FC_cc_mean)[0, 1]))
cc_fc_r = np.array(cc_fc_r)

print(f"  CC FC-r: {cc_fc_r.mean():.4f}±{cc_fc_r.std():.4f}")
print(f"  AD baseline FC-r: {np.mean(list(pat_fc_r_base.values())):.4f}")

# ── helper: G-score of arbitrary W for a given patient ────────────────────────
def w_to_lda(W, pid):
    wp = project_W(W, pid)
    g  = ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]
    return float(lda_B.transform(g.reshape(1, -1))[0])

# Also: FC-r of a W matrix's implied signal —
# here we compute FC from the original target signal (unchanged by W_int),
# so FC-r is just the original baseline (signal doesn't change with W_int in direct approach).
# Instead we compare the column-norm profile of W_int vs W_cc_mean as a proxy.

def w_fc_r(W, pid):
    """FC of the signal re-derived by TF with W as readout (approximation)."""
    # For the direct approach, the FC of Y = W.T @ X_original
    Xc  = patX_single[pid].astype(np.float64)
    Y   = (W.T.astype(np.float64) @ Xc.T)  # (N_sites, T_eff)
    fc  = np.nan_to_num(np.corrcoef(Y)).flatten()
    return float(np.corrcoef(fc, FC_cc_mean)[0, 1])

# ── perturbation loop (direct: no simulation) ──────────────────────────────────
print("\nPerturbation experiment (direct G-score of W_int) ...")

# results[pert_type][alpha_idx] = {pid: (lda_score, fc_r)}
results = {}
for pert_type, alphas in ALPHA_GRIDS.items():
    print(f"\n  [{pert_type}]  {len(alphas)} alpha values")
    results[pert_type] = []
    for ai, alpha in enumerate(alphas):
        alpha_res = {}
        for pid in ad_pids:
            Wp = pat_W[pid]
            dW = W_cc_mean - Wp

            if pert_type == "full_w":
                W_int = (1 - alpha) * Wp + alpha * W_cc_mean

            elif pert_type == "top5":
                norms = np.linalg.norm(dW, axis=0)
                top_k = np.argsort(norms)[::-1][:5]
                W_int = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]

            else:  # top1
                norms = np.linalg.norm(dW, axis=0)
                top_k = np.argsort(norms)[::-1][:1]
                W_int = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]

            lda_sc = w_to_lda(W_int, pid)
            fc_r   = w_fc_r(W_int, pid)
            alpha_res[pid] = (lda_sc, fc_r)
        results[pert_type].append(alpha_res)
        # Print a quick summary
        lda_vals = [alpha_res[pid][0] for pid in ad_pids]
        print(f"    α={alpha:.3f}  AD LDA mean={np.mean(lda_vals):.3f}  "
              f"std={np.std(lda_vals):.3f}", flush=True)

# ── save data ──────────────────────────────────────────────────────────────────
save_dict = {"ad_pids": np.array(ad_pids), "cc_pids": np.array(cc_pids),
             "patient_labels": patient_labels, "Z_base": Z_base,
             "cc_lda": cc_lda, "ad_lda": ad_lda,
             "cc_fc_r": cc_fc_r,
             "ad_fc_r_base": np.array([pat_fc_r_base[p] for p in ad_pids])}
for pert_type, alphas in ALPHA_GRIDS.items():
    alpha_data = results[pert_type]
    lda_arr = np.array([[alpha_data[ai][pid][0] for pid in ad_pids]
                         for ai in range(len(alphas))])
    fcr_arr = np.array([[alpha_data[ai][pid][1] for pid in ad_pids]
                         for ai in range(len(alphas))])
    save_dict[f"{pert_type}_alphas"] = alphas
    save_dict[f"{pert_type}_lda"]    = lda_arr
    save_dict[f"{pert_type}_fcr"]    = fcr_arr
np.savez(f"{OUT_DIR}/pertB_direct_data.npz", **save_dict)
print("\nSaved pertB_direct_data.npz")

# ── plotting ───────────────────────────────────────────────────────────────────
print("Plotting ...")

import matplotlib.gridspec as gridspec

PERT_LABELS = {
    "full_w": "Full-W  (all 121 sites)\nW_int = (1-α)·W_p + α·W_CC",
    "top5":   "Top-5 sites\n(largest ‖W_CC − W_p‖ columns)",
    "top1":   "Single site\n(largest ‖W_CC − W_p‖ column)",
}
COL_CC   = "#2196F3"
COL_AD   = "#E91E63"
COL_MEAN = "black"
COL_FC   = "#7B1FA2"

# Per-type colours for the comparison row
PERT_COLS = {"full_w": "#1B5E20", "top5": "#E65100", "top1": "#4A148C"}

# Pre-compute lda_mat and fcr_mat for each perturbation type
lda_mats = {}
fcr_mats = {}
for pert_type, alphas in ALPHA_GRIDS.items():
    alpha_data = results[pert_type]
    lda_mats[pert_type] = np.array([[alpha_data[ai][pid][0] for pid in ad_pids]
                                     for ai in range(len(alphas))])
    fcr_mats[pert_type] = np.array([[alpha_data[ai][pid][1] for pid in ad_pids]
                                     for ai in range(len(alphas))])

# Shared y limits for LDA panels
y_all = np.concatenate([m.flatten() for m in lda_mats.values()])
y_min = float(np.nanmin(y_all)) - 0.3
y_max = float(np.nanmax(y_all)) + 0.3
ad_cc_mid = 0.5 * (cc_lda.mean() + ad_lda.mean())

# ── figure layout: 3 rows × 3 cols ────────────────────────────────────────────
fig = plt.figure(figsize=(20, 17), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig,
                         height_ratios=[1.0, 0.85, 0.95],
                         hspace=0.45, wspace=0.30)

# ── rows 0-1: per-type panels (same as before) ────────────────────────────────
for ci, pert_type in enumerate(["full_w", "top5", "top1"]):
    alphas     = ALPHA_GRIDS[pert_type]
    alpha_data = results[pert_type]
    lda_mat    = lda_mats[pert_type]
    fcr_mat    = fcr_mats[pert_type]

    # row 0: LDA score
    ax = fig.add_subplot(gs[0, ci])
    ax.axhspan(cc_lda.mean() - cc_lda.std(), cc_lda.mean() + cc_lda.std(),
               alpha=0.20, color=COL_CC, label="CC ±1σ")
    ax.axhline(cc_lda.mean(), color=COL_CC, lw=2, ls="--")
    ax.axhspan(ad_lda.mean() - ad_lda.std(), ad_lda.mean() + ad_lda.std(),
               alpha=0.12, color=COL_AD)
    ax.axhline(ad_lda.mean(), color=COL_AD, lw=1.5, ls=":", alpha=0.6)
    ax.axhline(ad_cc_mid, color="gray", lw=0.8, ls="-.", alpha=0.5)

    for pid in ad_pids:
        traj = [alpha_data[ai][pid][0] for ai in range(len(alphas))]
        ax.plot(alphas, traj, "-", lw=0.8, color=PERT_COLS[pert_type], alpha=0.22)
    mean_t = lda_mat.mean(1); std_t = lda_mat.std(1)
    ax.fill_between(alphas, mean_t - std_t, mean_t + std_t,
                    alpha=0.22, color=PERT_COLS[pert_type])
    ax.plot(alphas, mean_t, "-o", ms=6, lw=2.5, color=PERT_COLS[pert_type],
            zorder=5, label="AD mean ±1σ")
    ax.set_ylim(y_min, y_max)
    ax.set_title(PERT_LABELS[pert_type], fontsize=10)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("LDA score  (cond. B, K=25)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # row 1: FC-r
    ax = fig.add_subplot(gs[1, ci])
    ax.axhspan(cc_fc_r.mean() - cc_fc_r.std(), cc_fc_r.mean() + cc_fc_r.std(),
               alpha=0.20, color=COL_CC, label="CC ±1σ")
    ax.axhline(cc_fc_r.mean(), color=COL_CC, lw=2, ls="--")
    for pid in ad_pids:
        traj = [alpha_data[ai][pid][1] for ai in range(len(alphas))]
        ax.plot(alphas, traj, "-", lw=0.8, color=COL_FC, alpha=0.22)
    mean_fc = fcr_mat.mean(1); std_fc = fcr_mat.std(1)
    ax.fill_between(alphas, mean_fc - std_fc, mean_fc + std_fc,
                    alpha=0.22, color=COL_FC)
    ax.plot(alphas, mean_fc, "-o", ms=6, lw=2.5, color=COL_FC,
            zorder=5, label="AD mean ±1σ")
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("FC-r (W·ᵀX vs CC template)", fontsize=9)
    ax.set_title("FC similarity to CC mean", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

# ── row 2: comparison panels ──────────────────────────────────────────────────
# Left (spans 2 cols): LDA gap-fraction vs alpha for all 3 types
# Right (1 col):       FC-r vs alpha for all 3 types

# Normalised LDA: 0 = AD baseline, 1 = CC mean  (fraction of gap closed)
gap      = ad_lda.mean() - cc_lda.mean()   # positive
norm_lda = lambda z: (ad_lda.mean() - z) / gap   # 0 at AD, 1 at CC

ax_cmp = fig.add_subplot(gs[2, 0:2])
ax_cmp.axhspan(0 - cc_lda.std()/gap, 0 + cc_lda.std()/gap,
               alpha=0.20, color=COL_CC)
ax_cmp.axhline(1.0, color=COL_CC, lw=2, ls="--", label="CC mean")
ax_cmp.axhline(0.0, color=COL_AD, lw=1.5, ls=":", alpha=0.6, label="AD baseline")
ax_cmp.axhline(0.5, color="gray", lw=0.8, ls="-.", alpha=0.5, label="midpoint")

for pert_type in ["full_w", "top5", "top1"]:
    alphas  = ALPHA_GRIDS[pert_type]
    mean_t  = lda_mats[pert_type].mean(1)
    std_t   = lda_mats[pert_type].std(1)
    norm_m  = norm_lda(mean_t)
    norm_lo = norm_lda(mean_t + std_t)   # +std → lower gap-fraction
    norm_hi = norm_lda(mean_t - std_t)
    col     = PERT_COLS[pert_type]
    ax_cmp.fill_between(alphas, norm_lo, norm_hi, alpha=0.18, color=col)
    ax_cmp.plot(alphas, norm_m, "-o", ms=6, lw=2.5, color=col,
                label=PERT_LABELS[pert_type].split("\n")[0])

ax_cmp.set_xlabel("alpha", fontsize=10)
ax_cmp.set_ylabel("Fraction of AD→CC gap closed\n(0 = AD, 1 = CC)", fontsize=10)
ax_cmp.set_title("Comparison: perturbation efficiency\n"
                 "(all 3 types on common scale)", fontsize=10)
ax_cmp.legend(fontsize=8, frameon=False, loc="upper left")
for sp in ["top","right"]: ax_cmp.spines[sp].set_visible(False)

ax_fc2 = fig.add_subplot(gs[2, 2])
ax_fc2.axhspan(cc_fc_r.mean() - cc_fc_r.std(), cc_fc_r.mean() + cc_fc_r.std(),
               alpha=0.20, color=COL_CC, label="CC ±1σ")
ax_fc2.axhline(cc_fc_r.mean(), color=COL_CC, lw=2, ls="--")
for pert_type in ["full_w", "top5", "top1"]:
    alphas  = ALPHA_GRIDS[pert_type]
    mean_fc = fcr_mats[pert_type].mean(1)
    ax_fc2.plot(alphas, mean_fc, "-o", ms=5, lw=2.0,
                color=PERT_COLS[pert_type],
                label=PERT_LABELS[pert_type].split("\n")[0])
ax_fc2.set_xlabel("alpha", fontsize=10)
ax_fc2.set_ylabel("FC-r vs CC template", fontsize=10)
ax_fc2.set_title("FC similarity comparison\n(mean over AD patients)", fontsize=10)
ax_fc2.legend(fontsize=7, frameon=False)
for sp in ["top","right"]: ax_fc2.spines[sp].set_visible(False)

fig.suptitle(
    "Direct perturbation — Condition B  (σ=0.05, K_LDA=25, sr=0.95)\n"
    "Rows 1–2: per-type individual trajectories  |  Row 3: all-type comparison",
    fontsize=11, fontweight="bold", y=1.005)

fig.savefig(f"{OUT_DIR}/pertB_direct_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_direct_results.png")

# ── per-patient heat-map ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor="white")

for col, pert_type in enumerate(["full_w", "top5", "top1"]):
    alphas     = ALPHA_GRIDS[pert_type]
    alpha_data = results[pert_type]
    ax         = axes[col]

    base_z = np.array([pid_to_z0[pid] for pid in ad_pids])
    order  = np.argsort(base_z)   # sort by baseline LDA score (low → high)

    lda_mat = np.array([[alpha_data[ai][ad_pids[pi]][0]
                          for ai in range(len(alphas))]
                         for pi in order])   # (n_ad_sorted, n_alpha)

    vabs = max(abs(np.nanmin(lda_mat)), abs(np.nanmax(lda_mat)),
               abs(cc_lda.mean()) + cc_lda.std(), 0.5)
    im   = ax.imshow(lda_mat, aspect="auto", origin="lower",
                     cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, shrink=0.8, label="LDA score")

    # CC mean as dashed line on colour scale
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2g}" for a in alphas],
                       rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("AD patient  (sorted by baseline LDA ↑)", fontsize=8)
    ax.set_title(f"{PERT_LABELS[pert_type]}", fontsize=9)

fig.suptitle("Per-patient LDA score — direct perturbation (Cond. B)\n"
             "Blue = CC-like, Red = AD-like", fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pertB_direct_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_direct_heatmap.png")

print("\nDone.")
