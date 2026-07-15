"""
perturbation_condB.py
=====================
Perturbation / stimulation experiment using Condition B (fair) setup:
  - Single-session W per patient (first session only)
  - sigma = 0.05  (W-fitting regularisation, LOPO-optimised)
  - K_LDA = 25   (G-space PCs → LDA, LOPO-optimised)
  - spectral_radius = 0.95

For each AD patient and each (perturbation_type, alpha):
  1. Build W_int  (interpolation of patient W toward CC-mean W)
  2. Start autonomous sim from warmup hidden state (first-session TF)
  3. Collect Y_sim — check stability
  4. Re-TF on Y_sim → X_aut
  5. Re-fit  W_new = pinv(X_aut + noise) @ Y_aut
  6. Project W_new into condition-B G-space → LDA score

Perturbation types & alpha ranges:
  full_w : interpolate ALL sites,  alpha ∈ [0 … 2]
  top5   : interpolate TOP-5 sites (largest ‖ΔW‖ columns), alpha ∈ [0 … 5]
  top1   : interpolate TOP-1 site,                          alpha ∈ [0 … 10]

Outputs:
  pertB_results.png          3×3 panel (LDA score / FC-r / stability)
  pertB_lda_heatmap.png      per-patient LDA trajectory heat-map
  pertB_data.npz             raw numerical results for follow-up analysis
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
RNG_SEED       = 42
N_CC_SAMPLE    = 40
N_SITES        = 121
N_PC_MODEL     = 50
K_PC           = 200
TIMES_SKIP     = 10
ff             = 0.1
N_HIDDEN       = 2000
SIGMA          = 0.05        # condition-B optimum
K_LDA          = 25          # condition-B optimum
SR             = 0.95
SIM_MAX_STABLE = 1e4
TS_ROOT        = "./timeseries"
OUT_DIR        = "."

ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    "top1":   np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
}

# ── data loading ───────────────────────────────────────────────────────────────
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
    tgt   = (s.T @ ev50 @ ev50.T).T      # PCA-projected signal (N_sites, T)
    res.T = T_s; res.reset()
    Xraw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        Xraw.append(res.X.copy())
    Xf          = np.array(Xraw)[TIMES_SKIP:]
    sess_X[idx] = Xf                                        # (T_eff, N_hidden)
    sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T  # (T_eff, N_sites)
    sess_tgt[idx] = tgt
print("  TF done.\n")

# ── condition B: single-session per patient ────────────────────────────────────
print("Condition B — single-session setup ...")
rng_w = np.random.default_rng(RNG_SEED + 1)

first_idx      = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single    = {pid: sess_X[first_idx[pid]]   for pid in unique_pids}
patY_single    = {pid: sess_Y[first_idx[pid]]   for pid in unique_pids}
pat_tgt_single = {pid: sess_tgt[first_idx[pid]] for pid in unique_pids}

# Pre-compute Vt_k for each patient (SVD of their X matrix) — reused many times
print("  Pre-computing X-space SVDs ...")
pat_Vtk = {}
for pid in tqdm(unique_pids, desc="  SVD", leave=False):
    Xca = patX_single[pid].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    pat_Vtk[pid] = Vtx[:kk]  # (kk, N_hidden)

def project_W(W, pid):
    """Project W (N_hidden, N_sites) into patient's X-subspace → flat vector."""
    Vt_k = pat_Vtk[pid]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

# Fit W per patient (sigma = 0.05)
print("  Fitting W (sigma=0.05) ...")
pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc   # (N_hidden, N_sites)

# ── G-space + LDA ──────────────────────────────────────────────────────────────
print("  Building G-space + LDA ...")
Wproj_B = np.array([project_W(pat_W[pid], pid) for pid in unique_pids])
Wmean_B = Wproj_B.mean(0)
Wcent_B = Wproj_B - Wmean_B
_, _, Vsvd_B = np.linalg.svd(Wcent_B, full_matrices=False)
Meff_B  = N_patients - 1
G_B     = Wcent_B @ Vsvd_B[:Meff_B].T   # (N_patients, Meff_B)

class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w  = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_   = w
        self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n   = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i,n,replace=False),
                          rng2.choice(c1i,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

Xlda, ylda = _balance(G_B[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_B = _LDA().fit(Xlda, ylda)
Z_base = lda_B.transform(G_B[:, :K_LDA])
if Z_base[patient_labels==0].mean() > Z_base[patient_labels==1].mean():
    lda_B.w_ *= -1; lda_B.thr_ *= -1
Z_base = lda_B.transform(G_B[:, :K_LDA])

cc_lda_scores = Z_base[patient_labels==0]
ad_lda_scores = Z_base[patient_labels==1]
cc_mean_lda, cc_std_lda = cc_lda_scores.mean(), cc_lda_scores.std()
ad_mean_lda, ad_std_lda = ad_lda_scores.mean(), ad_lda_scores.std()
auc_base = roc_auc_score(patient_labels, Z_base)
print(f"  Baseline: CC={cc_mean_lda:.3f}±{cc_std_lda:.3f}  "
      f"AD={ad_mean_lda:.3f}±{ad_std_lda:.3f}  AUROC={auc_base:.4f}")

# Map pid → baseline LDA score
pid_to_z0 = {pid: float(Z_base[i]) for i, pid in enumerate(unique_pids)}

# ── CC mean W + FC baseline ────────────────────────────────────────────────────
print("\nCC mean W and FC template ...")
W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)  # (N_hidden, N_sites)

fc_cc_list = []
for pid in cc_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc_cc_list.append(np.nan_to_num(np.corrcoef(tgt)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)

# AD baseline FC correlation with CC template (at alpha=0)
pat_fc_r_base = {}
for pid in ad_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc  = np.nan_to_num(np.corrcoef(tgt)).flatten()
    pat_fc_r_base[pid] = float(np.corrcoef(fc, FC_cc_mean)[0, 1])
print(f"  AD baseline FC-r with CC: {np.mean(list(pat_fc_r_base.values())):.4f}")

# ── warmup cache (AD patients only) ───────────────────────────────────────────
print("\nWarmup cache (AD patients) ...")
warmup_X = {}
for pid in tqdm(ad_pids, desc="  warm-up", leave=False):
    tgt = pat_tgt_single[pid]; T = tgt.shape[1]
    res.T = T; res.reset()
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
    warmup_X[pid] = (res.X.copy(), T)

# ── simulation + refit helpers ────────────────────────────────────────────────
rng_sim = np.random.default_rng(RNG_SEED + 99)

def w_to_lda(W_fitted, pid):
    """Project re-fitted W into condition-B G-space, return LDA score."""
    wp = project_W(W_fitted, pid)
    g  = ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]
    return float(lda_B.transform(g.reshape(1, -1))[0])

def run_sim_refit(W_int, pid):
    """
    Autonomous sim with perturbed readout W_int, then refit W.
    Returns (lda_score, fc_r, stable).
    """
    Xw, T = warmup_X[pid]
    res.T    = T
    res.X    = Xw.copy()
    res.Jout = W_int.T.copy()          # (N_sites, N_hidden)
    res.y    = res.Jout @ res.X

    Ys = []
    for _ in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Ys.append(res.y.copy())
    Ysim = np.array(Ys).T              # (N_sites, T-1)

    if not np.isfinite(Ysim).all() or np.abs(Ysim).max() > SIM_MAX_STABLE:
        return None, None, False

    # FC-r with CC template
    Yeff = Ysim[:, TIMES_SKIP:]
    fc_r = float(np.corrcoef(
        np.nan_to_num(np.corrcoef(Yeff)).flatten(), FC_cc_mean)[0, 1])

    # Re-TF on Y_sim → X_aut
    T2    = Ysim.shape[1]
    res.T = T2; res.reset()
    Xaut  = []
    for t in range(T2 - 1):
        res.step_rate(ff * Ysim[:, t], sigma_dyn=0.)
        Xaut.append(res.X.copy())
    Xaut = np.array(Xaut)[TIMES_SKIP:]
    Yaut = Ysim[:, TIMES_SKIP:TIMES_SKIP+len(Xaut)].T

    noise    = rng_sim.normal(0, SIGMA, Xaut.shape)
    W_fitted = np.linalg.pinv(Xaut + noise) @ Yaut
    lda_sc   = w_to_lda(W_fitted, pid)
    return lda_sc, fc_r, True

# ── perturbation loop ──────────────────────────────────────────────────────────
print("\nPerturbation experiment ...")

# Store: results[pert_type] = list over alphas of dict {pid: (lda, fc_r, stable)}
results = {}
for pert_type, alphas in ALPHA_GRIDS.items():
    print(f"\n  [{pert_type}]  {len(alphas)} alpha values, {len(ad_pids)} AD patients")
    results[pert_type] = []

    for ai, alpha in enumerate(alphas):
        alpha_res = {}
        for pid in ad_pids:
            Wp = pat_W[pid]                       # (N_hidden, N_sites)
            dW = W_cc_mean - Wp

            if pert_type == "full_w":
                W_int = (1 - alpha) * Wp + alpha * W_cc_mean

            elif pert_type == "top5":
                norms = np.linalg.norm(dW, axis=0)   # per-site norm (N_sites,)
                top_k = np.argsort(norms)[::-1][:5]
                W_int = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]

            else:   # top1
                norms = np.linalg.norm(dW, axis=0)
                top_k = np.argsort(norms)[::-1][:1]
                W_int = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]

            lda_sc, fc_r, stable = run_sim_refit(W_int, pid)
            alpha_res[pid] = (lda_sc, fc_r, stable)

        n_stable = sum(v[2] for v in alpha_res.values())
        print(f"    α={alpha:.3f}  →  {n_stable}/{len(ad_pids)} stable", flush=True)
        results[pert_type].append(alpha_res)

# ── save raw data ──────────────────────────────────────────────────────────────
save_dict = {"ad_pids": np.array(ad_pids), "patient_labels": patient_labels,
             "Z_base": Z_base, "cc_lda_scores": cc_lda_scores,
             "ad_lda_scores": ad_lda_scores}
for pert_type, alphas in ALPHA_GRIDS.items():
    alpha_data = results[pert_type]
    lda_arr  = np.full((len(alphas), len(ad_pids)), np.nan)
    fcr_arr  = np.full((len(alphas), len(ad_pids)), np.nan)
    stab_arr = np.zeros((len(alphas), len(ad_pids)), dtype=bool)
    for ai in range(len(alphas)):
        for pi, pid in enumerate(ad_pids):
            lda_sc, fc_r, stable = alpha_data[ai][pid]
            stab_arr[ai, pi] = stable
            if stable:
                lda_arr[ai, pi] = lda_sc
                fcr_arr[ai, pi] = fc_r
    save_dict[f"{pert_type}_alphas"] = alphas
    save_dict[f"{pert_type}_lda"]    = lda_arr
    save_dict[f"{pert_type}_fcr"]    = fcr_arr
    save_dict[f"{pert_type}_stable"] = stab_arr
np.savez(f"{OUT_DIR}/pertB_data.npz", **save_dict)
print("\nSaved pertB_data.npz")

# ── plotting ───────────────────────────────────────────────────────────────────
print("Plotting ...")

PERT_LABELS = {
    "full_w": "Full-W  (all 121 sites)\nα ∈ [0, 2]",
    "top5":   "Top-5 sites\nα ∈ [0, 5]",
    "top1":   "Single site (largest ‖ΔW‖)\nα ∈ [0, 10]",
}
COL_CC    = "#2196F3"
COL_AD    = "#E91E63"
COL_MEAN  = "black"
COL_FC    = "#7B1FA2"
COL_STAB  = "#388E3C"

fig, axes = plt.subplots(3, 3, figsize=(20, 15), facecolor="white")

for col, pert_type in enumerate(["full_w", "top5", "top1"]):
    alphas     = ALPHA_GRIDS[pert_type]
    alpha_data = results[pert_type]

    # ── row 0: LDA score vs alpha ─────────────────────────────────────────
    ax = axes[0, col]

    # CC distribution band
    ax.axhspan(cc_mean_lda - cc_std_lda, cc_mean_lda + cc_std_lda,
               alpha=0.20, color=COL_CC)
    ax.axhline(cc_mean_lda, color=COL_CC, lw=2, ls="--", label="CC mean")

    # AD distribution band
    ax.axhspan(ad_mean_lda - ad_std_lda, ad_mean_lda + ad_std_lda,
               alpha=0.15, color=COL_AD)
    ax.axhline(ad_mean_lda, color=COL_AD, lw=1.5, ls=":", alpha=0.7, label="AD baseline")

    # Per-patient trajectories
    for pid in ad_pids:
        traj = [alpha_data[ai][pid][0] if alpha_data[ai][pid][2] else np.nan
                for ai in range(len(alphas))]
        ax.plot(alphas, traj, "-o", ms=3, lw=0.9, color=COL_AD, alpha=0.30)

    # Mean trajectory (over stable patients)
    mean_traj = []
    for ai in range(len(alphas)):
        vals = [alpha_data[ai][pid][0] for pid in ad_pids if alpha_data[ai][pid][2]]
        mean_traj.append(np.nanmean(vals) if vals else np.nan)
    ax.plot(alphas, mean_traj, "-o", ms=7, lw=2.5, color=COL_MEAN, zorder=5, label="AD mean")

    ax.set_title(PERT_LABELS[pert_type], fontsize=10)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("LDA score (cond. B)", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # ── row 1: FC correlation vs alpha ────────────────────────────────────
    ax = axes[1, col]

    base_fc = np.array([pat_fc_r_base[pid] for pid in ad_pids])
    ax.axhline(base_fc.mean(), color=COL_AD, ls=":", lw=1.5, alpha=0.7, label="AD base FC-r")

    for pid in ad_pids:
        traj = [alpha_data[ai][pid][1] if alpha_data[ai][pid][2] else np.nan
                for ai in range(len(alphas))]
        ax.plot(alphas, traj, "-o", ms=3, lw=0.9, color=COL_FC, alpha=0.30)

    mean_fc = []
    for ai in range(len(alphas)):
        vals = [alpha_data[ai][pid][1] for pid in ad_pids if alpha_data[ai][pid][2]]
        mean_fc.append(np.nanmean(vals) if vals else np.nan)
    ax.plot(alphas, mean_fc, "-o", ms=7, lw=2.5, color=COL_MEAN, zorder=5, label="mean FC-r")

    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("FC-r vs CC template", fontsize=9)
    ax.set_title("FC similarity to CC mean", fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # ── row 2: stability + LDA per-patient heat-map ───────────────────────
    ax = axes[2, col]

    stab_frac = []
    for ai in range(len(alphas)):
        n_s = sum(1 for pid in ad_pids if alpha_data[ai][pid][2])
        stab_frac.append(n_s / len(ad_pids))

    ax2 = ax.twinx()
    ax.bar(range(len(alphas)), stab_frac, color=COL_STAB, alpha=0.55, label="stable frac")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Fraction stable", fontsize=9, color=COL_STAB)
    ax.tick_params(axis="y", labelcolor=COL_STAB)

    # overlay mean LDA on twin axis
    ax2.plot(range(len(alphas)), mean_traj, "-o", ms=5, lw=2, color=COL_MEAN, zorder=5)
    ax2.axhline(cc_mean_lda, color=COL_CC, lw=1.5, ls="--", alpha=0.8)
    ax2.set_ylabel("Mean LDA score", fontsize=9)

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2g}" for a in alphas], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_title("Stability & mean LDA", fontsize=9)
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    for sp in ["top"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    "Perturbation experiment — Condition B (single session, σ=0.05, K_LDA=25, sr=0.95)\n"
    "Blue band = CC distribution  |  Pink band = AD baseline  |  Black = AD mean trajectory",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pertB_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_results.png")

# ── per-patient LDA heat-map ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 8), facecolor="white")

for col, pert_type in enumerate(["full_w", "top5", "top1"]):
    alphas     = ALPHA_GRIDS[pert_type]
    alpha_data = results[pert_type]
    ax         = axes[col]

    # matrix: rows = AD patients (sorted by baseline LDA), cols = alphas
    lda_mat = np.full((len(ad_pids), len(alphas)), np.nan)
    base_z  = np.array([pid_to_z0[pid] for pid in ad_pids])
    order   = np.argsort(base_z)

    for pi_sorted, pi_orig in enumerate(order):
        pid = ad_pids[pi_orig]
        for ai in range(len(alphas)):
            lda_sc, _, stable = alpha_data[ai][pid]
            lda_mat[pi_sorted, ai] = lda_sc if stable else np.nan

    vabs = max(abs(np.nanmin(lda_mat)), abs(np.nanmax(lda_mat)), abs(cc_mean_lda), 0.1)
    im   = ax.imshow(lda_mat, aspect="auto", origin="lower",
                     cmap="RdBu_r", vmin=-vabs, vmax=vabs)
    plt.colorbar(im, ax=ax, shrink=0.8, label="LDA score")

    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.2g}" for a in alphas], rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("alpha", fontsize=9)
    ax.set_ylabel("AD patient (sorted by baseline LDA score ↑)", fontsize=8)
    ax.set_title(f"{PERT_LABELS[pert_type]}\n(red=high/AD-like, blue=CC-like)",
                 fontsize=9)
    ax.axhline(cc_mean_lda, color="white", ls="--", lw=1, alpha=0.5)

fig.suptitle("Per-patient LDA score heat-map — Perturbation experiment (Condition B)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pertB_lda_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_lda_heatmap.png")

print("\nDone.")
