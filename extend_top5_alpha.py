"""
extend_top5_alpha.py
====================
Re-runs the condition-B direct perturbation with an EXTENDED top-5 alpha grid
[0 → 10], keeping full_w and top1 unchanged.

Uses the IDENTICAL random seeds / params as perturbation_condB_direct.py so
the W matrices, G-space, and LDA are bit-for-bit the same.

Overwrites pertB_direct_data.npz with the merged result.
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

# ── identical hyper-parameters ─────────────────────────────────────────────────
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

# ── EXTENDED top-5 grid (everything else unchanged) ───────────────────────────
ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0,
                         4.0, 5.0, 6.0, 7.5, 10.0]),          # ← extended
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
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)),
                                replace=False))
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
print(f"  {N_patients} patients ({len(cc_pids)} CC, {len(ad_pids)} AD), "
      f"{N_subj} sessions")

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
print("TF pass ...")
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

# ── condition B setup ──────────────────────────────────────────────────────────
print("Fitting W + building G-space ...")
rng_w = np.random.default_rng(RNG_SEED + 1)

first_idx      = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single    = {pid: sess_X[first_idx[pid]]   for pid in unique_pids}
patY_single    = {pid: sess_Y[first_idx[pid]]   for pid in unique_pids}
pat_tgt_single = {pid: sess_tgt[first_idx[pid]] for pid in unique_pids}

pat_Vtk = {}
for pid in tqdm(unique_pids, desc="  SVD", leave=False):
    Xca = patX_single[pid].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    pat_Vtk[pid] = Vtx[:kk]

def project_W(W, pid):
    Vt_k = pat_Vtk[pid]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

# G-space + LDA
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
print(f"  Baseline: CC={cc_lda.mean():.3f}±{cc_lda.std():.3f}  "
      f"AD={ad_lda.mean():.3f}±{ad_lda.std():.3f}")

def w_to_lda(W, pid):
    wp = project_W(W, pid)
    g  = ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]
    return float(lda_B.transform(g.reshape(1, -1))[0])

# FC reference
W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

fc_cc_list = []
for pid in cc_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc_cc_list.append(np.nan_to_num(np.corrcoef(tgt)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)

cc_fc_r = np.array([
    float(np.corrcoef(
        np.nan_to_num(np.corrcoef(
            pat_tgt_single[pid][:, TIMES_SKIP:])).flatten(),
        FC_cc_mean)[0, 1])
    for pid in cc_pids])

ad_fc_r_base = np.array([
    float(np.corrcoef(
        np.nan_to_num(np.corrcoef(
            pat_tgt_single[pid][:, TIMES_SKIP:])).flatten(),
        FC_cc_mean)[0, 1])
    for pid in ad_pids])

def w_fc_r(W, pid):
    Xc = patX_single[pid].astype(np.float64)
    Y  = (W.T.astype(np.float64) @ Xc.T)
    fc = np.nan_to_num(np.corrcoef(Y)).flatten()
    return float(np.corrcoef(fc, FC_cc_mean)[0, 1])

# ── perturbation loop ──────────────────────────────────────────────────────────
print("\nPerturbation loop ...")
results = {}
for pert_type, alphas in ALPHA_GRIDS.items():
    print(f"\n  [{pert_type}]  {len(alphas)} alpha values  "
          f"(max α = {alphas.max():.1f})")
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
                W_int[:, top_k] = ((1-alpha)*Wp[:, top_k]
                                   + alpha*W_cc_mean[:, top_k])
            else:   # top1
                norms = np.linalg.norm(dW, axis=0)
                top_k = np.argsort(norms)[::-1][:1]
                W_int = Wp.copy()
                W_int[:, top_k] = ((1-alpha)*Wp[:, top_k]
                                   + alpha*W_cc_mean[:, top_k])
            alpha_res[pid] = (w_to_lda(W_int, pid), w_fc_r(W_int, pid))
        results[pert_type].append(alpha_res)
        lda_vals = [alpha_res[p][0] for p in ad_pids]
        print(f"    α={alpha:5.2f}  AD LDA mean={np.mean(lda_vals):+.3f}  "
              f"std={np.std(lda_vals):.3f}", flush=True)

# ── save ───────────────────────────────────────────────────────────────────────
save_dict = {
    "ad_pids": np.array(ad_pids), "cc_pids": np.array(cc_pids),
    "patient_labels": patient_labels, "Z_base": Z_base,
    "cc_lda": cc_lda, "ad_lda": ad_lda,
    "cc_fc_r": cc_fc_r, "ad_fc_r_base": ad_fc_r_base,
}
for pert_type, alphas in ALPHA_GRIDS.items():
    ad = results[pert_type]
    lda_arr = np.array([[ad[ai][pid][0] for pid in ad_pids]
                         for ai in range(len(alphas))])
    fcr_arr = np.array([[ad[ai][pid][1] for pid in ad_pids]
                         for ai in range(len(alphas))])
    save_dict[f"{pert_type}_alphas"] = alphas
    save_dict[f"{pert_type}_lda"]    = lda_arr
    save_dict[f"{pert_type}_fcr"]    = fcr_arr

np.savez(f"{OUT_DIR}/pertB_direct_data.npz", **save_dict)
print("\nSaved pertB_direct_data.npz")

# ── quick summary plot ─────────────────────────────────────────────────────────
ad_cc_mid = 0.5 * (cc_lda.mean() + ad_lda.mean())
PERT_COLS = {"full_w": "#1B5E20", "top5": "#E65100", "top1": "#6A1B9A"}
LABELS    = {"full_w": "Full-W (121 sites)",
             "top5":   "Top-5 sites",
             "top1":   "Top-1 site"}

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")

for ax, metric, ylabel, title in [
    (axes[0], "lda",
     "Mean LDA score  (K=25)", "LDA score vs α"),
    (axes[1], "fcr",
     "FC-r vs CC template", "FC similarity vs α"),
]:
    ax.axhspan(cc_lda.mean()-cc_lda.std(), cc_lda.mean()+cc_lda.std(),
               alpha=0.15, color="#1565C0")
    ax.axhline(cc_lda.mean(), color="#1565C0", lw=1.5, ls="--",
               label="CC mean ±1σ")
    ax.axhline(ad_lda.mean(), color="#C62828", lw=1.2, ls=":", alpha=0.7,
               label="AD baseline")
    if metric == "lda":
        ax.axhline(ad_cc_mid, color="gray", lw=0.8, ls="-.", alpha=0.5,
                   label="Midpoint")
    for pt, col in PERT_COLS.items():
        alphas = ALPHA_GRIDS[pt]
        if metric == "lda":
            mat  = np.array([[results[pt][ai][pid][0] for pid in ad_pids]
                              for ai in range(len(alphas))])
        else:
            mat  = np.array([[results[pt][ai][pid][1] for pid in ad_pids]
                              for ai in range(len(alphas))])
        m = mat.mean(1); s = mat.std(1)
        ax.fill_between(alphas, m-s, m+s, alpha=0.18, color=col)
        ax.plot(alphas, m, "-o", ms=5, lw=2.0, color=col, label=LABELS[pt])
    ax.set_xlabel("α", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    "Extended perturbation — top-5 up to α=10  (Cond. B, K=25, σ=0.05)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pertB_extended.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved pertB_extended.png")
print("\nDone.")
