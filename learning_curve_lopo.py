"""
learning_curve_lopo.py
======================
Learning curve: LOPO balanced accuracy vs number of patients per class.

For each N in N_GRID, repeatedly:
  1. Sample N CC and N AD patients at random.
  2. Run full LOPO-CV on those 2N patients:
     - For each left-out patient i:
       a. Compute G-space (kernel PCA on training 2N-1 patients only).
       b. Project test patient into that G-space.
       c. Train balanced LDA on 2N-1 training G-scores.
       d. Score test patient → collect all 2N scores.
  3. Compute BAL-ACC and AUROC from the 2N LOO scores.

G-space is computed from training patients only (proper LOPO),
using kernel PCA on the precomputed 76×76 Gram matrix — avoids
recomputing the SVD of the full 2N×242000 matrix per fold.

Condition B: sigma=0.05, K_LDA=25 (capped at n_train-2), sr=0.95.

Outputs:
  learning_curve_lopo.png
  learning_curve_lopo_data.npz
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

N_GRID = [4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35]
N_REPS = 30        # random repetitions per N value

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
cc_idx = np.where(patient_labels == 0)[0]   # indices into unique_pids
ad_idx = np.where(patient_labels == 1)[0]
print(f"  {N_patients} patients ({len(cc_idx)} CC, {len(ad_idx)} AD), max N/class = {min(len(cc_idx), len(ad_idx))}")

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
print("  TF done.")

# ── condition B: single-session W per patient ──────────────────────────────────
print("\nFitting W (sigma=0.05, single session) ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
first_idx   = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single = {pid: sess_X[first_idx[pid]] for pid in unique_pids}
patY_single = {pid: sess_Y[first_idx[pid]] for pid in unique_pids}

# Pre-compute per-patient SVD of X → Vtk (needed for project_W)
print("  Pre-computing X-space SVDs ...")
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

# ── precompute Gram matrix ─────────────────────────────────────────────────────
print("\nPrecomputing Gram matrix K (76×76) ...")
Wproj_all = np.array([project_W(pat_W[pid], pid) for pid in unique_pids],
                     dtype=np.float64)   # (N_patients, D)
K_raw = Wproj_all @ Wproj_all.T          # (N_patients, N_patients) — uncentered
print(f"  Wproj shape: {Wproj_all.shape}, K shape: {K_raw.shape}")

# ── LDA + balance helpers ──────────────────────────────────────────────────────
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

def kpca_lopo(S_global, y_S):
    """
    Kernel PCA LOPO on the patient subset S_global (array of global patient indices).
    Returns array of LOO LDA scores (length 2N).
    """
    n     = len(S_global)
    scores = np.full(n, np.nan)

    for i in range(n):
        mask    = np.arange(n) != i
        S_tr    = S_global[mask]           # (n-1,) global indices
        y_tr    = y_S[mask]
        n_tr    = len(S_tr)
        k_use   = min(K_LDA, n_tr - 2)    # LDA needs k < n_class_min
        if k_use < 1: continue

        # Training Gram matrix (centered)
        K_tr    = K_raw[np.ix_(S_tr, S_tr)]
        H_n     = np.eye(n_tr) - 1.0/n_tr
        K_tr_c  = H_n @ K_tr @ H_n

        # Eigendecomposition
        evals, evecs = np.linalg.eigh(K_tr_c)
        evals = np.maximum(evals, 0.0)
        order = np.argsort(evals)[::-1]
        evals = evals[order]; evecs = evecs[:, order]
        pos   = evals > 1e-10
        if not pos.any(): continue
        evals = evals[pos]; evecs = evecs[:, pos]
        k_use = min(k_use, len(evals))

        # Training G-scores (rows = patients, cols = components)
        G_tr = evecs[:, :k_use] * np.sqrt(evals[:k_use])

        # Test-point kernel (centered)
        t       = S_global[i]
        k_te    = K_raw[S_tr, t]               # (n_tr,)
        k_te_c  = (k_te
                   - k_te.mean()               # mean over training of K(x_l, x_test)
                   - K_tr.mean(1)              # mean over training of K(x_j, x_l)
                   + K_tr.mean())              # grand mean of training Gram
        g_test  = k_te_c @ evecs[:, :k_use] / np.sqrt(evals[:k_use])

        # Balanced LDA
        Xb, yb = _balance(G_tr, y_tr, seed=RNG_SEED)
        try:
            lda = _LDA().fit(Xb, yb)
        except Exception:
            continue

        # Force orientation: AD (1) > CC (0)
        z_tr = lda.transform(G_tr)
        if z_tr[y_tr==0].mean() > z_tr[y_tr==1].mean():
            lda.w_ *= -1

        scores[i] = lda.transform(g_test.reshape(1, -1))[0]

    return scores

# ── learning curve sweep ───────────────────────────────────────────────────────
print(f"\nLearning curve: N in {N_GRID}, {N_REPS} reps each ...")
N_max = min(len(cc_idx), len(ad_idx))
N_GRID_eff = [N for N in N_GRID if N <= N_max]

results_bal = {N: [] for N in N_GRID_eff}
results_auc = {N: [] for N in N_GRID_eff}

for N in N_GRID_eff:
    print(f"\n  N={N:3d}  (2N={2*N} patients per rep, {N_REPS} reps)", flush=True)
    for rep in range(N_REPS):
        rng_rep = np.random.default_rng(RNG_SEED + rep * 10000 + N)

        # Random balanced subsample
        sel_cc = rng_rep.choice(cc_idx, N, replace=False)
        sel_ad = rng_rep.choice(ad_idx, N, replace=False)
        S      = np.concatenate([sel_cc, sel_ad])   # global patient indices
        y_S    = patient_labels[S]

        sc = kpca_lopo(S, y_S)
        valid = np.isfinite(sc)
        if valid.sum() < 4:
            continue

        sc_v = sc[valid]; y_v = y_S[valid]
        thr  = 0.5*(sc_v[y_v==0].mean() + sc_v[y_v==1].mean())
        pred = (sc_v >= thr).astype(int)
        sens = np.mean(pred[y_v==1] == 1)
        spec = np.mean(pred[y_v==0] == 0)
        bal  = 0.5*(sens + spec)
        auc  = roc_auc_score(y_v, sc_v)

        results_bal[N].append(bal)
        results_auc[N].append(auc)

    ba = np.array(results_bal[N])
    au = np.array(results_auc[N])
    print(f"    BAL-ACC={ba.mean():.4f}±{ba.std():.4f}  "
          f"AUROC={au.mean():.4f}±{au.std():.4f}  "
          f"(k_eff={min(K_LDA, 2*N-2)})", flush=True)

# ── save ───────────────────────────────────────────────────────────────────────
np.savez(f"{OUT_DIR}/learning_curve_lopo_data.npz",
         N_grid   = np.array(N_GRID_eff),
         bal_mean = np.array([np.mean(results_bal[N]) for N in N_GRID_eff]),
         bal_std  = np.array([np.std(results_bal[N])  for N in N_GRID_eff]),
         auc_mean = np.array([np.mean(results_auc[N]) for N in N_GRID_eff]),
         auc_std  = np.array([np.std(results_auc[N])  for N in N_GRID_eff]),
         k_eff    = np.array([min(K_LDA, 2*N-2)       for N in N_GRID_eff]))
print("\nSaved learning_curve_lopo_data.npz")

# ── plot ───────────────────────────────────────────────────────────────────────
print("Plotting ...")

N_arr   = np.array(N_GRID_eff)
ba_mean = np.array([np.mean(results_bal[N]) for N in N_GRID_eff])
ba_std  = np.array([np.std(results_bal[N])  for N in N_GRID_eff])
au_mean = np.array([np.mean(results_auc[N]) for N in N_GRID_eff])
au_std  = np.array([np.std(results_auc[N])  for N in N_GRID_eff])
k_eff   = np.array([min(K_LDA, 2*N-2)       for N in N_GRID_eff])

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")

for ax, mean_v, std_v, ylabel, title, chance in [
        (axes[0], ba_mean, ba_std,
         "LOPO Balanced Accuracy", "BAL-ACC vs N patients / class", 0.50),
        (axes[1], au_mean, au_std,
         "LOPO AUROC", "AUROC vs N patients / class", 0.50)]:

    ax.axhline(chance, color="gray", ls="--", lw=1, alpha=0.6, label="chance (0.50)")

    # shade where K_LDA is capped
    capped = k_eff < K_LDA
    if capped.any():
        x_cap = N_arr[capped]
        ax.axvspan(x_cap.min()-0.5, x_cap.max()+0.5,
                   alpha=0.08, color="orange",
                   label=f"K_LDA capped (< {K_LDA})")

    ax.fill_between(N_arr, mean_v - std_v, mean_v + std_v,
                    alpha=0.25, color="#1565C0")
    ax.plot(N_arr, mean_v, "-o", ms=7, lw=2.5, color="#1565C0",
            label=f"mean ± std  ({N_REPS} reps)")

    # annotate k_eff
    for xi, (N, k, m) in enumerate(zip(N_arr, k_eff, mean_v)):
        ax.text(N, m + std_v[xi] + 0.005,
                f"k={k}", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xlabel("N patients per class", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(N_arr)
    ax.set_ylim(0.35, 1.05)
    ax.legend(fontsize=8, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    "Learning curve — LOPO CV with proper per-fold SVD (kernel PCA)\n"
    "Condition B: σ=0.05, K_LDA=25 (capped when N is small)",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/learning_curve_lopo.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved learning_curve_lopo.png")
print("\nDone.")
