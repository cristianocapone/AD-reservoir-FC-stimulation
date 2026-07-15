"""
learning_curve_direct.py
========================
Learning curve: LOPO BAL-ACC vs N patients per class.

G-space SVD strategy (user-selected):
  ALL 712 sessions treated independently.
  Each session → own W → own Wproj (via session-specific X-SVD).
  SVD on all 712 Wproj vectors (712×712 Gram matrix).
  Patient G-score = mean of their sessions' projections.

LDA uses patient-level G-scores; N varies per class (up to 40 AD).
Per-fold binary decisions → BAL-ACC chance = 0.50 always.

Condition B parameters: sigma=0.05, K_LDA=25, sr=0.95.
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
RNG_SEED   = 42
N_SITES    = 121
N_PC_MODEL = 50
K_PC       = 200
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
K_LDA      = 25
SR         = 0.95
TS_ROOT    = "./timeseries"
OUT_DIR    = "."

N_GRID = [4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
N_REPS = 30

# ── load ALL sessions (no cap) ─────────────────────────────────────────────────
print("Loading data (all sessions) ...")
signals, labels_raw, pid_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    for fname in sorted(f for f in os.listdir(folder) if f.endswith(".npy")):
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            pid_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

pid_raw    = np.array(pid_raw)
labels_raw = np.array(labels_raw)
N_subj     = len(signals)           # 712 sessions
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)   # 186 patients
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_idx = np.where(patient_labels == 0)[0]
ad_idx = np.where(patient_labels == 1)[0]
print(f"  {N_subj} sessions | {N_patients} patients "
      f"({len(cc_idx)} CC, {len(ad_idx)} AD)")

# ── population PCA from all sessions ──────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals_p, evecs_p = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs_p[:, np.argsort(evals_p)[::-1]][:, :N_PC_MODEL]

# ── reservoir ──────────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

# ── TF pass — all 712 sessions ─────────────────────────────────────────────────
print("TF pass (all sessions) ...")
sess_X, sess_Y = {}, {}
for idx in trange(N_subj, desc="  TF"):
    s = signals[idx]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    Xraw = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        Xraw.append(res.X.copy())
    Xf = np.array(Xraw)[TIMES_SKIP:]
    sess_X[idx] = Xf
    sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T
print("  TF done.")

# ── fit W + X-SVD for every session independently ─────────────────────────────
print("\nFitting W for every session ...")
rng_w    = np.random.default_rng(RNG_SEED + 1)
sess_Vtk = {}   # per-session X-space SVD
sess_W   = {}   # per-session W

print("  X-space SVDs ...")
for idx in trange(N_subj, desc="  SVD", leave=False):
    Xca = sess_X[idx].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    sess_Vtk[idx] = Vtx[:kk]

print("  W fits ...")
for idx in trange(N_subj, desc="  W-fit", leave=False):
    Xc = sess_X[idx]; Yc = sess_Y[idx]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    sess_W[idx] = np.linalg.pinv(Xc + noise) @ Yc

def project_W_sess(W, idx):
    Vt_k = sess_Vtk[idx]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

# ── G-space SVD from all 712 sessions ─────────────────────────────────────────
print("\nComputing G-space (SVD on all 712 sessions) ...")
Wproj_sess = np.array([project_W_sess(sess_W[idx], idx)
                       for idx in range(N_subj)], dtype=np.float64)  # (712, D)
Wmean_s    = Wproj_sess.mean(0)
Wproj_sc   = Wproj_sess - Wmean_s                                     # centred

# Gram matrix (712×712)
print("  Building 712×712 Gram matrix ...")
C_s = Wproj_sc @ Wproj_sc.T
evals_s, evecs_s = np.linalg.eigh(C_s)
order_s   = np.argsort(evals_s)[::-1]
evals_s   = np.maximum(evals_s[order_s], 0.0)
evecs_s   = evecs_s[:, order_s]

# Session-level G-scores: (712, K_LDA)
G_sess = evecs_s[:, :K_LDA] * np.sqrt(evals_s[:K_LDA])

var_frac = evals_s[:K_LDA].sum() / evals_s.sum() * 100
print(f"  G_sess shape: {G_sess.shape}  "
      f"(top {K_LDA} PCs: {var_frac:.1f}% of variance)")

# ── patient G-scores: mean over sessions ──────────────────────────────────────
print("Averaging sessions → patient G-scores ...")
G_all = np.zeros((N_patients, K_LDA), dtype=np.float64)
for pi, pid in enumerate(unique_pids):
    sess_indices = patient_sids[pid]          # session indices for this patient
    G_all[pi] = G_sess[sess_indices].mean(0)  # mean over sessions

print(f"  G_all shape: {G_all.shape}  "
      f"(sessions/patient: CC mean {len(cc_idx)}/{sum(len(patient_sids[unique_pids[i]]) for i in cc_idx):.0f}, "
      f"AD mean {len(ad_idx)}/{sum(len(patient_sids[unique_pids[i]]) for i in ad_idx):.0f})")

# ── helpers ────────────────────────────────────────────────────────────────────
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


def lopo_cv(S_global, y_S, k_use):
    """LOPO on patient subset S_global using fixed global G-space."""
    n     = len(S_global)
    G_sub = G_all[S_global, :k_use]
    preds  = np.full(n, np.nan)
    scores = np.full(n, np.nan)

    for i in range(n):
        mask = np.arange(n) != i
        G_tr = G_sub[mask]; y_tr = y_S[mask]
        G_te = G_sub[i]

        Xb, yb = _balance(G_tr, y_tr, seed=RNG_SEED)
        try:
            lda = _LDA().fit(Xb, yb)
        except Exception:
            continue

        z_tr = lda.transform(G_tr)
        if z_tr[y_tr==0].mean() > z_tr[y_tr==1].mean():
            lda.w_ *= -1
            z_tr    = lda.transform(G_tr)
        thr = 0.5*(z_tr[y_tr==0].mean() + z_tr[y_tr==1].mean())

        z_te      = lda.transform(G_te.reshape(1,-1))[0]
        preds[i]  = float(z_te >= thr)
        scores[i] = z_te - thr

    return preds, scores


# ── learning curve ─────────────────────────────────────────────────────────────
print(f"\nLearning curve: N in {N_GRID}, {N_REPS} reps ...")
N_max      = min(len(cc_idx), len(ad_idx))
N_GRID_eff = [N for N in N_GRID if N <= N_max]

results_bal = {N: [] for N in N_GRID_eff}
results_auc = {N: [] for N in N_GRID_eff}

for N in N_GRID_eff:
    k_use = min(K_LDA, 2*N - 2)
    print(f"\n  N={N:3d}  (2N={2*N}, k={k_use}, {N_REPS} reps)", flush=True)
    for rep in range(N_REPS):
        rng_rep = np.random.default_rng(RNG_SEED + rep*10000 + N)
        sel_cc  = rng_rep.choice(cc_idx, N, replace=False)
        sel_ad  = rng_rep.choice(ad_idx, N, replace=False)
        S       = np.concatenate([sel_cc, sel_ad])
        y_S     = patient_labels[S]

        pr, sc = lopo_cv(S, y_S, k_use)
        valid  = np.isfinite(pr)
        if valid.sum() < 4: continue

        pr_v = pr[valid]; sc_v = sc[valid]; y_v = y_S[valid]
        sens = np.mean(pr_v[y_v==1] == 1)
        spec = np.mean(pr_v[y_v==0] == 0)
        bal  = 0.5*(sens + spec)
        try:    auc = roc_auc_score(y_v, sc_v)
        except: auc = np.nan

        results_bal[N].append(bal)
        results_auc[N].append(auc)

    ba = np.array(results_bal[N]); au = np.array(results_auc[N])
    print(f"    BAL-ACC={ba.mean():.4f}±{ba.std():.4f}  "
          f"AUROC={au.mean():.4f}±{au.std():.4f}", flush=True)

# ── save ───────────────────────────────────────────────────────────────────────
k_eff_arr = np.array([min(K_LDA, 2*N-2) for N in N_GRID_eff])
np.savez(f"{OUT_DIR}/learning_curve_direct_data.npz",
         N_grid   = np.array(N_GRID_eff),
         bal_mean = np.array([np.mean(results_bal[N]) for N in N_GRID_eff]),
         bal_std  = np.array([np.std(results_bal[N])  for N in N_GRID_eff]),
         auc_mean = np.array([np.mean(results_auc[N]) for N in N_GRID_eff]),
         auc_std  = np.array([np.std(results_auc[N])  for N in N_GRID_eff]),
         k_eff    = k_eff_arr)
print("\nSaved learning_curve_direct_data.npz")

# ── plot ───────────────────────────────────────────────────────────────────────
print("Plotting ...")
N_arr   = np.array(N_GRID_eff)
ba_mean = np.array([np.mean(results_bal[N]) for N in N_GRID_eff])
ba_std  = np.array([np.std(results_bal[N])  for N in N_GRID_eff])
au_mean = np.array([np.mean(results_auc[N]) for N in N_GRID_eff])
au_std  = np.array([np.std(results_auc[N])  for N in N_GRID_eff])
REF_BA  = 0.6833; REF_AUC = 0.6722

fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="white")
for ax, mean_v, std_v, ref_val, ylabel, title in [
        (axes[0], ba_mean, ba_std, REF_BA,
         "LOPO Balanced Accuracy", "BAL-ACC vs N patients / class"),
        (axes[1], au_mean, au_std, REF_AUC,
         "LOPO AUROC", "AUROC vs N patients / class")]:

    ax.axhline(0.50, color="gray", ls="--", lw=1.0, alpha=0.6, label="chance (0.50)")
    ax.axhline(ref_val, color="#C62828", ls="-.", lw=1.5, alpha=0.85,
               label=f"previous (76-pt G-space): {ref_val:.3f}")

    capped = k_eff_arr < K_LDA
    if capped.any():
        ax.axvspan(N_arr[capped].min()-0.5, N_arr[capped].max()+0.5,
                   alpha=0.09, color="orange", label=f"K_LDA capped (< {K_LDA})")

    ax.fill_between(N_arr, mean_v - std_v, mean_v + std_v,
                    alpha=0.25, color="#1565C0")
    ax.plot(N_arr, mean_v, "-o", ms=7, lw=2.5, color="#1565C0",
            label="LOPO (SVD: all 712 sessions, pat. mean)")

    for xi, (Nv, k, m) in enumerate(zip(N_arr, k_eff_arr, mean_v)):
        ax.text(Nv, m + std_v[xi] + 0.007,
                f"k={k}", ha="center", va="bottom", fontsize=7, color="#555")

    ax.set_xlabel("N patients per class", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(N_arr)
    ax.set_ylim(0.30, 0.90)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    "Learning curve — G-space SVD from all 712 sessions (patient G-score = session mean)\n"
    "Condition B: σ=0.05, K_LDA=25  |  LDA: LOPO with per-fold threshold",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/learning_curve_direct.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved learning_curve_direct.png")
print("\nDone.")
