"""
klda_sweep.py
=============
Sweep K_LDA at fixed N=40 (all AD patients) using the 712-session G-space.
Shows whether more PCs improve LOPO BAL-ACC / AUROC.
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

RNG_SEED   = 42
N_SITES    = 121
N_PC_MODEL = 50
K_PC       = 200
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
SR         = 0.95
TS_ROOT    = "./timeseries"
OUT_DIR    = "."

N_REPS  = 30
N_FIXED = 40          # fix N = all AD patients

# K_LDA grid: from 5 to as large as N_FIXED-2 allows (38 max at N=40)
# but also check if more PCs beyond that help by testing at N=40 with larger K
# (at N=40, balanced training = 39 per class → k_use capped at 38 anyway)
K_GRID = [5, 10, 15, 20, 25, 30, 35, 38]

# ── load all sessions ──────────────────────────────────────────────────────────
print("Loading data ...")
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
N_subj     = len(signals)
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_idx = np.where(patient_labels == 0)[0]
ad_idx = np.where(patient_labels == 1)[0]
print(f"  {N_subj} sessions | {N_patients} patients ({len(cc_idx)} CC, {len(ad_idx)} AD)")

# ── population PCA ─────────────────────────────────────────────────────────────
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

# ── TF pass ────────────────────────────────────────────────────────────────────
print("TF pass ...")
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

# ── W + X-SVD per session ──────────────────────────────────────────────────────
print("Fitting W (all sessions) ...")
rng_w    = np.random.default_rng(RNG_SEED + 1)
sess_Vtk = {}; sess_W = {}
for idx in trange(N_subj, desc="  SVD+W", leave=False):
    Xca = sess_X[idx].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    sess_Vtk[idx] = Vtx[:kk]
    noise = rng_w.normal(0, SIGMA, sess_X[idx].shape)
    sess_W[idx] = np.linalg.pinv(sess_X[idx] + noise) @ sess_Y[idx]

def project_W_sess(W, idx):
    Vt_k = sess_Vtk[idx]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

# ── full G-space SVD (all sessions, keep ALL eigenvectors) ────────────────────
print("\nBuilding G-space (all sessions, keeping all PCs) ...")
Wproj_sess = np.array([project_W_sess(sess_W[idx], idx)
                       for idx in range(N_subj)], dtype=np.float64)
Wmean_s  = Wproj_sess.mean(0)
Wproj_sc = Wproj_sess - Wmean_s

print("  Computing Gram matrix ...")
C_s = Wproj_sc @ Wproj_sc.T                   # (N_subj, N_subj)
evals_s, evecs_s = np.linalg.eigh(C_s)
order_s  = np.argsort(evals_s)[::-1]
evals_s  = np.maximum(evals_s[order_s], 0.0)
evecs_s  = evecs_s[:, order_s]

# Print variance explained
K_max = min(N_FIXED - 2, len(evals_s))   # hard cap at what LDA can use
cum_var = np.cumsum(evals_s) / evals_s.sum() * 100
print(f"  Variance explained:")
for k in [10, 20, 25, 30, 35, 38, 50, 75, 100]:
    if k <= len(evals_s):
        print(f"    top {k:3d} PCs: {cum_var[k-1]:.1f}%")

# Full session G-scores (keep up to K_max columns for efficiency)
G_sess_full = evecs_s[:, :K_max] * np.sqrt(evals_s[:K_max])   # (N_subj, K_max)

# Patient G-scores for each K: mean over sessions
print("Averaging sessions → patient G-scores ...")
G_pat_full = np.zeros((N_patients, K_max), dtype=np.float64)
for pi, pid in enumerate(unique_pids):
    G_pat_full[pi] = G_sess_full[patient_sids[pid]].mean(0)

# ── LDA helpers ───────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w
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
    n     = len(S_global)
    G_sub = G_pat_full[S_global, :k_use]
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
            z_tr = lda.transform(G_tr)
        thr = 0.5*(z_tr[y_tr==0].mean() + z_tr[y_tr==1].mean())
        z_te = lda.transform(G_te.reshape(1,-1))[0]
        preds[i]  = float(z_te >= thr)
        scores[i] = z_te - thr
    return preds, scores

# ── K_LDA sweep at fixed N=N_FIXED ────────────────────────────────────────────
print(f"\nK_LDA sweep at N={N_FIXED}, {N_REPS} reps ...")
K_GRID_eff = [k for k in K_GRID if k <= K_max]

ba_list, ba_std_list = [], []
au_list, au_std_list = [], []

for k in K_GRID_eff:
    bals, aucs = [], []
    for rep in range(N_REPS):
        rng_rep = np.random.default_rng(RNG_SEED + rep*10000 + k)
        sel_cc  = rng_rep.choice(cc_idx, N_FIXED, replace=False)
        sel_ad  = rng_rep.choice(ad_idx, N_FIXED, replace=False)
        S       = np.concatenate([sel_cc, sel_ad])
        y_S     = patient_labels[S]

        pr, sc = lopo_cv(S, y_S, k)
        valid  = np.isfinite(pr)
        if valid.sum() < 4: continue
        pr_v = pr[valid]; sc_v = sc[valid]; y_v = y_S[valid]
        sens = np.mean(pr_v[y_v==1] == 1)
        spec = np.mean(pr_v[y_v==0] == 0)
        bals.append(0.5*(sens+spec))
        try:    aucs.append(roc_auc_score(y_v, sc_v))
        except: pass

    ba = np.array(bals); au = np.array(aucs)
    ba_list.append(ba.mean()); ba_std_list.append(ba.std())
    au_list.append(au.mean()); au_std_list.append(au.std())
    print(f"  K={k:3d}  BAL-ACC={ba.mean():.4f}±{ba.std():.4f}  "
          f"AUROC={au.mean():.4f}±{au.std():.4f}  "
          f"(var={cum_var[k-1]:.1f}%)", flush=True)

# ── plot ───────────────────────────────────────────────────────────────────────
K_arr = np.array(K_GRID_eff)
ba_arr = np.array(ba_list); ba_std_arr = np.array(ba_std_list)
au_arr = np.array(au_list); au_std_arr = np.array(au_std_list)

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
for ax, mean_v, std_v, ylabel, title in [
        (axes[0], ba_arr, ba_std_arr, "LOPO BAL-ACC", f"BAL-ACC vs K_LDA  (N={N_FIXED}/class)"),
        (axes[1], au_arr, au_std_arr, "LOPO AUROC",   f"AUROC vs K_LDA  (N={N_FIXED}/class)")]:

    ax.axhline(0.50, color="gray", ls="--", lw=1, alpha=0.6, label="chance")
    ax.fill_between(K_arr, mean_v-std_v, mean_v+std_v, alpha=0.25, color="#1565C0")
    ax.plot(K_arr, mean_v, "-o", ms=7, lw=2.5, color="#1565C0",
            label=f"mean±std ({N_REPS} reps)")

    # annotate variance explained
    for ki, (k, m) in enumerate(zip(K_arr, mean_v)):
        ax.text(k, m + std_v[ki] + 0.005,
                f"{cum_var[k-1]:.0f}%", ha="center", va="bottom",
                fontsize=7, color="#555")

    ax.set_xlabel("K_LDA (number of PCs)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(K_arr)
    ax.set_ylim(0.35, 0.85)
    ax.legend(fontsize=8, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    f"K_LDA sweep — N={N_FIXED}/class, G-space: all 712 sessions\n"
    "Annotations: % variance explained by top K PCs",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/klda_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved klda_sweep.png")
print("Done.")
