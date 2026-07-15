"""
noise_ba_sweep.py
=================
Sweep noise_size for per-patient W and report AUROC at each level.
Tests: [0.025, 0.05, 0.10, 0.50]
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"

NOISE_VALS  = [0.025, 0.05, 0.10, 0.50]
K_MAX       = 15   # k-sweep range for LOPO LDA

# ── LDA + helpers ─────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = (X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1) + 1e-6*np.eye(X0.shape[1])
        w  = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def predict(self, X):
        return np.where(X @ self.w_ >= self.thr_, self.classes_[1], self.classes_[0])
    def score(self, X):
        """Continuous LDA score (higher = class 1 / AD)."""
        return (X @ self.w_).ravel()

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    if n == 0: return X, y
    sel = np.concatenate([rng2.choice(c0,n,replace=False), rng2.choice(c1,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

def _auc(yt, ys):
    """AUROC from true labels and continuous LDA scores."""
    try:
        return roc_auc_score(yt, ys)
    except Exception:
        return 0.5

def _class_acc(yt, yp):
    cc = (yp[yt==0]==0).mean() if (yt==0).any() else float('nan')
    ad = (yp[yt==1]==1).mean() if (yt==1).any() else float('nan')
    return cc, ad

def lopo_flat(G, y, k):
    """LOPO LDA — returns (true labels, predicted labels, LDA scores)."""
    n = len(y); all_t, all_p, all_s = [], [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xb, yb = _balance(G[tr, :k], y[tr], seed=RNG_SEED + i)
        if len(np.unique(yb)) < 2: continue
        lda = _LDA().fit(Xb, yb)
        all_t.extend(y[[i]].tolist())
        all_p.extend(lda.predict(G[[i], :k]).tolist())
        all_s.extend(lda.score(G[[i], :k]).tolist())
    return np.array(all_t), np.array(all_p), np.array(all_s)

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, patient_ids_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            patient_ids_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

patient_ids_raw = np.array(patient_ids_raw)
labels_raw      = np.array(labels_raw)
N_subj          = len(signals)
unique_pids     = np.unique(patient_ids_raw)
N_patients      = len(unique_pids)
patient_sids    = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
patient_labels  = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
print(f"  Sessions: {N_subj}   Patients: {N_patients}  "
      f"(CC={( patient_labels==0).sum()}, AD={(patient_labels==1).sum()})")

# ── Population PCA ─────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Reservoir + teacher-forced X,Y (done once) ────────────────────────────────
print("Reservoir + teacher-forced pass ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

sess_X, sess_Y = {}, {}
for idx in trange(N_subj, desc="  TF pass"):
    s      = signals[idx]
    T_s    = s.shape[1]
    target = (s.T @ ev50 @ ev50.T).T
    res.T  = T_s; res.reset()
    X_raw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    X_fit = np.array(X_raw)[TIMES_SKIP:]
    Y_fit = target[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T
    sess_X[idx] = X_fit
    sess_Y[idx] = Y_fit

# ── Sweep noise ────────────────────────────────────────────────────────────────
print(f"\n{'noise':>8}  {'best_k':>7}  {'AUC':>7}  {'CC':>6}  {'AD':>6}")
print("-" * 44)

results = []
for noise_size in NOISE_VALS:
    rng_p = np.random.default_rng(RNG_SEED + 1)
    W_proj_list = []

    for pid in tqdm(unique_pids, desc=f"  noise={noise_size}", leave=False):
        idxs   = patient_sids[pid]
        X_coll = np.vstack([sess_X[i] for i in idxs])
        Y_coll = np.vstack([sess_Y[i] for i in idxs])
        noise  = rng_p.normal(0, noise_size, X_coll.shape)
        W      = np.linalg.pinv(X_coll + noise) @ Y_coll

        W_T = W.T.astype(np.float64)
        _, sx, Vtx = np.linalg.svd(X_coll.astype(np.float64), full_matrices=False)
        k_pc = min(K_PC, int((sx > 1e-8).sum()))
        Vt_k = Vtx[:k_pc]
        W_proj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

    W_stack = np.array(W_proj_list)
    Wcent   = W_stack - W_stack.mean(0)
    _, _, Vt_svd = np.linalg.svd(Wcent, full_matrices=False)
    M_eff   = N_patients - 1
    G_pat   = Wcent @ Vt_svd[:M_eff].T

    # k-sweep (AUROC)
    kv = list(range(1, K_MAX + 1))
    auc_k = []
    for k in kv:
        yt, yp, ys = lopo_flat(G_pat, patient_labels, k)
        auc_k.append(_auc(yt, ys))
    auc_k = np.array(auc_k)

    best_k = kv[int(np.argmax(auc_k))]
    yt, yp, ys = lopo_flat(G_pat, patient_labels, best_k)
    auc    = _auc(yt, ys)
    cc, ad = _class_acc(yt, yp)

    print(f"  {noise_size:>6.3f}  {best_k:>7}  {auc:.4f}  {cc*100:>5.0f}%  {ad*100:>5.0f}%")
    results.append((noise_size, best_k, auc, cc, ad, auc_k.copy()))

print("\n[top-3 k values per noise]")
for noise_size, best_k, auc, cc, ad, auc_k in results:
    top3 = sorted(range(len(auc_k)), key=lambda i: auc_k[i], reverse=True)[:3]
    top3_str = "  ".join(f"k={kv[i]}:{auc_k[i]:.4f}" for i in top3)
    print(f"  noise={noise_size}  {top3_str}")
