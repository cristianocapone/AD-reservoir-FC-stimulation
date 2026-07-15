"""
check_leakage.py
================
Proper nested LOPO to check whether the 84.7% BA for per-patient G-scores
holds without k-selection bias.

Pipeline:
  1. Outer LOPO: leave out patient i
  2. Inner LOPO on training patients (j != i): select best k
  3. Train final LDA on all training patients with best_k_inner
  4. Evaluate on patient i

Compares:
  A) Flat k-sweep (current approach, k selected on all test folds combined)
  B) Nested LOPO  (k selected inside each outer fold — no leakage)
  C) Fixed k=1, k=2, k=3  (no hyperparameter search at all)
"""

import os, sys, warnings
import numpy as np
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
noise_size  = 0.025
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"

# ── LDA ───────────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw  = (X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1) + 1e-6*np.eye(X0.shape[1])
        w   = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def predict(self, X):
        return np.where(X @ self.w_ >= self.thr_, self.classes_[1], self.classes_[0])

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    if n == 0: return X, y
    sel = np.concatenate([rng2.choice(c0,n,replace=False), rng2.choice(c1,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

def _bal_acc(yt, yp):
    return np.mean([(yp[yt==c]==c).mean() for c in np.unique(yt)])

def _class_acc(yt, yp):
    cc = (yp[yt==0]==0).mean() if (yt==0).any() else np.nan
    ad = (yp[yt==1]==1).mean() if (yt==1).any() else np.nan
    return cc, ad

# ── Load data ─────────────────────────────────────────────────────────────────
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

unique_pids    = np.unique(patient_ids_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])

print(f"  Sessions: {N_subj}  (CC={(labels_raw==0).sum()}, AD={(labels_raw==1).sum()})")
print(f"  Patients: {N_patients}  (CC={(patient_labels==0).sum()}, AD={(patient_labels==1).sum()})")

# ── Population PCA ────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Build reservoir + per-session X, Y ───────────────────────────────────────
print("Building reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

rng_r = np.random.default_rng(RNG_SEED)
sess_X, sess_Y = {}, {}

for idx in trange(N_subj, desc="Teacher-forced pass"):
    s = signals[idx]; T_s = s.shape[1]
    res.T = T_s; res.reset()
    target = (s.T @ ev50 @ ev50.T).T
    X_raw = []
    for t in range(T_s - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    X_fit = np.array(X_raw)[TIMES_SKIP:]
    Y_fit = target[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T
    sess_X[idx] = X_fit
    sess_Y[idx] = Y_fit

# ── Per-patient W + G-scores (global SVD) ────────────────────────────────────
print("Per-patient W ...")
rng_p = np.random.default_rng(RNG_SEED + 1)
W_proj_list = []

for pid in tqdm(unique_pids, desc="  Patients"):
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

W_stack  = np.array(W_proj_list)              # (N_patients, D)
W_mean   = W_stack.mean(0)
Wcent    = W_stack - W_mean
_, sv, Vt_svd = np.linalg.svd(Wcent, full_matrices=False)
M_eff    = N_patients - 1                     # = 75
G_pat    = Wcent @ Vt_svd[:M_eff].T           # (N_patients, 75)

print(f"  G_pat shape: {G_pat.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# A) FLAT k-SWEEP (current approach — k selected post-hoc using all test folds)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Flat k-sweep (current approach) ...")

kv = list(range(1, M_eff + 1))

def lopo_flat(G, y, k):
    n = len(y); all_t, all_p = [], []
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        Xtr, ytr = G[tr, :k], y[tr]
        Xte, yte = G[[i], :k], y[[i]]
        Xb, yb   = _balance(Xtr, ytr, seed=RNG_SEED + i)
        if len(np.unique(yb)) < 2: continue
        lda = _LDA().fit(Xb, yb)
        all_t.extend(yte.tolist()); all_p.extend(lda.predict(Xte).tolist())
    return np.array(all_t), np.array(all_p)

acc_flat = []
for k in tqdm(kv, desc="  k-sweep"):
    yt, yp = lopo_flat(G_pat, patient_labels, k)
    acc_flat.append(_bal_acc(yt, yp))
acc_flat = np.array(acc_flat)

best_k_flat = kv[int(np.argmax(acc_flat))]
ba_flat = acc_flat.max()
yt_flat, yp_flat = lopo_flat(G_pat, patient_labels, best_k_flat)
cc_flat, ad_flat = _class_acc(yt_flat, yp_flat)
print(f"  best k={best_k_flat}  BA={ba_flat*100:.1f}%  CC={cc_flat*100:.0f}%  AD={ad_flat*100:.0f}%")
print(f"  [top 5 BA values: {sorted(acc_flat*100, reverse=True)[:5]}]")

# ══════════════════════════════════════════════════════════════════════════════
# B) FIXED k (no hyperparameter search at all)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[B] Fixed k (no search) ...")
print(f"  {'k':>4}  {'BA':>7}  {'CC':>6}  {'AD':>6}")
for k_fix in [1, 2, 3, 4, 5]:
    yt, yp = lopo_flat(G_pat, patient_labels, k_fix)
    ba = _bal_acc(yt, yp); cc, ad = _class_acc(yt, yp)
    print(f"  k={k_fix}  BA={ba*100:.1f}%  CC={cc*100:.0f}%  AD={ad*100:.0f}%")

# ══════════════════════════════════════════════════════════════════════════════
# C) NESTED LOPO (proper: k selected inside each outer fold)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C] Nested LOPO (k selected inside each outer fold) ...")
print("  This takes a while (~N^2 * K LDA fits) ...")

kv_inner = list(range(1, 16))   # inner search over k=1..15 (covers the typical range)
# Expanding to k=1..75 inner would be too slow; k=1..15 captures the plausible optimum

all_t_nested, all_p_nested, k_chosen = [], [], []

for i in tqdm(range(N_patients), desc="  Outer fold"):
    tr_idx = np.array([j for j in range(N_patients) if j != i])
    G_tr   = G_pat[tr_idx]
    y_tr   = patient_labels[tr_idx]
    G_te   = G_pat[[i]]
    y_te   = patient_labels[[i]]

    # Inner LOPO to pick best k
    inner_accs = []
    n_in = len(y_tr)
    for k in kv_inner:
        it, ip = [], []
        for j in range(n_in):
            jtr = [jj for jj in range(n_in) if jj != j]
            Xii = G_tr[jtr, :k]; yii = y_tr[jtr]
            Xij = G_tr[[j],  :k]; yij = y_tr[[j]]
            Xib, yib = _balance(Xii, yii, seed=RNG_SEED + j)
            if len(np.unique(yib)) < 2: continue
            lda = _LDA().fit(Xib, yib)
            it.extend(yij.tolist()); ip.extend(lda.predict(Xij).tolist())
        if it:
            inner_accs.append(_bal_acc(np.array(it), np.array(ip)))
        else:
            inner_accs.append(0.5)

    best_k_in = kv_inner[int(np.argmax(inner_accs))]
    k_chosen.append(best_k_in)

    # Train on all training patients with best_k_in, test on patient i
    Xb, yb = _balance(G_tr[:, :best_k_in], y_tr, seed=RNG_SEED + i)
    if len(np.unique(yb)) < 2: continue
    lda = _LDA().fit(Xb, yb)
    all_t_nested.extend(y_te.tolist())
    all_p_nested.extend(lda.predict(G_te[:, :best_k_in]).tolist())

yt_n = np.array(all_t_nested); yp_n = np.array(all_p_nested)
ba_n = _bal_acc(yt_n, yp_n); cc_n, ad_n = _class_acc(yt_n, yp_n)

print(f"\n  Nested LOPO  BA={ba_n*100:.1f}%  CC={cc_n*100:.0f}%  AD={ad_n*100:.0f}%")
print(f"  k selected per fold: {sorted(set(k_chosen))} "
      f"(most common: {max(set(k_chosen), key=k_chosen.count)})")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Leakage audit — per-patient G-scores, LOPO LDA")
print("=" * 65)
print(f"  Flat k-sweep  (k selected on all test folds):  BA = {ba_flat*100:.1f}%  (best k={best_k_flat})")
print(f"  Nested LOPO   (k selected inside each fold):   BA = {ba_n*100:.1f}%")
print(f"  Fixed k=2     (no search):                     BA = {acc_flat[1]*100:.1f}%")
print(f"  Fixed k=1     (no search):                     BA = {acc_flat[0]*100:.1f}%")
print("=" * 65)
print("\nConclusion:")
delta = (ba_flat - ba_n) * 100
if delta < 2:
    print(f"  k-selection bias is negligible ({delta:.1f}pp). Result is robust.")
elif delta < 8:
    print(f"  Mild k-selection bias ({delta:.1f}pp). Nested estimate is more conservative.")
else:
    print(f"  Substantial k-selection bias ({delta:.1f}pp). Report nested estimate.")
