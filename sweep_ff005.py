"""
sweep_ff005.py
Quick evaluation of recurrent_factor=0.05 with k=4 G-scores LOPO.
Compare against the known result: ff=0.10, k=4 → BA=60.1%
"""
import os, sys, warnings
import numpy as np
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED         = 42
N_CC_SAMPLE      = 40
N_SITES          = 121
N_PC_MODEL       = 50
K_PC             = 200
M_ARCH           = 600
noise_size       = 0.025
TIMES_SKIP       = 10
TS_ROOT          = "./timeseries"

# ── LDA ──────────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw  = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)
        Sw += 1e-6 * np.eye(Sw.shape[0])
        w   = np.linalg.solve(Sw, mu1 - mu0)
        w  /= np.linalg.norm(w) + 1e-12
        thr = 0.5 * ((X0 @ w).mean() + (X1 @ w).mean())
        self.w_, self.thr_ = w, thr
        return self
    def predict(self, X):
        return np.where(X @ self.w_ >= self.thr_, self.classes_[1], self.classes_[0])

def balance_train(X, y, seed=0):
    rng = np.random.default_rng(seed)
    c0, c1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(c0), len(c1))
    if n == 0: return X, y
    sel = np.concatenate([rng.choice(c0, n, replace=False),
                          rng.choice(c1, n, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]

def lopo_splits(patient_ids):
    unique_pids = np.unique(patient_ids)
    return [(np.where(patient_ids != pid)[0], np.where(patient_ids == pid)[0])
            for pid in unique_pids]

def lopo_eval(X_feat, y, splits):
    all_true, all_pred = [], []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat[tr_idx], y[tr_idx]
        X_te, y_te = X_feat[te_idx], y[te_idx]
        X_tr_b, y_tr_b = balance_train(X_tr, y_tr, seed=RNG_SEED + fold_i)
        if len(np.unique(y_tr_b)) < 2: continue
        lda = _LDA().fit(X_tr_b, y_tr_b)
        all_true.extend(y_te.tolist())
        all_pred.extend(lda.predict(X_te).tolist())
    yt, yp = np.array(all_true), np.array(all_pred)
    ba = np.mean([(yp[yt == c] == c).mean() for c in np.unique(yt)])
    cc = (yp[yt == 0] == 0).mean()
    ad = (yp[yt == 1] == 1).mean()
    return ba, cc, ad

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
ctrl_idx = np.where(labels_raw == 0)[0]
ad_idx   = np.where(labels_raw == 1)[0]
print(f"  Sessions: {N_subj}  (CC={len(ctrl_idx)}, AD={len(ad_idx)})")

# ── Population PCA ────────────────────────────────────────────────────────────
print("PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(axis=0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Run reservoir for both ff values and collect G-scores ─────────────────────
def build_gscores(ff_val, signals, ev50, N_subj):
    N, I_d, O_d, TIME = 2000, N_SITES, N_SITES, 600
    dt = 0.005; tau_m = 0.0001 * dt
    par = dict(tau_m_f=tau_m, tau_m_s=tau_m, N=N, T=TIME, dt=dt,
               sigma_input=0.01, shape=(N, I_d, O_d, TIME))
    res = RESERVOIRE_SIMPLE(par)
    sr  = max(abs(np.linalg.eigvals(res.J)))
    res.J *= 0.95 / sr

    W_proj_list = []
    rng_fit = np.random.default_rng(RNG_SEED)

    for idx in trange(N_subj, desc=f"  ff={ff_val:.2f} fit"):
        sig = signals[idx]
        res.T = sig.shape[1]; res.reset()
        pc_sc  = sig.T @ ev50
        target = (pc_sc @ ev50.T).T   # (N_sites, T)

        # teacher-forced pass
        for t in range(target.shape[1] - 1):
            res.step_rate(ff_val * target[:, t], sigma_dyn=0.)

        X_raw = []
        res.reset()
        for t in range(target.shape[1] - 1):
            res.step_rate(ff_val * target[:, t], sigma_dyn=0.)
            X_raw.append(res.X.copy())

        X = np.array(X_raw)[TIMES_SKIP:]
        Y = target[:, TIMES_SKIP:TIMES_SKIP + len(X)].T

        noise = rng_fit.normal(0, noise_size, X.shape)
        W     = np.linalg.pinv(X + noise).dot(Y)   # (N_res, N_sites)

        W_T = W.T.astype(np.float64)
        _, sx, Vtx = np.linalg.svd(X.astype(np.float64), full_matrices=False)
        k_pc = min(K_PC, int((sx > 1e-8).sum()))
        Vt_k = Vtx[:k_pc]
        W_proj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

    W_stack = np.array(W_proj_list)
    W_mean  = W_stack.mean(0)
    _, _, Vt_svd = np.linalg.svd(W_stack - W_mean, full_matrices=False)
    M_eff  = min(M_ARCH, W_stack.shape[0] - 1)
    return (W_stack - W_mean) @ Vt_svd[:M_eff].T   # (N_subj, M_eff)

splits = lopo_splits(patient_ids_raw)

print("\n" + "=" * 55)
print(f"{'ff':>6}  {'k':>4}  {'BA':>7}  {'CC sens':>8}  {'AD sens':>8}")
print("-" * 55)

for ff_val in [0.05, 0.10]:
    print(f"\nBuilding G-scores for ff={ff_val} ...")
    G = build_gscores(ff_val, signals, ev50, N_subj)
    for k in [4]:
        ba, cc, ad = lopo_eval(G[:, :k], labels_raw, splits)
        print(f"  ff={ff_val:.2f}  k={k}  ->  BA={ba*100:.1f}%  CC={cc*100:.0f}%  AD={ad*100:.0f}%")

print("=" * 55)
