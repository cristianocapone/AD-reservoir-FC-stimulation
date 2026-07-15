"""
per_patient_W.py
================
Per-patient W estimation: concatenate X,Y from ALL sessions of a patient,
then fit a single W per patient (more data -> better conditioned regression).

One teacher-forced pass per session (reset between sessions), X,Y pooled,
then W = pinv(X_coll) @ Y_coll.

Evaluations:
  1. FC-r (delay=0): closed-loop sim with per-patient W vs per-session W
  2. LOPO BA on per-patient G-scores (k-sweep)
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── Config ────────────────────────────────────────────────────────────────────
RNG_SEED         = 42
N_CC_SAMPLE      = 40
N_SITES          = 121
N_PC_MODEL       = 50
K_PC             = 200
noise_size       = 0.025
TIMES_SKIP       = 10
ff               = 0.1
TS_ROOT          = "./timeseries"
OUT_DIR          = "./summary_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LDA ───────────────────────────────────────────────────────────────────────
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

def balanced_acc(yt, yp):
    return np.mean([(yp[yt == c] == c).mean() for c in np.unique(yt)])

def class_accs(yt, yp):
    cc = (yp[yt == 0] == 0).mean() if (yt == 0).any() else np.nan
    ad = (yp[yt == 1] == 1).mean() if (yt == 1).any() else np.nan
    return cc, ad

# ── LOPO on per-patient array (one row per patient) ───────────────────────────
def lopo_patient_eval(X_feat, y):
    n = len(y)
    all_true, all_pred = [], []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        X_tr, y_tr = X_feat[tr], y[tr]
        X_te, y_te = X_feat[[i]], y[[i]]
        X_tr_b, y_tr_b = balance_train(X_tr, y_tr, seed=RNG_SEED + i)
        if len(np.unique(y_tr_b)) < 2: continue
        lda = _LDA().fit(X_tr_b, y_tr_b)
        all_true.extend(y_te.tolist())
        all_pred.extend(lda.predict(X_te).tolist())
    yt, yp = np.array(all_true), np.array(all_pred)
    return yt, yp

def lopo_patient_sweep(X_feat, y, k_values):
    accs = []
    for k in tqdm(k_values, desc="  k-sweep", leave=False):
        yt, yp = lopo_patient_eval(X_feat[:, :k], y)
        accs.append(balanced_acc(yt, yp))
    return np.array(accs)

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
print(f"  Sessions: {N_subj}  (CC={(labels_raw==0).sum()}, AD={(labels_raw==1).sum()})")

# ── Population PCA ────────────────────────────────────────────────────────────
print("PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(axis=0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Build reservoir ───────────────────────────────────────────────────────────
print("Initialising reservoir ...")
N_H, I_d, O_d, TIME = 2000, N_SITES, N_SITES, 600
dt = 0.005; tau_m = 0.0001 * dt
par = dict(tau_m_f=tau_m, tau_m_s=tau_m, N=N_H, T=TIME, dt=dt,
           sigma_input=0.01, shape=(N_H, I_d, O_d, TIME))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

# ── Single teacher-forced pass: collect X, Y per session ─────────────────────
print("\nTeacher-forced pass (all sessions) ...")
sess_X      = {}   # idx -> (T_eff, N_H) states (post-skip)
sess_Y      = {}   # idx -> (T_eff, N_sites) targets (post-skip)
sess_target = {}   # idx -> (N_sites, T) full target (for closed-loop eval)
sess_X_warm = {}   # idx -> final reservoir state after teacher-forcing

rng_noise = np.random.default_rng(RNG_SEED)
for idx in trange(N_subj, desc="  TF pass"):
    sig    = signals[idx]
    res.T  = sig.shape[1]; res.reset()
    pc_sc  = sig.T @ ev50
    target = (pc_sc @ ev50.T).T          # (N_sites, T)
    sess_target[idx] = target

    X_raw = []
    for t in range(target.shape[1] - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())

    sess_X_warm[idx] = res.X.copy()      # warm state at end of TF pass
    X_all = np.array(X_raw)              # (T-1, N_H)
    T_eff = len(X_all) - TIMES_SKIP
    sess_X[idx] = X_all[TIMES_SKIP:]     # (T_eff, N_H)
    sess_Y[idx] = target[:, TIMES_SKIP:TIMES_SKIP + T_eff].T  # (T_eff, N_sites)

# ── Group sessions by patient ─────────────────────────────────────────────────
unique_pids = np.unique(patient_ids_raw)
N_patients  = len(unique_pids)
patient_sessions = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
n_sess_counts = np.array([len(v) for v in patient_sessions.values()])

print(f"\n  {N_patients} unique patients")
print(f"  Sessions per patient: min={n_sess_counts.min()}  "
      f"max={n_sess_counts.max()}  mean={n_sess_counts.mean():.1f}")

# ── Fit per-session W (baseline) and per-patient W (new) ─────────────────────
print("\nFitting W matrices ...")

sess_W    = {}   # idx -> (N_H, N_sites)   per-session (baseline)
patient_W = {}   # pid -> (N_H, N_sites)   per-patient (new)

for idx in range(N_subj):
    X = sess_X[idx]; Y = sess_Y[idx]
    noise = rng_noise.normal(0, noise_size, X.shape)
    sess_W[idx] = np.linalg.pinv(X + noise) @ Y

for pid in tqdm(unique_pids, desc="  Per-patient W"):
    idxs   = patient_sessions[pid]
    X_coll = np.vstack([sess_X[i] for i in idxs])   # pool sessions
    Y_coll = np.vstack([sess_Y[i] for i in idxs])
    noise  = rng_noise.normal(0, noise_size, X_coll.shape)
    patient_W[pid] = np.linalg.pinv(X_coll + noise) @ Y_coll

# ── FC-r evaluation (closed-loop) ─────────────────────────────────────────────
print("\nClosed-loop FC-r evaluation ...")

def closed_loop_fc_r(res, target, W, ff, skip):
    """Teacher-force -> keep warm state -> closed-loop -> return FC-r."""
    T = target.shape[1]
    res.T = T; res.reset()
    for t in range(T - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
    # keep warm state, switch to closed-loop with given W
    res.Jout = W.T.copy()       # (N_sites, N_H)
    res.y    = res.Jout @ res.X

    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())

    Y_sim    = np.array(Y_sim)[skip:].T          # (N_sites, T-1-skip)
    FC_emp   = np.nan_to_num(np.corrcoef(target[:, skip:]))
    FC_sim   = np.nan_to_num(np.corrcoef(Y_sim))
    iu       = np.triu_indices(N_SITES, k=1)
    r        = np.corrcoef(FC_emp[iu], FC_sim[iu])[0, 1]
    return float(r)

r_per_sess_W  = []   # baseline: per-session W
r_per_pat_W   = []   # new:      per-patient W

for idx in trange(N_subj, desc="  FC-r"):
    target = sess_target[idx]
    pid    = patient_ids_raw[idx]

    r_s = closed_loop_fc_r(res, target, sess_W[idx],    ff, TIMES_SKIP)
    r_p = closed_loop_fc_r(res, target, patient_W[pid], ff, TIMES_SKIP)

    r_per_sess_W.append(r_s)
    r_per_pat_W.append(r_p)

r_per_sess_W = np.array(r_per_sess_W)
r_per_pat_W  = np.array(r_per_pat_W)

print(f"\n  FC-r per-session W  : {r_per_sess_W.mean():.3f} +/- {r_per_sess_W.std():.3f}")
print(f"  FC-r per-patient W  : {r_per_pat_W.mean():.3f} +/- {r_per_pat_W.std():.3f}")

# ── Per-patient G-scores ──────────────────────────────────────────────────────
print("\nBuilding per-patient G-scores ...")

W_proj_list = []
for pid in unique_pids:
    idxs   = patient_sessions[pid]
    X_coll = np.vstack([sess_X[i] for i in idxs])   # (T_coll, N_H)
    W      = patient_W[pid]                           # (N_H, N_sites)

    W_T  = W.T.astype(np.float64)                    # (N_sites, N_H)
    _, sx, Vtx = np.linalg.svd(X_coll.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    W_proj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

W_stack  = np.array(W_proj_list)                     # (N_patients, N_sites*N_H)
W_mean   = W_stack.mean(0)
_, _, Vt_svd = np.linalg.svd(W_stack - W_mean, full_matrices=False)
M_eff    = W_stack.shape[0] - 1                      # = N_patients - 1 = 75
G_scores = (W_stack - W_mean) @ Vt_svd[:M_eff].T    # (N_patients, M_eff)

patient_labels = np.array([labels_raw[patient_sessions[pid][0]] for pid in unique_pids])
print(f"  G-scores: {G_scores.shape}  "
      f"(CC={( patient_labels==0).sum()}, AD={(patient_labels==1).sum()} patients)")

# ── Per-session G-scores (baseline) ──────────────────────────────────────────
print("Building per-session G-scores (baseline) ...")
W_proj_s = []
for idx in range(N_subj):
    X = sess_X[idx]; W = sess_W[idx]
    W_T = W.T.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    W_proj_s.append((W_T @ Vt_k.T @ Vt_k).flatten())

W_stack_s  = np.array(W_proj_s)
W_mean_s   = W_stack_s.mean(0)
_, _, Vt_s = np.linalg.svd(W_stack_s - W_mean_s, full_matrices=False)
M_s        = W_stack_s.shape[0] - 1
G_sess     = (W_stack_s - W_mean_s) @ Vt_s[:M_s].T    # (N_subj, M_s)

# ── LOPO k-sweep: per-patient G-scores ───────────────────────────────────────
print("\nLOPO k-sweep — per-patient G-scores ...")
k_vals_p = list(range(1, M_eff + 1))
acc_curve_p = lopo_patient_sweep(G_scores, patient_labels, k_vals_p)
best_k_p    = k_vals_p[int(np.argmax(acc_curve_p))]
yt_p, yp_p  = lopo_patient_eval(G_scores[:, :best_k_p], patient_labels)
ba_p        = balanced_acc(yt_p, yp_p)
cc_p, ad_p  = class_accs(yt_p, yp_p)

# ── LOPO k-sweep: per-session G-scores (baseline, session-level LOPO) ────────
print("LOPO k-sweep — per-session G-scores (baseline) ...")

def lopo_splits_per_session(patient_ids):
    unique = np.unique(patient_ids)
    return [(np.where(patient_ids != pid)[0], np.where(patient_ids == pid)[0])
            for pid in unique]

splits_s = lopo_splits_per_session(patient_ids_raw)

def lopo_sess_eval(X_feat, y, splits):
    all_true, all_pred = [], []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat[tr_idx], y[tr_idx]
        X_te, y_te = X_feat[te_idx], y[te_idx]
        X_tr_b, y_tr_b = balance_train(X_tr, y_tr, seed=RNG_SEED + fold_i)
        if len(np.unique(y_tr_b)) < 2: continue
        lda = _LDA().fit(X_tr_b, y_tr_b)
        all_true.extend(y_te.tolist())
        all_pred.extend(lda.predict(X_te).tolist())
    return np.array(all_true), np.array(all_pred)

k_vals_s    = list(range(1, min(50, M_s) + 1))
acc_curve_s = []
for k in tqdm(k_vals_s, desc="  k-sweep (session)", leave=False):
    yt, yp = lopo_sess_eval(G_sess[:, :k], labels_raw, splits_s)
    acc_curve_s.append(balanced_acc(yt, yp))
acc_curve_s = np.array(acc_curve_s)
best_k_s    = k_vals_s[int(np.argmax(acc_curve_s))]
yt_s, yp_s  = lopo_sess_eval(G_sess[:, :best_k_s], labels_raw, splits_s)
ba_s        = balanced_acc(yt_s, yp_s)
cc_s, ad_s  = class_accs(yt_s, yp_s)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 58)
print(f"{'Method':<30} {'FC-r':>6}  {'BA':>6}  {'CC':>5}  {'AD':>5}  {'k':>4}")
print("-" * 58)
print(f"  {'Per-session W (baseline)':<28} {r_per_sess_W.mean():.3f}  "
      f"{ba_s*100:5.1f}%  {cc_s*100:4.0f}%  {ad_s*100:4.0f}%  {best_k_s:4d}")
print(f"  {'Per-patient W (new)':<28} {r_per_pat_W.mean():.3f}  "
      f"{ba_p*100:5.1f}%  {cc_p*100:4.0f}%  {ad_p*100:4.0f}%  {best_k_p:4d}")
print("=" * 58)
print("FC-r = Pearson r between upper-triangle FC (empirical vs simulated)")
print("BA   = balanced accuracy (mean CC + AD sensitivity), LOPO")

# ── Plot k-sweep comparison ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4), facecolor="white")

ax = axes[0]
ax.plot(k_vals_s, acc_curve_s * 100, color="#2196F3", lw=2, label=f"Per-session W (best k={best_k_s})")
ax.axvline(best_k_s, color="#2196F3", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("k (G-score dims)"); ax.set_ylabel("Balanced accuracy (%)")
ax.set_title("Per-session W — k-sweep (session-level LOPO)")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
ax.plot(k_vals_p, acc_curve_p * 100, color="#E91E63", lw=2, label=f"Per-patient W (best k={best_k_p})")
ax.axvline(best_k_p, color="#E91E63", lw=1, ls="--", alpha=0.6)
ax.set_xlabel("k (G-score dims)"); ax.set_ylabel("Balanced accuracy (%)")
ax.set_title("Per-patient W — k-sweep (patient-level LOPO)")
ax.legend(); ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/per_patient_W_sweep.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {OUT_DIR}/per_patient_W_sweep.png")
