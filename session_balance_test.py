"""
session_balance_test.py
=======================
Tests whether the baseline AUROC is inflated by unequal sessions per patient.

Three conditions compared:

  A) Multi-session per patient  (current approach)
     W fitted from ALL sessions concatenated per patient → 76 G-scores → AUROC

  B) Single-session per patient  (balanced)
     W fitted from SESSION-0 only per patient → 76 G-scores → AUROC

  C) Per-session  (notebook-style)
     W fitted per session → one G-score per session → AUROC
     (uses all sessions, classified by patient label)

Prints session counts per class, AUROC for each condition, and
saves session_balance_auroc.png.
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── constants ──────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
NOISE_SIZE  = 0.025
TS_ROOT     = "./timeseries"
OUT_DIR     = "."
K_LDA       = 2

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

pid_raw        = np.array(pid_raw)
labels_raw     = np.array(labels_raw)
N_subj         = len(signals)

unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_idx = np.where(patient_labels == 0)[0]
ad_idx = np.where(patient_labels == 1)[0]

# ── session counts per patient ─────────────────────────────────────────────────
n_sess_cc = [len(patient_sids[unique_pids[i]]) for i in cc_idx]
n_sess_ad = [len(patient_sids[unique_pids[i]]) for i in ad_idx]
print(f"\n  Total sessions : {N_subj}")
print(f"  CC patients    : {len(cc_idx)}  |  sessions: {sum(n_sess_cc)} "
      f"(mean={np.mean(n_sess_cc):.2f}, max={max(n_sess_cc)})")
print(f"  AD patients    : {len(ad_idx)}  |  sessions: {sum(n_sess_ad)} "
      f"(mean={np.mean(n_sess_ad):.2f}, max={max(n_sess_ad)})")

# session-count histogram
sess_per_pat = np.array(n_sess_cc + n_sess_ad)
label_vec    = np.array([0]*len(n_sess_cc) + [1]*len(n_sess_ad))
print(f"\n  Sessions/patient distribution (CC|AD):")
for k in sorted(set(sess_per_pat)):
    nc = np.sum((sess_per_pat == k) & (label_vec == 0))
    na = np.sum((sess_per_pat == k) & (label_vec == 1))
    print(f"    {k} session(s): {nc} CC, {na} AD")

# ── population PCA ─────────────────────────────────────────────────────────────
print("\nPopulation PCA ...")
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
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

# ── teacher-forced pass (all sessions) ────────────────────────────────────────
print("TF pass (all sessions) ...")
sess_X, sess_Y = {}, {}
for idx in trange(N_subj, desc="  TF"):
    s      = signals[idx]
    T_s    = s.shape[1]
    tgt    = (s.T @ ev50 @ ev50.T).T
    res.T  = T_s; res.reset()
    X_raw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    Xf        = np.array(X_raw)[TIMES_SKIP:]
    sess_X[idx] = Xf
    sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T

# ── LDA helper ─────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = (X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1) + 1e-6*np.eye(X0.shape[1])
        w  = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_   = w
        self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_
    def predict(self, X):
        return np.where(self.transform(X) >= self.thr_,
                        self.classes_[1], self.classes_[0])

def _accuracy(lda, G, labels):
    """Returns (accuracy, balanced_accuracy) given a fitted LDA and G-scores."""
    pred = lda.predict(G[:, :K_LDA])
    acc  = float(np.mean(pred == labels))
    c0, c1 = np.unique(labels)
    sens = float(np.mean(pred[labels==c1] == c1))   # AD recall
    spec = float(np.mean(pred[labels==c0] == c0))   # CC recall
    bal  = 0.5 * (sens + spec)
    return acc, bal, sens, spec

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([rng2.choice(c0,n,replace=False), rng2.choice(c1,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

def fit_W_and_Gscores(pat_X_map, pat_Y_map, label_str):
    """
    Fit per-patient W from provided X/Y maps, compute G-scores & AUROC.
    Returns (AUROC, G_pat, lda, W_mean, Vt_svd, M_eff).
    """
    print(f"  Fitting W and G-scores [{label_str}] ...")
    rng_p = np.random.default_rng(RNG_SEED + 1)
    pat_W = {}
    for pid in tqdm(unique_pids, desc="    W-fit", leave=False):
        Xc = pat_X_map[pid]
        Yc = pat_Y_map[pid]
        noise = rng_p.normal(0, NOISE_SIZE, Xc.shape)
        pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

    Wproj_list = []
    for pid in unique_pids:
        W_T = pat_W[pid].T.astype(np.float64)
        Xc  = pat_X_map[pid].astype(np.float64)
        _, sx, Vtx = np.linalg.svd(Xc, full_matrices=False)
        kk  = min(K_PC, int((sx > 1e-8).sum()))
        Vt_k = Vtx[:kk]
        Wproj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

    Wstack = np.array(Wproj_list)
    Wmean  = Wstack.mean(0)
    Wcent  = Wstack - Wmean
    _, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
    Meff   = N_patients - 1
    G_pat  = Wcent @ Vsvd[:Meff].T

    Xlda, ylda = _balance(G_pat[:, :K_LDA], patient_labels, seed=RNG_SEED)
    lda = _LDA().fit(Xlda, ylda)
    Z0  = lda.transform(G_pat[:, :K_LDA])
    if Z0[patient_labels==0].mean() > Z0[patient_labels==1].mean():
        lda.w_ *= -1; lda.thr_ *= -1
    Z0  = lda.transform(G_pat[:, :K_LDA])
    auc = roc_auc_score(patient_labels, Z0)
    acc, bal, sens, spec = _accuracy(lda, G_pat, patient_labels)

    print(f"    AUROC={auc:.4f}  ACC={acc:.4f}  BAL-ACC={bal:.4f}  "
          f"(sens={sens:.3f}, spec={spec:.3f})  "
          f"CC_LDA={Z0[patient_labels==0].mean():.3f}  "
          f"AD_LDA={Z0[patient_labels==1].mean():.3f}")
    return auc, acc, bal, G_pat, lda, Wmean, Vsvd, Meff

# ══════════════════════════════════════════════════════════════════════════════
# Condition A — multi-session per patient (current approach)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Multi-session per patient (all sessions concatenated)")
patX_multi = {pid: np.vstack([sess_X[i] for i in patient_sids[pid]])
              for pid in unique_pids}
patY_multi = {pid: np.vstack([sess_Y[i] for i in patient_sids[pid]])
              for pid in unique_pids}
auc_A, acc_A, bal_A, G_A, lda_A, _, _, _ = fit_W_and_Gscores(patX_multi, patY_multi, "multi-session")

# ══════════════════════════════════════════════════════════════════════════════
# Condition B — single session per patient (session index 0)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[B] Single session per patient (first session only)")
patX_single = {pid: sess_X[patient_sids[pid][0]] for pid in unique_pids}
patY_single = {pid: sess_Y[patient_sids[pid][0]] for pid in unique_pids}
auc_B, acc_B, bal_B, G_B, lda_B, _, _, _ = fit_W_and_Gscores(patX_single, patY_single, "single-session")

# also report how many CC vs AD have >1 session
print(f"    (CC with >1 sess: {sum(k>1 for k in n_sess_cc)}  "
      f"AD with >1 sess: {sum(k>1 for k in n_sess_ad)})")

# ══════════════════════════════════════════════════════════════════════════════
# Condition C — per-session (notebook style)
# Each session → one W → one G-score, labelled by the patient's class
# LDA trained on all per-session G-scores (balanced), AUROC per session
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C] Per-session (notebook style) — one G-score per session")
rng_p = np.random.default_rng(RNG_SEED + 1)

# fit W per session
sess_W = {}
for idx in tqdm(range(N_subj), desc="  W-fit per sess", leave=False):
    Xc = sess_X[idx]
    Yc = sess_Y[idx]
    noise = rng_p.normal(0, NOISE_SIZE, Xc.shape)
    sess_W[idx] = np.linalg.pinv(Xc + noise) @ Yc

# We need a G-space basis: use all per-session W projections
print("  G-scores per session ...")
# NOTE: population SVD on per-session W stack (same as notebook approach)
# X-space basis: use per-session X for projection (like notebook)
Wproj_sess = []
for idx in range(N_subj):
    W_T = sess_W[idx].T.astype(np.float64)
    Xc  = sess_X[idx].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xc, full_matrices=False)
    kk  = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:kk]
    Wproj_sess.append((W_T @ Vt_k.T @ Vt_k).flatten())

Wstack_s   = np.array(Wproj_sess)
Wmean_s    = Wstack_s.mean(0)
Wcent_s    = Wstack_s - Wmean_s
_, _, Vsvd_s = np.linalg.svd(Wcent_s, full_matrices=False)
Meff_s     = N_subj - 1
G_sess_all = Wcent_s @ Vsvd_s[:Meff_s].T   # (N_subj, Meff_s)

# labels per session
labels_sess = labels_raw   # (N_subj,)

# LDA on per-session G-scores
Xlda_s, ylda_s = _balance(G_sess_all[:, :K_LDA], labels_sess, seed=RNG_SEED)
lda_s = _LDA().fit(Xlda_s, ylda_s)
Z_s   = lda_s.transform(G_sess_all[:, :K_LDA])
if Z_s[labels_sess==0].mean() > Z_s[labels_sess==1].mean():
    lda_s.w_ *= -1; lda_s.thr_ *= -1
Z_s   = lda_s.transform(G_sess_all[:, :K_LDA])

# AUROC C1: per-session (all sessions, possibly multiple per patient)
auc_C1 = roc_auc_score(labels_sess, Z_s)
acc_C1, bal_C1, _, _ = _accuracy(lda_s, G_sess_all, labels_sess)
print(f"  [C1] Per-session AUROC (all sessions): {auc_C1:.4f}  "
      f"ACC={acc_C1:.4f}  BAL-ACC={bal_C1:.4f}  "
      f"(N_CC_sess={int((labels_sess==0).sum())}, N_AD_sess={int((labels_sess==1).sum())})")

# AUROC C2: per-patient, averaging G-score across sessions before LDA
#   (this is a third natural way: fit per-session W, average G-scores, evaluate per patient)
print("  [C2] Per-session W → average G per patient ...")
G_pat_avg = np.zeros((N_patients, K_LDA))
for pi, pid in enumerate(unique_pids):
    idxs = patient_sids[pid]
    g_list = []
    for si in idxs:
        W_T = sess_W[si].T.astype(np.float64)
        Xc  = sess_X[si].astype(np.float64)
        _, sx, Vtx = np.linalg.svd(Xc, full_matrices=False)
        kk  = min(K_PC, int((sx > 1e-8).sum()))
        Vt_k = Vtx[:kk]
        wp = (W_T @ Vt_k.T @ Vt_k).flatten()
        g = (wp - Wmean_s) @ Vsvd_s[:Meff_s].T
        g_list.append(g[:K_LDA])
    G_pat_avg[pi] = np.mean(g_list, axis=0)

Xlda_avg, ylda_avg = _balance(G_pat_avg, patient_labels, seed=RNG_SEED)
lda_avg = _LDA().fit(Xlda_avg, ylda_avg)
Z_avg   = lda_avg.transform(G_pat_avg)
if Z_avg[patient_labels==0].mean() > Z_avg[patient_labels==1].mean():
    lda_avg.w_ *= -1; lda_avg.thr_ *= -1
Z_avg   = lda_avg.transform(G_pat_avg)
auc_C2  = roc_auc_score(patient_labels, Z_avg)
acc_C2, bal_C2, _, _ = _accuracy(lda_avg, G_pat_avg, patient_labels)
print(f"  [C2] Per-session W → averaged G per patient AUROC: {auc_C2:.4f}  "
      f"ACC={acc_C2:.4f}  BAL-ACC={bal_C2:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*75)
print(f"{'Condition':<40}  {'AUROC':>6}  {'ACC':>6}  {'BAL-ACC':>7}  {'N':>5}")
print("-"*75)
print(f"  A) Multi-session W → per-patient AUROC  {auc_A:>6.4f}  {acc_A:>6.4f}  {bal_A:>7.4f}  {N_patients:>5}")
print(f"  B) Single-session W → per-patient AUROC {auc_B:>6.4f}  {acc_B:>6.4f}  {bal_B:>7.4f}  {N_patients:>5}")
print(f"  C1) Per-session W → per-session AUROC   {auc_C1:>6.4f}  {acc_C1:>6.4f}  {bal_C1:>7.4f}  {N_subj:>5}")
print(f"  C2) Per-session W → avg-G per patient   {auc_C2:>6.4f}  {acc_C2:>6.4f}  {bal_C2:>7.4f}  {N_patients:>5}")
print("="*75)
delta_AB = auc_A - auc_B
print(f"\n  A - B = {delta_AB:+.4f} AUROC  |  {acc_A - acc_B:+.4f} ACC  |  {bal_A - bal_B:+.4f} BAL-ACC")
print(f"  {'A higher → multi-session inflates metrics' if delta_AB > 0.01 else 'A ~ B → session count not the driver' if abs(delta_AB) <= 0.01 else 'B higher → unexpected'}")

# ══════════════════════════════════════════════════════════════════════════════
# Plot
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")

col_cc = "#2196F3"; col_ad = "#E91E63"
xs_base = np.linspace(-1, 1, 300)

def plot_kde(ax, Z, labels, title, auc, acc=None, bal=None):
    from scipy.stats import gaussian_kde
    for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
        pts = Z[labels == cls]
        if len(pts) < 3: continue
        try: kde = gaussian_kde(pts, bw_method="scott")
        except: kde = gaussian_kde(pts + np.random.normal(0, 1e-4*pts.std()+1e-8, pts.shape))
        xs = np.linspace(Z.min()-0.2, Z.max()+0.2, 400)
        ax.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
        ax.plot(xs, kde(xs), color=col, lw=2)
        ax.axvline(pts.mean(), color=col, ls="--", lw=1.5)
    metrics = f"AUROC={auc:.4f}"
    if acc  is not None: metrics += f"  ACC={acc:.4f}"
    if bal  is not None: metrics += f"  Bal={bal:.4f}"
    ax.set_title(f"{title}\n{metrics}", fontsize=10)
    ax.set_xlabel("LDA score", fontsize=10)
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9)
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)

Z_A = lda_A.transform(G_A[:, :K_LDA])
if Z_A[patient_labels==0].mean() > Z_A[patient_labels==1].mean(): Z_A *= -1
plot_kde(axes[0], Z_A, patient_labels,
         "A) Multi-session W\n(all sessions concatenated)", auc_A, acc_A, bal_A)

Z_B = lda_B.transform(G_B[:, :K_LDA])
if Z_B[patient_labels==0].mean() > Z_B[patient_labels==1].mean(): Z_B *= -1
plot_kde(axes[1], Z_B, patient_labels,
         "B) Single-session W\n(first session only)", auc_B, acc_B, bal_B)

Z_avg2 = lda_avg.transform(G_pat_avg)
if Z_avg2[patient_labels==0].mean() > Z_avg2[patient_labels==1].mean(): Z_avg2 *= -1
plot_kde(axes[2], Z_avg2, patient_labels,
         f"C2) Per-session W → avg G\n(per-patient; C1 sess-level AUROC={auc_C1:.3f} ACC={acc_C1:.3f})",
         auc_C2, acc_C2, bal_C2)

fig.suptitle("Session-balance test: is the 0.85 AUROC inflated by unequal sessions?",
             fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/session_balance_auroc.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved session_balance_auroc.png")
print("Done.")
