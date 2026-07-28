"""
fc_lag_classification.py
========================
Classify CC vs AD using lagged functional connectivity features.

Features (per patient, averaged across sessions):
  Lag 0 : standard FC  ->  upper triangle (no diag)   = N*(N-1)/2 = 7260
  Lag 1 : lagged FC    ->  full N×N matrix             = N²        = 14641 each
  ...
  Lag 5 : lagged FC    ->  full N×N                    = 14641

  C[i,j,k] = corr( region_i(t),  region_j(t+k) )

Total features (lags 0-5): 7260 + 5*14641 = 80 465

Dimensionality reduction:
  Gram-matrix SVD over ALL 183 patients (transductive - test points included).
  G = U * sqrt(Lambda)  in R^{183 x K}

Classifiers:
  1. LOPO LDA   - balanced training; orientation fix.
  2. LOPO RF    - RandomForestClassifier with class_weight='balanced',
                  n_estimators=500, min_samples_leaf=3, max_features='sqrt'

Feature cache: fc_lag_features_cache.npz (skip recomputation on re-runs)

Outputs:
  fc_lag_results.npz
  fc_lag_classification.png  (300 DPI, paper-ready)
"""
import os, sys, warnings, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import warnings as _w; _w.filterwarnings("ignore")

# ── settings ──────────────────────────────────────────────────────────────────
RNG_SEED  = 42
N_SITES   = 121
MAX_LAG   = 5
TS_ROOT   = "./timeseries"
K_VALUES  = [5, 10, 15, 17, 20, 22, 25, 28, 30, 40, 50, 75]
CACHE_FILE = "fc_lag_features_cache.npz"
OUT_DIR   = "."

# RF hyperparameters (tuned for 183-patient, 3.6:1 imbalanced, small-N setting)
# Strategy: balanced subsampling per fold (equal CC/AD), no class_weight,
# threshold=0.5.  Mirrors LDA's balanced training protocol and avoids the
# probability-scale distortion introduced by class_weight='balanced'.
RF_PARAMS = dict(
    n_estimators     = 500,
    max_features     = "sqrt",
    min_samples_leaf = 3,
    max_depth        = None,     # depth controlled via min_samples_leaf
    class_weight     = None,     # handled by balanced subsampling in lopo_rf
    random_state     = RNG_SEED,
    n_jobs           = -1,
)

# ── lagged FC helpers ──────────────────────────────────────────────────────────
def lagged_corrcoef(S, lag):
    """
    S   : (T, N_sites)  time series (already as rows=time)
    lag : int  (0 = standard FC)
    Returns (N_sites, N_sites)  Pearson correlation matrix at given lag.
    C[i,j] = corr( S[:,i],  S[lag:,j] )
    """
    if lag == 0:
        return np.corrcoef(S.T)          # (N_sites, N_sites)
    T = S.shape[0]
    A = S[:T-lag, :].astype(np.float64)  # (T-lag, N_sites)  - sources at t
    B = S[lag:,   :].astype(np.float64)  # (T-lag, N_sites)  - targets at t+lag
    A -= A.mean(0); B -= B.mean(0)
    std_A = A.std(0) + 1e-12
    std_B = B.std(0) + 1e-12
    A /= std_A; B /= std_B
    n = T - lag
    return (A.T @ B) / n                 # (N_sites, N_sites)

def session_features(S, max_lag):
    """
    S        : (T, N_sites)
    max_lag  : int
    Returns 1-D feature vector:
      lag 0 : upper triangle (no diag)   len = N*(N-1)//2
      lag k>0: full N×N matrix           len = N²  each
    """
    feats = []
    for lag in range(max_lag + 1):
        fc = lagged_corrcoef(S, lag)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        if lag == 0:
            idx = np.triu_indices(N_SITES, k=1)
            feats.append(fc[idx])
        else:
            feats.append(fc.flatten())
    return np.concatenate(feats)

# ── load or compute patient feature matrix ─────────────────────────────────────
if os.path.exists(CACHE_FILE):
    print(f"Loading cached features from {CACHE_FILE} ...")
    cache = np.load(CACHE_FILE, allow_pickle=True)
    X_raw          = cache["X_raw"]
    patient_labels = cache["patient_labels"]
    unique_pids    = list(cache["unique_pids"])
    N_patients     = len(unique_pids)
    cc_idx = np.where(patient_labels == 0)[0]
    ad_idx = np.where(patient_labels == 1)[0]
    print(f"  {N_patients} patients  (CC={len(cc_idx)}, AD={len(ad_idx)})")
    D = X_raw.shape[1]
    print(f"  Feature dim = {D}")
else:
    print("Loading timeseries and computing lagged FC ...")
    pid_feats  = defaultdict(list)
    pid_labels = {}

    for grp, label in [("CN", 0), ("AD", 1)]:
        folder = os.path.join(TS_ROOT, grp)
        files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        for fname in files:
            arr = np.load(os.path.join(folder, fname)).T   # (T, N_sites)
            if arr.shape[1] != N_SITES or arr.shape[0] < MAX_LAG + 2:
                continue
            pid = fname.split("_ses-")[0]
            feat = session_features(arr, MAX_LAG)
            pid_feats[pid].append(feat)
            pid_labels[pid] = label

    unique_pids    = sorted(pid_feats.keys())
    N_patients     = len(unique_pids)
    patient_labels = np.array([pid_labels[p] for p in unique_pids])
    cc_idx         = np.where(patient_labels == 0)[0]
    ad_idx         = np.where(patient_labels == 1)[0]
    print(f"  {N_patients} patients  (CC={len(cc_idx)}, AD={len(ad_idx)})")

    # Per-patient feature = mean over sessions
    X_raw = np.array([np.mean(pid_feats[p], axis=0)
                      for p in unique_pids], dtype=np.float64)   # (N_patients, D)
    D = X_raw.shape[1]
    print(f"  Feature vector dim = {D}  "
          f"(lag0={N_SITES*(N_SITES-1)//2}  "
          f"+ {MAX_LAG}x{N_SITES**2} = "
          f"{N_SITES*(N_SITES-1)//2 + MAX_LAG*N_SITES**2})")

    # Save cache
    np.savez(CACHE_FILE,
             X_raw=X_raw,
             patient_labels=patient_labels,
             unique_pids=np.array(unique_pids))
    print(f"  Cached to {CACHE_FILE}")

# ── global SVD (transductive) ──────────────────────────────────────────────────
print("Global SVD (all patients, transductive) ...")
X_mean = X_raw.mean(0)
X_c    = X_raw - X_mean                             # (N_patients, D)

print(f"  Building {N_patients}x{N_patients} Gram matrix ...")
C = X_c @ X_c.T                                     # (N_patients, N_patients)
evals, evecs = np.linalg.eigh(C)
order  = np.argsort(evals)[::-1]
evals  = np.maximum(evals[order], 0.0)
evecs  = evecs[:, order]
G_full = evecs * np.sqrt(evals)                     # (N_patients, N_patients)
cum_var = np.cumsum(evals) / evals.sum() * 100
print("  Variance explained:")
for k in [5, 10, 15, 20, 25, 30, 50, 75]:
    if k <= N_patients:
        print(f"    top {k:3d} PCs: {cum_var[k-1]:.1f}%")

# ── per-lag-subset G matrices ──────────────────────────────────────────────────
print("\nBuilding per-lag-subset feature matrices ...")
lag0_d = N_SITES * (N_SITES - 1) // 2
lagk_d = N_SITES ** 2

def G_for_lags(max_lag_use):
    """Build G matrix using only lags 0..max_lag_use."""
    cols = lag0_d if max_lag_use == 0 else lag0_d + max_lag_use * lagk_d
    Xs = X_raw[:, :cols]
    Xs_c = Xs - Xs.mean(0)
    Cg = Xs_c @ Xs_c.T
    ev, evec = np.linalg.eigh(Cg)
    order = np.argsort(ev)[::-1]
    ev = np.maximum(ev[order], 0)
    evec = evec[:, order]
    return evec * np.sqrt(ev)

# Pre-compute all lag-subset G matrices for experiment 2
G_lag_cache = {}
for ml in range(MAX_LAG + 1):
    G_lag_cache[ml] = G_for_lags(ml)

# ── LOPO helpers ───────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0)
        w /= np.linalg.norm(w) + 1e-12
        self.w_ = w
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i = np.where(y==0)[0]; c1i = np.where(y==1)[0]
    n   = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i, n, replace=False),
                          rng2.choice(c1i, n, replace=False)])
    rng2.shuffle(sel)
    return X[sel], y[sel]

def lopo_lda(G, y, k):
    """LOPO LDA using first k cols of G. Returns (bal_acc, auroc)."""
    n     = len(y)
    Gk    = G[:, :k]
    preds = np.full(n, np.nan)
    scores= np.full(n, np.nan)
    for i in range(n):
        mask = np.arange(n) != i
        G_tr = Gk[mask]; y_tr = y[mask]; G_te = Gk[i]
        Xb, yb = _balance(G_tr, y_tr, seed=RNG_SEED)
        try:
            lda = _LDA().fit(Xb, yb)
        except Exception:
            continue
        z_tr = lda.transform(G_tr)
        if z_tr[y_tr==0].mean() > z_tr[y_tr==1].mean():
            lda.w_ *= -1
            z_tr = lda.transform(G_tr)
        thr     = 0.5*(z_tr[y_tr==0].mean() + z_tr[y_tr==1].mean())
        z_te    = lda.transform(G_te.reshape(1,-1))[0]
        preds[i]  = float(z_te >= thr)
        scores[i] = z_te - thr
    valid = np.isfinite(preds)
    if valid.sum() < 4:
        return np.nan, np.nan
    sens = np.mean(preds[valid & (y==1)] == 1)
    spec = np.mean(preds[valid & (y==0)] == 0)
    bal  = 0.5*(sens+spec)
    try:   auc = roc_auc_score(y[valid], scores[valid])
    except: auc = np.nan
    return bal, auc

def lopo_rf(G, y, k, rng_seed=RNG_SEED):
    """
    LOPO Random Forest using first k cols of G. Returns (bal_acc, auroc).

    Protocol mirrors LDA: subsample training fold to equal CC/AD counts,
    fit RF on balanced set, apply threshold=0.5 to predict_proba output.
    This avoids class_weight distortions and the threshold calibration trap
    (calibrating on imbalanced training data produces degenerate thresholds).
    """
    n     = len(y)
    Gk    = G[:, :k]
    preds = np.full(n, np.nan)
    scores= np.full(n, np.nan)
    rng2  = np.random.default_rng(rng_seed)

    for i in range(n):
        mask = np.arange(n) != i
        G_tr = Gk[mask]; y_tr = y[mask]; G_te = Gk[i]

        # Balanced subsample: equal CC / AD
        c0i = np.where(y_tr == 0)[0]; c1i = np.where(y_tr == 1)[0]
        n_min = min(len(c0i), len(c1i))
        sel   = np.concatenate([rng2.choice(c0i, n_min, replace=False),
                                rng2.choice(c1i, n_min, replace=False)])
        G_bal = G_tr[sel]; y_bal = y_tr[sel]

        rf = RandomForestClassifier(**RF_PARAMS)
        try:
            rf.fit(G_bal, y_bal)
        except Exception:
            continue

        proba_te  = rf.predict_proba(G_te.reshape(1, -1))[0, 1]  # P(AD)
        preds[i]  = float(proba_te >= 0.5)
        scores[i] = proba_te

    valid = np.isfinite(preds)
    if valid.sum() < 4:
        return np.nan, np.nan
    sens = np.mean(preds[valid & (y==1)] == 1)
    spec = np.mean(preds[valid & (y==0)] == 0)
    bal  = 0.5*(sens+spec)
    try:   auc = roc_auc_score(y[valid], scores[valid])
    except: auc = np.nan
    return bal, auc

# ── Experiment 1: K sweep ─────────────────────────────────────────────────────
print(f"\nExperiment 1: K sweep (lags 0-{MAX_LAG}) ...")
print(f"  RF params: {RF_PARAMS}")
ba_k_lda = []; au_k_lda = []
ba_k_rf  = []; au_k_rf  = []

for k in K_VALUES:
    t0 = time.time()
    ba_l, au_l = lopo_lda(G_full, patient_labels, k)
    ba_r, au_r = lopo_rf (G_full, patient_labels, k)
    elapsed = time.time() - t0
    ba_k_lda.append(ba_l); au_k_lda.append(au_l)
    ba_k_rf .append(ba_r); au_k_rf .append(au_r)
    print(f"  K={k:3d}  "
          f"LDA: BAL={ba_l:.4f} AUC={au_l:.4f}  |  "
          f"RF:  BAL={ba_r:.4f} AUC={au_r:.4f}  "
          f"({elapsed:.1f}s)")

ba_k_lda = np.array(ba_k_lda); au_k_lda = np.array(au_k_lda)
ba_k_rf  = np.array(ba_k_rf);  au_k_rf  = np.array(au_k_rf)

best_k_lda = K_VALUES[np.argmax(ba_k_lda)]
best_k_rf  = K_VALUES[np.argmax(ba_k_rf)]
print(f"\n  Best K (LDA by BAL): {best_k_lda}")
print(f"  Best K (RF  by BAL): {best_k_rf}")

# Use best LDA K for incremental lag (consistent with previous results)
best_k = best_k_lda

# ── Experiment 2: incremental lag comparison ──────────────────────────────────
print(f"\nExperiment 2: incremental lags (K={best_k}) ...")
ba_lag_lda = []; au_lag_lda = []
ba_lag_rf  = []; au_lag_rf  = []

for ml in range(MAX_LAG + 1):
    G_lag = G_lag_cache[ml]
    ba_l, au_l = lopo_lda(G_lag, patient_labels, best_k)
    ba_r, au_r = lopo_rf (G_lag, patient_labels, best_k)
    ba_lag_lda.append(ba_l); au_lag_lda.append(au_l)
    ba_lag_rf .append(ba_r); au_lag_rf .append(au_r)
    n_feat = lag0_d + ml * lagk_d if ml > 0 else lag0_d
    print(f"  lags 0-{ml}  ({n_feat} feats)  "
          f"LDA: BAL={ba_l:.4f} AUC={au_l:.4f}  |  "
          f"RF:  BAL={ba_r:.4f} AUC={au_r:.4f}")

ba_lag_lda = np.array(ba_lag_lda); au_lag_lda = np.array(au_lag_lda)
ba_lag_rf  = np.array(ba_lag_rf);  au_lag_rf  = np.array(au_lag_rf)

# ── Reference G-space LDA (N=40) ─────────────────────────────────────────────
ref_ba = ref_au = None
try:
    lc = np.load("rf_results.npz", allow_pickle=True)
    idx40 = np.where(lc["N_grid"] == 40)[0]
    if len(idx40):
        ref_ba = float(lc["LDA_bal_mean"][idx40[0]])
        ref_au = float(lc["LDA_auc_mean"][idx40[0]])
        print(f"\nG-space reference (N=40, K=15): "
              f"BAL-ACC={ref_ba:.4f}  AUROC={ref_au:.4f}")
except Exception:
    pass

# ── save ───────────────────────────────────────────────────────────────────────
np.savez("fc_lag_results.npz",
         K_values      = np.array(K_VALUES),
         ba_k_lda      = ba_k_lda,  au_k_lda = au_k_lda,
         ba_k_rf       = ba_k_rf,   au_k_rf  = au_k_rf,
         lag_values    = np.arange(MAX_LAG+1),
         ba_lag_lda    = ba_lag_lda, au_lag_lda = au_lag_lda,
         ba_lag_rf     = ba_lag_rf,  au_lag_rf  = au_lag_rf,
         best_k_lda    = np.array(best_k_lda),
         best_k_rf     = np.array(best_k_rf),
         cum_var       = cum_var[:N_patients],
         n_cc          = np.array(len(cc_idx)),
         n_ad          = np.array(len(ad_idx)))
print("\nSaved fc_lag_results.npz")

# ── figure ─────────────────────────────────────────────────────────────────────
print("Plotting ...")
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

LDA_COL = "#1565C0"   # blue
RF_COL  = "#2E7D32"   # green
REF_COL = "#E65100"   # orange

fig = plt.figure(figsize=(15, 10), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.54, wspace=0.42)

# ── A: K sweep — BAL-ACC ──────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")
if ref_ba:
    ax_a.axhline(ref_ba, color=REF_COL, ls="--", lw=1.5, alpha=0.8,
                 label=f"G-space LDA ({ref_ba:.3f})")
ax_a.plot(K_VALUES, ba_k_lda, "-o", ms=5.5, lw=2.0, color=LDA_COL, label="LDA")
ax_a.plot(K_VALUES, ba_k_rf,  "-s", ms=5.5, lw=2.0, color=RF_COL,  label="RF")
for k, b in zip(K_VALUES, ba_k_lda):
    ax_a.text(k, b + 0.006, f"{b:.3f}", ha="center", va="bottom",
              fontsize=6, color=LDA_COL)
for k, b in zip(K_VALUES, ba_k_rf):
    ax_a.text(k, b - 0.018, f"{b:.3f}", ha="center", va="top",
              fontsize=6, color=RF_COL)
ax_a.set_xlabel("K  (SVD components)")
ax_a.set_ylabel("LOPO Balanced Accuracy")
ax_a.set_title(f"BAL-ACC vs K  (all lags 0-{MAX_LAG})")
ax_a.set_xticks(K_VALUES); ax_a.set_xticklabels(K_VALUES, rotation=45, ha="right")
ax_a.set_ylim(0.35, 0.90)
ax_a.legend(frameon=False, fontsize=7.5)

# ── B: K sweep — AUROC ───────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")
if ref_au:
    ax_b.axhline(ref_au, color=REF_COL, ls="--", lw=1.5, alpha=0.8,
                 label=f"G-space LDA ({ref_au:.3f})")
ax_b.plot(K_VALUES, au_k_lda, "-o", ms=5.5, lw=2.0, color=LDA_COL, label="LDA")
ax_b.plot(K_VALUES, au_k_rf,  "-s", ms=5.5, lw=2.0, color=RF_COL,  label="RF")
for k, a in zip(K_VALUES, au_k_lda):
    ax_b.text(k, a + 0.006, f"{a:.3f}", ha="center", va="bottom",
              fontsize=6, color=LDA_COL)
for k, a in zip(K_VALUES, au_k_rf):
    ax_b.text(k, a - 0.018, f"{a:.3f}", ha="center", va="top",
              fontsize=6, color=RF_COL)
ax_b.set_xlabel("K  (SVD components)")
ax_b.set_ylabel("LOPO AUROC")
ax_b.set_title(f"AUROC vs K  (all lags 0-{MAX_LAG})")
ax_b.set_xticks(K_VALUES); ax_b.set_xticklabels(K_VALUES, rotation=45, ha="right")
ax_b.set_ylim(0.35, 0.90)
ax_b.legend(frameon=False, fontsize=7.5)

# ── C: Cumulative variance explained ─────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
k_plot = np.arange(1, min(N_patients, 80) + 1)
ax_c.plot(k_plot, cum_var[:len(k_plot)], "-", lw=2.0, color="#37474F")
for k in K_VALUES:
    if k <= len(cum_var):
        ax_c.scatter([k], [cum_var[k-1]], s=45, zorder=4,
                     color=LDA_COL, edgecolors="white", linewidths=0.5)
ax_c.axvline(best_k_lda, color=LDA_COL, lw=1.5, ls="--", alpha=0.7,
             label=f"Best K_LDA={best_k_lda} ({cum_var[best_k_lda-1]:.1f}%)")
ax_c.axvline(best_k_rf,  color=RF_COL,  lw=1.5, ls=":",  alpha=0.7,
             label=f"Best K_RF ={best_k_rf}  ({cum_var[best_k_rf-1]:.1f}%)")
ax_c.set_xlabel("K  (SVD components)")
ax_c.set_ylabel("Cumulative variance (%)")
ax_c.set_title("Variance explained\n(lagged FC Gram, all patients)")
ax_c.legend(frameon=False, fontsize=7.5)

# ── D: Incremental lag — BAL-ACC ─────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
ax_d.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")
if ref_ba:
    ax_d.axhline(ref_ba, color=REF_COL, ls="--", lw=1.5, alpha=0.8,
                 label=f"G-space LDA")
lags_x = np.arange(MAX_LAG + 1)
ax_d.plot(lags_x, ba_lag_lda, "-o", ms=7, lw=2.2, color=LDA_COL,
          label=f"LDA  (K={best_k_lda})")
ax_d.plot(lags_x, ba_lag_rf,  "-s", ms=7, lw=2.2, color=RF_COL,
          label=f"RF   (K={best_k_lda})")
for x, b in zip(lags_x, ba_lag_lda):
    ax_d.text(x, b + 0.007, f"{b:.3f}", ha="center", va="bottom",
              fontsize=7, color=LDA_COL)
for x, b in zip(lags_x, ba_lag_rf):
    ax_d.text(x, b - 0.022, f"{b:.3f}", ha="center", va="top",
              fontsize=7, color=RF_COL)
ax_d.set_xlabel("Max lag included")
ax_d.set_ylabel("LOPO Balanced Accuracy")
ax_d.set_xticks(lags_x)
ax_d.set_xticklabels([f"Lag 0" if l==0 else f"0-{l}" for l in lags_x],
                      rotation=25, ha="right")
ax_d.set_title(f"Incremental lag benefit  (K={best_k_lda})")
ax_d.set_ylim(0.35, 0.90)
ax_d.legend(frameon=False, fontsize=7.5)

# ── E: Incremental lag — AUROC ───────────────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
ax_e.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.55, label="Chance")
if ref_au:
    ax_e.axhline(ref_au, color=REF_COL, ls="--", lw=1.5, alpha=0.8,
                 label=f"G-space LDA")
ax_e.plot(lags_x, au_lag_lda, "-o", ms=7, lw=2.2, color=LDA_COL,
          label=f"LDA  (K={best_k_lda})")
ax_e.plot(lags_x, au_lag_rf,  "-s", ms=7, lw=2.2, color=RF_COL,
          label=f"RF   (K={best_k_lda})")
for x, a in zip(lags_x, au_lag_lda):
    ax_e.text(x, a + 0.007, f"{a:.3f}", ha="center", va="bottom",
              fontsize=7, color=LDA_COL)
for x, a in zip(lags_x, au_lag_rf):
    ax_e.text(x, a - 0.022, f"{a:.3f}", ha="center", va="top",
              fontsize=7, color=RF_COL)
ax_e.set_xlabel("Max lag included")
ax_e.set_ylabel("LOPO AUROC")
ax_e.set_xticks(lags_x)
ax_e.set_xticklabels([f"Lag 0" if l==0 else f"0-{l}" for l in lags_x],
                      rotation=25, ha="right")
ax_e.set_title(f"Incremental lag benefit  (K={best_k_lda})")
ax_e.set_ylim(0.35, 0.90)
ax_e.legend(frameon=False, fontsize=7.5)

# ── F: Summary comparison bar chart ──────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])

# Best values: FC-only (lag 0), and best all-lags
best_ba_lda_fc0 = ba_lag_lda[0]; best_au_lda_fc0 = au_lag_lda[0]
best_ba_lda_all = ba_k_lda.max(); best_au_lda_all = au_k_lda.max()
best_ba_rf_fc0  = ba_lag_rf[0];  best_au_rf_fc0  = au_lag_rf[0]
best_ba_rf_all  = ba_k_rf.max(); best_au_rf_all  = au_k_rf.max()

cats   = ["FC only\n(lag 0)", f"FC + lags\n1-{MAX_LAG}  (best K)"]
x      = np.arange(len(cats))
bw     = 0.20

bar_lda_ba = ax_f.bar(x - 1.5*bw, [best_ba_lda_fc0, best_ba_lda_all],
                       bw, color=LDA_COL, alpha=0.85, label="LDA BAL")
bar_lda_au = ax_f.bar(x - 0.5*bw, [best_au_lda_fc0, best_au_lda_all],
                       bw, color=LDA_COL, alpha=0.45, label="LDA AUC",
                       hatch="///", edgecolor=LDA_COL)
bar_rf_ba  = ax_f.bar(x + 0.5*bw, [best_ba_rf_fc0,  best_ba_rf_all],
                       bw, color=RF_COL,  alpha=0.85, label="RF  BAL")
bar_rf_au  = ax_f.bar(x + 1.5*bw, [best_au_rf_fc0,  best_au_rf_all],
                       bw, color=RF_COL,  alpha=0.45, label="RF  AUC",
                       hatch="///", edgecolor=RF_COL)

all_bars = list(bar_lda_ba) + list(bar_lda_au) + list(bar_rf_ba) + list(bar_rf_au)
all_vals = ([best_ba_lda_fc0, best_ba_lda_all] +
            [best_au_lda_fc0, best_au_lda_all] +
            [best_ba_rf_fc0,  best_ba_rf_all]  +
            [best_au_rf_fc0,  best_au_rf_all])
for bar, val in zip(all_bars, all_vals):
    ax_f.text(bar.get_x()+bar.get_width()/2, val+0.004,
              f"{val:.3f}", ha="center", va="bottom", fontsize=6.5,
              rotation=90)

if ref_ba:
    ax_f.axhline(ref_ba, color=REF_COL, ls="--", lw=1.5, alpha=0.8,
                 label=f"G-space LDA ({ref_ba:.3f})")
ax_f.axhline(0.5, color="gray", ls="--", lw=1.0, alpha=0.5)
ax_f.set_xticks(x)
ax_f.set_xticklabels(cats, fontsize=8)
ax_f.set_ylabel("Performance")
ax_f.set_title("LDA vs RF summary\n(FC-only vs all lags, best K)")
ax_f.set_ylim(0.35, 0.95)
ax_f.legend(frameon=False, fontsize=7, ncol=2)

# ── panel labels ───────────────────────────────────────────────────────────────
for ax, lbl in zip([ax_a,ax_b,ax_c,ax_d,ax_e,ax_f],
                   ["A","B","C","D","E","F"]):
    ax.text(-0.14, 1.05, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

rf_str = (f"RF: n_est=500, min_leaf=3, max_feat=sqrt, "
          f"class_weight=balanced")
fig.suptitle(
    f"Lagged FC — LDA vs RF  |  {len(cc_idx)} CC, {len(ad_idx)} AD  "
    f"(all patients, LOPO)\n"
    f"Features: lag-0 ({N_SITES*(N_SITES-1)//2}) + "
    f"lags 1-{MAX_LAG} ({MAX_LAG}x{N_SITES**2}) = {D} total  |  "
    f"Transductive SVD\n{rf_str}",
    fontsize=8, y=1.02)

fig.savefig("fc_lag_classification.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved fc_lag_classification.png")

# ── print summary tables ───────────────────────────────────────────────────────
print("\n" + "="*70)
print(f"K SWEEP  (all lags 0-{MAX_LAG})")
print("="*70)
print(f"{'K':>5}  {'LDA-BAL':>8}  {'LDA-AUC':>8}  {'RF-BAL':>8}  {'RF-AUC':>8}  {'var%':>6}")
print("-"*70)
for k, bl, al, br, ar in zip(K_VALUES, ba_k_lda, au_k_lda, ba_k_rf, au_k_rf):
    s_l = " *" if k == best_k_lda else "  "
    s_r = " *" if k == best_k_rf  else "  "
    print(f"{k:>5}  {bl:.4f}{s_l}  {al:.4f}    {br:.4f}{s_r}  {ar:.4f}   "
          f"{cum_var[k-1]:>5.1f}%")

print()
print(f"INCREMENTAL LAG  (K={best_k_lda})")
print("="*70)
print(f"{'Lags':>8}  {'n_feat':>7}  {'LDA-BAL':>8}  {'LDA-AUC':>8}  "
      f"{'RF-BAL':>8}  {'RF-AUC':>8}")
print("-"*70)
for ml in range(MAX_LAG + 1):
    nf = lag0_d + ml * lagk_d if ml > 0 else lag0_d
    bl = ba_lag_lda[ml]; al = au_lag_lda[ml]
    br = ba_lag_rf[ml];  ar = au_lag_rf[ml]
    print(f"{'0-'+str(ml):>8}  {nf:>7d}  {bl:.4f}    {al:.4f}    {br:.4f}    {ar:.4f}")

if ref_ba:
    print(f"\nG-space LDA ref (N=40, K=15): BAL={ref_ba:.4f}  AUC={ref_au:.4f}")

print("\nDone.")
