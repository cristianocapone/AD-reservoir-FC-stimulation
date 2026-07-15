"""
Reservoir + FC-lag classification on ABIDE I (ASD vs Control).

Mirrors the AD paper pipeline:
  1. Load ABIDE Harvard-Oxford time series (from abide_timeseries.npz)
  2. Truncate all sessions to a common length T_common
  3. PCA-project to top-50 population components (per paper)
  4. Fit a ridge-regularised linear read-out per session (reservoir teacher-forcing)
  5. Run closed-loop reconstruction; compute lagged-FC features (lags 0-2)
  6. Build G-space (SVD archetype geometry across subjects)
  7. Classify ASD vs Control with balanced LDA and RF
     under repeated stratified 5-fold subject-level CV
  8. Tangent-space SVM benchmark (train-only reference, nested 5x5)

Outputs
-------
  ABIDE/abide_reservoir_results.npz   - all scores and folds
  ABIDE/abide_reservoir_results.png   - AUROC curves
"""

import sys, warnings, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf
from scipy.linalg import logm, sqrtm, inv as la_inv
from tqdm import trange, tqdm

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

WORK  = os.path.dirname(os.path.abspath(__file__))
NPZ   = os.path.join(WORK, "ABIDE", "abide_timeseries.npz")
OUT   = os.path.join(WORK, "ABIDE", "abide_reservoir_results")

# ── Reservoir hyper-parameters (same as AD paper) ─────────────────────────────
N_RES      = 2000      # reservoir units
SIGMA_IN   = 0.01      # input weight scale
SR         = 0.95      # spectral radius
DT         = 0.005     # discrete time step (dimensionless, ~1/TR)
TAU        = 1e-4 * DT # effectively instantaneous (paper: tau << dt)
SIGMA_FIT  = 0.025     # read-out regularisation noise (paper default)
FF         = 0.1       # feedback gain during teacher-forcing and closed-loop
N_PCA_POP  = 50        # population PCA components (paper: 50)
N_DRIVE    = 5         # closed-loop warm-up steps
K_READOUT  = 200       # G-space: project W onto top-K reservoir singular vecs
K_SVD      = 25        # G-space / FC-lag embedding dimension for classification
N_FOLDS    = 5         # outer CV folds
N_REPEATS  = 10        # CV repeats
RNG        = 42

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("Loading ABIDE time series (from nilearn cache)...")
from nilearn.datasets import fetch_abide_pcp

abide   = fetch_abide_pcp(
    data_dir=os.path.join(WORK, "ABIDE"),
    pipeline="cpac", band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_ho"], quality_checked=True, verbose=0,
)
ts_list = abide.rois_ho        # list of (T_i, 111) arrays
pheno   = abide.phenotypic     # proper structured array

dx      = np.array(pheno["DX_GROUP"]).astype(int)   # 1=ASD, 2=Control
y       = (dx == 1).astype(int)                      # 1=ASD, 0=Control
sites   = np.array([s.decode() if isinstance(s, bytes) else str(s)
                    for s in pheno["SITE_ID"]])
sub_ids = np.array([str(s) for s in pheno["SUB_ID"]])

# ── 2. Truncate to common length ───────────────────────────────────────────────
T_common = int(np.percentile([ts.shape[0] for ts in ts_list], 10))
T_common = max(T_common, 100)     # at least 100 volumes
print(f"  N={len(ts_list)}  ASD={y.sum()}  Ctrl={(1-y).sum()}")
print(f"  Truncating sessions to T={T_common} volumes (10th percentile)")

# keep only sessions at least T_common long
keep    = np.array([ts.shape[0] >= T_common for ts in ts_list])
ts_list = [ts_list[i][:T_common, :].T for i in range(len(ts_list)) if keep[i]]
# shape after: each is (N_p=111, T_common), matching the paper's (N_p, T)
y       = y[keep]
sites   = sites[keep]
sub_ids = sub_ids[keep]
N_p     = ts_list[0].shape[0]     # 111 parcels

print(f"  Kept {len(ts_list)} sessions (>= {T_common} vol)  "
      f"ASD={y.sum()}  Ctrl={(1-y).sum()}")
print(f"  Parcels: {N_p}")

# ── 3. Population PCA projection ───────────────────────────────────────────────
print("\nPopulation PCA projection...")
all_ts   = np.hstack(ts_list)                  # (N_p, N * T)
all_ts  -= all_ts.mean(axis=1, keepdims=True)
U, S, Vt = np.linalg.svd(all_ts, full_matrices=False)
V50      = U[:, :N_PCA_POP]                    # (N_p, 50) — top eigenvectors

targets = []
for ts in tqdm(ts_list, desc="  projecting"):
    s  = ts - ts.mean(axis=1, keepdims=True)
    t_ = V50 @ (V50.T @ s)                     # PCA reconstruction, (N_p, T)
    targets.append(t_)

# ── 4. Build reservoir (fixed random, shared across all sessions) ──────────────
print("\nBuilding reservoir...")
rng = np.random.default_rng(RNG)

J    = rng.standard_normal((N_RES, N_RES)) / np.sqrt(N_RES)
eigs = np.linalg.eigvals(J)
J   *= SR / np.abs(eigs).max()               # rescale to spectral radius

Jin  = rng.standard_normal((N_RES, N_p)) * SIGMA_IN

decay = np.exp(-DT / TAU)                   # ~0 (instantaneous)
gain  = 1.0 - decay

# ── 5. Read-out fitting per session ───────────────────────────────────────────
def fit_readout(target, sigma=SIGMA_FIT):
    """Teacher-force reservoir on target; fit read-out W by ridge-like LS."""
    T   = target.shape[1]
    x   = np.zeros(N_RES)
    X   = np.empty((T - 1, N_RES))
    for t in range(T - 1):
        inp = FF * target[:, t]
        x   = decay * x + gain * (J @ x + Jin @ inp)
        X[t] = x

    Y   = target[:, 1:].T               # (T-1, N_p) targets
    E   = rng.standard_normal(X.shape) * sigma
    W   = np.linalg.lstsq(X + E, Y, rcond=None)[0]  # (N_RES, N_p)
    return W, X


def run_closedloop(W, target):
    """Run reservoir closed-loop: drive N_DRIVE steps then free-run."""
    T   = target.shape[1]
    x   = np.zeros(N_RES)
    Y   = np.empty((T - 1, N_p))
    for t in range(T - 1):
        if t < N_DRIVE:
            inp = FF * target[:, t]
        else:
            inp = FF * (W.T @ x)         # closed-loop
        x    = decay * x + gain * (J @ x + Jin @ inp)
        Y[t] = W.T @ x
    return Y                             # (T-1, N_p)


def lagged_fc_features(Y, max_lag=2):
    """Concatenate lagged correlation matrices as feature vector."""
    T, N = Y.shape
    feats = []
    # lag 0: upper triangle
    C0 = np.corrcoef(Y.T)
    idx = np.triu_indices(N, k=1)
    feats.append(C0[idx])
    # lags 1..max_lag: full matrix flattened
    for lag in range(1, max_lag + 1):
        C = np.corrcoef(Y[:-lag].T, Y[lag:].T)[:N, N:]
        feats.append(C.flatten())
    return np.concatenate(feats)


print("Fitting reservoir read-outs per session (this takes a few minutes)...")
W_all    = []      # read-outs
X_all    = []      # reservoir states (for G-space)
FC_lag_all = []    # lagged-FC feature vectors

for i in trange(len(targets)):
    W, X_driven = fit_readout(targets[i])
    W_all.append(W)                    # (N_RES, N_p)
    X_all.append(X_driven)            # (T-1, N_RES)
    Y_cl   = run_closedloop(W, targets[i])
    FC_lag_all.append(lagged_fc_features(Y_cl))

W_all      = np.array(W_all)          # (n, N_RES, N_p)
FC_lag_all = np.array(FC_lag_all)     # (n, n_features)
print(f"  W shape: {W_all.shape}")
print(f"  FC-lag feature dim: {FC_lag_all.shape[1]}")

# ── 6. G-space read-out archetype embedding ────────────────────────────────────
print("\nBuilding G-space embedding...")

# Project each W onto its own reservoir state singular vectors
W_proj = []
for i in range(len(W_all)):
    _, _, Vt_res = np.linalg.svd(X_all[i], full_matrices=False)
    V_k = Vt_res[:K_READOUT].T                         # (N_RES, K_READOUT)
    w_proj = (W_all[i].T @ V_k @ V_k.T).flatten()     # project & vectorise
    W_proj.append(w_proj)

W_proj = np.array(W_proj)    # (n, N_p * N_RES) — large; centre and SVD
W_c    = W_proj - W_proj.mean(axis=0, keepdims=True)
_, _, Vt_g = np.linalg.svd(W_c, full_matrices=False)
G_scores = W_c @ Vt_g[:K_SVD].T    # (n, K_SVD) — G-space coordinates

# FC-lag Gram SVD embedding
FC_c   = FC_lag_all - FC_lag_all.mean(axis=0, keepdims=True)
_, _, Vt_fc = np.linalg.svd(FC_c, full_matrices=False)
FC_scores   = FC_c @ Vt_fc[:K_SVD].T    # (n, K_SVD)

print(f"  G-space:  {G_scores.shape}")
print(f"  FC-lag:   {FC_scores.shape}")

# ── 7. Repeated stratified 5-fold CV ──────────────────────────────────────────
print("\nClassification (repeated 5-fold CV)...")

def cv_score(X, y, n_folds=N_FOLDS, n_repeats=N_REPEATS, clf_type="lda"):
    aucs, bacs = [], []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                              random_state=RNG + rep)
        for tr, te in skf.split(X, y):
            X_tr, X_te = X[tr], X[te]
            y_tr, y_te = y[tr], y[te]
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X_tr)
            Xte = sc.transform(X_te)
            if clf_type == "lda":
                clf = LinearDiscriminantAnalysis()
            else:
                clf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                             min_samples_leaf=3,
                                             class_weight="balanced",
                                             random_state=RNG)
            clf.fit(Xtr, y_tr)
            ys  = (clf.predict_proba(Xte)[:, 1] if hasattr(clf, "predict_proba")
                   else clf.decision_function(Xte))
            if len(np.unique(y_te)) < 2:
                continue
            aucs.append(roc_auc_score(y_te, ys))
            bacs.append(balanced_accuracy_score(y_te, clf.predict(Xte)))
    return np.array(aucs), np.array(bacs)

results = {}
for feat_name, Xfeat in [("G-space", G_scores), ("FC-lag", FC_scores)]:
    for clf_name in ["lda", "rf"]:
        key = f"{feat_name}_{clf_name}"
        aucs, bacs = cv_score(Xfeat, y, clf_type=clf_name)
        results[key] = {"aucs": aucs, "bacs": bacs}
        print(f"  {key:20s}  AUROC={aucs.mean():.3f}+/-{aucs.std():.3f}"
              f"  BAcc={bacs.mean():.3f}+/-{bacs.std():.3f}")

# ── 8. Tangent-space SVM benchmark (subject-level nested 5x5) ─────────────────
print("\nTangent-space SVM benchmark (nested 5x5, subject-level)...")

covs = np.array([LedoitWolf(assume_centered=False).fit(ts.T).covariance_
                 for ts in tqdm(targets, desc="  LW covs")])

def tangent_vecs(covs_sub, ref):
    Ms = sqrtm(ref).real; Mi = la_inv(Ms)
    idx = np.triu_indices(ref.shape[0], k=0); off = idx[0] != idx[1]
    out = []
    for C in covs_sub:
        S = Mi @ C @ Mi; T_ = logm(S).real; T_ = (T_ + T_.T) / 2
        v = T_[idx].copy(); v[off] *= np.sqrt(2)
        out.append(v)
    return np.array(out)

sgkf  = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RNG)
param_grid = [(n, C) for n in [30, 50, 75] for C in [0.01, 0.1, 1.0]]
tan_aucs, tan_oof_y, tan_oof_s = [], [], []

for tr, te in sgkf.split(covs, y, sub_ids):
    cov_tr, y_tr = covs[tr], y[tr]
    # inner 5-fold to pick n_pca, C
    best_auc, best_p = -1, param_grid[0]
    inner_splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0)
                        .split(cov_tr, y_tr, sub_ids[tr]))
    for n, C in param_grid:
        iauc = []
        for itr, ival in inner_splits:
            ref  = cov_tr[itr].mean(0)
            Ttr  = StandardScaler().fit_transform(tangent_vecs(cov_tr[itr], ref))
            Tval_raw = tangent_vecs(cov_tr[ival], ref)
            sc2  = StandardScaler().fit(Ttr)
            Tval = sc2.transform(Tval_raw)
            Ttr  = sc2.transform(Ttr)
            pca  = PCA(n, random_state=RNG).fit(Ttr)
            clf  = SVC(kernel="linear", C=C, probability=True,
                       class_weight="balanced", random_state=RNG)
            clf.fit(pca.transform(Ttr), y_tr[itr])
            if len(np.unique(y_tr[ival])) < 2:
                continue
            iauc.append(roc_auc_score(y_tr[ival],
                        clf.predict_proba(pca.transform(Tval))[:, 1]))
        if iauc and np.mean(iauc) > best_auc:
            best_auc, best_p = np.mean(iauc), (n, C)
    n, C = best_p
    ref  = cov_tr.mean(0)
    sc   = StandardScaler()
    Ttr  = sc.fit_transform(tangent_vecs(cov_tr, ref))
    Tte  = sc.transform(tangent_vecs(covs[te], ref))
    pca  = PCA(n, random_state=RNG).fit(Ttr)
    clf  = SVC(kernel="linear", C=C, probability=True,
               class_weight="balanced", random_state=RNG)
    clf.fit(pca.transform(Ttr), y_tr)
    ys = clf.predict_proba(pca.transform(Tte))[:, 1]
    tan_aucs.append(roc_auc_score(y[te], ys))
    tan_oof_y.extend(y[te]); tan_oof_s.extend(ys)
    print(f"    fold AUC={tan_aucs[-1]:.3f}  best_p={best_p}")

tan_aucs = np.array(tan_aucs)
print(f"  Tangent SVM: AUROC={tan_aucs.mean():.3f}+/-{tan_aucs.std():.3f}")

# ── 9. Save and plot ───────────────────────────────────────────────────────────
np.savez(OUT + ".npz", y=y, sites=sites, sub_ids=sub_ids,
         G_scores=G_scores, FC_scores=FC_scores,
         tan_oof_y=np.array(tan_oof_y), tan_oof_s=np.array(tan_oof_s),
         tan_fold_aucs=tan_aucs,
         **{k + "_aucs": v["aucs"] for k, v in results.items()},
         **{k + "_bacs": v["bacs"] for k, v in results.items()})
print(f"\nSaved {OUT}.npz")

fig, ax = plt.subplots(figsize=(9, 5))
colors = {"G-space_lda": "#1f77b4", "G-space_rf": "#aec7e8",
          "FC-lag_lda": "#d62728",  "FC-lag_rf": "#f7b6b2"}
for key, res in results.items():
    mu, sd = res["aucs"].mean(), res["aucs"].std()
    ax.bar(key, mu, yerr=sd, color=colors[key], capsize=4,
           alpha=0.85, edgecolor="k", linewidth=0.6,
           label=f"{key}  {mu:.3f}+-{sd:.3f}")

tan_mu, tan_sd = tan_aucs.mean(), tan_aucs.std()
ax.axhline(tan_mu, color="k", lw=1.5, ls="--",
           label=f"Tangent SVM  {tan_mu:.3f}+-{tan_sd:.3f}")
ax.fill_between([-0.5, len(results) - 0.5],
                [tan_mu - tan_sd] * 2, [tan_mu + tan_sd] * 2,
                color="k", alpha=0.08)
ax.axhline(0.5, color="grey", lw=0.8, ls=":")
ax.set_ylim(0.4, 0.9)
ax.set_ylabel("AUROC (mean +/- SD over CV folds)", fontsize=11)
ax.set_title(f"ABIDE I: ASD vs Control  N={len(y)}\n"
             f"Reservoir + FC-lag / G-space  |  {N_FOLDS}-fold x {N_REPEATS} repeats",
             fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(OUT + ".png", dpi=180, bbox_inches="tight")
plt.savefig(OUT + ".pdf", bbox_inches="tight")
print(f"Saved {OUT}.png / .pdf")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"  ABIDE I  ASD vs Control  N={len(y)}  T={T_common}  parcels={N_p}")
print("=" * 60)
for key, res in results.items():
    print(f"  {key:22s}  AUROC {res['aucs'].mean():.3f} +/- {res['aucs'].std():.3f}")
print(f"  {'Tangent SVM':22s}  AUROC {tan_mu:.3f} +/- {tan_sd:.3f}")
print("=" * 60)
