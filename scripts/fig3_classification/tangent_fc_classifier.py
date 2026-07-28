"""
Tangent-space FC classifier: CN vs AD (ADNI, 121 parcels).

Pipeline
--------
1. Load CN + AD resting-state timeseries from ./data/timeseries/
2. Estimate per-session covariance with Ledoit-Wolf shrinkage
3. Nested 5×5 patient-level CV:
     inner: project to tangent space (train-only reference), StandardScaler,
            PCA, SVM or LR — sweep (n_pca, C, classifier)
     outer: evaluate best hyper-params on held-out patients
4. Report AUROC and save ROC curve

Expected result: AUC ≈ 0.787 ± 0.104
"""

import os, warnings
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import logm, sqrtm, inv as la_inv
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
from tqdm import tqdm

warnings.filterwarnings('ignore')

WORK_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(WORK_DIR, 'data', 'timeseries')
if not os.path.isdir(DATA_ROOT):                       # fall back to repo-root data
    DATA_ROOT = os.path.join(WORK_DIR, 'timeseries_GSR')   # GSR preproc (paper benchmark)
    if not os.path.isdir(DATA_ROOT):
        DATA_ROOT = os.path.join(WORK_DIR, 'timeseries')
OUT       = os.path.join(WORK_DIR, 'tangent_fc_results')

N_OUTER    = 5
N_INNER    = 5
FOLDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'tangent_fc_folds.npz')


def generate_folds(y, g, n_outer, n_inner, seed_out=42, seed_in=0):
    """Generate nested fold indices with sklearn StratifiedGroupKFold."""
    from sklearn.model_selection import StratifiedGroupKFold
    sgkf_out = StratifiedGroupKFold(n_splits=n_outer, shuffle=True, random_state=seed_out)
    sgkf_in  = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed_in)
    outer, inner = [], []
    for otr, ote in sgkf_out.split(y, y, g):
        outer.append((otr, ote))
        fold_inner = []
        for itr, ival in sgkf_in.split(y[otr], y[otr], g[otr]):
            fold_inner.append((itr, ival))   # local indices into otr
        inner.append(fold_inner)
    return outer, inner


def load_or_generate_folds(y, g):
    """Load saved fold indices if available, else generate and save them."""
    if os.path.exists(FOLDS_FILE):
        f = np.load(FOLDS_FILE, allow_pickle=True)
        outer = [(f[f'otr{i}'], f[f'ote{i}']) for i in range(N_OUTER)]
        inner = [[(f[f'itr{i}_{j}'], f[f'ival{i}_{j}']) for j in range(N_INNER)]
                 for i in range(N_OUTER)]
        print(f"Loaded fold indices from {FOLDS_FILE}")
    else:
        print("Generating fold indices with sklearn and saving...")
        outer, inner = generate_folds(y, g, N_OUTER, N_INNER)
        save = {}
        for i, (otr, ote) in enumerate(outer):
            save[f'otr{i}'] = otr; save[f'ote{i}'] = ote
            for j, (itr, ival) in enumerate(inner[i]):
                save[f'itr{i}_{j}'] = itr; save[f'ival{i}_{j}'] = ival
        np.savez(FOLDS_FILE, **save)
        print(f"Saved: {FOLDS_FILE}  (copy this file alongside the script for reproducibility)")
    return outer, inner

# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data():
    raw, pids, lbls = [], [], []
    for subfolder, lbl in [('CN', 0), ('AD', 1)]:
        folder = os.path.join(DATA_ROOT, subfolder)
        if not os.path.isdir(folder):
            subfolder = 'CC'; folder = os.path.join(DATA_ROOT, subfolder)
        for fname in sorted(f for f in os.listdir(folder) if f.endswith('.npy')):
            arr = np.load(os.path.join(folder, fname)).T      # (N,T) → (T,N)
            if arr.shape[0] > 100 and arr.shape[1] == 121:
                raw.append(arr)
                pids.append(fname.split('_ses-')[0])
                lbls.append(lbl)
    return raw, np.array(pids), np.array(lbls)

print("Loading data...", flush=True)
signals, g, y = load_data()
print(f"  {len(signals)} sessions  CN={sum(y==0)}  AD={sum(y==1)}"
      f"  patients CN={len(set(g[y==0]))}  AD={len(set(g[y==1]))}")

# ── 2. LW covariances ─────────────────────────────────────────────────────────
print("Computing Ledoit-Wolf covariances...", flush=True)
covs = np.array([LedoitWolf(assume_centered=False).fit(s).covariance_
                 for s in tqdm(signals)])   # (702, 121, 121)

# ── 3. Tangent-space helpers ──────────────────────────────────────────────────
def tangent_vectors(covs_subset, ref):
    """Map covariance matrices to tangent vectors at SPD manifold point ref."""
    Ms  = sqrtm(ref).real
    Mi  = la_inv(Ms)
    idx = np.triu_indices(ref.shape[0], k=0)
    off = idx[0] != idx[1]
    vecs = []
    for C in covs_subset:
        S = Mi @ C @ Mi
        T = logm(S).real
        T = (T + T.T) / 2        # symmetrise (numerical safety)
        v = T[idx].copy()
        v[off] *= np.sqrt(2)      # off-diagonal × √2 → isometric embedding
        vecs.append(v)
    return np.array(vecs)         # (n, 121*122//2 = 7381)

# ── 4. Nested 5×5 patient-level CV ───────────────────────────────────────────
# Fold indices are saved to disk on first run; load them on every subsequent run
# so results are identical regardless of sklearn version on any machine.
outer_folds, inner_folds_all = load_or_generate_folds(y, g)

param_grid = [(n, C, ct)
              for n  in [30, 50, 75, 100, 125]
              for C  in [0.01, 0.05, 0.1, 0.5, 1.0]
              for ct in ['svm', 'lr']]

print(f"\nNested {N_OUTER}×{N_INNER} CV  |  param grid: {len(param_grid)} combos",
      flush=True)
print("="*60)

outer_aucs = []; outer_bacs = []; oof_yt = []; oof_ys = []

for fold_o, (otr, ote) in enumerate(outer_folds):
    cov_o, y_o = covs[otr], y[otr]

    # ── inner CV: pre-compute tangent per inner split ──────────────────────
    inner_data = []
    for itr, ival in inner_folds_all[fold_o]:
        ref   = cov_o[itr].mean(axis=0)          # train-only reference
        T_tr  = tangent_vectors(cov_o[itr], ref)
        T_val = tangent_vectors(cov_o[ival], ref)
        sc    = StandardScaler()
        T_tr  = sc.fit_transform(T_tr)
        T_val = sc.transform(T_val)
        inner_data.append((T_tr, T_val, y_o[itr], y_o[ival]))

    # ── inner CV: sweep hyper-params ───────────────────────────────────────
    best_auc = -1; best_p = param_grid[0]
    for n, C, ct in param_grid:
        iauc = []
        for (T_tr, T_val, yi_tr, yi_val) in inner_data:
            if len(np.unique(yi_val)) < 2: continue
            pca   = PCA(n, random_state=42)
            Xp_tr = pca.fit_transform(T_tr)
            Xp_val = pca.transform(T_val)
            clf   = (SVC(kernel='linear', C=C, probability=True,
                         class_weight='balanced', random_state=42)
                     if ct == 'svm' else
                     LogisticRegression(C=C, max_iter=2000,
                                        class_weight='balanced', random_state=42))
            clf.fit(Xp_tr, yi_tr)
            iauc.append(roc_auc_score(yi_val, clf.predict_proba(Xp_val)[:, 1]))
        if iauc and np.mean(iauc) > best_auc:
            best_auc = np.mean(iauc); best_p = (n, C, ct)

    # ── outer evaluation ───────────────────────────────────────────────────
    n, C, ct = best_p
    ref    = cov_o.mean(axis=0)
    T_tr   = tangent_vectors(cov_o, ref)
    T_te   = tangent_vectors(covs[ote], ref)
    sc     = StandardScaler()
    T_tr   = sc.fit_transform(T_tr); T_te = sc.transform(T_te)
    pca    = PCA(n, random_state=42)
    T_tr   = pca.fit_transform(T_tr); T_te = pca.transform(T_te)
    clf    = (SVC(kernel='linear', C=C, probability=True,
                  class_weight='balanced', random_state=42)
              if ct == 'svm' else
              LogisticRegression(C=C, max_iter=2000,
                                 class_weight='balanced', random_state=42))
    clf.fit(T_tr, y_o)
    ys = clf.predict_proba(T_te)[:, 1]
    yp = clf.predict(T_te)

    auc = roc_auc_score(y[ote], ys)
    bac = balanced_accuracy_score(y[ote], yp)
    outer_aucs.append(auc); outer_bacs.append(bac)
    oof_yt.extend(y[ote].tolist()); oof_ys.extend(ys.tolist())
    print(f"  fold {fold_o+1}: AUC={auc:.3f}  BAcc={bac:.3f}  best={best_p}",
          flush=True)

outer_aucs = np.array(outer_aucs); outer_bacs = np.array(outer_bacs)
oof_yt = np.array(oof_yt); oof_ys = np.array(oof_ys)

print("="*60)
print(f"  AUC  = {outer_aucs.mean():.3f} ± {outer_aucs.std():.3f}")
print(f"  BAcc = {outer_bacs.mean():.3f} ± {outer_bacs.std():.3f}")
print(f"  OOF AUC = {roc_auc_score(oof_yt, oof_ys):.3f}")

# ── save artifacts for the paper's classification figure (Fig 3) ──────────────
ref_all = covs.mean(axis=0)
T_all   = StandardScaler().fit_transform(tangent_vectors(covs, ref_all))
pcs2    = PCA(2, random_state=42).fit_transform(T_all)   # illustrative 2-D clouds
np.savez(os.path.join(WORK_DIR, "tangent_fc_cv.npz"),
         oof_y=oof_yt, oof_scores=oof_ys, fold_aucs=outer_aucs, fold_bacs=outer_bacs,
         pcs=pcs2, y=y, n_cc=int((y == 0).sum()), n_ad=int((y == 1).sum()),
         n_pat_cc=len(set(g[y == 0])), n_pat_ad=len(set(g[y == 1])))
print("Saved tangent_fc_cv.npz")

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
fpr, tpr, _ = roc_curve(oof_yt, oof_ys)
ax.plot(fpr, tpr, color='steelblue', lw=2,
        label=f"Tangent FC  AUC={roc_auc_score(oof_yt, oof_ys):.3f}")
ax.fill_between(fpr, tpr, alpha=0.15, color='steelblue')
ax.plot([0, 1], [0, 1], 'k:', alpha=0.4)
ax.set_xlabel('FPR', fontsize=12); ax.set_ylabel('TPR', fontsize=12)
ax.set_title('Tangent-space FC — CN vs AD\n'
             f'(nested {N_OUTER}×{N_INNER} CV, patient-level groups)', fontsize=11)
ax.legend(fontsize=11, loc='lower right'); ax.grid(alpha=0.3)

ax2 = axes[1]
ax2.bar(range(N_OUTER), outer_aucs, color='steelblue', alpha=0.8,
        edgecolor='k', lw=0.7)
ax2.axhline(outer_aucs.mean(), color='navy', lw=2, ls='--',
            label=f"mean={outer_aucs.mean():.3f}")
ax2.axhline(0.5, color='grey', lw=1, ls=':')
for i, v in enumerate(outer_aucs):
    ax2.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom',
             fontsize=10, fontweight='bold')
ax2.set_xticks(range(N_OUTER))
ax2.set_xticklabels([f'fold {i+1}' for i in range(N_OUTER)])
ax2.set_ylim(0.4, 1.0)
ax2.set_ylabel('AUROC', fontsize=12)
ax2.set_title('Per-fold AUC', fontsize=11)
ax2.legend(fontsize=10); ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
for ext in ('pdf', 'png'):
    plt.savefig(f'{OUT}.{ext}', bbox_inches='tight', dpi=200)
    print(f"Saved: {OUT}.{ext}")
plt.close()
