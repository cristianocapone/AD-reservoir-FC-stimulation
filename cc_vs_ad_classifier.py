"""
CC vs AD classifier — AUC ≈ 0.76 (5-fold patient-level CV)
===========================================================

Pipeline
--------
1. Load parcellated fMRI timeseries (shape 121 × 140 per session)
2. Compute Ledoit-Wolf regularised covariance per session
3. Project to tangent space at the Euclidean mean (Riemannian embedding)
4. PCA(75) + linear SVM(C=0.05, class_weight='balanced')
5. Evaluate with StratifiedGroupKFold(5) keeping all sessions of the
   same patient in the same fold (no data leakage)

Expected data layout
--------------------
<DATA_ROOT>/
    timeseries_GSR/
        CN/   sub-..._ses-..._task-rest_bold_timeseries.npy   (121, 140)
        AD/   ...

Dependencies
------------
numpy>=2.4, scipy>=1.17, scikit-learn>=1.8, matplotlib>=3.10
    pip install numpy scipy scikit-learn matplotlib tqdm

Usage
-----
    python cc_vs_ad_classifier.py                        # uses ./data default
    python cc_vs_ad_classifier.py --data /path/to/data  # custom root
"""

import argparse
import os
import warnings
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import inv as la_inv, logm, sqrtm
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Constants ─────────────────────────────────────────────────────────────────
N_PARCELS   = 121
N_TIMEPOINTS = 140
TARGET_SHAPE = (N_PARCELS, N_TIMEPOINTS)

# Best hyperparameters found by grid search
N_PCA_COMPONENTS = 75
SVM_C            = 0.05
CV_FOLDS         = 5
RANDOM_STATE     = 42


# ── Data loading ──────────────────────────────────────────────────────────────

def load_sessions(data_root: str):
    """
    Load all valid (121, 140) sessions from CN and AD subfolders.

    Returns
    -------
    signals : list of np.ndarray, shape (121, 140) each
    patient_ids : list of str  — unique per patient, shared across sessions
    labels : list of int       — 0 = CN/CC, 1 = AD
    """
    gsr_root = os.path.join(data_root, "timeseries_GSR")
    signals, patient_ids, labels = [], [], []

    for group, label in [("CN", 0), ("AD", 1)]:
        folder = os.path.join(gsr_root, group)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"Expected folder not found: {folder}")
        fnames = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        for fname in fnames:
            arr = np.load(os.path.join(folder, fname))
            if arr.shape != TARGET_SHAPE:
                continue
            signals.append(arr)
            patient_ids.append(fname.split("_ses-")[0])
            labels.append(label)

    print(f"Loaded {len(signals)} sessions  "
          f"(CC={sum(l==0 for l in labels)}, AD={sum(l==1 for l in labels)})")
    n_cc_patients = len(set(p for p, l in zip(patient_ids, labels) if l == 0))
    n_ad_patients = len(set(p for p, l in zip(patient_ids, labels) if l == 1))
    print(f"Unique patients: CC={n_cc_patients}, AD={n_ad_patients}")
    return signals, patient_ids, labels


# ── Feature extraction ────────────────────────────────────────────────────────

def lw_covariance(ts: np.ndarray) -> np.ndarray:
    """
    Ledoit-Wolf shrinkage covariance.

    Parameters
    ----------
    ts : (N_PARCELS, N_TIMEPOINTS)

    Returns
    -------
    cov : (N_PARCELS, N_PARCELS)
    """
    return LedoitWolf(assume_centered=False).fit(ts.T).covariance_


def tangent_space_vectors(cov_mats: list[np.ndarray],
                          ref_cov: np.ndarray | None = None) -> np.ndarray:
    """
    Project a list of SPD covariance matrices to the tangent space
    at ref_cov (Euclidean mean if None).

    Each matrix C is mapped to:
        S   = M^{-1/2} @ C @ M^{-1/2}
        T   = logm(S)          (matrix logarithm)
        vec = upper-triangle of T, off-diag scaled by sqrt(2)

    This gives Euclidean coordinates where standard linear classifiers
    are geometrically appropriate for SPD matrices.

    Returns
    -------
    X : (n_sessions, n_features)   n_features = N_PARCELS*(N_PARCELS+1)//2
    """
    if ref_cov is None:
        ref_cov = np.mean(cov_mats, axis=0)

    M_sqrt = sqrtm(ref_cov).real
    M_inv  = la_inv(M_sqrt)

    n   = ref_cov.shape[0]
    idx = np.triu_indices(n, k=0)
    off = idx[0] != idx[1]

    vecs = []
    for C in cov_mats:
        S = M_inv @ C @ M_inv
        T = logm(S).real
        T = (T + T.T) / 2          # enforce symmetry after numerical noise
        v = T[idx].copy()
        v[off] *= np.sqrt(2)       # standard tangent-space normalisation
        vecs.append(v)

    return np.array(vecs)


def extract_features(signals: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute LW covariance and tangent-space features for all sessions.

    Returns
    -------
    X      : (n_sessions, n_features)
    ref_cov: (N_PARCELS, N_PARCELS)  — Euclidean mean used as tangent point
    """
    print("Computing Ledoit-Wolf covariances...")
    cov_mats = [lw_covariance(ts) for ts in tqdm(signals)]
    ref_cov  = np.mean(cov_mats, axis=0)

    print("Projecting to tangent space...")
    X = tangent_space_vectors(cov_mats, ref_cov)

    print(f"Feature matrix: {X.shape}")
    return X, ref_cov


# ── Cross-validation ──────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Best pipeline from grid search:
      StandardScaler → PCA(75) → linear SVM(C=0.05, balanced)
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)),
        ("clf",    SVC(kernel="linear", C=SVM_C, probability=True,
                       class_weight="balanced", random_state=RANDOM_STATE)),
    ])


def run_cross_validation(X: np.ndarray,
                         y: np.ndarray,
                         groups: np.ndarray) -> dict:
    """
    5-fold patient-level stratified group CV.

    All sessions of the same patient are kept in the same fold, so the
    model never sees any session of a test patient during training.

    Returns
    -------
    dict with per-fold arrays and concatenated OOF predictions.
    """
    sgkf = StratifiedGroupKFold(n_splits=CV_FOLDS,
                                shuffle=True,
                                random_state=RANDOM_STATE)

    fold_accs, fold_bacs, fold_aucs = [], [], []
    oof_y, oof_scores = [], []

    for fold, (tr, te) in enumerate(sgkf.split(X, y, groups)):
        pipe = build_pipeline()
        pipe.fit(X[tr], y[tr])

        y_pred  = pipe.predict(X[te])
        y_score = pipe.predict_proba(X[te])[:, 1]

        fold_accs.append(accuracy_score(y[te], y_pred))
        fold_bacs.append(balanced_accuracy_score(y[te], y_pred))
        fold_aucs.append(roc_auc_score(y[te], y_score))
        oof_y.extend(y[te].tolist())
        oof_scores.extend(y_score.tolist())

        print(f"  Fold {fold+1}: AUC={fold_aucs[-1]:.3f}  "
              f"BAcc={fold_bacs[-1]:.3f}  Acc={fold_accs[-1]:.3f}")

    return {
        "fold_accs":  np.array(fold_accs),
        "fold_bacs":  np.array(fold_bacs),
        "fold_aucs":  np.array(fold_aucs),
        "oof_y":      np.array(oof_y),
        "oof_scores": np.array(oof_scores),
    }


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(cv_results: dict, out_prefix: str = "cc_vs_ad"):
    oof_y, oof_scores = cv_results["oof_y"], cv_results["oof_scores"]
    fold_aucs = cv_results["fold_aucs"]

    fpr, tpr, _ = roc_curve(oof_y, oof_scores)
    auc_oof = roc_auc_score(oof_y, oof_scores)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ROC curve (concatenated OOF predictions)
    ax = axes[0]
    ax.plot(fpr, tpr, color="steelblue", lw=2.0,
            label=f"OOF AUC = {auc_oof:.3f}")
    ax.fill_between(fpr, tpr, alpha=0.15, color="steelblue")
    ax.plot([0, 1], [0, 1], "k:", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("CC vs AD — ROC curve\n"
                 "(Tangent GSR · PCA75 · linear SVM, "
                 f"patient-level {CV_FOLDS}-fold CV)")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)

    # Per-fold AUC bar chart
    ax2 = axes[1]
    fold_labels = [f"Fold {i+1}" for i in range(len(fold_aucs))]
    ax2.bar(fold_labels, fold_aucs, color="steelblue", alpha=0.75, edgecolor="navy")
    ax2.axhline(fold_aucs.mean(), color="firebrick", lw=1.8, linestyle="--",
                label=f"Mean AUC = {fold_aucs.mean():.3f} ± {fold_aucs.std():.3f}")
    ax2.axhline(0.70, color="orange", lw=1.2, linestyle=":",
                label="0.70 reference")
    ax2.set_ylim(0.4, 1.0)
    ax2.set_ylabel("AUROC")
    ax2.set_title("Per-fold AUROCs")
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_prefix}_roc.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    plt.close()


def plot_feature_importance(X: np.ndarray, y: np.ndarray,
                            out_prefix: str = "cc_vs_ad"):
    """
    Train on all data and back-project the linear SVM weights to
    the original tangent-space feature space → FC edge weight matrix.
    """
    pipe = build_pipeline()
    pipe.fit(X, y)

    # back-project: clf weights → PCA space → original feature space
    w_pca = pipe["clf"].coef_.ravel()
    w     = pipe["pca"].components_.T @ w_pca   # (n_features,)

    # reconstruct symmetric weight matrix from upper-triangle (incl. diagonal)
    W_mat = np.zeros((N_PARCELS, N_PARCELS))
    idx_ut = np.triu_indices(N_PARCELS, k=0)
    W_mat[idx_ut] = w[: len(idx_ut[0])]
    W_mat = (W_mat + W_mat.T) / 2
    np.fill_diagonal(W_mat, 0)

    node_w = np.abs(W_mat).sum(axis=1)
    top5   = np.argsort(node_w)[::-1][:5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    lim = np.abs(W_mat).max()
    im  = axes[0].imshow(W_mat, cmap="RdBu_r", aspect="auto",
                          vmin=-lim, vmax=lim)
    plt.colorbar(im, ax=axes[0], fraction=0.046)
    axes[0].set_title("FC weight matrix — AD+ direction\n"
                       "(red = stronger in AD, blue = stronger in CC)")
    axes[0].set_xlabel("Parcel index")
    axes[0].set_ylabel("Parcel index")

    bars = axes[1].bar(range(N_PARCELS), node_w, color="steelblue", alpha=0.75)
    for i in top5:
        bars[i].set_color("firebrick")
    axes[1].set_xlabel("Parcel index")
    axes[1].set_ylabel("Σ|weight| over edges")
    axes[1].set_title(f"Node importance  (top-5 highlighted: {list(top5)})")
    axes[1].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        path = f"{out_prefix}_importance.{ext}"
        plt.savefig(path, bbox_inches="tight", dpi=200)
        print(f"Saved: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="CC vs AD fMRI classifier (Tangent-GSR · PCA75 · SVM)"
    )
    parser.add_argument(
        "--data", default="./data",
        help="Root directory containing timeseries_GSR/CN and timeseries_GSR/AD "
             "(default: ./data)"
    )
    parser.add_argument(
        "--out", default="cc_vs_ad",
        help="Prefix for output figures (default: cc_vs_ad)"
    )
    args = parser.parse_args()

    np.random.seed(RANDOM_STATE)

    # 1. Load
    signals, patient_ids, labels = load_sessions(args.data)
    y      = np.array(labels)
    groups = np.array(patient_ids)

    # 2. Extract tangent-space features
    X, _ = extract_features(signals)

    # 3. Cross-validate
    print(f"\nRunning {CV_FOLDS}-fold patient-level stratified group CV...")
    cv = run_cross_validation(X, y, groups)

    # 4. Print summary
    print("\n" + "=" * 55)
    print("RESULTS SUMMARY")
    print("=" * 55)
    print(f"  AUC  (mean ± std):  {cv['fold_aucs'].mean():.3f} ± {cv['fold_aucs'].std():.3f}")
    print(f"  BAcc (mean ± std):  {cv['fold_bacs'].mean():.3f} ± {cv['fold_bacs'].std():.3f}")
    print(f"  Acc  (mean ± std):  {cv['fold_accs'].mean():.3f} ± {cv['fold_accs'].std():.3f}")
    print(f"  OOF AUC:            {roc_auc_score(cv['oof_y'], cv['oof_scores']):.3f}")
    print("=" * 55)

    # 5. Save CV results (for integration into other figures)
    np.savez(f"{args.out}_cv.npz",
             fold_aucs=cv["fold_aucs"], fold_bacs=cv["fold_bacs"],
             fold_accs=cv["fold_accs"],
             oof_y=cv["oof_y"], oof_scores=cv["oof_scores"])
    print(f"Saved: {args.out}_cv.npz")

    # 6. Figures
    plot_results(cv, out_prefix=args.out)
    plot_feature_importance(X, y, out_prefix=args.out)


if __name__ == "__main__":
    main()
