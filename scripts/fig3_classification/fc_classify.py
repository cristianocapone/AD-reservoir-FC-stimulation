"""
Functional connectivity (FC) classification: AD vs CN (and AD+MCI vs CN).

Steps:
  1. Load timeseries from timeseries/{AD,MCI,CN}/*.npy
     Files are (N_parcels, T); keep only (121, 140) shapes.
  2. Compute FC = Pearson correlation matrix (121x121) per subject,
     vectorise upper triangle (7260 features).
  3. Classify with:
       a) Linear SVM (L2)
       b) Logistic Regression (L2)
     using stratified 5-fold CV (repeated 5x).
  4. Report accuracy, balanced accuracy, ROC-AUC (one-vs-rest for multiclass).
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
from pathlib import Path
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, balanced_accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# ── Paths ─────────────────────────────────────────────────────────────────────
TS_BASE    = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries")
N_PARCELS  = 121
T_STANDARD = 140
TRIU_IDX   = np.triu_indices(N_PARCELS, k=1)   # 7260 pairs

# ── Load data ─────────────────────────────────────────────────────────────────
X_list, y_list, ids = [], [], []

for group, label in [("AD", 1), ("MCI", 2), ("CN", 0)]:
    folder = TS_BASE / group
    if not folder.exists():
        continue
    for fpath in sorted(folder.glob("*.npy")):
        arr = np.load(fpath)                       # (N, T)
        if arr.shape != (N_PARCELS, T_STANDARD):
            print(f"  skip {fpath.name} shape={arr.shape}")
            continue
        fc = np.corrcoef(arr)                      # (121, 121)
        X_list.append(fc[TRIU_IDX])               # 7260 features
        y_list.append(label)
        ids.append((group, fpath.stem))

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

groups_present = {g for g, _ in ids}
counts = {g: int((y == l).sum()) for g, l in [("AD",1),("MCI",2),("CN",0)]}
print(f"\nLoaded {len(X)} subjects: AD={counts['AD']}  MCI={counts['MCI']}  CN={counts['CN']}")
print(f"FC feature size: {X.shape[1]}")

# ── Helper ─────────────────────────────────────────────────────────────────────
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

def run_cv(X_sub, y_sub, label):
    n = len(np.unique(y_sub))
    multi = n > 2
    scoring = {
        "acc":      "accuracy",
        "bal_acc":  "balanced_accuracy",
    }
    if not multi:
        scoring["roc_auc"] = "roc_auc"

    pipe_svm = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(LinearSVC(C=0.01, max_iter=2000), cv=3)
    )
    pipe_lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
    )

    print(f"\n{'-'*55}")
    print(f"  {label}  (n={len(y_sub)}, classes={np.unique(y_sub).tolist()})")
    print(f"{'-'*55}")

    for name, pipe in [("Linear SVM", pipe_svm), ("Logistic Reg", pipe_lr)]:
        res = cross_validate(pipe, X_sub, y_sub, cv=cv, scoring=scoring,
                             n_jobs=-1, return_train_score=False)
        line = (f"  {name:14s}  acc={res['test_acc'].mean():.3f}±{res['test_acc'].std():.3f}"
                f"  bal_acc={res['test_bal_acc'].mean():.3f}±{res['test_bal_acc'].std():.3f}")
        if "roc_auc" in res:
            line += f"  AUC={res['test_roc_auc'].mean():.3f}±{res['test_roc_auc'].std():.3f}"
        print(line)

# ── Task 1: AD vs CN ──────────────────────────────────────────────────────────
mask_adcn = y != 2
if mask_adcn.sum() >= 10:
    run_cv(X[mask_adcn], y[mask_adcn], "AD vs CN")
else:
    print("\nNot enough AD+CN samples for AD vs CN classification.")

# ── Task 2: AD vs MCI vs CN ──────────────────────────────────────────────────
if len(np.unique(y)) == 3 and len(y) >= 15:
    run_cv(X, y, "AD vs MCI vs CN (3-class)")
else:
    print("\nNot enough samples for 3-class classification.")

# ── Task 3: AD+MCI vs CN ─────────────────────────────────────────────────────
y_binary = (y > 0).astype(int)   # AD=1, MCI=1, CN=0
if y_binary.sum() >= 5 and (y_binary == 0).sum() >= 5:
    run_cv(X, y_binary, "AD+MCI vs CN")

print("\nDone.")
