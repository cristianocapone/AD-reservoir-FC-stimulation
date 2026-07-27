"""
struct_gm_classify.py
=====================
AD-vs-control classification from parcel-wise grey-matter volume (same 121-parcel
atlas as the functional analyses), as a structural benchmark for the FC-based
classifiers and as a positive control on the diagnostic labels.

Arms
  1. hippocampus only          — the canonical structural AD marker (2 features)
  2. all 121 parcel volumes    — matched in dimensionality to the FC parcellation
  3. contaminated labelling    — the AD subjects ALSO entered as controls, which is
                                 what the CN/AD timeseries folders currently do;
                                 quantifies the damage that mislabelling causes.

Protocol: repeated (x10) stratified 5-fold CV at subject level, features
z-scored inside each fold, logistic regression with L2. AUROC on pooled
out-of-fold scores plus per-repeat spread.
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from scipy.stats import mannwhitneyu
import warnings; warnings.filterwarnings("ignore")

SEED = 42
L_HIPP, R_HIPP = 108, 118          # HO-9 / HO-19 -> 0-based parcel indices

d = np.load("struct_gm_features.npz", allow_pickle=True)
V, tot, y, sid = d["vol"], d["total_gm"], d["labels"], d["subjects"]
parcels = d["parcels"]

# head-size normalisation: parcel GM as a fraction of the subject's total GM
Vn = V / tot[:, None]

print(f"cohort: {len(y)}  CN {(y==0).sum()}  AD {(y==1).sum()}  MCI {(y==2).sum()}")
print(f"hippocampal parcels: {parcels[L_HIPP]}, {parcels[R_HIPP]}")


def cv_auc(X, yy, tag, n_repeats=10):
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, C=1.0, class_weight="balanced"))
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=n_repeats, random_state=SEED)
    oof = np.full(len(yy), np.nan); aucs = []; bals = []
    for i, (tr, te) in enumerate(rskf.split(X, yy)):
        clf.fit(X[tr], yy[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        if (i + 1) % 5 == 0:
            aucs.append(roc_auc_score(yy, oof))
            bals.append(balanced_accuracy_score(yy, (oof > 0.5).astype(int)))
            oof[:] = np.nan
    print(f"  {tag:44s} AUROC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}   "
          f"bal-acc = {np.mean(bals):.3f}")
    return np.mean(aucs)


# ── arm 0: raw group difference in hippocampal volume ─────────────────────────
m = y != 2
hip = Vn[:, [L_HIPP, R_HIPP]].sum(1)
u, p = mannwhitneyu(hip[y == 0], hip[y == 1])
print(f"\nHippocampal GM fraction: CN {hip[y==0].mean()*100:.3f}% "
      f"vs AD {hip[y==1].mean()*100:.3f}%   Mann-Whitney p = {p:.2g}")
print(f"  absolute volume: CN {V[y==0][:,[L_HIPP,R_HIPP]].sum(1).mean():.0f} mm^3 "
      f"vs AD {V[y==1][:,[L_HIPP,R_HIPP]].sum(1).mean():.0f} mm^3")
print(f"  total GM:        CN {tot[y==0].mean()/1000:.0f} cm^3 "
      f"vs AD {tot[y==1].mean()/1000:.0f} cm^3")

# ── arms 1 & 2: clean labels ──────────────────────────────────────────────────
print("\nCLEAN LABELS (AD_bids subjects vs CN_bids minus AD/MCI)")
cv_auc(Vn[m][:, [L_HIPP, R_HIPP]], y[m], "1. hippocampus only (2 features)")
cv_auc(Vn[m], y[m], "2. all 121 parcel GM volumes")
cv_auc(np.c_[Vn[m], tot[m]], y[m], "2b. 121 parcels + total GM")

# ── arm 3: reproduce the contaminated labelling of the timeseries folders ─────
# every AD subject also appears as a control, exactly as timeseries/CN does
Xc = np.vstack([Vn[m], Vn[y == 1]])
yc = np.concatenate([y[m], np.zeros((y == 1).sum(), dtype=int)])
print(f"\nCONTAMINATED LABELS (each AD subject also entered as a control): "
      f"n={len(yc)}, CN {(yc==0).sum()}, AD {(yc==1).sum()}")
cv_auc(Xc[:, [L_HIPP, R_HIPP]], yc, "3. hippocampus only, contaminated")
cv_auc(Xc, yc, "3b. all 121 parcels, contaminated")

# ── leave-one-site-out: does the structural signal survive scanner change? ────
def loso(X, yy, sids, tag):
    site = np.array([s.replace("sub-", "").split("S")[0] for s in sids])
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=5000, class_weight="balanced"))
    z = np.full(len(yy), np.nan)
    for s_ in np.unique(site):
        te = site == s_; tr = ~te
        if len(np.unique(yy[tr])) < 2: continue
        clf.fit(X[tr], yy[tr])
        ptr = clf.predict_proba(X[tr])[:, 1]
        # standardise on the training distribution so folds are poolable
        z[te] = (clf.predict_proba(X[te])[:, 1] - ptr.mean()) / (ptr.std() + 1e-12)
    k = ~np.isnan(z)
    print(f"  {tag:44s} AUROC = {roc_auc_score(yy[k], z[k]):.3f}   "
          f"({len(np.unique(site))} sites, n={k.sum()})")

print("\nLEAVE-ONE-SITE-OUT (clean labels)")
loso(np.c_[Vn[m], tot[m]], y[m], sid[m], "4. 121 parcels + total GM, LOSO")

# ── MCI as an intermediate check (biological sanity) ──────────────────────────
print("\nMCI positioning (should sit between CN and AD if labels are real)")
for lb, nm in [(0, "CN"), (2, "MCI"), (1, "AD")]:
    print(f"  {nm:4s} n={int((y==lb).sum()):3d}  hippocampal GM fraction "
          f"{hip[y==lb].mean()*100:.3f}%")
