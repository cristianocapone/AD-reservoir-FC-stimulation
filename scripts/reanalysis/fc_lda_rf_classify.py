"""
fc_lda_rf_classify.py
=====================
AD-vs-control classification from FUNCTIONAL features only (no structural), on
the clean cohort (cohort_clean.npz: authoritative labels, one preprocessing
batch per subject, one session per subject).

Two feature families:
  * empirical FC : Ledoit-Wolf covariance -> tangent space at a train-only
                   reference mean -> vectorised (the paper's strong FC baseline)
  * FC-lag       : reservoir read-out features (the paper's own representation),
                   cached in clean_cohort_features.npz

Each classified by LDA and Random Forest. Everything that sees the data - the
tangent reference, the PCA basis, the scaler, the classifier - is fit on the
training fold only; test subjects are projected in.

Protocol: repeated (x10) stratified 5-fold.
Metric: AUROC on pooled out-of-fold scores, plus balanced accuracy.
"""
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.covariance import LedoitWolf
from nilearn.connectome import ConnectivityMeasure
import warnings; warnings.filterwarnings("ignore")

SEED, N_PC, K_FC, N_SITES = 42, 25, 25, 121

# ── clean cohort, one session per subject (earliest) ──────────────────────────
C = np.load("cohort_clean.npz", allow_pickle=True)
aid, lab, site, path, sess = (C["adni_id"], C["label"], C["site"],
                              C["path"], C["session"])
first = {}
for i in np.argsort(sess):
    first.setdefault(aid[i], i)
idx = np.array([first[a] for a in sorted(first)])
sub, y, st, pt = aid[idx], lab[idx], site[idx], path[idx]

D = np.load("clean_cohort_features.npz", allow_pickle=True)
assert list(D["sub"]) == list(sub), "reservoir feature cache is stale"
fb = D["fb"]                                              # FC-lag features

series = [(a if a.shape[0] == N_SITES else a.T) for a in (np.load(p) for p in pt)]
print(f"clean cohort: {len(y)} subjects ({(y==0).sum()} CN, {(y==1).sum()} AD), "
      f"{len(np.unique(st))} sites   (functional features only)")


def clf(name):
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def embed(kind, tr, te):
    """Leakage-free feature reduction, fit on train rows only."""
    if kind == "FC-lag":
        fm = fb[tr].mean(0); fcc = fb[tr] - fm
        ev, evec = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(ev)[::-1]
        ev = np.maximum(ev[o], 0); evec = evec[:, o]
        Gtr = (evec * np.sqrt(ev))[:, :K_FC]
        Gte = (((fb[te] - fm) @ fcc.T @ evec) / (np.sqrt(ev) + 1e-12))[:, :K_FC]
    else:                                                # empirical tangent FC
        cm = ConnectivityMeasure(cov_estimator=LedoitWolf(), kind="tangent",
                                 vectorize=True)
        Ftr = cm.fit_transform([series[i].T for i in tr])
        Fte = cm.transform([series[i].T for i in te])
        pca = PCA(n_components=min(N_PC, len(tr) - 1), random_state=SEED).fit(Ftr)
        Gtr, Gte = pca.transform(Ftr), pca.transform(Fte)
    sc = StandardScaler().fit(Gtr)
    return sc.transform(Gtr), sc.transform(Gte)


def proba(model, Xtr, Xte, ytr):
    m = clf(model).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def evaluate(kind, model):
    # site-mixed repeated CV
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    oof = np.full(len(y), np.nan); aucs = []; bals = []
    for j, (tr, te) in enumerate(rskf.split(fb, y)):
        Xtr, Xte = embed(kind, tr, te)
        oof[te] = proba(model, Xtr, Xte, y[tr])
        if (j + 1) % 5 == 0:
            aucs.append(roc_auc_score(y, oof))
            bals.append(balanced_accuracy_score(y, (oof > 0.5).astype(int)))
            oof[:] = np.nan
    return np.mean(aucs), np.std(aucs), np.mean(bals)


print(f"\n{'feature':10s} {'classifier':13s} {'CV AUROC':16s} {'bal-acc':9s}")
print("-" * 62)
res = {}
for kind in ["FC-lag", "empirical FC"]:
    for model in ["LDA", "RandomForest"]:
        a, s, b = evaluate(kind, model)
        res[(kind, model)] = (a, s, b)
        print(f"{kind:10s} {model:13s} {a:.3f} +/- {s:.3f}    {b:.3f}")

np.savez("fc_lda_rf_results.npz",
         **{f"{k}_{m}".replace(" ", ""): np.array(v) for (k, m), v in res.items()})
print("\nSaved fc_lda_rf_results.npz")
