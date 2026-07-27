"""
fc_sc_fusion_empirical.py
=========================
Companion to fc_sc_fusion.py. Same 115-subject clean-label cohort, but the
functional block is the EMPIRICAL tangent-space FC (Ledoit-Wolf covariance ->
tangent space at a train-only reference), i.e. the paper's own strong FC
baseline, rather than the reservoir FC-lag features.

Purpose: separate two explanations for the weak functional result.
  * if empirical FC also fails on clean labels -> the cohort carries little
    functional signal, and the reservoir is not at fault;
  * if empirical FC works where the reservoir features do not -> the reservoir
    read-out is discarding disease-relevant information.
"""
import os
import numpy as np
from nilearn.connectome import ConnectivityMeasure
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings; warnings.filterwarnings("ignore")

SEED, N_PC = 42, 25
MIN_VOL, N_SITES = 139, 121


def clean_labels():
    def subs(d):
        return set(x.replace("_S_", "S") for x in os.listdir(d) if x.startswith("sub-"))
    cn_all, ad, mci = subs("CN_bids"), subs("AD_bids"), subs("MCI_bids")
    return cn_all - ad - mci, ad, mci


# ── locate one timeseries per subject (AD folder wins for AD subjects) ────────
# The CN and AD timeseries folders are two different preprocessing batches of
# overlapping scans. "adfolder" gives AD subjects their AD-batch file (what the
# paper does, but then batch and diagnosis are confounded); "cnfolder" gives
# every subject a CN-batch file, so both groups share one preprocessing batch
# and any remaining separation cannot be a batch signature.
FC_COPY = os.environ.get("FC_COPY", "adfolder")


def index_timeseries():
    cn_files, ad_files = {}, {}
    for folder, dest in [("timeseries/CN", cn_files), ("timeseries/AD", ad_files)]:
        for fn in sorted(os.listdir(folder)):
            if fn.endswith(".npy"):
                s = fn.split("_ses-")[0].replace("_S_", "S")
                dest.setdefault(s, os.path.join(folder, fn))
    idx = dict(cn_files)
    if FC_COPY == "adfolder":
        idx.update(ad_files)                 # AD-batch file wins where it exists
    else:
        for s, p in ad_files.items():        # only for subjects absent from CN
            idx.setdefault(s, p)
    return idx


S = np.load("struct_gm_features.npz", allow_pickle=True)
sc_id = S["subjects"]
sc_X = np.c_[S["vol"] / S["total_gm"][:, None], S["total_gm"]]
si = {s: i for i, s in enumerate(sc_id)}

cn, ad, _ = clean_labels()
ts_idx = index_timeseries()

ids, series = [], []
for s in sorted(sc_id):
    if s not in ts_idx or not (s in cn or s in ad):
        continue
    a = np.load(ts_idx[s])
    a = a if a.shape[0] == N_SITES else a.T          # -> (121, T)
    if a.shape[0] != N_SITES or a.shape[1] < MIN_VOL:
        continue
    ids.append(s); series.append(a.T)                # (T, 121) for nilearn

ids = np.array(ids)
y = np.array([1 if s in ad else 0 for s in ids])
site = np.array([s.replace("sub-", "").split("S")[0] for s in ids])
SCX = np.array([sc_X[si[s]] for s in ids])
n_adbatch = sum("timeseries" + os.sep + "AD" in ts_idx[s] or "timeseries/AD" in ts_idx[s]
                for s in ids[y == 1])
print(f"cohort: {len(y)} ({(y==0).sum()} CN, {(y==1).sum()} AD), "
      f"{len(np.unique(site))} sites")
print(f"  FC copy policy '{FC_COPY}': {n_adbatch}/{(y==1).sum()} AD subjects "
      f"read from the AD preprocessing batch")


def lr():
    return LogisticRegression(max_iter=5000, class_weight="balanced")


def fold_scores(tr, te):
    """Train-only tangent reference, train-only PCA, train-only scalers."""
    out = {}
    ssc = StandardScaler().fit(SCX[tr])
    Xtr, Xte = ssc.transform(SCX[tr]), ssc.transform(SCX[te])
    m_sc = lr().fit(Xtr, y[tr])
    out["SC"] = (m_sc.decision_function(Xtr), m_sc.decision_function(Xte))

    cm = ConnectivityMeasure(cov_estimator=LedoitWolf(), kind="tangent",
                             vectorize=True)
    Ftr = cm.fit_transform([series[i] for i in tr])
    Fte = cm.transform([series[i] for i in te])
    pca = PCA(n_components=min(N_PC, len(tr) - 1), random_state=SEED).fit(Ftr)
    sfc = StandardScaler().fit(pca.transform(Ftr))
    Gtr, Gte = sfc.transform(pca.transform(Ftr)), sfc.transform(pca.transform(Fte))
    m_fc = lr().fit(Gtr, y[tr])
    out["FC-emp"] = (m_fc.decision_function(Gtr), m_fc.decision_function(Gte))

    m_e = lr().fit(np.c_[Xtr, Gtr], y[tr])
    out["early"] = (m_e.decision_function(np.c_[Xtr, Gtr]),
                    m_e.decision_function(np.c_[Xte, Gte]))

    def zpair(a, b):
        mu, sd = a.mean(), a.std() + 1e-12
        return (a - mu) / sd, (b - mu) / sd
    ztr_s, zte_s = zpair(*out["SC"]); ztr_f, zte_f = zpair(*out["FC-emp"])
    out["late"] = (0.5 * (ztr_s + ztr_f), 0.5 * (zte_s + zte_f))
    return out


ARMS = ["SC", "FC-emp", "early", "late"]

print("\nREPEATED STRATIFIED 5-FOLD (x5)")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
oof = {a: np.full(len(y), np.nan) for a in ARMS}
acc = {a: [] for a in ARMS}; bal = {a: [] for a in ARMS}
for i, (tr, te) in enumerate(rskf.split(SCX, y)):
    fs = fold_scores(tr, te)
    for a in ARMS:
        oof[a][te] = fs[a][1]
    if (i + 1) % 5 == 0:
        for a in ARMS:
            acc[a].append(roc_auc_score(y, oof[a]))
            bal[a].append(balanced_accuracy_score(
                y, (oof[a] > np.median(oof[a])).astype(int)))
            oof[a][:] = np.nan
for a in ARMS:
    print(f"  {a:8s} AUROC = {np.mean(acc[a]):.3f} +/- {np.std(acc[a]):.3f}   "
          f"bal-acc = {np.mean(bal[a]):.3f}")

print("\nLEAVE-ONE-SITE-OUT")
z = {a: np.full(len(y), np.nan) for a in ARMS}
for s_ in np.unique(site):
    te = np.where(site == s_)[0]; tr = np.where(site != s_)[0]
    if len(np.unique(y[tr])) < 2 or len(tr) < 10:
        continue
    fs = fold_scores(tr, te)
    for a in ARMS:
        a_tr, a_te = fs[a]
        z[a][te] = (a_te - a_tr.mean()) / (a_tr.std() + 1e-12)
for a in ARMS:
    k = ~np.isnan(z[a])
    print(f"  {a:8s} AUROC = {roc_auc_score(y[k], z[a][k]):.3f}   (n={k.sum()})")

np.savez("fc_sc_fusion_empirical_results.npz", subjects=ids, labels=y, sites=site,
         **{f"kfold_{a}": np.array(acc[a]) for a in ARMS},
         **{f"loso_{a}": z[a] for a in ARMS})
print("\nSaved fc_sc_fusion_empirical_results.npz")
