"""
fc_sc_fusion.py
===============
Multimodal AD classification: reservoir FC-lag features (the paper's functional
read-out) combined with parcel-wise grey-matter volume (struct_gm_features.py),
on the subjects that have BOTH modalities and CLEAN diagnostic labels
(controls = CN_bids minus every AD_bids and MCI_bids subject).

Note on nomenclature: no diffusion data exist in this project, so the structural
modality here is morphometry (GM volume in the same 121 parcels), not
tractography-derived structural connectivity.

Every step that sees the data - the FC Gram/SVD embedding, the scalers, the
classifiers - is fit inside the training fold only; test subjects are projected
in. Three fusion levels are compared against each single modality on the
identical subject set:

  early   : concatenate z-scored [SC | FC-embedding] -> one classifier
  late    : average the two single-modality decision scores (equal weight,
            each standardised on its own training distribution)
  stacked : inner-CV logistic regression on the two training scores

Protocols: repeated (x10) stratified 5-fold, and leave-one-site-out.
"""
import os
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import warnings; warnings.filterwarnings("ignore")

SEED = 42
K_FC = 25                      # paper's FC-lag operating point


# ── cohort: clean labels, both modalities ─────────────────────────────────────
def clean_labels():
    def subs(d):
        return set(x.replace("_S_", "S") for x in os.listdir(d) if x.startswith("sub-"))
    cn_all, ad, mci = subs("CN_bids"), subs("AD_bids"), subs("MCI_bids")
    return cn_all - ad - mci, ad, mci


S = np.load("struct_gm_features.npz", allow_pickle=True)
F = np.load("loso_fb_cache.npz", allow_pickle=True)

sc_id = S["subjects"]
sc_X = np.c_[S["vol"] / S["total_gm"][:, None], S["total_gm"]]
fc_id = np.array([u.replace("_S_", "S") for u in F["upid"]])
fc_raw = np.array([u for u in F["upid"]])
fc_X = F["fb"]

# Which preprocessing copy to use for the 17 AD subjects that exist in BOTH
# folders? "adfolder" takes the AD-folder file (what the paper uses), "cnfolder"
# takes the CN-folder file. The two are the same scan run through different
# preprocessing batches, so comparing the arms tests whether the FC signal is a
# batch signature rather than disease.
FC_COPY = os.environ.get("FC_COPY", "adfolder")

cn, ad, _ = clean_labels()
common = [s for s in sc_id if s in set(fc_id) and (s in cn or s in ad)]
si = {s: i for i, s in enumerate(sc_id)}
fi = {}
for i, s in enumerate(fc_id):
    is_adfolder = "_S_" in fc_raw[i]
    want = is_adfolder if FC_COPY == "adfolder" else (not is_adfolder)
    if s not in fi or want:            # prefer the requested preprocessing copy
        if s in fi and not want:
            continue
        fi[s] = i

ids = np.array(sorted(common))
SCX = np.array([sc_X[si[s]] for s in ids])
FCX = np.array([fc_X[fi[s]] for s in ids])
y = np.array([1 if s in ad else 0 for s in ids])
site = np.array([s.replace("sub-", "").split("S")[0] for s in ids])

print(f"cohort with both modalities and clean labels: {len(y)} "
      f"({(y==0).sum()} CN, {(y==1).sum()} AD), {len(np.unique(site))} sites")
print(f"  SC block {SCX.shape}, FC block {FCX.shape}")
n_ad_from_adfolder = sum("_S_" in fc_raw[fi[s]] for s in ids[y == 1])
print(f"  FC copy policy '{FC_COPY}': {n_ad_from_adfolder}/{(y==1).sum()} AD "
      f"subjects taken from the AD folder")


# ── leakage-free FC embedding (fit on train rows only) ────────────────────────
def fc_embed(tr_X, te_X, k=K_FC):
    fm = tr_X.mean(0); fcc = tr_X - fm
    ev, evec = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(ev)[::-1]
    ev = np.maximum(ev[o], 0); evec = evec[:, o]
    Gtr = (evec * np.sqrt(ev))[:, :k]
    Gte = (((te_X - fm) @ fcc.T @ evec) / (np.sqrt(ev) + 1e-12))[:, :k]
    return Gtr, Gte


def lr():
    return LogisticRegression(max_iter=5000, class_weight="balanced")


def fold_scores(tr, te):
    """Return out-of-fold decision scores for every arm, all fit on `tr` only."""
    out = {}
    # --- single modality: SC
    ssc = StandardScaler().fit(SCX[tr])
    Xtr, Xte = ssc.transform(SCX[tr]), ssc.transform(SCX[te])
    m_sc = lr().fit(Xtr, y[tr])
    out["SC"] = (m_sc.decision_function(Xtr), m_sc.decision_function(Xte))
    # --- single modality: FC
    Gtr, Gte = fc_embed(FCX[tr], FCX[te])
    sfc = StandardScaler().fit(Gtr)
    Gtr_s, Gte_s = sfc.transform(Gtr), sfc.transform(Gte)
    m_fc = lr().fit(Gtr_s, y[tr])
    out["FC"] = (m_fc.decision_function(Gtr_s), m_fc.decision_function(Gte_s))
    # --- early fusion: concatenate the two standardised blocks
    Etr, Ete = np.c_[Xtr, Gtr_s], np.c_[Xte, Gte_s]
    m_e = lr().fit(Etr, y[tr])
    out["early"] = (m_e.decision_function(Etr), m_e.decision_function(Ete))
    # --- late fusion: equal-weight mean of z-scored single-modality scores
    def zpair(a, b):
        mu, sd = a.mean(), a.std() + 1e-12
        return (a - mu) / sd, (b - mu) / sd
    ztr_sc, zte_sc = zpair(*out["SC"])
    ztr_fc, zte_fc = zpair(*out["FC"])
    out["late"] = (0.5 * (ztr_sc + ztr_fc), 0.5 * (zte_sc + zte_fc))
    # --- stacked: LR on the two training scores (inner 5-fold to build them)
    inner = StratifiedKFold(5, shuffle=True, random_state=SEED)
    Ztr = np.zeros((len(tr), 2))
    for itr, ite in inner.split(SCX[tr], y[tr]):
        a, b = tr[itr], tr[ite]
        s2 = StandardScaler().fit(SCX[a])
        Ztr[ite, 0] = lr().fit(s2.transform(SCX[a]), y[a]).decision_function(
            s2.transform(SCX[b]))
        g1, g2 = fc_embed(FCX[a], FCX[b])
        s3 = StandardScaler().fit(g1)
        Ztr[ite, 1] = lr().fit(s3.transform(g1), y[a]).decision_function(s3.transform(g2))
    meta = lr().fit(Ztr, y[tr])
    Zte = np.c_[out["SC"][1], out["FC"][1]]
    out["stacked"] = (meta.decision_function(Ztr), meta.decision_function(Zte))
    return out


ARMS = ["SC", "FC", "early", "late", "stacked"]

# ── protocol 1: repeated stratified 5-fold ────────────────────────────────────
print("\nREPEATED STRATIFIED 5-FOLD (x10)")
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
oof = {a: np.full(len(y), np.nan) for a in ARMS}
acc = {a: [] for a in ARMS}; bal = {a: [] for a in ARMS}
for i, (tr, te) in enumerate(rskf.split(SCX, y)):
    fs = fold_scores(tr, te)
    for a in ARMS:
        oof[a][te] = fs[a][1]
    if (i + 1) % 5 == 0:
        for a in ARMS:
            acc[a].append(roc_auc_score(y, oof[a]))
            bal[a].append(balanced_accuracy_score(y, (oof[a] > np.median(oof[a])).astype(int)))
            oof[a][:] = np.nan
for a in ARMS:
    print(f"  {a:10s} AUROC = {np.mean(acc[a]):.3f} +/- {np.std(acc[a]):.3f}   "
          f"bal-acc = {np.mean(bal[a]):.3f}")

# ── protocol 2: leave-one-site-out ────────────────────────────────────────────
print("\nLEAVE-ONE-SITE-OUT")
z = {a: np.full(len(y), np.nan) for a in ARMS}
for s_ in np.unique(site):
    te = np.where(site == s_)[0]; tr = np.where(site != s_)[0]
    if len(np.unique(y[tr])) < 2: continue
    fs = fold_scores(tr, te)
    for a in ARMS:                       # standardise on the training distribution
        str_, ste_ = fs[a]
        z[a][te] = (ste_ - str_.mean()) / (str_.std() + 1e-12)
for a in ARMS:
    k = ~np.isnan(z[a])
    print(f"  {a:10s} AUROC = {roc_auc_score(y[k], z[a][k]):.3f}   (n={k.sum()})")

np.savez("fc_sc_fusion_results.npz", subjects=ids, labels=y, sites=site,
         **{f"kfold_{a}": np.array(acc[a]) for a in ARMS},
         **{f"loso_{a}": z[a] for a in ARMS})
print("\nSaved fc_sc_fusion_results.npz")
