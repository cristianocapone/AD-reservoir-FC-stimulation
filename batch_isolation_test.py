"""
batch_isolation_test.py
=======================
Isolate the preprocessing-batch effect from every other factor.

The 17 AD subjects that exist in BOTH preprocessing batches are scored twice
against the SAME control group, with the SAME labels, the SAME subjects and the
SAME sample size. The only thing that changes is which batch the patients' own
timeseries were read from:

  arm AD-batch : patients from timeseries/AD, controls from timeseries/CN
                 -> batch is perfectly confounded with diagnosis (paper's setup)
  arm CN-batch : patients and controls both from timeseries/CN
                 -> one shared batch, so batch cannot carry label information

Any difference between the arms is attributable to preprocessing batch alone.
"""
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.covariance import LedoitWolf
from nilearn.connectome import ConnectivityMeasure
import warnings; warnings.filterwarnings("ignore")
from external_oasis_validate import (
    build_reservoir, teacher_force, fit_W, feat, RNG_SEED, N_PC_MODEL, N_SITES)

SEED, K_FC, N_PC = 42, 25, 25

C = np.load("cohort_clean_allbatches.npz", allow_pickle=True)
aid, lab, site, batch, path, sess = (C["adni_id"], C["label"], C["site"],
                                     C["batch"], C["path"], C["session"])

# every retained session, indexed by (subject, batch) -> earliest session path
bykey = {}
for i in np.argsort(sess):
    bykey.setdefault((aid[i], batch[i]), path[i])

subs = sorted(set(aid))
cn = [s for s in subs if lab[aid == s][0] == 0 and (s, "CN") in bykey]
ad_both = [s for s in subs if lab[aid == s][0] == 1
           and (s, "CN") in bykey and (s, "AD") in bykey]
print(f"controls (CN batch): {len(cn)}")
print(f"AD patients present in BOTH batches: {len(ad_both)}")

# the AD-batch arm may also use patients that only exist in the AD batch, but to
# keep n identical across arms we restrict to the paired subjects.
site_of = {s: site[aid == s][0] for s in subs}


def load(p):
    a = np.load(p)
    return a if a.shape[0] == N_SITES else a.T


def build(arm):
    paths = [bykey[(s, "CN")] for s in cn] + \
            [bykey[(s, "AD" if arm == "AD-batch" else "CN")] for s in ad_both]
    y = np.r_[np.zeros(len(cn), int), np.ones(len(ad_both), int)]
    st = np.array([site_of[s] for s in cn + ad_both])
    return [load(p) for p in paths], y, st


def lr():
    return LogisticRegression(max_iter=5000, class_weight="balanced")


def reservoir_features(series):
    allsig = np.concatenate([s.T for s in series], 0)
    evv, evec = np.linalg.eigh(np.cov((allsig - allsig.mean(0)).T))
    ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]
    res = build_reservoir(); rw = np.random.default_rng(RNG_SEED + 1)
    out = []
    for s in tqdm(series, desc="   teacher-force", leave=False):
        X, tgt = teacher_force(res, s, ev50)
        out.append(feat(fit_W(X, tgt, rw), X))
    return np.array(out)


def fc_embed(tr_X, te_X, k=K_FC):
    fm = tr_X.mean(0); fcc = tr_X - fm
    ev, evec = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(ev)[::-1]
    ev = np.maximum(ev[o], 0); evec = evec[:, o]
    return ((evec * np.sqrt(ev))[:, :k],
            (((te_X - fm) @ fcc.T @ evec) / (np.sqrt(ev) + 1e-12))[:, :k])


def run(arm):
    print(f"\n--- {arm} ---")
    series, y, st = build(arm)
    fb = reservoir_features(series)
    res = {}
    for sel in ["reservoir", "empirical"]:
        def sc(tr, te):
            if sel == "reservoir":
                Gtr, Gte = fc_embed(fb[tr], fb[te])
            else:
                cm = ConnectivityMeasure(cov_estimator=LedoitWolf(),
                                         kind="tangent", vectorize=True)
                Ftr = cm.fit_transform([series[i].T for i in tr])
                Fte = cm.transform([series[i].T for i in te])
                pca = PCA(n_components=min(N_PC, len(tr) - 1),
                          random_state=SEED).fit(Ftr)
                Gtr, Gte = pca.transform(Ftr), pca.transform(Fte)
            s = StandardScaler().fit(Gtr)
            Gtr, Gte = s.transform(Gtr), s.transform(Gte)
            m = lr().fit(Gtr, y[tr])
            return m.decision_function(Gtr), m.decision_function(Gte)

        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        oof = np.full(len(y), np.nan); aucs = []
        for j, (tr, te) in enumerate(rskf.split(fb, y)):
            oof[te] = sc(tr, te)[1]
            if (j + 1) % 5 == 0:
                aucs.append(roc_auc_score(y, oof)); oof[:] = np.nan
        z = np.full(len(y), np.nan)
        for s_ in np.unique(st):
            te = np.where(st == s_)[0]; tr = np.where(st != s_)[0]
            if len(np.unique(y[tr])) < 2 or len(tr) < 10: continue
            a, b = sc(tr, te)
            z[te] = (b - a.mean()) / (a.std() + 1e-12)
        k = ~np.isnan(z)
        lo = roc_auc_score(y[k], z[k]) if len(np.unique(y[k])) == 2 else np.nan
        res[sel] = (np.mean(aucs), np.std(aucs), lo)
        print(f"   {sel:10s} CV AUROC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}"
              f"     LOSO AUROC = {lo:.3f}")
    return res


r_ad = run("AD-batch")
r_cn = run("CN-batch")

print("\n" + "=" * 68)
print(f"Identical subjects ({len(cn)} CN, {len(ad_both)} AD) and labels in both arms;")
print("only the patients' preprocessing batch differs.")
print("=" * 68)
for sel in ["reservoir", "empirical"]:
    print(f"  {sel:10s} CV  {r_ad[sel][0]:.3f} (batch confounded) -> "
          f"{r_cn[sel][0]:.3f} (shared batch)   delta = {r_cn[sel][0]-r_ad[sel][0]:+.3f}")
    print(f"  {'':10s} LOSO {r_ad[sel][2]:.3f} (batch confounded) -> "
          f"{r_cn[sel][2]:.3f} (shared batch)   delta = {r_cn[sel][2]-r_ad[sel][2]:+.3f}")

np.savez("batch_isolation_results.npz",
         **{f"{a}_{s}": np.array(v[s]) for a, v in
            [("adbatch", r_ad), ("cnbatch", r_cn)] for s in v})
print("\nSaved batch_isolation_results.npz")
