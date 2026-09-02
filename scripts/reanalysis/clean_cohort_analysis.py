"""
clean_cohort_analysis.py
========================
Rerun the paper's core quantitative claims on the rebuilt cohort
(cohort_clean.npz: authoritative CN/AD labels, one preprocessing batch per
subject), so that the numbers can be compared with the published ones.

Three questions, in the order they matter:
  A. classification  - what AUROC does the reservoir FC-lag read-out reach when
     labels are correct? Reported next to empirical tangent-space FC on the same
     subjects, under site-mixed CV.
  B. batch check     - does the residual preprocessing-batch imbalance among AD
     subjects drive the result? Repeated on the batch-uniform subset.
  C. pathology map   - is the per-site read-out deviation dW disease-specific,
     i.e. larger in AD than the control-vs-control null?

Nothing here writes to paper/.
"""

# --- path bootstrap (auto-added): import shared modules from scripts/common ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "scripts" / "common").is_dir():
        _sys.path.insert(0, str(_p / "scripts" / "common")); break
# --- end bootstrap ---
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr, mannwhitneyu
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
CACHE = "clean_cohort_features.npz"

C = np.load("cohort_clean.npz", allow_pickle=True)
aid, lab, site, batch, path = (C["adni_id"], C["label"], C["site"],
                               C["batch"], C["path"])

# one session per subject (earliest), to match the paper's convention
order = np.argsort(C["session"])
first = {}
for i in order:
    first.setdefault(aid[i], i)
idx = np.array([first[a] for a in sorted(first)])
sub = aid[idx]; y = lab[idx]; st = site[idx]; bt = batch[idx]; pt = path[idx]
print(f"clean cohort: {len(y)} subjects ({(y==0).sum()} CN, {(y==1).sum()} AD), "
      f"{len(np.unique(st))} sites")
print(f"  preprocessing batch: CN {(bt[y==0]=='CN').mean()*100:.0f}% CN-batch, "
      f"AD {(bt[y==1]=='CN').mean()*100:.0f}% CN-batch")

series = []
for p in pt:
    a = np.load(p)
    series.append((a if a.shape[0] == N_SITES else a.T))          # (121, T)

# ── reservoir features + read-outs (cached) ───────────────────────────────────
if os.path.exists(CACHE):
    d = np.load(CACHE, allow_pickle=True)
    fb, W_all = d["fb"], d["W_all"]
    assert list(d["sub"]) == list(sub), "cache is stale - delete it and rerun"
    print(f"loaded cached reservoir features {fb.shape}")
else:
    allsig = np.concatenate([s.T for s in series], 0)
    evv, evec = np.linalg.eigh(np.cov((allsig - allsig.mean(0)).T))
    ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]
    res = build_reservoir(); rw = np.random.default_rng(RNG_SEED + 1)
    fb, W_all = [], []
    for s in tqdm(series, desc="teacher-force + read-out"):
        X, tgt = teacher_force(res, s, ev50)
        W = fit_W(X, tgt, rw)
        W_all.append(W); fb.append(feat(W, X))
    fb = np.array(fb); W_all = np.array(W_all)
    np.savez(CACHE, fb=fb, W_all=W_all, sub=sub)
    print(f"computed + cached reservoir features {fb.shape}")


def lr():
    return LogisticRegression(max_iter=5000, class_weight="balanced")


def fc_embed(tr_X, te_X, k=K_FC):
    fm = tr_X.mean(0); fcc = tr_X - fm
    ev, evec = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(ev)[::-1]
    ev = np.maximum(ev[o], 0); evec = evec[:, o]
    return ((evec * np.sqrt(ev))[:, :k],
            (((te_X - fm) @ fcc.T @ evec) / (np.sqrt(ev) + 1e-12))[:, :k])


def scores(tr, te, sel):
    """Return (train, test) decision scores for one feature family."""
    if sel == "reservoir":
        Gtr, Gte = fc_embed(fb[tr], fb[te])
    else:
        cm = ConnectivityMeasure(cov_estimator=LedoitWolf(), kind="tangent",
                                 vectorize=True)
        Ftr = cm.fit_transform([series[i].T for i in tr])
        Fte = cm.transform([series[i].T for i in te])
        pca = PCA(n_components=min(N_PC, len(tr) - 1), random_state=SEED).fit(Ftr)
        Gtr, Gte = pca.transform(Ftr), pca.transform(Fte)
    sc = StandardScaler().fit(Gtr)
    Gtr, Gte = sc.transform(Gtr), sc.transform(Gte)
    m = lr().fit(Gtr, y[tr])
    return m.decision_function(Gtr), m.decision_function(Gte)


def evaluate(mask, tag):
    ii = np.where(mask)[0]
    if len(np.unique(y[ii])) < 2 or (y[ii] == 1).sum() < 5:
        print(f"  {tag}: too few patients, skipped"); return
    print(f"\n  -- {tag}: n={len(ii)} ({(y[ii]==0).sum()} CN, {(y[ii]==1).sum()} AD)")
    for sel in ["reservoir", "empirical"]:
        # site-mixed repeated CV
        rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
        oof = np.full(len(ii), np.nan); aucs = []
        for j, (tr, te) in enumerate(rskf.split(ii, y[ii])):
            oof[te] = scores(ii[tr], ii[te], sel)[1]
            if (j + 1) % 5 == 0:
                aucs.append(roc_auc_score(y[ii], oof)); oof[:] = np.nan
        print(f"     {sel:10s} CV AUROC = {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")


print("\n=== A. CLASSIFICATION ON CLEAN LABELS ===")
evaluate(np.ones(len(y), bool), "all clean-label subjects")
print("\n=== B. BATCH CHECK (subjects from the CN preprocessing batch only) ===")
evaluate(bt == "CN", "batch-uniform subset")

# ── C. is the pathology map disease-specific? ─────────────────────────────────
print("\n=== C. PATHOLOGY MAP SPECIFICITY ===")
cc_i = np.where(y == 0)[0]; ad_i = np.where(y == 1)[0]
Wcc = W_all[cc_i].mean(0)
DW_AD = np.array([np.linalg.norm(Wcc - W_all[i], axis=0) for i in ad_i])
DW_CC = np.array([np.linalg.norm(W_all[[j for j in cc_i if j != i]].mean(0)
                                 - W_all[i], axis=0) for i in cc_i])
r, p = spearmanr(DW_AD.mean(0), DW_CC.mean(0))
print(f"  AD map vs control-null map: rho = {r:+.3f} (p={p:.3g}); "
      f"amplitude ratio = {DW_AD.mean()/DW_CC.mean():.3f}")
pv = np.array([mannwhitneyu(DW_AD[:, i], DW_CC[:, i])[1] for i in range(N_SITES)])
q = pv * N_SITES / (np.argsort(np.argsort(pv)) + 1)
eff = (DW_AD.mean(0) - DW_CC.mean(0)) / (DW_CC.std(0) + 1e-12)
print(f"  sites raw p<0.05: {(pv<0.05).sum()}/{N_SITES};  "
      f"BH-FDR q<0.05: {(q<0.05).sum()}/{N_SITES}")
print(f"  effect sizes: {eff.min():+.2f} to {eff.max():+.2f} SD "
      f"({(eff>0).sum()} sites positive)")
n = np.load("node_strength_cache.npz")
s_cc = np.abs(n["cc_fc"] * (1 - np.eye(N_SITES))).sum(1)
print(f"  raw dW map vs FC node strength: rho = {spearmanr(DW_AD.mean(0), s_cc)[0]:+.3f}")
print(f"  null-corrected map vs node strength: rho = {spearmanr(eff, s_cc)[0]:+.3f}")

np.savez("clean_cohort_results.npz", sub=sub, y=y, site=st, batch=bt,
         DW_AD=DW_AD, DW_CC=DW_CC, eff=eff, pv=pv)
print("\nSaved clean_cohort_results.npz")
