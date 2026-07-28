"""
loso_combat_validate.py
=======================
Decisive cross-site generalization test for the FC-lag AD classifier.

Two arms, SAME full ADNI cohort and SAME leave-one-site-out (LOSO) protocol:
  A. RAW     — LOSO on the FC-lag features as-is.
  B. ComBat  — LOSO after harmonising the FC-lag features across scanner sites.

ComBat is fit LABEL-FREE (batch = site, no diagnosis covariate). That is the
conservative choice: it cannot inject label information, and if diagnosis is
partly confounded with site it will remove some real signal too. So a recovery
of AUROC under arm B is strong evidence for a genuine, site-invariant disease
signal; no recovery means the ~0.70 k-fold number was largely scanner effects.

Cohort: ALL CN + AD subjects (one session each), unlike the paper's balanced
40/40 — more subjects per site makes both LOSO and ComBat far more stable.
Sites with <2 subjects are dropped (ComBat cannot estimate a site scale from
one sample); both arms use the identical retained cohort so they are comparable.

Saves: loso_combat_scores.npz
"""

# --- path bootstrap (auto-added): import shared modules from scripts/common ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "scripts" / "common").is_dir():
        _sys.path.insert(0, str(_p / "scripts" / "common")); break
# --- end bootstrap ---
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from external_oasis_validate import (
    feat, teacher_force, fit_W, build_reservoir, LDA, balm,
    K_LDA, RNG_SEED, N_PC_MODEL, N_SITES, MIN_VOL, ADNI_TS_ROOT)

def site_of(pid):                       # "sub-002S0295" -> "002"
    return pid.split("sub-")[-1].split("S")[0]

# ── full cohort: ALL CN + AD subjects, first session each ─────────────────────
def load_adni_full():
    signals, labs, pids = [], [], []
    for sub, lb in [("CN", 0), ("AD", 1)]:
        folder = os.path.join(ADNI_TS_ROOT, sub)
        for fn in sorted(f for f in os.listdir(folder) if f.endswith(".npy")):
            a = np.load(os.path.join(folder, fn)).T
            if a.shape[1] == N_SITES and a.shape[0] >= MIN_VOL:
                signals.append(a.T); pids.append(fn.split("_ses-")[0]); labs.append(lb)
    pids = np.array(pids); labs = np.array(labs); upid = np.unique(pids)
    first = {p: np.where(pids == p)[0][0] for p in upid}
    plabel = np.array([labs[first[p]] for p in upid])
    return [signals[first[p]] for p in upid], plabel, upid

sig, plabel, upid = load_adni_full()
sites = np.array([site_of(p) for p in upid])
print(f"Full cohort: {len(upid)} subjects ({int((plabel==0).sum())} CN, "
      f"{int((plabel==1).sum())} AD) across {len(np.unique(sites))} sites")

# ── features once (fixed unsupervised ev50 + fixed reservoir) ─────────────────
all_sig = np.concatenate([s.T for s in sig], 0)
evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]

res = build_reservoir()
fb = []
for i, s in enumerate(tqdm(sig, desc="TF+feat")):
    X, tgt = teacher_force(res, s, ev50)
    W = fit_W(X, tgt, np.random.default_rng(RNG_SEED + 100 + i))
    fb.append(feat(W, X))
fb = np.array(fb)
print(f"feature matrix: {fb.shape}")

# ── keep sites with >=2 subjects (ComBat needs a within-site scale) ───────────
cnt = {s_: int((sites == s_).sum()) for s_ in np.unique(sites)}
keep = np.array([cnt[s_] >= 2 for s_ in sites])
fb, plabel, sites, upid = fb[keep], plabel[keep], sites[keep], upid[keep]
print(f"Retained (sites with >=2 subj): {keep.sum()} subjects, "
      f"{len(np.unique(sites))} sites  "
      f"({int((plabel==0).sum())} CN, {int((plabel==1).sum())} AD)")

# ── LOSO (identical protocol for both arms) ──────────────────────────────────
def loso(F, y, st):
    scores = np.full(len(y), np.nan); preds = np.full(len(y), -1)
    for s_ in np.unique(st):
        te = np.where(st == s_)[0]; tr = np.where(st != s_)[0]
        ytr = y[tr]
        if len(np.unique(ytr)) < 2: continue
        fm = F[tr].mean(0); fcc = F[tr] - fm
        evf, evecf = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(evf)[::-1]
        evf = np.maximum(evf[o], 0); evecf = evecf[:, o]; Gtr = evecf * np.sqrt(evf)
        Xl, yl = balm(Gtr[:, :K_LDA], ytr, RNG_SEED)
        lda = LDA().fit(Xl, yl); Ztr = lda.tr(Gtr[:, :K_LDA])
        if Ztr[ytr == 0].mean() > Ztr[ytr == 1].mean(): lda.w *= -1; Ztr = -Ztr
        thr = 0.5 * (Ztr[ytr == 0].mean() + Ztr[ytr == 1].mean())
        gte = ((F[te] - fm) @ fcc.T @ evecf) / (np.sqrt(evf) + 1e-12)
        ste = gte[:, :K_LDA] @ lda.w
        scores[te] = ste; preds[te] = (ste > thr).astype(int)
    return scores, preds

def report(scores, preds, y, tag):
    m = ~np.isnan(scores); yy = y[m]; ss = scores[m]; pp = preds[m]
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(yy, ss)
    except Exception:
        auc = np.nan
    tpr = pp[yy == 1].mean(); tnr = (1 - pp[yy == 0]).mean()
    print(f"  {tag:26s} n={m.sum():3d}  AUROC={auc:.3f}  bal-acc={0.5*(tpr+tnr):.3f} "
          f"(TPR={tpr:.2f}, TNR={tnr:.2f})")
    return float(auc), float(0.5 * (tpr + tnr))

print("\nRunning LOSO ...")
s_raw, p_raw = loso(fb, plabel, sites)

# ── ComBat harmonisation across sites (label-free) ───────────────────────────
print("ComBat harmonising across sites (label-free) ...")
from neuroCombat import neuroCombat
covars = pd.DataFrame({"site": sites})
try:
    fb_h = neuroCombat(dat=fb.T, covars=covars, batch_col="site")["data"].T
except Exception as e:
    print(f"  neuroCombat with batch only failed ({e}); retrying with dummy covariate")
    covars["dummy"] = np.arange(len(sites)) % 2
    fb_h = neuroCombat(dat=fb.T, covars=covars, batch_col="site")["data"].T
print(f"  harmonised: {fb_h.shape}")
s_cmb, p_cmb = loso(fb_h, plabel, sites)

# ── report ────────────────────────────────────────────────────────────────────
print("\nLEAVE-ONE-SITE-OUT (pooled held-out predictions, full cohort)")
a_raw, b_raw = report(s_raw, p_raw, plabel, "A. raw features")
a_cmb, b_cmb = report(s_cmb, p_cmb, plabel, "B. ComBat-harmonised")
print(f"\n  reference: paper's site-mixed k-fold CV FC-lag AUROC ~0.70")
print(f"  reference: 40/40-cohort raw LOSO (earlier run)        ~0.55")

np.savez("loso_combat_scores.npz", scores_raw=s_raw, preds_raw=p_raw,
         scores_combat=s_cmb, preds_combat=p_cmb, labels=plabel, sites=sites,
         subjects=upid, auc_raw=a_raw, auc_combat=a_cmb,
         bal_raw=b_raw, bal_combat=b_cmb)
print("Saved loso_combat_scores.npz")
