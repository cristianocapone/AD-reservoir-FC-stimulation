"""
loso_site_validate.py
=====================
Leave-one-SITE-out (LOSO) generalization of the FC-lag AD classifier on ADNI.

ADNI is multi-site; the scanner site is encoded in the subject id
(sub-<SITE>S<id>, e.g. sub-002S0295 -> site 002). LOSO holds out ALL subjects
from one site, rebuilds the FC-lag embedding + LDA on the remaining sites, and
projects the held-out site's subjects through that frozen (per-fold) embedding.
Every subject is therefore scored exactly once, by a model that never saw any
subject from its scanner. This is stricter than the paper's k-fold CV (which can
place same-site subjects in both train and test) and is the closest in-hand
proxy for external validation while OASIS-3 is being obtained.

Efficiency: ev50 (unsupervised population PCA) and the reservoir are fixed, so
teacher-forcing + W-fit + FC-lag features are computed ONCE; each fold is then
cheap linear algebra on the cached feature matrix. Reuses the validated machinery
from external_oasis_validate.py.

Saves: loso_site_scores.npz   (per-subject held-out score, label, site)
"""
import numpy as np
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from external_oasis_validate import (
    load_adni, feat, teacher_force, fit_W, build_reservoir, LDA, balm,
    K_LDA, RNG_SEED, N_PC_MODEL)

def site_of(pid):                       # "sub-002S0295" -> "002"
    return pid.split("sub-")[-1].split("S")[0]

# ── cohort (40 CC + 40 AD, one session each) — same as the paper's classifier ──
signals, first, upid, plabel = load_adni()
sig   = [signals[first[p]] for p in upid]
sites = np.array([site_of(p) for p in upid])
uniq_sites = np.unique(sites)
print(f"Cohort: {len(upid)} subjects ({int((plabel==0).sum())} CN, "
      f"{int((plabel==1).sum())} AD) across {len(uniq_sites)} sites")

# ── fixed ev50 (unsupervised) + reservoir; TF + W-fit + features ONCE ──────────
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

# ── leave-one-site-out ────────────────────────────────────────────────────────
scores = np.full(len(upid), np.nan)     # held-out transfer score per subject
preds  = np.full(len(upid), -1)         # held-out prediction (fold's own thr)
skipped = []
for st in uniq_sites:
    te = np.where(sites == st)[0]; tr = np.where(sites != st)[0]
    ytr = plabel[tr]
    if len(np.unique(ytr)) < 2:         # need both classes to train
        skipped.append(st); continue
    fm  = fb[tr].mean(0); fcc = fb[tr] - fm
    evf, evecf = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(evf)[::-1]
    evf = np.maximum(evf[o], 0); evecf = evecf[:, o]; Gtr = evecf * np.sqrt(evf)
    Xl, yl = balm(Gtr[:, :K_LDA], ytr, RNG_SEED)
    lda = LDA().fit(Xl, yl); Ztr = lda.tr(Gtr[:, :K_LDA])
    if Ztr[ytr == 0].mean() > Ztr[ytr == 1].mean(): lda.w *= -1; Ztr = -Ztr
    thr = 0.5 * (Ztr[ytr == 0].mean() + Ztr[ytr == 1].mean())   # train-only thr
    gte = ((fb[te] - fm) @ fcc.T @ evecf) / (np.sqrt(evf) + 1e-12)
    ste = gte[:, :K_LDA] @ lda.w
    scores[te] = ste; preds[te] = (ste > thr).astype(int)

# ── pooled report ─────────────────────────────────────────────────────────────
m = ~np.isnan(scores)
y = plabel[m]; s = scores[m]; p = preds[m]
try:
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, s)
except Exception:
    auc = np.nan
tpr = p[y == 1].mean(); tnr = (1 - p[y == 0]).mean(); bal = 0.5 * (tpr + tnr)
print(f"\nLEAVE-ONE-SITE-OUT (pooled held-out predictions)")
print(f"  scored {m.sum()}/{len(upid)} subjects "
      f"({len(skipped)} single-class sites skipped: {skipped})")
print(f"  AUROC = {auc:.3f}   balanced-acc = {bal:.3f}   "
      f"(TPR={tpr:.2f}, TNR={tnr:.2f})")
print(f"  reference: paper's repeated k-fold CV FC-lag AUROC ~0.70")

np.savez("loso_site_scores.npz", scores=scores, preds=preds, labels=plabel,
         sites=sites, subjects=upid, auc=float(auc), bal=float(bal))
print("Saved loso_site_scores.npz")
