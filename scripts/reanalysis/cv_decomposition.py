"""
cv_decomposition.py
===================
Decompose the FC-lag classifier's optimism into its two sources, on ONE cohort
with ONE code path (K=25 fixed for all arms):

  1. TRANSDUCTIVE k-fold   — embedding built ONCE on ALL subjects, then random
                             stratified CV splits that embedding.  (This is what
                             figure3_classification.py:_eval does -> the ~0.70.)
  2. NON-TRANSDUCTIVE k-fold — random stratified CV, but the embedding is REBUILT
                             per fold from training subjects only and the test
                             subjects are projected in.  Still site-mixed.
Gap 1->2 = optimism from the transductive embedding.

Read-only w.r.t. the manuscript: nothing here touches paper/.
Saves: cv_decomposition_results.npz  (+ caches the feature matrix)
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
import warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
from external_oasis_validate import (
    feat, teacher_force, fit_W, build_reservoir, LDA, balm,
    K_LDA, RNG_SEED, N_PC_MODEL, N_SITES, MIN_VOL, ADNI_TS_ROOT)

FB_CACHE = "fb_features_cache.npz"
N_SPLITS, N_REPEATS = 5, 10

def site_of(pid): return pid.split("sub-")[-1].split("S")[0]

# ── cohort + features (cached) ────────────────────────────────────────────────
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

if os.path.exists(FB_CACHE):
    d = np.load(FB_CACHE, allow_pickle=True)
    fb, y, sites, upid = d["fb"], d["y"], d["sites"], d["upid"]
    print(f"Loaded cached features {fb.shape}")
else:
    sig, y, upid = load_adni_full()
    sites = np.array([site_of(p) for p in upid])
    all_sig = np.concatenate([s.T for s in sig], 0)
    evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
    ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]
    res = build_reservoir(); fb = []
    for i, s in enumerate(tqdm(sig, desc="TF+feat")):
        X, tgt = teacher_force(res, s, ev50)
        W = fit_W(X, tgt, np.random.default_rng(RNG_SEED + 100 + i))
        fb.append(feat(W, X))
    fb = np.array(fb)
    np.savez(FB_CACHE, fb=fb, y=y, sites=sites, upid=upid)
    print(f"Computed + cached features {fb.shape}")

# retained cohort: sites with >=2 subjects
cnt = {s_: int((sites == s_).sum()) for s_ in np.unique(sites)}
keep = np.array([cnt[s_] >= 2 for s_ in sites])
fb, y, sites, upid = fb[keep], y[keep], sites[keep], upid[keep]
print(f"Cohort: {len(y)} subjects ({int((y==0).sum())} CN, {int((y==1).sum())} AD), "
      f"{len(np.unique(sites))} sites")

# ── embedding helpers ─────────────────────────────────────────────────────────
def embed_fit(F):
    """Build the Gram/SVD embedding on F (training rows). Returns basis + G."""
    fm = F.mean(0); fcc = F - fm
    evf, evecf = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(evf)[::-1]
    evf = np.maximum(evf[o], 0); evecf = evecf[:, o]
    return fm, fcc, evf, evecf, evecf * np.sqrt(evf)

def embed_apply(F_new, fm, fcc, evf, evecf):
    return ((F_new - fm) @ fcc.T @ evecf) / (np.sqrt(evf) + 1e-12)

def lda_train_score(Gtr, ytr, Gte):
    """Balanced Fisher LDA on training embedding; return oriented test scores."""
    Xl, yl = balm(Gtr[:, :K_LDA], ytr, RNG_SEED)
    lda = LDA().fit(Xl, yl); Ztr = lda.tr(Gtr[:, :K_LDA])
    if Ztr[ytr == 0].mean() > Ztr[ytr == 1].mean(): lda.w *= -1
    return Gte[:, :K_LDA] @ lda.w

# ── arms 1 & 2: repeated stratified k-fold ────────────────────────────────────
print(f"\nRunning {N_REPEATS}x{N_SPLITS}-fold CV (arms 1 & 2) ...")
_, _, _, _, G_all = embed_fit(fb)          # transductive: built on ALL subjects
rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                               random_state=RNG_SEED)
auc_t, auc_n = [], []
oof_t = np.full(len(y), np.nan); oof_n = np.full(len(y), np.nan)
for i, (tr, te) in enumerate(tqdm(list(rskf.split(fb, y)), desc="  folds")):
    # 1. transductive — reuse the all-subject embedding
    oof_t[te] = lda_train_score(G_all[tr], y[tr], G_all[te])
    # 2. non-transductive — rebuild embedding on train only, project test
    fm, fcc, evf, evecf, Gtr = embed_fit(fb[tr])
    oof_n[te] = lda_train_score(Gtr, y[tr], embed_apply(fb[te], fm, fcc, evf, evecf))
    if (i + 1) % N_SPLITS == 0:            # one full repeat complete
        auc_t.append(roc_auc_score(y, oof_t)); auc_n.append(roc_auc_score(y, oof_n))
        oof_t[:] = np.nan; oof_n[:] = np.nan

# ── report ────────────────────────────────────────────────────────────────────
t_m, t_s = np.mean(auc_t), np.std(auc_t)
n_m, n_s = np.mean(auc_n), np.std(auc_n)
print("\n" + "=" * 66)
print(f"FC-lag AUROC decomposition  (n={len(y)}, K={K_LDA})")
print("=" * 66)
print(f"  1. transductive k-fold      AUROC = {t_m:.3f} +/- {t_s:.3f}   "
      f"(paper's protocol)")
print(f"  2. non-transductive k-fold  AUROC = {n_m:.3f} +/- {n_s:.3f}   "
      f"(site-mixed, honest embedding)")
print("-" * 66)
print(f"  gap 1->2 (transductive embedding optimism) = {t_m - n_m:+.3f}")
print("=" * 66)

np.savez("cv_decomposition_results.npz", auc_transductive=np.array(auc_t),
         auc_nontransductive=np.array(auc_n),
         labels=y, sites=sites, subjects=upid)
print("Saved cv_decomposition_results.npz")
