"""
dw_null_check.py
================
Specificity check on the paper's pathology map.

site_dw_norm (pert_dw_site_gridsearch.py:120) is the mean over AD patients of the
per-site column norm of dW_p = Wbar_CC - W_p. It is read throughout the paper as
"how much this site's read-out has departed from the control template", i.e. as a
map of pathology.

But the same quantity computed between the control template and individual
CONTROLS measures ordinary between-subject read-out variability, with no disease
involved. If the two maps agree, site_dw_norm is not disease-specific.

Controls use a leave-one-out template (Wbar over the other controls) so that a
subject never contributes to the template it is compared against; AD patients are
compared to the full control template exactly as the paper does.
"""
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr
import warnings; warnings.filterwarnings("ignore")
from external_oasis_validate import (
    build_reservoir, teacher_force, fit_W, load_adni, RNG_SEED, N_PC_MODEL)

OUT = "dw_null_check.npz"

signals, first, upid, plabel = load_adni()
cc = [p for p, l in zip(upid, plabel) if l == 0]
ad = [p for p, l in zip(upid, plabel) if l == 1]
print(f"cohort: {len(cc)} CC, {len(ad)} AD")

all_sig = np.concatenate([signals[first[p]].T for p in upid], 0)
evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]

res = build_reservoir()
rw = np.random.default_rng(RNG_SEED + 1)
patW = {}
for p in tqdm(upid, desc="teacher-force + read-out"):
    X, tgt = teacher_force(res, signals[first[p]], ev50)
    patW[p] = fit_W(X, tgt, rw)

Wcc_all = np.mean([patW[p] for p in cc], 0)

# AD side: exactly the paper's quantity, kept per patient
DW_AD = np.array([np.linalg.norm(Wcc_all - patW[p], axis=0) for p in ad])   # (40,121)

# control side: leave-one-out template, so no self-contribution
DW_CC = np.array([
    np.linalg.norm(np.mean([patW[q] for q in cc if q != p], 0) - patW[p], axis=0)
    for p in cc])                                                            # (36,121)

dw_ad, dw_cc = DW_AD.mean(0), DW_CC.mean(0)

# the control template's own column norm, as a third reference
w_norm = np.linalg.norm(Wcc_all, axis=0)

np.savez(OUT, dw_ad=dw_ad, dw_cc=dw_cc, w_norm=w_norm,
         DW_AD=DW_AD, DW_CC=DW_CC)

cached = np.load("pert_dw_site_gridsearch_data.npz")["site_dw_norm"]
n = np.load("node_strength_cache.npz")
s_cc = np.abs(n["cc_fc"] * (1 - np.eye(121))).sum(1)

print("\n=== reproduction check ===")
print(f"  recomputed AD map vs cached site_dw_norm: r = "
      f"{np.corrcoef(dw_ad, cached)[0,1]:.4f}")

print("\n=== is the pathology map disease-specific? ===")
r, p = spearmanr(dw_ad, dw_cc)
print(f"  AD-vs-template map  vs  CONTROL-vs-template map:  rho = {r:+.3f}  p = {p:.3g}")
print(f"  amplitude ratio (mean AD / mean control deviation): "
      f"{dw_ad.mean()/dw_cc.mean():.3f}")

print("\n=== each map's relation to FC node strength ===")
for nm, v in [("AD deviation map", dw_ad), ("control deviation map", dw_cc),
              ("control template column norm", w_norm)]:
    r, p = spearmanr(v, s_cc)
    print(f"  {nm:30s} vs node strength: rho = {r:+.3f}  p = {p:.3g}")

print("\n=== do the two maps pick the same top sites? ===")
for K in [10, 20, 35]:
    a = set(np.argsort(dw_ad)[::-1][:K]); b = set(np.argsort(dw_cc)[::-1][:K])
    print(f"  top-{K:<3d} overlap: {len(a & b)}/{K}")

# ── can a disease-specific map be recovered by testing AD against the null? ──
from scipy.stats import mannwhitneyu
print("\n=== corrected map: per-site AD vs control deviation ===")
pv = np.array([mannwhitneyu(DW_AD[:, i], DW_CC[:, i])[1] for i in range(121)])
sd = DW_CC.std(0) + 1e-12
eff = (DW_AD.mean(0) - DW_CC.mean(0)) / sd          # per-site effect size
order = np.argsort(pv)
q = pv * 121 / (np.argsort(np.argsort(pv)) + 1)     # Benjamini-Hochberg
print(f"  sites with raw p < 0.05      : {(pv < 0.05).sum()}/121")
print(f"  sites surviving BH-FDR q<0.05: {(q < 0.05).sum()}/121")
print(f"  effect size range            : {eff.min():+.2f} to {eff.max():+.2f} SD")
print(f"  strongest 5 sites by p-value : "
      f"{[(int(i), round(float(eff[i]), 2), float(f'{pv[i]:.3g}')) for i in order[:5]]}")
r, p = spearmanr(eff, s_cc)
print(f"  corrected map vs node strength: rho = {r:+.3f}  p = {p:.3g}")
old_top = set(np.argsort(dw_ad)[::-1][:20])
new_top = set(np.argsort(eff)[::-1][:20])
print(f"  overlap of corrected top-20 with the paper's top-20: "
      f"{len(old_top & new_top)}/20")
print(f"\nSaved {OUT}")
