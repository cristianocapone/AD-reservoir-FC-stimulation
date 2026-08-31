"""
atlas_control.py  -- ADNI ONLY, no other cohorts.

Is the paper's headline dW result biology, or an artefact of the parcellation?

The 121-parcel atlas is 100 Schaefer CORTICAL parcels (idx 0-99) + 21
Harvard-Oxford SUBCORTICAL parcels (idx 100-120). The subcortical parcels are
small and weakly coupled, so they may dominate ||dW|| for estimation reasons
rather than pathological ones. This recomputes the two central claims restricted
to cortex.

Reads the clean-rerun caches; writes nothing over existing results.
"""
import os, numpy as np
from scipy.stats import spearmanr, mannwhitneyu

N = 121
CORT = np.arange(0, 100)
SUB  = np.arange(100, 121)

dw = np.load("pert_dw_site_gridsearch_data.npz", allow_pickle=True)["site_dw_norm"]

# FC node strength recomputed on the CLEAN cohort
mats = []
for g in ("CN", "AD"):
    p = os.path.join("timeseries_clean", g)
    for fn in sorted(os.listdir(p)):
        if fn.endswith(".npy"):
            a = np.load(os.path.join(p, fn))
            if a.shape[0] != N:
                a = a.T
            if a.shape[0] != N:          # a few sessions lost parcels in extraction;
                continue                 # the main pipeline excludes them too
            mats.append(np.nan_to_num(np.corrcoef(a)))
FC = np.mean(mats, 0)
S = np.abs(FC * (1 - np.eye(N))).sum(1)
print(f"(node strength from {len(mats)} clean sessions)\n")

print("=== CLAIM 1: dW magnitude vs FC node strength ===")
for name, ix in [("ALL 121 parcels", np.arange(N)),
                 ("CORTICAL only (100)", CORT),
                 ("SUBCORTICAL only (21)", SUB)]:
    r, p = spearmanr(dw[ix], S[ix])
    print(f"  {name:24s} rho = {r:+.3f}   p = {p:.2g}")

print("\n=== Is dW simply tagging 'subcortical'? ===")
_, p1 = mannwhitneyu(dw[SUB], dw[CORT])
_, p2 = mannwhitneyu(S[SUB], S[CORT])
print(f"  mean ||dW||   subcortical {dw[SUB].mean():.4f}  cortical {dw[CORT].mean():.4f}   p={p1:.2g}")
print(f"  mean strength subcortical {S[SUB].mean():7.2f}  cortical {S[CORT].mean():7.2f}   p={p2:.2g}")

sites = np.load("pert_sites_data.npz", allow_pickle=True)
comp  = np.load("pert_compare3_data.npz", allow_pickle=True)
path = np.where(sites["top5_site_counts"] > 0)[0]
ther = np.where(comp["pers_counts"] > 0)[0]
sp = lambda ix: (int((ix < 100).sum()), int((ix >= 100).sum()))
print("\n=== CLAIM 2: composition of the two site sets ===")
print(f"  most-affected (dW) : {len(path):3d} sites -> cortical {sp(path)[0]}, subcortical {sp(path)[1]}")
print(f"  effective targets  : {len(ther):3d} sites -> cortical {sp(ther)[0]}, subcortical {sp(ther)[1]}")
print(f"  shared (all parcels): {len(set(path) & set(ther))}")

pc, tc = set(path[path < 100]), set(ther[ther < 100])
inter = pc & tc
den = min(len(pc), len(tc))
print("\n=== THE CONTROL: both sets restricted to cortex ===")
print(f"  cortical most-affected {len(pc)} | cortical targets {len(tc)} | shared {len(inter)}"
      f"  ({100*len(inter)/den if den else 0:.0f}% of smaller set)")
print("\nInterpretation: if rho collapses within cortex and the cortical sets")
print("overlap substantially, the dissociation is largely parcellation-driven.")
