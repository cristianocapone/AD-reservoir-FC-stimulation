"""
Clinical correlate (ADNI only): does the model's disease score track cognition?

Uses out-of-fold (cross-validated) FC-lag discriminant scores, so the score is
never fit on the subject it is evaluated on, and MMSE is never seen by the model.
Each imaging session is matched to the MMSE visit nearest in time.
Writes only to clinical_results/.
"""
import os, numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
import warnings; warnings.filterwarnings("ignore")

MM = r"C:\Users\user\.claude\uploads\6e43a621-22c2-45ea-a643-0050c4d24963\d9d0cc53-MMSE_28Jan2025.xlsx"
DM = r"C:\Users\user\.claude\uploads\6e43a621-22c2-45ea-a643-0050c4d24963\ec6b87dc-Demographics.xlsx"
OUT = "clinical_results"; os.makedirs(OUT, exist_ok=True)
K_FC, SEED = 25, 42

C = np.load("cohort_clean.npz", allow_pickle=True)
D = np.load("clean_cohort_features.npz", allow_pickle=True)
fb, sub = D["fb"], D["sub"]
lab = {a: int(C["label"][C["adni_id"] == a][0]) for a in sub}
ses = {a: str(C["session"][C["adni_id"] == a][0]) for a in sub}
y = np.array([lab[a] for a in sub])
print(f"cohort: {len(sub)} subjects ({(y==0).sum()} CN, {(y==1).sum()} AD)")

# ── out-of-fold FC-lag discriminant score (leakage-free embedding per fold) ───
def embed(tr, te, k=K_FC):
    fm = fb[tr].mean(0); fcc = fb[tr] - fm
    ev, evec = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(ev)[::-1]
    ev = np.maximum(ev[o], 0); evec = evec[:, o]
    return (evec*np.sqrt(ev))[:, :k], (((fb[te]-fm) @ fcc.T @ evec)/(np.sqrt(ev)+1e-12))[:, :k]

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
acc = np.zeros(len(sub)); cnt = np.zeros(len(sub))
for tr, te in rskf.split(fb, y):
    Xtr, Xte = embed(tr, te)
    sc = StandardScaler().fit(Xtr)
    m = LinearDiscriminantAnalysis().fit(sc.transform(Xtr), y[tr])
    acc[te] += m.decision_function(sc.transform(Xte)); cnt[te] += 1
Z = acc/cnt                                    # higher = more AD-like
from sklearn.metrics import roc_auc_score
print(f"  out-of-fold AUROC = {roc_auc_score(y, Z):.3f}")

# ── MMSE nearest to each imaging session ─────────────────────────────────────
mm = pd.read_excel(MM)[["PTID","VISDATE","MMSCORE"]].dropna(subset=["MMSCORE"])
mm["VISDATE"] = pd.to_datetime(mm["VISDATE"], errors="coerce")
mmse, dgap = [], []
for a in sub:
    t = pd.to_datetime(ses[a], format="%Y%m%d", errors="coerce")
    r = mm[mm.PTID == a].dropna(subset=["VISDATE"])
    if len(r) == 0 or pd.isna(t):
        mmse.append(np.nan); dgap.append(np.nan); continue
    j = (r.VISDATE - t).abs().idxmin()
    mmse.append(float(r.loc[j,"MMSCORE"])); dgap.append(abs((r.loc[j,"VISDATE"]-t).days))
mmse = np.array(mmse); dgap = np.array(dgap)
ok = ~np.isnan(mmse)
print(f"  matched MMSE for {ok.sum()}/{len(sub)}  (median |scan-visit| gap {np.nanmedian(dgap):.0f} d)")

dm = pd.read_excel(DM)[["PTID","PTGENDER","PTEDUCAT","PTDOBYY"]].drop_duplicates("PTID")
dmap = dm.set_index("PTID")
age = np.array([ (int(ses[a][:4]) - float(dmap.loc[a,"PTDOBYY"])) if a in dmap.index and not pd.isna(dmap.loc[a,"PTDOBYY"]) else np.nan for a in sub])
edu = np.array([ float(dmap.loc[a,"PTEDUCAT"]) if a in dmap.index and not pd.isna(dmap.loc[a,"PTEDUCAT"]) else np.nan for a in sub])
sex = np.array([ float(dmap.loc[a,"PTGENDER"]) if a in dmap.index and not pd.isna(dmap.loc[a,"PTGENDER"]) else np.nan for a in sub])

print("\n=== MMSE by group (label sanity) ===")
print(f"  CN {np.nanmean(mmse[ok&(y==0)]):.1f}+/-{np.nanstd(mmse[ok&(y==0)]):.1f}   "
      f"AD {np.nanmean(mmse[ok&(y==1)]):.1f}+/-{np.nanstd(mmse[ok&(y==1)]):.1f}   "
      f"p={mannwhitneyu(mmse[ok&(y==0)],mmse[ok&(y==1)])[1]:.2g}")

print("\n=== MODEL SCORE vs MMSE (out-of-fold; model never saw MMSE) ===")
for name, m in [("all subjects", ok), ("within AD only", ok&(y==1)), ("within CN only", ok&(y==0))]:
    if m.sum() < 8: continue
    r,p = spearmanr(Z[m], mmse[m]); rp,pp = pearsonr(Z[m], mmse[m])
    print(f"  {name:16s} n={m.sum():3d}  Spearman rho={r:+.3f} p={p:.3g}   Pearson r={rp:+.3f} p={pp:.3g}")

print("\n=== confounds: is the score just age / education / sex? ===")
for nm, v in [("age", age), ("education", edu), ("sex", sex)]:
    m = ok & ~np.isnan(v)
    r,p = spearmanr(Z[m], v[m]); print(f"  score vs {nm:10s} rho={r:+.3f} p={p:.3g}")
m = ok & ~np.isnan(age) & ~np.isnan(edu)
try:
    import statsmodels.api as sm
    X = sm.add_constant(np.column_stack([mmse[m], age[m], edu[m]]))
    res = sm.OLS(Z[m], X).fit()
    print(f"\n  OLS  score ~ MMSE + age + education   (n={m.sum()})")
    for nm,b,pv in zip(["const","MMSE","age","edu"], res.params, res.pvalues):
        print(f"    {nm:6s} beta={b:+.4f}  p={pv:.3g}")
    print(f"    R^2 = {res.rsquared:.3f}")
except ImportError:
    print("  (statsmodels not installed; skipping partial regression)")

np.savez(os.path.join(OUT,"mmse_correlation.npz"), sub=sub, y=y, Z=Z,
         mmse=mmse, gap=dgap, age=age, edu=edu, sex=sex)
print(f"\nSaved {OUT}/mmse_correlation.npz")
