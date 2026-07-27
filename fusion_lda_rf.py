"""
fusion_lda_rf.py
================
Multimodal AD-vs-control classification on the CLEAN cohort (authoritative
labels, one preprocessing batch/session per subject), comparing each modality
alone with early-fusion combinations, under LDA and Random Forest.

Modalities
  FC-lag : reservoir read-out features (clean_cohort_features.npz)
  FC     : empirical Ledoit-Wolf tangent-space FC (paper's strong baseline)
  struct : parcel-wise grey-matter VOLUME morphometry (struct_gm_features.npz).
           NB there is no diffusion data in this project, so this is structural
           morphometry, not a tractography structural connectome.

All subjects are restricted to those carrying ALL modalities so every arm is
scored on the identical set. Each modality is reduced by a train-only PCA
(<=20 comps), standardised on train rows, then classified; fusion concatenates
the reduced blocks. Nothing that sees the data is fit outside the training fold.

Protocols: repeated (x10) stratified 5-fold, and leave-one-site-out.
"""
import re
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.covariance import LedoitWolf
from nilearn.connectome import ConnectivityMeasure
import warnings; warnings.filterwarnings("ignore")

SEED, N_PC, K_FC, N_SITES = 42, 20, 25, 121


def to_adni(s):
    s = s.replace("sub-", "").split("_ses-")[0]
    if "_S_" in s:
        return s
    m = re.match(r"^(\d+)S(\d+)$", s)
    return f"{m.group(1)}_S_{m.group(2)}" if m else None


# ── clean functional cohort, one session per subject ──────────────────────────
C = np.load("cohort_clean.npz", allow_pickle=True)
aid, lab, site, path, sess = (C["adni_id"], C["label"], C["site"],
                              C["path"], C["session"])
first = {}
for i in np.argsort(sess):
    first.setdefault(aid[i], i)
idx = np.array([first[a] for a in sorted(first)])
sub, y_all, st_all, pt = aid[idx], lab[idx], site[idx], path[idx]

D = np.load("clean_cohort_features.npz", allow_pickle=True)
assert list(D["sub"]) == list(sub), "reservoir cache stale"
fb_all = D["fb"]

# ── structural GM, keyed by ADNI id ───────────────────────────────────────────
S = np.load("struct_gm_features.npz", allow_pickle=True)
struct = {}
sx = np.c_[S["vol"] / S["total_gm"][:, None], S["total_gm"]]
for i, s in enumerate(S["subjects"]):
    a = to_adni(s)
    if a:
        struct[a] = sx[i]

# ── restrict to subjects with ALL modalities ──────────────────────────────────
keep = np.array([s in struct for s in sub])
sub, y, st, pt, fb = sub[keep], y_all[keep], st_all[keep], pt[keep], fb_all[keep]
SX = np.array([struct[s] for s in sub])
series = [(a if a.shape[0] == N_SITES else a.T) for a in (np.load(p) for p in pt)]
print(f"subjects with FC + FC-lag + structural, clean labels: {len(y)} "
      f"({(y==0).sum()} CN, {(y==1).sum()} AD), {len(np.unique(st))} sites")


def clf(name):
    if name == "LDA":
        return LinearDiscriminantAnalysis()
    return RandomForestClassifier(n_estimators=500, min_samples_leaf=2,
                                  class_weight="balanced_subsample",
                                  random_state=SEED, n_jobs=-1)


def block(mod, tr, te):
    """One modality's train-only reduced+standardised features."""
    if mod == "struct":
        Btr, Bte = SX[tr], SX[te]
    elif mod == "FC-lag":
        fm = fb[tr].mean(0); fcc = fb[tr] - fm
        ev, evec = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(ev)[::-1]
        ev = np.maximum(ev[o], 0); evec = evec[:, o]
        Btr = (evec * np.sqrt(ev))[:, :K_FC]
        Bte = (((fb[te] - fm) @ fcc.T @ evec) / (np.sqrt(ev) + 1e-12))[:, :K_FC]
    else:                                                # empirical tangent FC
        cm = ConnectivityMeasure(cov_estimator=LedoitWolf(), kind="tangent",
                                 vectorize=True)
        Btr = cm.fit_transform([series[i].T for i in tr])
        Bte = cm.transform([series[i].T for i in te])
    pca = PCA(n_components=min(N_PC, len(tr) - 1, Btr.shape[1]),
              random_state=SEED).fit(Btr)
    Btr, Bte = pca.transform(Btr), pca.transform(Bte)
    sc = StandardScaler().fit(Btr)
    return sc.transform(Btr), sc.transform(Bte)


def features(mods, tr, te):
    tr_blocks, te_blocks = zip(*(block(m, tr, te) for m in mods))
    return np.concatenate(tr_blocks, 1), np.concatenate(te_blocks, 1)


def evaluate(mods, model):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    oof = np.full(len(y), np.nan); aucs = []; bals = []
    for j, (tr, te) in enumerate(rskf.split(fb, y)):
        Xtr, Xte = features(mods, tr, te)
        oof[te] = clf(model).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        if (j + 1) % 5 == 0:
            aucs.append(roc_auc_score(y, oof))
            bals.append(balanced_accuracy_score(y, (oof > 0.5).astype(int)))
            oof[:] = np.nan
    z = np.full(len(y), np.nan)
    for s_ in np.unique(st):
        te = np.where(st == s_)[0]; tr = np.where(st != s_)[0]
        if len(np.unique(y[tr])) < 2 or len(tr) < 10:
            continue
        Xtr, Xte = features(mods, tr, te)
        p = clf(model).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1]
        z[te] = p - p.mean()
    k = ~np.isnan(z)
    lo = roc_auc_score(y[k], z[k]) if len(np.unique(y[k])) == 2 else np.nan
    return np.mean(aucs), np.std(aucs), np.mean(bals), lo


ARMS = [("struct",), ("FC",), ("FC-lag",),
        ("struct", "FC"), ("struct", "FC-lag"),
        ("struct", "FC", "FC-lag")]
print(f"\n{'modalities':22s} {'clf':13s} {'CV AUROC':16s} {'bal':7s} {'LOSO'}")
print("-" * 66)
rows = {}
for mods in ARMS:
    for model in ["LDA", "RandomForest"]:
        a, s, b, lo = evaluate(mods, model)
        rows[(mods, model)] = (a, s, b, lo)
        print(f"{'+'.join(mods):22s} {model:13s} {a:.3f} +/- {s:.3f}    "
              f"{b:.3f}   {lo:.3f}")

np.savez("fusion_lda_rf_results.npz",
         **{f"{'_'.join(m)}__{c}": np.array(v) for (m, c), v in rows.items()})
print("\nSaved fusion_lda_rf_results.npz")
