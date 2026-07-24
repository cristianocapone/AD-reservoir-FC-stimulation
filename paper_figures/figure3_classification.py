"""
figure3_classification.py
=========================
Figure 3 — FC vs G-space classification.

Both classifiers are derived from the SAME per-patient (single-session) readout
W, fit with regularisation noise sigma:
  • G-space  : project W into the row-space of X, SVD across patients -> G-scores
  • FC-lag   : reconstruct signals  Y = W.T @ X, lagged-FC features (lags 0-2),
               Gram-SVD across patients -> FC-scores

Each is classified CC vs AD with a balanced Fisher LDA (transductive, the
established G-space methodology), and evaluated by balanced accuracy + AUROC.

Swept:
  • K     : number of SVD components fed to the LDA
  • sigma : W-fit regularisation noise

Panels (2 x 2):
  A  BAL-ACC vs K      (at best sigma)     B  AUROC vs K      (at best sigma)
  C  BAL-ACC vs sigma  (at best K)         D  AUROC vs sigma  (at best K)

COMPUTE STEP ONLY -- this script produces the sweep data. The MANUSCRIPT
Figure 3 (6 panels A-F, incl. the tangent-space benchmark ROC and the FC PCA
point clouds) is rendered by  replot_figure3.py  from the .npz saved here.
The figure this script draws is an outdated 4-panel version and is written to
figure3_legacy_4panel.{png,pdf} so it can never overwrite the real figure.

Saves: figure3_classification_data.npz  (consumed by replot_figure3.py)
       figure3_legacy_4panel.png / .pdf  (legacy render, NOT used in the paper)
"""
import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

from res import RESERVOIRE_SIMPLE

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

RNG_SEED   = 42
N_CC_SAMP  = 40
N_SITES    = 121
N_PC_MODEL = 50
K_PC       = 200
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SR         = 0.95
MAX_LAG    = 2
TS_ROOT    = "../timeseries"

SIGMA_GRID = np.array([0.003, 0.007, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5])
K_GRID     = np.array([2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50])

N_SPLITS  = 5     # 80% train / 20% test per fold
N_REPEATS = 10    # repeated stratified CV for stable estimates

G_COL  = "#7B1FA2"    # G-space  — purple
FC_COL = "#00838F"    # FC-lag   — teal

# ══════════════════════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════════════════════
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, pid_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMP, len(files)),
                                replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            pid_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

pid_raw    = np.array(pid_raw)
labels_raw = np.array(labels_raw)
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
n_cc = int((patient_labels == 0).sum()); n_ad = int((patient_labels == 1).sum())
print(f"  {N_patients} patients ({n_cc} CC, {n_ad} AD)")

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── reservoir (fresh, established classification setup) ───────────────────────
print("Reservoir TF pass (first session / patient) ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

first_idx = {pid: patient_sids[pid][0] for pid in unique_pids}
patX, patY = {}, {}
for pid in tqdm(unique_pids, desc="  TF"):
    s = signals[first_idx[pid]]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    X_raw = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    Xf = np.array(X_raw)[TIMES_SKIP:]
    patX[pid] = Xf
    patY[pid] = tgt[:, TIMES_SKIP:TIMES_SKIP + len(Xf)].T

# precompute X row-space (W -> G projection),  sigma-independent
patVt = {}
for pid in unique_pids:
    _, sx, Vtx = np.linalg.svd(patX[pid].astype(np.float64), full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    patVt[pid] = Vtx[:kk]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.thr_ = 0.5*((X0@w).mean() + (X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_
    def predict(self, X):
        return np.where(self.transform(X) >= self.thr_,
                        self.classes_[1], self.classes_[0])

def _balance(X, y, seed=0):
    r = np.random.default_rng(seed)
    c0, c1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([r.choice(c0, n, replace=False),
                          r.choice(c1, n, replace=False)])
    r.shuffle(sel); return X[sel], y[sel]

_rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                random_state=RNG_SEED)

def _eval(G, labels, k, clf):
    """Repeated stratified 80/20 CV on a transductive embedding.
    clf = 'lda' (balanced Fisher LDA) or 'rf' (random forest).
    Returns (bal_mean, bal_std, auc_mean, auc_std) over folds."""
    k  = min(k, G.shape[1])
    Gk = G[:, :k]
    bals, aucs = [], []
    for fold, (tr, te) in enumerate(_rskf.split(Gk, labels)):
        Xtr, ytr = Gk[tr], labels[tr]
        Xte, yte = Gk[te], labels[te]
        if clf == "lda":
            Xb, yb = _balance(Xtr, ytr, seed=RNG_SEED + fold)
            if len(np.unique(yb)) < 2:
                continue
            lda = _LDA().fit(Xb, yb)
            if lda.transform(Xtr)[ytr == 0].mean() > \
               lda.transform(Xtr)[ytr == 1].mean():
                lda.w_ *= -1; lda.thr_ *= -1
            score = lda.transform(Xte); pred = lda.predict(Xte)
        else:  # random forest
            rf = RandomForestClassifier(n_estimators=300, max_features="sqrt",
                                        min_samples_leaf=3, class_weight="balanced",
                                        random_state=fold, n_jobs=-1)
            rf.fit(Xtr, ytr)
            score = rf.predict_proba(Xte)[:, 1]; pred = rf.predict(Xte)
        bals.append(balanced_accuracy_score(yte, pred))
        if len(np.unique(yte)) == 2:
            aucs.append(roc_auc_score(yte, score))
    return (float(np.mean(bals)),  float(np.std(bals)),
            float(np.mean(aucs)),  float(np.std(aucs)))

def project_W(W, pid):
    Vt = patVt[pid]
    return (W.T.astype(np.float64) @ Vt.T @ Vt).flatten()

def build_G(Wproj_list):
    Wstack = np.array(Wproj_list)
    Wcent  = Wstack - Wstack.mean(0)
    _, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
    Meff = Wstack.shape[0] - 1
    return Wcent @ Vsvd[:Meff].T

def lagged_corr(S, lag):
    if lag == 0:
        return np.corrcoef(S.T)
    T = S.shape[0]
    A = S[:T-lag].astype(np.float64); B = S[lag:].astype(np.float64)
    A -= A.mean(0); B -= B.mean(0)
    A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - lag)

def fclag_feat(S):
    feats = []
    for lag in range(MAX_LAG + 1):
        fc = np.nan_to_num(lagged_corr(S, lag))
        feats.append(fc[np.triu_indices(N_SITES, k=1)] if lag == 0
                     else fc.flatten())
    return np.concatenate(feats)

def build_gram(feat_list):
    F = np.array(feat_list)
    Fc = F - F.mean(0)
    ev, evec = np.linalg.eigh(Fc @ Fc.T)
    o = np.argsort(ev)[::-1]
    ev = np.maximum(ev[o], 0); evec = evec[:, o]
    return evec * np.sqrt(ev)

# ══════════════════════════════════════════════════════════════════════════════
# SWEEP
# ══════════════════════════════════════════════════════════════════════════════
nS, nK = len(SIGMA_GRID), len(K_GRID)
# results[(fspace, clf)] = dict of (nS,nK) arrays for bal/auc mean & std
COMBOS = [("G",  "lda"), ("G",  "rf"), ("FC", "lda"), ("FC", "rf")]
R = {c: {m: np.zeros((nS, nK)) for m in
         ["bal_m", "bal_s", "auc_m", "auc_s"]} for c in COMBOS}

print(f"\nSweep: {nS} sigma x {nK} K  x {{G,FC}} x {{LDA,RF}}  "
      f"({N_REPEATS}x{N_SPLITS}-fold) ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
for si, sigma in enumerate(SIGMA_GRID):
    Wproj, fcs = [], []
    for pid in unique_pids:
        Xc = patX[pid]; Yc = patY[pid]
        W  = np.linalg.pinv(Xc + rng_w.normal(0, sigma, Xc.shape)) @ Yc
        Wproj.append(project_W(W, pid))
        S = (W.T.astype(np.float64) @ Xc.T.astype(np.float64)).T
        fcs.append(fclag_feat(S))
    emb = {"G": build_G(Wproj), "FC": build_gram(fcs)}
    for ki, k in enumerate(K_GRID):
        for (fs, clf) in COMBOS:
            bm, bs, am, as_ = _eval(emb[fs], patient_labels, k, clf)
            R[(fs, clf)]["bal_m"][si, ki] = bm
            R[(fs, clf)]["bal_s"][si, ki] = bs
            R[(fs, clf)]["auc_m"][si, ki] = am
            R[(fs, clf)]["auc_s"][si, ki] = as_
    msg = "  ".join(f"{fs}-{clf}:auc={R[(fs,clf)]['auc_m'][si].max():.3f}"
                    for fs, clf in COMBOS)
    print(f"  sigma={sigma:.3f}  {msg}", flush=True)

# Reference slices: robust marginal optima (averaged over the other axis &
# all four combos) — avoids chasing a single noisy (sigma,K) cell.
auc_stack = np.stack([R[c]["auc_m"] for c in COMBOS])   # (4, nS, nK)
mean_sk   = auc_stack.mean(0)                            # (nS, nK)
g_ki = int(mean_sk.mean(0).argmax())                    # best K  (marg. over sigma)
g_si = int(mean_sk.mean(1).argmax())                    # best sigma (marg. over K)
best_sigma = SIGMA_GRID[g_si]; best_K = K_GRID[g_ki]
print(f"\nReference slices (robust): sigma={best_sigma}, K={best_K}")
for (fs, clf) in COMBOS:
    am = R[(fs, clf)]["auc_m"]; bm = R[(fs, clf)]["bal_m"]
    i = np.unravel_index(am.argmax(), am.shape)
    print(f"  {fs:2s}-{clf:3s}  best AUROC={am[i]:.3f}  BAL={bm[i]:.3f}  "
          f"@ sigma={SIGMA_GRID[i[0]]}, K={K_GRID[i[1]]}")

np.savez("figure3_classification_data.npz",
         sigma_grid=SIGMA_GRID, k_grid=K_GRID,
         best_sigma=best_sigma, best_K=best_K, n_cc=n_cc, n_ad=n_ad,
         **{f"{fs}_{clf}_{m}": R[(fs, clf)][m]
            for (fs, clf) in COMBOS
            for m in ["bal_m", "bal_s", "auc_m", "auc_s"]})

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering ...")
fig = plt.figure(figsize=(12, 9.5), facecolor="white")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30,
                        top=0.89, bottom=0.08, left=0.09, right=0.97)

def _tag(ax, t):
    ax.text(-0.13, 1.05, t, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")

# style per combo: colour=feature space, marker/linestyle=classifier
STYLE = {
    ("G",  "lda"): dict(color=G_COL,  ls="-",  marker="o", label="G-space · LDA"),
    ("G",  "rf"):  dict(color=G_COL,  ls="--", marker="^", label="G-space · RF"),
    ("FC", "lda"): dict(color=FC_COL, ls="-",  marker="s", label="FC-lag · LDA"),
    ("FC", "rf"):  dict(color=FC_COL, ls="--", marker="D", label="FC-lag · RF"),
}
YL = (0.42, 0.82)

def _curve(ax, x, m, s, st):
    ax.fill_between(x, m - s, m + s, color=st["color"], alpha=0.10)
    ax.plot(x, m, st["ls"], marker=st["marker"], ms=4.5, lw=1.8,
            color=st["color"], label=st["label"])

# A: BAL-ACC vs K  (at best sigma)
ax = fig.add_subplot(gs[0, 0])
for c in COMBOS:
    d = R[c]; _curve(ax, K_GRID, d["bal_m"][g_si], d["bal_s"][g_si], STYLE[c])
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("K  (# SVD components)"); ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Accuracy vs. # SVD components  (σ = {best_sigma:g})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7.5); _tag(ax, "A")

# B: AUROC vs K
ax = fig.add_subplot(gs[0, 1])
for c in COMBOS:
    d = R[c]; _curve(ax, K_GRID, d["auc_m"][g_si], d["auc_s"][g_si], STYLE[c])
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax.set_xlabel("K  (# SVD components)"); ax.set_ylabel("AUROC")
ax.set_title(f"AUROC vs. # SVD components  (σ = {best_sigma:g})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7.5); _tag(ax, "B")

# C: BAL-ACC vs sigma  (at best K)
ax = fig.add_subplot(gs[1, 0])
for c in COMBOS:
    d = R[c]; _curve(ax, SIGMA_GRID, d["bal_m"][:, g_ki], d["bal_s"][:, g_ki], STYLE[c])
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("W-fit regularisation noise  σ"); ax.set_ylabel("Balanced accuracy")
ax.set_title(f"Accuracy vs. noise  (K = {best_K})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7.5); _tag(ax, "C")

# D: AUROC vs sigma
ax = fig.add_subplot(gs[1, 1])
for c in COMBOS:
    d = R[c]; _curve(ax, SIGMA_GRID, d["auc_m"][:, g_ki], d["auc_s"][:, g_ki], STYLE[c])
ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.6)
ax.set_xscale("log")
ax.set_xlabel("W-fit regularisation noise  σ"); ax.set_ylabel("AUROC")
ax.set_title(f"AUROC vs. noise  (K = {best_K})")
ax.set_ylim(*YL); ax.legend(frameon=False, fontsize=7.5); _tag(ax, "D")

fig.suptitle(
    "CC vs AD classification: G-space geometry vs. lagged-FC of reconstruction\n"
    f"(same per-patient W,  LDA & RF,  repeated {N_SPLITS}-fold 80/20 CV,  "
    f"N={n_cc} CC + {n_ad} AD,  lags 0–{MAX_LAG})",
    fontsize=10.5, fontweight="bold", y=0.975)

for ext in ("png", "pdf"):
    out = f"figure3_legacy_4panel.{ext}"     # NOT the manuscript figure
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
plt.close()
