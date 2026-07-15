"""
rf_lc.py
========
Compare LDA vs Random Forest classifiers on the G-space learning curve.

Pipeline: identical to learning_curve_direct.py / klda_sweep.py
  - 702-session G-space (each session → own W + X-SVD)
  - Patient G-score = mean over sessions
  - G-space cached in g_space_cache.npz after first run (keeps all PCs)

Classifiers compared (all trained on balanced subset, evaluated LOPO):
  1. LDA          (baseline)
  2. RF-100       n_estimators=100, max_depth=None,  class_weight='balanced'
  3. RF-100-reg   n_estimators=100, max_depth=5,     class_weight='balanced',
                  min_samples_leaf=2  (regularised for small N)

Experiments:
  A) Learning curve at K=15 (LDA optimum), N_GRID=[4..40], N_REPS=30
  B) K-sweep at N=40 for RF-100 vs LDA  (K in [5,10,15,20,25,30,35,38])

Outputs: rf_lc_curve.png, rf_klda_sweep.png, rf_results.npz
"""
import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from tqdm import trange
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── hyper-parameters ───────────────────────────────────────────────────────────
RNG_SEED   = 42
N_SITES    = 121
N_PC_MODEL = 50
K_PC       = 200
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
SR         = 0.95
TS_ROOT    = "./timeseries"
OUT_DIR    = "."
CACHE_FILE = "g_space_cache.npz"

K_USE   = 15          # LDA optimal from klda_sweep
K_MAX   = 38          # max PCs to keep in cache (= N_FIXED-2 at N=40)
N_GRID  = [4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 35, 40]
N_REPS  = 30
K_SWEEP = [5, 10, 15, 20, 25, 30, 35, 38]   # for experiment B
N_FIXED = 40

# RF configurations
RF_DEFAULT = RandomForestClassifier(
    n_estimators=100, max_depth=None, min_samples_leaf=1,
    class_weight="balanced", random_state=RNG_SEED, n_jobs=-1)
RF_REG = RandomForestClassifier(
    n_estimators=100, max_depth=5, min_samples_leaf=2,
    class_weight="balanced", random_state=RNG_SEED, n_jobs=-1)

# ── G-space: load cache or compute ────────────────────────────────────────────
if os.path.exists(CACHE_FILE):
    print(f"Loading cached G-space from {CACHE_FILE} ...")
    cache          = np.load(CACHE_FILE)
    G_pat_full     = cache["G_pat_full"]      # (N_patients, K_MAX)
    patient_labels = cache["patient_labels"]
    cc_idx         = cache["cc_idx"].astype(int)
    ad_idx         = cache["ad_idx"].astype(int)
    cum_var        = cache["cum_var"]
    N_patients     = G_pat_full.shape[0]
    print(f"  {N_patients} patients, {G_pat_full.shape[1]} PCs  "
          f"(CC={len(cc_idx)}, AD={len(ad_idx)})")

else:
    print("No cache found — running full pipeline ...")

    # ── load sessions ──────────────────────────────────────────────────────────
    print("Loading data ...")
    signals, labels_raw, pid_raw = [], [], []
    for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
        folder = os.path.join(TS_ROOT, subfolder)
        for fname in sorted(f for f in os.listdir(folder) if f.endswith(".npy")):
            arr = np.load(os.path.join(folder, fname)).T
            if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
                signals.append(arr.T)
                pid_raw.append(fname.split("_ses-")[0])
                labels_raw.append(0 if label == "CC" else 1)

    pid_raw    = np.array(pid_raw)
    labels_raw = np.array(labels_raw)
    N_subj     = len(signals)
    unique_pids    = np.unique(pid_raw)
    N_patients     = len(unique_pids)
    patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
    patient_labels = np.array([labels_raw[patient_sids[pid][0]]
                                for pid in unique_pids])
    cc_idx = np.where(patient_labels == 0)[0]
    ad_idx = np.where(patient_labels == 1)[0]
    print(f"  {N_subj} sessions | {N_patients} patients "
          f"({len(cc_idx)} CC, {len(ad_idx)} AD)")

    # ── population PCA ────────────────────────────────────────────────────────
    print("Population PCA ...")
    all_sig  = np.concatenate([s.T for s in signals], axis=0)
    centered = all_sig - all_sig.mean(0)
    evals_p, evecs_p = np.linalg.eigh(np.cov(centered.T))
    ev50 = evecs_p[:, np.argsort(evals_p)[::-1]][:, :N_PC_MODEL]

    # ── reservoir ─────────────────────────────────────────────────────────────
    print("Initialising reservoir ...")
    par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
               T=139, dt=0.005, sigma_input=0.01,
               shape=(N_HIDDEN, N_SITES, N_SITES, 139))
    res = RESERVOIRE_SIMPLE(par)
    res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

    # ── TF pass ───────────────────────────────────────────────────────────────
    print("TF pass ...")
    sess_X, sess_Y = {}, {}
    for idx in trange(N_subj, desc="  TF"):
        s = signals[idx]; T_s = s.shape[1]
        tgt = (s.T @ ev50 @ ev50.T).T
        res.T = T_s; res.reset()
        Xraw = []
        for t in range(T_s - 1):
            res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
            Xraw.append(res.X.copy())
        Xf = np.array(Xraw)[TIMES_SKIP:]
        sess_X[idx] = Xf
        sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T
    print("  TF done.")

    # ── W + X-SVD per session ─────────────────────────────────────────────────
    print("Fitting W (all sessions) ...")
    rng_w    = np.random.default_rng(RNG_SEED + 1)
    sess_Vtk = {}; sess_W = {}
    for idx in trange(N_subj, desc="  SVD+W", leave=False):
        Xca = sess_X[idx].astype(np.float64)
        _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
        kk = min(K_PC, int((sx > 1e-8).sum()))
        sess_Vtk[idx] = Vtx[:kk]
        noise = rng_w.normal(0, SIGMA, sess_X[idx].shape)
        sess_W[idx] = np.linalg.pinv(sess_X[idx] + noise) @ sess_Y[idx]

    def project_W_sess(W, idx):
        Vt_k = sess_Vtk[idx]
        return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

    # ── G-space (all sessions, keep K_MAX PCs) ────────────────────────────────
    print(f"\nBuilding G-space (all sessions, keeping {K_MAX} PCs) ...")
    Wproj_sess = np.array([project_W_sess(sess_W[idx], idx)
                           for idx in range(N_subj)], dtype=np.float64)
    Wmean_s  = Wproj_sess.mean(0)
    Wproj_sc = Wproj_sess - Wmean_s
    print("  Computing Gram matrix ...")
    C_s = Wproj_sc @ Wproj_sc.T
    evals_s, evecs_s = np.linalg.eigh(C_s)
    order_s  = np.argsort(evals_s)[::-1]
    evals_s  = np.maximum(evals_s[order_s], 0.0)
    evecs_s  = evecs_s[:, order_s]

    cum_var = np.cumsum(evals_s) / evals_s.sum() * 100
    print("  Variance explained:")
    for k in [5, 10, 15, 20, 25, 30, 35, 38]:
        if k <= len(evals_s):
            print(f"    top {k:3d} PCs: {cum_var[k-1]:.1f}%")

    G_sess_full = evecs_s[:, :K_MAX] * np.sqrt(evals_s[:K_MAX])

    print("Averaging sessions → patient G-scores ...")
    G_pat_full = np.zeros((N_patients, K_MAX), dtype=np.float64)
    for pi, pid in enumerate(unique_pids):
        G_pat_full[pi] = G_sess_full[patient_sids[pid]].mean(0)

    # ── save cache ────────────────────────────────────────────────────────────
    np.savez(CACHE_FILE,
             G_pat_full=G_pat_full,
             patient_labels=patient_labels,
             cc_idx=cc_idx, ad_idx=ad_idx,
             cum_var=cum_var)
    print(f"  Cache saved → {CACHE_FILE}")


# ── helpers ────────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i, n, replace=False),
                          rng2.choice(c1i, n, replace=False)])
    rng2.shuffle(sel)
    return X[sel], y[sel]


def lopo_lda(S_global, y_S, k_use):
    n      = len(S_global)
    G_sub  = G_pat_full[S_global, :k_use]
    preds  = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    for i in range(n):
        mask = np.arange(n) != i
        G_tr = G_sub[mask]; y_tr = y_S[mask]
        G_te = G_sub[i]
        Xb, yb = _balance(G_tr, y_tr, seed=RNG_SEED)
        try:
            lda = _LDA().fit(Xb, yb)
        except Exception:
            continue
        z_tr = lda.transform(G_tr)
        if z_tr[y_tr==0].mean() > z_tr[y_tr==1].mean():
            lda.w_ *= -1
            z_tr    = lda.transform(G_tr)
        thr = 0.5*(z_tr[y_tr==0].mean() + z_tr[y_tr==1].mean())
        z_te      = lda.transform(G_te.reshape(1,-1))[0]
        preds[i]  = float(z_te >= thr)
        scores[i] = z_te - thr
    return preds, scores


def lopo_rf(S_global, y_S, k_use, clf_proto):
    n      = len(S_global)
    G_sub  = G_pat_full[S_global, :k_use]
    preds  = np.full(n, np.nan)
    scores = np.full(n, np.nan)
    for i in range(n):
        mask = np.arange(n) != i
        G_tr = G_sub[mask]; y_tr = y_S[mask]
        G_te = G_sub[i]
        Xb, yb = _balance(G_tr, y_tr, seed=RNG_SEED)
        try:
            rf = clone(clf_proto).fit(Xb, yb)
        except Exception:
            continue
        ad_col = list(rf.classes_).index(1)
        prob_te   = rf.predict_proba(G_te.reshape(1,-1))[0, ad_col]
        preds[i]  = float(prob_te >= 0.5)
        scores[i] = prob_te - 0.5
    return preds, scores


def run_metrics(pr, sc, y_S):
    valid = np.isfinite(pr)
    if valid.sum() < 4:
        return np.nan, np.nan
    pr_v = pr[valid]; sc_v = sc[valid]; y_v = y_S[valid]
    sens = np.mean(pr_v[y_v==1] == 1)
    spec = np.mean(pr_v[y_v==0] == 0)
    bal  = 0.5*(sens + spec)
    try:    auc = roc_auc_score(y_v, sc_v)
    except: auc = np.nan
    return bal, auc


# ── Experiment A: Learning curve at K=15 ──────────────────────────────────────
print(f"\n{'='*65}")
print(f"Experiment A: Learning curve at K={K_USE}, {N_REPS} reps")
print(f"{'='*65}")

N_max      = min(len(cc_idx), len(ad_idx))
N_GRID_eff = [N for N in N_GRID if N <= N_max]

clfs = {
    "LDA":       (lopo_lda,  None,       "#1565C0"),
    "RF-100":    (lopo_rf,   RF_DEFAULT, "#2E7D32"),
    "RF-reg":    (lopo_rf,   RF_REG,     "#E65100"),
}

results = {name: {"bal": {N: [] for N in N_GRID_eff},
                  "auc": {N: [] for N in N_GRID_eff}}
           for name in clfs}

for N in N_GRID_eff:
    k_use = min(K_USE, 2*N - 2)
    print(f"\n  N={N:2d}  k_use={k_use}", flush=True)
    for rep in range(N_REPS):
        rng_rep = np.random.default_rng(RNG_SEED + rep*10000 + N)
        sel_cc  = rng_rep.choice(cc_idx, N, replace=False)
        sel_ad  = rng_rep.choice(ad_idx, N, replace=False)
        S       = np.concatenate([sel_cc, sel_ad])
        y_S     = patient_labels[S]

        for name, (fn, clf, _) in clfs.items():
            if clf is None:
                pr, sc = fn(S, y_S, k_use)
            else:
                pr, sc = fn(S, y_S, k_use, clf)
            bal, auc = run_metrics(pr, sc, y_S)
            if np.isfinite(bal):
                results[name]["bal"][N].append(bal)
                results[name]["auc"][N].append(auc)

    for name in clfs:
        ba = np.array(results[name]["bal"][N])
        au = np.array(results[name]["auc"][N])
        print(f"    {name:10s}  BAL-ACC={ba.mean():.4f}±{ba.std():.4f}  "
              f"AUROC={au.mean():.4f}±{au.std():.4f}", flush=True)


# ── Experiment B: K-sweep at N=40 ─────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"Experiment B: K-sweep at N={N_FIXED}, {N_REPS} reps")
print(f"{'='*65}")

K_SWEEP_eff = [k for k in K_SWEEP if k <= K_MAX]
sweep_res = {name: {"bal": [], "bal_std": [], "auc": [], "auc_std": []}
             for name in ["LDA", "RF-100"]}

for k in K_SWEEP_eff:
    bals_lda, aucs_lda = [], []
    bals_rf,  aucs_rf  = [], []
    for rep in range(N_REPS):
        rng_rep = np.random.default_rng(RNG_SEED + rep*10000 + k)
        sel_cc  = rng_rep.choice(cc_idx, N_FIXED, replace=False)
        sel_ad  = rng_rep.choice(ad_idx, N_FIXED, replace=False)
        S       = np.concatenate([sel_cc, sel_ad])
        y_S     = patient_labels[S]

        pr, sc = lopo_lda(S, y_S, k)
        bal, auc = run_metrics(pr, sc, y_S)
        if np.isfinite(bal): bals_lda.append(bal); aucs_lda.append(auc)

        pr, sc = lopo_rf(S, y_S, k, RF_DEFAULT)
        bal, auc = run_metrics(pr, sc, y_S)
        if np.isfinite(bal): bals_rf.append(bal); aucs_rf.append(auc)

    for arr, name in [(bals_lda,"LDA"),(bals_rf,"RF-100")]:
        a = np.array(arr)
        sweep_res[name]["bal"].append(a.mean())
        sweep_res[name]["bal_std"].append(a.std())
        sweep_res[name]["auc"].append(
            np.nanmean(aucs_lda if name=="LDA" else aucs_rf))
        sweep_res[name]["auc_std"].append(
            np.nanstd(aucs_lda if name=="LDA" else aucs_rf))

    cum = cum_var[k-1] if k <= len(cum_var) else 0.0
    print(f"  K={k:3d}  LDA  BAL={np.mean(bals_lda):.4f}±{np.std(bals_lda):.4f} "
          f"AUC={np.nanmean(aucs_lda):.4f}  |  "
          f"RF   BAL={np.mean(bals_rf):.4f}±{np.std(bals_rf):.4f} "
          f"AUC={np.nanmean(aucs_rf):.4f}  "
          f"(var={cum:.1f}%)", flush=True)


# ── save results ───────────────────────────────────────────────────────────────
save_dict = {"N_grid": np.array(N_GRID_eff), "K_sweep": np.array(K_SWEEP_eff)}
for name in clfs:
    tag = name.replace("-", "_")
    save_dict[f"{tag}_bal_mean"] = np.array([np.mean(results[name]["bal"][N]) for N in N_GRID_eff])
    save_dict[f"{tag}_bal_std"]  = np.array([np.std( results[name]["bal"][N]) for N in N_GRID_eff])
    save_dict[f"{tag}_auc_mean"] = np.array([np.nanmean(results[name]["auc"][N]) for N in N_GRID_eff])
    save_dict[f"{tag}_auc_std"]  = np.array([np.nanstd( results[name]["auc"][N]) for N in N_GRID_eff])
for name in ["LDA", "RF-100"]:
    tag = name.replace("-", "_")
    for key in ["bal", "bal_std", "auc", "auc_std"]:
        save_dict[f"sweep_{tag}_{key}"] = np.array(sweep_res[name][key])
np.savez(f"{OUT_DIR}/rf_results.npz", **save_dict)
print("\nSaved rf_results.npz")


# ── plot A: learning curve comparison ─────────────────────────────────────────
N_arr = np.array(N_GRID_eff)
fig, axes = plt.subplots(1, 2, figsize=(15, 6), facecolor="white")
metric_pairs = [("bal", "LOPO Balanced Accuracy", f"BAL-ACC vs N  (K={K_USE})"),
                ("auc", "LOPO AUROC",              f"AUROC vs N  (K={K_USE})")]

for ax, (mkey, ylabel, title) in zip(axes, metric_pairs):
    ax.axhline(0.50, color="gray", ls="--", lw=1.0, alpha=0.6, label="chance")
    # orange capped region
    k_eff_arr = np.array([min(K_USE, 2*N-2) for N in N_GRID_eff])
    capped = k_eff_arr < K_USE
    if capped.any():
        ax.axvspan(N_arr[capped].min()-0.5, N_arr[capped].max()+0.5,
                   alpha=0.08, color="orange", label=f"K capped (<{K_USE})")

    for name, (_, _, col) in clfs.items():
        mn = np.array([np.nanmean(results[name][mkey][N]) for N in N_GRID_eff])
        sd = np.array([np.nanstd( results[name][mkey][N]) for N in N_GRID_eff])
        ax.fill_between(N_arr, mn-sd, mn+sd, alpha=0.18, color=col)
        ax.plot(N_arr, mn, "-o", ms=6, lw=2.0, color=col, label=name)

    ax.set_xlabel("N patients per class", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(N_arr)
    ax.set_ylim(0.30, 0.90)
    ax.legend(fontsize=9, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig.suptitle(f"Learning curve: LDA vs Random Forest  (G-space: 702 sessions, K={K_USE})\n"
             "RF-100: 100 trees, balanced  |  RF-reg: 100 trees, depth≤5, min_leaf=2",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/rf_lc_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved rf_lc_curve.png")


# ── plot B: K-sweep comparison ────────────────────────────────────────────────
K_arr = np.array(K_SWEEP_eff)
cv_at_k = cum_var[K_arr - 1] if K_arr.max() <= len(cum_var) else np.zeros(len(K_arr))

fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")
colors_sweep = {"LDA": "#1565C0", "RF-100": "#2E7D32"}

for ax, mkey, ylabel, title in [
        (axes[0], "bal",  "LOPO BAL-ACC", f"BAL-ACC vs K_LDA  (N={N_FIXED}/class)"),
        (axes[1], "auc",  "LOPO AUROC",   f"AUROC vs K_LDA  (N={N_FIXED}/class)")]:
    ax.axhline(0.50, color="gray", ls="--", lw=1, alpha=0.6)
    for name, col in colors_sweep.items():
        mn  = np.array(sweep_res[name][mkey])
        sd  = np.array(sweep_res[name][f"{mkey}_std"])
        ax.fill_between(K_arr, mn-sd, mn+sd, alpha=0.18, color=col)
        ax.plot(K_arr, mn, "-o", ms=7, lw=2.5, color=col, label=name)
    # variance annotations
    for ki, (k, cv) in enumerate(zip(K_arr, cv_at_k)):
        ax.text(k, 0.35, f"{cv:.0f}%", ha="center", va="bottom",
                fontsize=7, color="#777")
    ax.set_xlabel("K_LDA / K_RF  (number of PCs)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(K_arr)
    ax.set_ylim(0.33, 0.82)
    ax.legend(fontsize=9, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    f"K-sweep: LDA vs RF  (N={N_FIXED}/class, 30 reps)\n"
    "Bottom annotations: % variance explained",
    fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/rf_klda_sweep.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved rf_klda_sweep.png")
print("\nDone.")
