"""
perturbation_sim_refit_v2.py
============================
Extends v1 with:
  1. Single-site (top-1) perturbation type
  2. Four (sigma, spectral_radius) configs:
       (0.025, 0.95)  baseline
       (0.100, 0.95)  more regularised W
       (0.500, 0.95)  heavily regularised W
       (0.025, 0.85)  smaller spectral radius

Warm-up states cached per patient per sr to avoid redundant TF passes.
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace",
                              line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── constants ──────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"
OUT_DIR     = "."
SIM_MAX_STABLE = 100.0
K_LDA       = 2

ALPHA_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 7.5, 10.0]

CONFIGS = [
    dict(sigma=0.025, sr=0.95, label="s=0.025,sr=0.95"),
    dict(sigma=0.100, sr=0.95, label="s=0.100,sr=0.95"),
    dict(sigma=0.500, sr=0.95, label="s=0.500,sr=0.95"),
    dict(sigma=0.025, sr=0.85, label="s=0.025,sr=0.85"),
]
CFG_COLORS  = ["#1565C0", "#E64A19", "#2E7D32", "#7B1FA2"]
CFG_MARKERS = ["o", "s", "^", "D"]

PERT_TYPES = ["full_w", "top5", "top1"]
PERT_NAMES = {"full_w": "Full-W therapeutic",
              "top5":   "Top-5-site",
              "top1":   "Single-site (top-1)"}

# ── load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, pid_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            pid_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

pid_raw        = np.array(pid_raw)
labels_raw     = np.array(labels_raw)
N_subj         = len(signals)
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_idx = np.where(patient_labels == 0)[0]
ad_idx = np.where(patient_labels == 1)[0]
N_AD   = len(ad_idx)
print(f"  Sessions={N_subj}  Patients={N_patients}  CC={len(cc_idx)}  AD={N_AD}")

# ── population PCA ─────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── LDA helper ─────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = (X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1) + 1e-6*np.eye(X0.shape[1])
        w  = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_   = w
        self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_
    def predict(self, X):
        return np.where(self.transform(X) >= self.thr_, self.classes_[1], self.classes_[0])

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([rng2.choice(c0,n,replace=False), rng2.choice(c1,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

# ── group configs by sr ────────────────────────────────────────────────────────
sr_groups = {}
for cfg in CONFIGS:
    sr_groups.setdefault(cfg["sr"], []).append(cfg)

# main results store
all_results = {}   # cfg_label -> pert_type -> alpha -> dict

# ══════════════════════════════════════════════════════════════════════════════
# Process each sr group (TF pass is shared within a group)
# ══════════════════════════════════════════════════════════════════════════════
for sr_val in sorted(sr_groups.keys()):
    cfgs_in_group = sr_groups[sr_val]
    print(f"\n{'='*70}")
    print(f"sr={sr_val}  sigma values: {[c['sigma'] for c in cfgs_in_group]}")
    print(f"{'='*70}")

    # ── build reservoir ────────────────────────────────────────────────────────
    print(f"  Init reservoir (sr={sr_val}) ...")
    par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
               T=139, dt=0.005, sigma_input=0.01,
               shape=(N_HIDDEN, N_SITES, N_SITES, 139))
    res = RESERVOIRE_SIMPLE(par)
    sr_raw = max(abs(np.linalg.eigvals(res.J)))
    res.J *= sr_val / sr_raw
    print(f"    sr set to {sr_val} (was {sr_raw:.4f})")

    # ── teacher-forced pass (shared for all sigma in this sr group) ────────────
    print(f"  TF pass (sr={sr_val}) ...")
    sess_X, sess_Y, sess_tgt = {}, {}, {}
    for idx in trange(N_subj, desc="    TF"):
        s     = signals[idx]; T_s = s.shape[1]
        tgt   = (s.T @ ev50 @ ev50.T).T
        res.T = T_s; res.reset()
        X_raw = []
        for t in range(T_s - 1):
            res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
            X_raw.append(res.X.copy())
        Xf = np.array(X_raw)[TIMES_SKIP:]
        sess_X[idx]   = Xf
        sess_Y[idx]   = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T
        sess_tgt[idx] = tgt

    pat_tgt = {pid: np.concatenate([sess_tgt[i] for i in patient_sids[pid]], axis=1)
               for pid in unique_pids}
    pat_X_sr = {pid: np.vstack([sess_X[i] for i in patient_sids[pid]])
                for pid in unique_pids}

    # ── cache warm-up hidden states (independent of W/sigma) ──────────────────
    print(f"  Caching warm-up states ({N_patients} patients) ...")
    warmup_X = {}
    for pid in tqdm(unique_pids, desc="    warm-up", leave=False):
        tgt = pat_tgt[pid]; T = tgt.shape[1]
        res.T = T; res.reset()
        for t in range(T - 1):
            res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        warmup_X[pid] = (res.X.copy(), T)

    # ── per-sigma configs ──────────────────────────────────────────────────────
    for cfg in cfgs_in_group:
        sigma = cfg["sigma"]; lbl = cfg["label"]
        print(f"\n  --- {lbl} ---")

        pat_X = pat_X_sr   # same TF pass → same X space

        # fit per-patient W
        print(f"    Fitting W (sigma={sigma}) ...")
        rng_p = np.random.default_rng(RNG_SEED + 1)
        pat_W = {}
        for pid in tqdm(unique_pids, desc="    W-fit", leave=False):
            Xc    = pat_X[pid]
            Yc    = np.vstack([sess_Y[i] for i in patient_sids[pid]])
            noise = rng_p.normal(0, sigma, Xc.shape)
            pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

        # G-scores
        print("    G-scores ...")
        Wproj_list = []
        for pid in unique_pids:
            W_T = pat_W[pid].T.astype(np.float64)
            Xc  = pat_X[pid].astype(np.float64)
            _, sx, Vtx = np.linalg.svd(Xc, full_matrices=False)
            kk  = min(K_PC, int((sx > 1e-8).sum()))
            Vt_k = Vtx[:kk]
            Wproj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())
        Wstack   = np.array(Wproj_list)
        Wmean    = Wstack.mean(0)
        Wcent    = Wstack - Wmean
        _, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
        Meff     = N_patients - 1
        G_pat    = Wcent @ Vsvd[:Meff].T

        # capture current config's projection params in defaults
        def w_to_g(W, Xc, _Wm=Wmean, _Vs=Vsvd, _M=Meff):
            W_T = W.T.astype(np.float64)
            Xca = Xc.astype(np.float64)
            _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
            kk   = min(K_PC, int((sx > 1e-8).sum()))
            Vt_k = Vtx[:kk]
            wp   = (W_T @ Vt_k.T @ Vt_k).flatten()
            return ((wp - _Wm) @ _Vs[:_M].T)[:K_LDA]

        # LDA
        Xlda, ylda = _balance(G_pat[:, :K_LDA], patient_labels, seed=RNG_SEED)
        lda = _LDA().fit(Xlda, ylda)
        Z0  = lda.transform(G_pat[:, :K_LDA])
        if Z0[patient_labels==0].mean() > Z0[patient_labels==1].mean():
            lda.w_ *= -1; lda.thr_ *= -1
        Z0  = lda.transform(G_pat[:, :K_LDA])
        auc0 = roc_auc_score(patient_labels, Z0)
        print(f"    baseline AUROC={auc0:.4f}  "
              f"CC={Z0[patient_labels==0].mean():.3f}  "
              f"AD={Z0[patient_labels==1].mean():.3f}")

        # CC mean W and CC mean FC
        W_cc_mean = np.mean([pat_W[unique_pids[i]] for i in cc_idx], axis=0)

        print("    CC mean FC (closed-loop) ...")
        fc_cc_list = []
        for i in tqdm(cc_idx, desc="    CC-FC", leave=False):
            pid   = unique_pids[i]
            Xw, T = warmup_X[pid]
            res.T = T; res.X = Xw.copy()
            res.Jout = pat_W[pid].T.copy(); res.y = res.Jout @ res.X
            Ys = []
            for t in range(T - 1):
                res.step_rate(ff * res.y, sigma_dyn=0.)
                Ys.append(res.y.copy())
            Ys = np.array(Ys)[TIMES_SKIP:].T
            fc_cc_list.append(np.nan_to_num(np.corrcoef(Ys)).flatten())
        FC_cc_mean = np.mean(fc_cc_list, axis=0)

        print("    Baseline FC-r (all patients) ...")
        pat_fc_base = {}
        for pid in tqdm(unique_pids, desc="    base-FC", leave=False):
            Xw, T = warmup_X[pid]
            res.T = T; res.X = Xw.copy()
            res.Jout = pat_W[pid].T.copy(); res.y = res.Jout @ res.X
            Ys = []
            for t in range(T - 1):
                res.step_rate(ff * res.y, sigma_dyn=0.)
                Ys.append(res.y.copy())
            Ys = np.array(Ys)[TIMES_SKIP:].T
            FC = np.nan_to_num(np.corrcoef(Ys))
            pat_fc_base[pid] = float(np.corrcoef(FC.flatten(), FC_cc_mean)[0, 1])

        # sim+refit closure (captures current config state via defaults)
        rng_sim = np.random.default_rng(RNG_SEED + 2)

        def run_sim_refit(W_int, pid,
                          _sig=sigma, _FC=FC_cc_mean,
                          _pX=pat_X, _wtg=w_to_g, _rng=rng_sim):
            Xw, T = warmup_X[pid]
            res.T  = T; res.X = Xw.copy()
            res.Jout = W_int.T.copy()
            res.y    = res.Jout @ res.X
            Ys_list  = []
            for t in range(T - 1):
                res.step_rate(ff * res.y, sigma_dyn=0.)
                Ys_list.append(res.y.copy())
            Ysim = np.array(Ys_list).T           # (N_sites, T-1)
            if not np.isfinite(Ysim).all() or np.abs(Ysim).max() > SIM_MAX_STABLE:
                return None, None, False
            Yeff = Ysim[:, TIMES_SKIP:]
            fc_r = float(np.corrcoef(
                np.nan_to_num(np.corrcoef(Yeff)).flatten(), _FC)[0, 1])
            # re-teacher-force on Y_sim → X_aut, Y_aut
            T2 = Ysim.shape[1]
            res.T = T2; res.reset()
            Xaut = []
            for t in range(T2 - 1):
                res.step_rate(ff * Ysim[:, t], sigma_dyn=0.)
                Xaut.append(res.X.copy())
            Xaut = np.array(Xaut)[TIMES_SKIP:]
            Yaut = Ysim[:, TIMES_SKIP:TIMES_SKIP+len(Xaut)].T
            noise    = _rng.normal(0, _sig, Xaut.shape)
            W_fitted = np.linalg.pinv(Xaut + noise) @ Yaut
            g_new    = _wtg(W_fitted, _pX[pid])
            return g_new, fc_r, True

        # ── main loop over pert_types × alpha ──────────────────────────────────
        cfg_res = {pt: {} for pt in PERT_TYPES}

        for pert_type in PERT_TYPES:
            print(f"\n    [{PERT_NAMES[pert_type]}]")
            for alpha in ALPHA_GRID:
                g_list, n_stable = [], 0

                for pid_i, pid in enumerate(unique_pids):
                    is_ad   = (patient_labels[pid_i] == 1)
                    Wp      = pat_W[pid]

                    if not is_ad or alpha == 0.0:
                        g_list.append(G_pat[pid_i, :K_LDA])
                        continue

                    # build W_int for this pert type
                    if pert_type == "full_w":
                        W_int = (1-alpha)*Wp + alpha*W_cc_mean
                    else:
                        dW    = W_cc_mean - Wp
                        norms = np.linalg.norm(dW, axis=0)
                        k     = 5 if pert_type == "top5" else 1
                        top_k = np.argsort(norms)[::-1][:k]
                        W_int = Wp.copy()
                        W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]

                    g_new, _, stable = run_sim_refit(W_int, pid)
                    if stable:
                        g_list.append(g_new); n_stable += 1
                    else:
                        g_list.append(w_to_g(W_int, pat_X[pid]))

                G_new   = np.array(g_list)
                z_new   = lda.transform(G_new)
                auc_val = roc_auc_score(patient_labels, z_new)
                cfg_res[pert_type][alpha] = dict(
                    auc=auc_val, n_stable=n_stable,
                    z=z_new,
                    ad_lda=float(z_new[patient_labels==1].mean()),
                    cc_lda=float(z_new[patient_labels==0].mean()),
                )
                print(f"      alpha={alpha:5.2f}  stable={n_stable:3}/{N_AD}  "
                      f"AUROC={auc_val:.4f}  AD_LDA={z_new[patient_labels==1].mean():.3f}")

        all_results[lbl] = cfg_res
        print(f"\n  Config {lbl} complete.")

# ═══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════════════════════════════
print("\nPlotting ...")
cfg_labels = [c["label"] for c in CONFIGS]

def _plot_panel(ax, pert_type, metric, ylabel, ylim, legend_loc="best"):
    for ci, (lbl, col, mk) in enumerate(zip(cfg_labels, CFG_COLORS, CFG_MARKERS)):
        if lbl not in all_results: continue
        rpt = all_results[lbl][pert_type]
        alphas = sorted(rpt.keys())
        if metric == "auc":
            vals = [rpt[a]["auc"]        for a in alphas]
        else:
            vals = [rpt[a]["n_stable"] / N_AD for a in alphas if a > 0]
            alphas = [a for a in alphas if a > 0]
        ax.plot(alphas, vals, f"-{mk}", color=col, lw=2, ms=5, label=lbl)
    ax.axhline(0.5, color="gray", ls=":", lw=1.2)
    ax.set_title(PERT_NAMES[pert_type], fontsize=11)
    ax.set_xlabel("Dose α", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(*ylim)
    ax.legend(frameon=True, framealpha=0.9, fontsize=7, loc=legend_loc)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

# ── combined 2×3 figure ────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 9), facecolor="white")

for j, pt in enumerate(PERT_TYPES):
    _plot_panel(axes[0, j], pt, "auc",      "AUROC",                 (0.0, 1.05))
    _plot_panel(axes[1, j], pt, "stability","Stable AD sim fraction", (-0.05, 1.10),
                legend_loc="lower right")

axes[0, 0].set_title("Full-W therapeutic", fontsize=12, fontweight="bold")
axes[0, 1].set_title("Top-5-site",         fontsize=12, fontweight="bold")
axes[0, 2].set_title("Single-site (top-1)",fontsize=12, fontweight="bold")
for j in range(3): axes[0, j].axhline(0.5, color="gray", ls=":", lw=1.2)

fig.suptitle("sim+re-fit: AUROC (top) and stability (bottom) vs α\n"
             "Comparing sigma and spectral-radius configurations",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefitv2_combined.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefitv2_combined.png")

# ── separate AUROC plot ────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white", sharey=True)
for j, pt in enumerate(PERT_TYPES):
    _plot_panel(axes[j], pt, "auc", "AUROC", (0.0, 1.05))
fig.suptitle("AUROC vs α — all configs and perturbation types",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefitv2_auroc.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefitv2_auroc.png")

# ── stability plot ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor="white", sharey=True)
for j, pt in enumerate(PERT_TYPES):
    _plot_panel(axes[j], pt, "stability", "Stable AD fraction",
                (-0.05, 1.10), legend_loc="lower right")
    axes[j].axhline(1.0, color="gray", ls=":", lw=1.2)
    axes[j].axhline(0.5, color="gray", ls="--", lw=0.8, alpha=0.5)
fig.suptitle("Stable simulation fraction vs α — all configs",
             fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefitv2_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefitv2_stability.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*76)
print(f"{'Config':<22}  {'PertType':<10}  "
      f"{'Best-α':>6}  {'AUROC':>6}  {'Stable':>9}  {'AD_LDA':>7}")
print("-"*76)
for lbl in cfg_labels:
    if lbl not in all_results: continue
    for pt in PERT_TYPES:
        rpt = all_results[lbl][pt]
        best_a = max(rpt.keys(), key=lambda a: rpt[a]["auc"])
        br = rpt[best_a]
        print(f"  {lbl:<20}  {pt:<10}  {best_a:>6.2f}  "
              f"{br['auc']:>6.4f}  {br['n_stable']:>4}/{N_AD}  "
              f"{br['ad_lda']:>7.3f}")
print("="*76)

# also print full dose-response for each config/pert_type
print("\nFull dose-response:")
for lbl in cfg_labels:
    if lbl not in all_results: continue
    print(f"\n  {lbl}")
    for pt in PERT_TYPES:
        print(f"    {PERT_NAMES[pt]}")
        rpt = all_results[lbl][pt]
        for a in sorted(rpt.keys()):
            r = rpt[a]
            print(f"      α={a:5.2f}  stable={r['n_stable']:3}/{N_AD}  "
                  f"AUROC={r['auc']:.4f}  AD_LDA={r['ad_lda']:.3f}")

print("\nDone. Saved: simrefitv2_combined.png  simrefitv2_auroc.png  simrefitv2_stability.png")
