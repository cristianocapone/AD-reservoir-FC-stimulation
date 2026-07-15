"""
pert_fclag_stimulation.py
=========================
Redoes the W-matrix stimulation experiment using TWO classifiers in parallel:

  1. G-space LDA (K=25) — same as original pertB_direct_data.npz
  2. FC-lag LDA (lags 0-2, K=25) — NEW: computed from predicted time series
                                    Y = W.T @ X_res

Both classifiers are trained on the SAME data (reservoir-predicted signals),
so they are directly comparable.  The FC-lag LDA uses lags 0, 1, 2 of the
Pearson cross-correlation matrix of Y, giving 7260 + 14641 + 14641 = 36542
features per patient, then Gram-matrix SVD → LDA.

Perturbation types:  full_w, top5, top1
Alpha grids match top5_alpha20_physio.py

Outputs:
  pert_fclag_data.npz          — both classifier scores at all (type, alpha, patient)
  pert_fclag_figure.png        — 3-row x 3-col comparison figure (300 DPI)
  paper_figures/figure4_fclag.png — paper-quality 2x2 (top5 perturbation, both
                                    classifiers in each panel)
"""
import os, sys, warnings, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from tqdm import trange, tqdm
import warnings as _w; _w.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── params ─────────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
SIGMA       = 0.05
K_LDA       = 25
SR          = 0.95
MAX_LAG_FL  = 2        # lags 0, 1, 2  for FC-lag LDA
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

LAG0_D  = N_SITES * (N_SITES - 1) // 2   # 7260
LAGK_D  = N_SITES ** 2                    # 14641
FCLAG_D = LAG0_D + MAX_LAG_FL * LAGK_D   # 36542

ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]),
    "top1":   np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
}

# ── data loading ───────────────────────────────────────────────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, pid_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)),
                                replace=False))
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
cc_pids = [unique_pids[i] for i in np.where(patient_labels == 0)[0]]
ad_pids = [unique_pids[i] for i in np.where(patient_labels == 1)[0]]
n_ad = len(ad_pids)
print(f"  {N_patients} patients ({len(cc_pids)} CC, {n_ad} AD), {N_subj} sessions")

# ── PCA ────────────────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals_pca, evecs_pca = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs_pca[:, np.argsort(evals_pca)[::-1]][:, :N_PC_MODEL]

# ── reservoir ──────────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

# ── TF pass ────────────────────────────────────────────────────────────────────
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

# ── W fitting ──────────────────────────────────────────────────────────────────
print("W fitting ...")
rng_w    = np.random.default_rng(RNG_SEED + 1)
first_idx     = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single   = {pid: sess_X[first_idx[pid]] for pid in unique_pids}
patY_single   = {pid: sess_Y[first_idx[pid]] for pid in unique_pids}

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc   # (N_res, N_sites)

W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

# ── helper: lagged correlation ─────────────────────────────────────────────────
def lagged_corrcoef(S, lag):
    """S: (T, N_sites). Returns (N_sites, N_sites)."""
    if lag == 0:
        return np.corrcoef(S.T)
    T = S.shape[0]
    A = S[:T-lag].astype(np.float64)
    B = S[lag:].astype(np.float64)
    A -= A.mean(0); B -= B.mean(0)
    A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - lag)

def compute_fclag_feat(W, pid):
    """Lags 0-2 FC features from predicted signal Y = W.T @ X_res."""
    Xc = patX_single[pid].astype(np.float64)  # (T_eff, N_res)
    Y  = W.T.astype(np.float64) @ Xc.T        # (N_sites, T_eff)
    S  = Y.T                                    # (T_eff, N_sites)
    feats = []
    for lag in range(MAX_LAG_FL + 1):
        fc = lagged_corrcoef(S, lag)
        fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
        if lag == 0:
            feats.append(fc[np.triu_indices(N_SITES, k=1)])
        else:
            feats.append(fc.flatten())
    return np.concatenate(feats)

# ── G-space (same as original) ─────────────────────────────────────────────────
print("Building G-space + LDA ...")
pat_Vtk = {}
for pid in tqdm(unique_pids, desc="  SVD-G", leave=False):
    Xca = patX_single[pid].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    pat_Vtk[pid] = Vtx[:kk]

def project_W(W, pid):
    Vt_k = pat_Vtk[pid]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

Wproj_B = np.array([project_W(pat_W[pid], pid) for pid in unique_pids])
Wmean_B = Wproj_B.mean(0)
Wcent_B = Wproj_B - Wmean_B
_, _, Vsvd_B = np.linalg.svd(Wcent_B, full_matrices=False)
Meff_B  = N_patients - 1
G_B     = Wcent_B @ Vsvd_B[:Meff_B].T

class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_   = w
        self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n   = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i, n, replace=False),
                          rng2.choice(c1i, n, replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

# G-space LDA (K=25)
Xlda_g, ylda_g = _balance(G_B[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_g = _LDA().fit(Xlda_g, ylda_g)
Z_g   = lda_g.transform(G_B[:, :K_LDA])
if Z_g[patient_labels==0].mean() > Z_g[patient_labels==1].mean():
    lda_g.w_ *= -1; lda_g.thr_ *= -1
    Z_g = lda_g.transform(G_B[:, :K_LDA])
cc_lda_g = Z_g[patient_labels==0]
ad_lda_g = Z_g[patient_labels==1]
thr_g    = 0.5*(cc_lda_g.mean() + ad_lda_g.mean())
print(f"  G-space: CC={cc_lda_g.mean():.3f}±{cc_lda_g.std():.3f}  "
      f"AD={ad_lda_g.mean():.3f}±{ad_lda_g.std():.3f}  thr={thr_g:.3f}")

def w_to_glda(W, pid):
    wp = project_W(W, pid)
    g  = ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]
    return float(lda_g.transform(g.reshape(1, -1))[0])

# ── FC-lag SVD + LDA (trained on predicted signals) ────────────────────────────
print("Computing baseline FC-lag features (predicted signals) ...")
t0 = time.time()
fclag_base = np.array([compute_fclag_feat(pat_W[pid], pid)
                        for pid in tqdm(unique_pids, desc="  FC-lag", leave=False)])
print(f"  done in {time.time()-t0:.1f}s  shape={fclag_base.shape}")

fclag_mean = fclag_base.mean(0)
fclag_c    = fclag_base - fclag_mean              # (N_patients, FCLAG_D)

# Gram matrix SVD
print("  Gram SVD ...")
C_fl   = fclag_c @ fclag_c.T                      # (N_patients, N_patients)
ev_fl, evec_fl = np.linalg.eigh(C_fl)
ord_fl  = np.argsort(ev_fl)[::-1]
ev_fl   = np.maximum(ev_fl[ord_fl], 0)
evec_fl = evec_fl[:, ord_fl]
G_fl    = evec_fl * np.sqrt(ev_fl)                # (N_patients, N_patients)

cum_fl = np.cumsum(ev_fl) / ev_fl.sum() * 100
print(f"  Variance explained: K=25 → {cum_fl[24]:.1f}%")

# FC-lag LDA (K=25, balanced)
Xlda_fl, ylda_fl = _balance(G_fl[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_fl = _LDA().fit(Xlda_fl, ylda_fl)
Z_fl   = lda_fl.transform(G_fl[:, :K_LDA])
if Z_fl[patient_labels==0].mean() > Z_fl[patient_labels==1].mean():
    lda_fl.w_ *= -1; lda_fl.thr_ *= -1
    Z_fl = lda_fl.transform(G_fl[:, :K_LDA])
cc_lda_fl = Z_fl[patient_labels==0]
ad_lda_fl = Z_fl[patient_labels==1]
thr_fl    = 0.5*(cc_lda_fl.mean() + ad_lda_fl.mean())
print(f"  FC-lag: CC={cc_lda_fl.mean():.3f}±{cc_lda_fl.std():.3f}  "
      f"AD={ad_lda_fl.mean():.3f}±{ad_lda_fl.std():.3f}  thr={thr_fl:.3f}")

def w_to_fclda(W_int, pid):
    """FC-lag LDA score for perturbed W_int."""
    feat = compute_fclag_feat(W_int, pid)
    feat_c = feat - fclag_mean
    cross  = feat_c @ fclag_c.T                   # (N_patients,)
    g      = (cross @ evec_fl) / (np.sqrt(ev_fl) + 1e-12)  # (N_patients,)
    return float(lda_fl.transform(g[:K_LDA].reshape(1, -1))[0])

# ── also compute FC reference for diagnostics ──────────────────────────────────
fc_cc_list = []
for pid in cc_pids:
    Xc = patX_single[pid].astype(np.float64)
    Y  = (pat_W[pid].T @ Xc.T)
    fc_cc_list.append(np.nan_to_num(np.corrcoef(Y)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)

def w_fc_r(W, pid):
    Xc = patX_single[pid].astype(np.float64)
    Y  = (W.T.astype(np.float64) @ Xc.T)
    fc = np.nan_to_num(np.corrcoef(Y)).flatten()
    return float(np.corrcoef(fc, FC_cc_mean)[0, 1])

cc_fc_r      = np.array([w_fc_r(pat_W[pid], pid) for pid in cc_pids])
ad_fc_r_base = np.array([w_fc_r(pat_W[pid], pid) for pid in ad_pids])

# ── perturbation loop ──────────────────────────────────────────────────────────
print("\nPerturbation loop ...")
results = {}  # results[pert_type][ai][pid] = (g_lda, fl_lda, fc_r)

for pert_type, alphas in ALPHA_GRIDS.items():
    print(f"\n  [{pert_type}]  alpha in [{alphas.min():.2g}, {alphas.max():.2g}]"
          f"  ({len(alphas)} points)")
    results[pert_type] = []
    for ai, alpha in enumerate(alphas):
        alpha_res = {}
        for pid in ad_pids:
            Wp = pat_W[pid]
            dW = W_cc_mean - Wp
            norms = np.linalg.norm(dW, axis=0)

            if pert_type == "full_w":
                W_int = (1 - alpha) * Wp + alpha * W_cc_mean
            elif pert_type == "top5":
                top_k = np.argsort(norms)[::-1][:5]
                W_int = Wp.copy()
                W_int[:, top_k] = ((1-alpha)*Wp[:, top_k]
                                   + alpha*W_cc_mean[:, top_k])
            else:   # top1
                top_k = np.argsort(norms)[::-1][:1]
                W_int = Wp.copy()
                W_int[:, top_k] = ((1-alpha)*Wp[:, top_k]
                                   + alpha*W_cc_mean[:, top_k])

            g_lda  = w_to_glda(W_int, pid)
            fl_lda = w_to_fclda(W_int, pid)
            fc_r_v = w_fc_r(W_int, pid)
            alpha_res[pid] = (g_lda, fl_lda, fc_r_v)

        results[pert_type].append(alpha_res)
        g_vals  = [alpha_res[p][0] for p in ad_pids]
        fl_vals = [alpha_res[p][1] for p in ad_pids]
        print(f"    alpha={alpha:5.2f}  G-LDA={np.mean(g_vals):+.3f}  "
              f"FL-LDA={np.mean(fl_vals):+.3f}", flush=True)

# ── save ───────────────────────────────────────────────────────────────────────
print("\nSaving ...")
save_dict = {
    "ad_pids"         : np.array(ad_pids),
    "cc_pids"         : np.array(cc_pids),
    "patient_labels"  : patient_labels,
    # G-space baseline
    "cc_lda_g"        : cc_lda_g,
    "ad_lda_g"        : ad_lda_g,
    # FC-lag baseline
    "cc_lda_fl"       : cc_lda_fl,
    "ad_lda_fl"       : ad_lda_fl,
    # FC reference
    "cc_fc_r"         : cc_fc_r,
    "ad_fc_r_base"    : ad_fc_r_base,
}
for pert_type, alphas in ALPHA_GRIDS.items():
    ad = results[pert_type]
    g_arr  = np.array([[ad[ai][pid][0] for pid in ad_pids]
                        for ai in range(len(alphas))])
    fl_arr = np.array([[ad[ai][pid][1] for pid in ad_pids]
                        for ai in range(len(alphas))])
    fcr_arr= np.array([[ad[ai][pid][2] for pid in ad_pids]
                        for ai in range(len(alphas))])
    save_dict[f"{pert_type}_alphas"]  = alphas
    save_dict[f"{pert_type}_glda"]    = g_arr
    save_dict[f"{pert_type}_fllda"]   = fl_arr
    save_dict[f"{pert_type}_fcr"]     = fcr_arr

np.savez(f"{OUT_DIR}/pert_fclag_data.npz", **save_dict)
print("Saved pert_fclag_data.npz")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURES
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

CC_COL  = "#1565C0"
AD_COL  = "#C62828"
FW_COL  = "#1B5E20"
T5_COL  = "#E65100"
T1_COL  = "#6A1B9A"
GCOL    = "#37474F"    # G-space LDA color
FLCOL   = "#0097A7"    # FC-lag LDA color
PHYSIO  = 4.0          # physiological limit for top5

PERT_COLORS = {"full_w": FW_COL, "top5": T5_COL, "top1": T1_COL}
PERT_LABELS = {"full_w": "Full-W (121 sites)", "top5": "Top-5 sites",
               "top1": "Top-1 site"}

# ── helper: add physio boundary for top-5 ─────────────────────────────────────
def shade_physio(ax, x_max=20, top5_only=True):
    ax.axvspan(PHYSIO, x_max, alpha=0.06, color="#B71C1C", zorder=0)
    ax.axvline(PHYSIO, color="#B71C1C", lw=1.0, ls=":", alpha=0.55)

# ── Figure 1: Full diagnostic (3×3) ───────────────────────────────────────────
fig = plt.figure(figsize=(16, 13), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.54, wspace=0.40)

t5_alphas = ALPHA_GRIDS["top5"]
t5_g  = np.array([[results["top5"][ai][pid][0] for pid in ad_pids]
                    for ai in range(len(t5_alphas))])
t5_fl = np.array([[results["top5"][ai][pid][1] for pid in ad_pids]
                    for ai in range(len(t5_alphas))])

# Row 0, Col 0: Baseline distributions
ax = fig.add_subplot(gs[0, 0])
for xi, (vals_g, vals_fl, lbl, col) in enumerate(
        [(cc_lda_g, cc_lda_fl, "CC", CC_COL),
         (ad_lda_g, ad_lda_fl, "AD", AD_COL)]):
    for vals, offset, marker in [(vals_g, -0.12, "o"), (vals_fl, +0.12, "s")]:
        ax.scatter(np.full(len(vals), xi) + offset
                   + np.random.default_rng(xi).uniform(-0.06,0.06,len(vals)),
                   vals, s=12, c=col, alpha=0.55, marker=marker,
                   edgecolors="white", linewidths=0.3, zorder=3)
    ax.errorbar([xi-0.12, xi+0.12],
                [vals_g.mean(), vals_fl.mean()],
                [vals_g.std(),  vals_fl.std()],
                fmt="none", color=col, lw=2.0, capsize=4, zorder=4)

ax.axhline(thr_g,  color=GCOL, ls="--", lw=1.0, alpha=0.7)
ax.axhline(thr_fl, color=FLCOL, ls="--", lw=1.0, alpha=0.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["CC", "AD"])
ax.set_ylabel("LDA score")
ax.set_title("Baseline scores\n(circles=G-space, squares=FC-lag)")
legend_els = [Line2D([0],[0],marker='o',color='gray',label='G-space LDA',ls='none',ms=7),
              Line2D([0],[0],marker='s',color='gray',label='FC-lag LDA (lags 0-2)',ls='none',ms=7)]
ax.legend(handles=legend_els, frameon=False, fontsize=7)

# Row 0, Col 1 & 2: Dose-response for both classifiers (top-5 only)
for col_idx, (scores, cc_lda, ad_lda, thr, label, color) in enumerate([
        (t5_g,  cc_lda_g, ad_lda_g, thr_g,  "G-space LDA",    GCOL),
        (t5_fl, cc_lda_fl, ad_lda_fl, thr_fl, "FC-lag LDA 0-2", FLCOL),
]):
    ax = fig.add_subplot(gs[0, col_idx+1])
    shade_physio(ax)
    ax.axhspan(cc_lda.mean()-cc_lda.std(), cc_lda.mean()+cc_lda.std(),
               alpha=0.15, color=CC_COL)
    ax.axhline(cc_lda.mean(), color=CC_COL, lw=1.5, ls="--", alpha=0.9,
               label="CC mean ±1σ")
    ax.axhline(ad_lda.mean(), color=AD_COL, lw=1.2, ls=":", alpha=0.6,
               label="AD baseline")
    ax.axhline(thr, color="gray", lw=0.8, ls="-.", alpha=0.5,
               label=f"Midpoint = {thr:.2f}")
    for pi in range(n_ad):
        ax.plot(t5_alphas, scores[:, pi], lw=0.5, color=color, alpha=0.15)
    m = scores.mean(1); s = scores.std(1)
    ax.fill_between(t5_alphas, m-s, m+s, alpha=0.22, color=color)
    ax.plot(t5_alphas, m, "-o", ms=4, lw=2.0, color=color,
            label=f"AD mean ±1σ  (n={n_ad})")
    ax.set_xlabel("alpha")
    ax.set_ylabel("LDA score")
    ax.set_title(f"Top-5 trajectories — {label}")
    ax.legend(frameon=False, fontsize=7)

# Row 1: Dose-response (mean ± std, all 3 perturbation types) for both classifiers
for row_idx, (data_key, cc_lda, ad_lda, thr, label, color) in enumerate([
        ("glda",  cc_lda_g,  ad_lda_g,  thr_g,  "G-space LDA",    GCOL),
        ("fllda", cc_lda_fl, ad_lda_fl, thr_fl, "FC-lag LDA 0-2", FLCOL),
]):
    ax = fig.add_subplot(gs[1, row_idx+1])
    ax.axhspan(cc_lda.mean()-cc_lda.std(), cc_lda.mean()+cc_lda.std(),
               alpha=0.15, color=CC_COL)
    ax.axhline(cc_lda.mean(), color=CC_COL, lw=1.5, ls="--", alpha=0.9,
               label="CC mean ±1σ")
    ax.axhline(ad_lda.mean(), color=AD_COL, lw=1.2, ls=":", alpha=0.6,
               label="AD baseline")
    ax.axhline(thr, color="gray", lw=0.8, ls="-.", alpha=0.5,
               label="Midpoint")
    for pert_type, pcol, plbl in [("full_w", FW_COL, "Full-W"),
                                   ("top5",   T5_COL, "Top-5"),
                                   ("top1",   T1_COL, "Top-1")]:
        alphas  = ALPHA_GRIDS[pert_type]
        mat     = np.array([[results[pert_type][ai][pid][row_idx]
                              for pid in ad_pids]
                             for ai in range(len(alphas))])
        m = mat.mean(1); s = mat.std(1)
        ax.fill_between(alphas, m-s, m+s, alpha=0.15, color=pcol)
        ax.plot(alphas, m, "-o", ms=4, lw=2.0, color=pcol, label=plbl)
    ax.set_xlabel("alpha")
    ax.set_ylabel("LDA score (mean ±1σ, 40 AD)")
    ax.set_title(f"Dose-response (all strategies)\n{label}")
    ax.legend(frameon=False, fontsize=7)

# Row 1, Col 0: Reclassification rate comparison
ax = fig.add_subplot(gs[1, 0])
for data_key, thr, label, lc, ms_ in [
        ("glda",  thr_g,  "G-space LDA",    GCOL,  "o"),
        ("fllda", thr_fl, "FC-lag LDA 0-2", FLCOL, "s"),
]:
    for pert_type, pcol in [("full_w", FW_COL), ("top5", T5_COL), ("top1", T1_COL)]:
        alphas = ALPHA_GRIDS[pert_type]
        mat    = np.array([[results[pert_type][ai][pid][0 if data_key=="glda" else 1]
                             for pid in ad_pids]
                            for ai in range(len(alphas))])
        frac = (mat < thr).mean(1) * 100
        ls_ = "-" if data_key == "glda" else "--"
        ax.plot(alphas, frac, marker=ms_, ms=3.5, lw=1.5, color=pcol, ls=ls_,
                label=f"{label[:3]}/{PERT_LABELS[pert_type][:5]}")
ax.axhline(50, color="gray", ls="--", lw=1.0, alpha=0.5)
ax.set_xlabel("alpha"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification rate\n(solid=G-space, dashed=FC-lag)")
ax.set_ylim(-2, 105)
# Minimal legend: just show classifier type
from matplotlib.lines import Line2D as L2D
ax.legend(handles=[L2D([],[],ls="-",color=GCOL,label="G-space LDA"),
                    L2D([],[],ls="--",color=FLCOL,label="FC-lag LDA"),
                    L2D([],[],color=FW_COL,label="Full-W"),
                    L2D([],[],color=T5_COL,label="Top-5"),
                    L2D([],[],color=T1_COL,label="Top-1")],
           frameon=False, fontsize=6.5, ncol=2)

# Row 2: Side-by-side reclassification for each pert type (clean comparison)
for col_idx, pert_type in enumerate(["full_w", "top5", "top1"]):
    ax = fig.add_subplot(gs[2, col_idx])
    alphas = ALPHA_GRIDS[pert_type]
    for data_key, thr, label, color, ms_ in [
            ("glda",  thr_g,  "G-space LDA",    GCOL,  "o"),
            ("fllda", thr_fl, "FC-lag LDA 0-2", FLCOL, "s"),
    ]:
        mat  = np.array([[results[pert_type][ai][pid][0 if data_key=="glda" else 1]
                           for pid in ad_pids]
                          for ai in range(len(alphas))])
        frac = (mat < thr).mean(1) * 100
        ax.plot(alphas, frac, marker=ms_, ms=4.5, lw=2.0, color=color, label=label)
    ax.axhline(50, color="gray", ls="--", lw=1.0, alpha=0.5)
    if pert_type == "top5":
        shade_physio(ax)
    ax.set_xlabel("alpha")
    ax.set_ylabel("AD reclassified (%)")
    ax.set_title(f"{PERT_LABELS[pert_type]}")
    ax.set_ylim(-2, 105)
    ax.legend(frameon=False, fontsize=7)

for ax, lbl in zip(fig.axes, list("ABCDEFGHI")):
    ax.text(-0.14, 1.05, lbl, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="bottom", ha="left")

fig.suptitle(
    "W-matrix stimulation — G-space LDA vs FC-lag LDA (lags 0-2, K=25)\n"
    "W_int = (1-alpha)*W_AD + alpha*W_CC_mean  |  Both classifiers trained on "
    "predicted signals Y = W.T @ X_res",
    fontsize=9, y=1.01)
fig.savefig(f"{OUT_DIR}/pert_fclag_figure.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved pert_fclag_figure.png")

# ── Paper figure: 2×2, top-5 perturbation, both classifiers ───────────────────
fig2 = plt.figure(figsize=(9.5, 7.5), facecolor="white")
gs2  = gridspec.GridSpec(2, 2, figure=fig2, hspace=0.50, wspace=0.42)

# A: Baseline distributions (violin-style)
ax_a = fig2.add_subplot(gs2[0, 0])
for xi, (label, lbl_str) in enumerate([(0, "CC"), (1, "AD")]):
    col = CC_COL if label == 0 else AD_COL
    for vals, offset, marker, mlbl in [
            (Z_g[patient_labels==label],   -0.13, "o", "G-space"),
            (Z_fl[patient_labels==label],  +0.13, "s", "FC-lag"),
    ]:
        jit = np.random.default_rng(xi*13+label).uniform(-0.08, 0.08, len(vals))
        ax_a.scatter(xi + offset + jit, vals, s=13, c=col, alpha=0.55,
                     marker=marker, edgecolors="white", linewidths=0.3, zorder=3)
        ax_a.errorbar([xi+offset], [vals.mean()], [vals.std()], fmt="none",
                      color=col, lw=2.2, capsize=5, zorder=5)

ax_a.axhline(thr_g,  color=GCOL,  ls="--", lw=1.0, alpha=0.7, label=f"G thr={thr_g:.2f}")
ax_a.axhline(thr_fl, color=FLCOL, ls="--", lw=1.0, alpha=0.7, label=f"FC thr={thr_fl:.2f}")
ax_a.set_xticks([0, 1]); ax_a.set_xticklabels(["CC", "AD"])
ax_a.set_ylabel("LDA score (K=25)")
ax_a.set_title("Baseline LDA scores")
ax_a.legend(handles=[
    L2D([],[],marker='o',color='gray',label='G-space',ls='none',ms=7),
    L2D([],[],marker='s',color='gray',label='FC-lag',ls='none',ms=7)],
    frameon=False, fontsize=7)

# B: Dose-response for both classifiers (mean ± std, top-5)
ax_b = fig2.add_subplot(gs2[0, 1])
shade_physio(ax_b)
ax_b.axhline(cc_lda_g.mean(),  color=CC_COL, lw=1.2, ls="--", alpha=0.7,
             label="CC mean (G-space)")
ax_b.axhline(ad_lda_g.mean(),  color=AD_COL, lw=1.0, ls=":", alpha=0.5)
ax_b.axhline(thr_g,  color=GCOL,  lw=0.8, ls="-.", alpha=0.5)
ax_b.axhline(thr_fl, color=FLCOL, lw=0.8, ls="-.", alpha=0.5)
for scores, thr, col, lbl in [
        (t5_g,  thr_g,  GCOL,  "G-space LDA"),
        (t5_fl, thr_fl, FLCOL, "FC-lag LDA (0-2)"),
]:
    m = scores.mean(1); s = scores.std(1)
    ax_b.fill_between(t5_alphas, m-s, m+s, alpha=0.18, color=col)
    ax_b.plot(t5_alphas, m, "-o", ms=4, lw=2.0, color=col, label=lbl)
ax_b.set_xlabel("Perturbation strength  alpha")
ax_b.set_ylabel("LDA score (mean ±1σ)")
ax_b.set_title("Dose-response  (top-5 sites)")
ax_b.legend(frameon=False, fontsize=7)

# C: Individual trajectories (top-5, G-space LDA)
ax_c = fig2.add_subplot(gs2[1, 0])
shade_physio(ax_c)
ax_c.axhspan(cc_lda_g.mean()-cc_lda_g.std(), cc_lda_g.mean()+cc_lda_g.std(),
             alpha=0.15, color=CC_COL)
ax_c.axhline(cc_lda_g.mean(), color=CC_COL, lw=1.5, ls="--", alpha=0.9)
ax_c.axhline(thr_g, color="gray", lw=0.8, ls="-.", alpha=0.5)
for pi in range(n_ad):
    ax_c.plot(t5_alphas, t5_g[:, pi], lw=0.6, color=GCOL, alpha=0.18)
mg = t5_g.mean(1); sg = t5_g.std(1)
ax_c.fill_between(t5_alphas, mg-sg, mg+sg, alpha=0.25, color=GCOL)
ax_c.plot(t5_alphas, mg, "-o", ms=4, lw=2.2, color=GCOL,
          label=f"G-space LDA  (n={n_ad})")
ax_c.set_xlabel("alpha"); ax_c.set_ylabel("G-space LDA score")
ax_c.set_title(f"Per-patient trajectories — G-space")
ax_c.legend(frameon=False, fontsize=7)

# D: Individual trajectories (top-5, FC-lag LDA)
ax_d = fig2.add_subplot(gs2[1, 1])
shade_physio(ax_d)
ax_d.axhspan(cc_lda_fl.mean()-cc_lda_fl.std(), cc_lda_fl.mean()+cc_lda_fl.std(),
             alpha=0.15, color=CC_COL)
ax_d.axhline(cc_lda_fl.mean(), color=CC_COL, lw=1.5, ls="--", alpha=0.9)
ax_d.axhline(thr_fl, color="gray", lw=0.8, ls="-.", alpha=0.5)
for pi in range(n_ad):
    ax_d.plot(t5_alphas, t5_fl[:, pi], lw=0.6, color=FLCOL, alpha=0.18)
mfl = t5_fl.mean(1); sfl = t5_fl.std(1)
ax_d.fill_between(t5_alphas, mfl-sfl, mfl+sfl, alpha=0.25, color=FLCOL)
ax_d.plot(t5_alphas, mfl, "-o", ms=4, lw=2.2, color=FLCOL,
          label=f"FC-lag LDA  (n={n_ad})")
ax_d.set_xlabel("alpha"); ax_d.set_ylabel("FC-lag LDA score")
ax_d.set_title(f"Per-patient trajectories — FC-lag")
ax_d.legend(frameon=False, fontsize=7)

for ax, lbl in zip([ax_a, ax_b, ax_c, ax_d], ["A", "B", "C", "D"]):
    ax.text(-0.14, 1.04, lbl, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="bottom", ha="left")

fig2.suptitle(
    "In-silico stimulation — G-space LDA vs FC-lag LDA (lags 0-2, K=25)\n"
    "W_int = (1-alpha)*W_AD + alpha*W̄_CC  |  Top-5 site perturbation\n"
    "Both classifiers trained on predicted signals  Y = W.T @ X_res",
    fontsize=9, y=1.01)

out_paper = os.path.join("paper_figures", "figure4_fclag.png")
fig2.savefig(out_paper, dpi=300, bbox_inches="tight", facecolor="white")
plt.close()
print(f"Saved {out_paper}")

# ── print summary table ────────────────────────────────────────────────────────
print("\n" + "="*70)
print("RECLASSIFICATION AT PHYSIO LIMIT (top-5, alpha=4)")
print("="*70)
t5_a = ALPHA_GRIDS["top5"]
safe_idx = np.where(t5_a <= PHYSIO)[0][-1]
print(f"  alpha = {t5_a[safe_idx]:.1f}")
g_recl  = (t5_g[safe_idx]  < thr_g).mean()  * 100
fl_recl = (t5_fl[safe_idx] < thr_fl).mean() * 100
print(f"  G-space LDA:  {g_recl:.0f}% reclassified as CC")
print(f"  FC-lag LDA:   {fl_recl:.0f}% reclassified as CC")

print("\nGAP CLOSED (% of AD→CC distance) at alpha=4  (top-5)")
g_gap_closed  = (ad_lda_g.mean()  - t5_g[safe_idx].mean())  / (ad_lda_g.mean()  - cc_lda_g.mean())  * 100
fl_gap_closed = (ad_lda_fl.mean() - t5_fl[safe_idx].mean()) / (ad_lda_fl.mean() - cc_lda_fl.mean()) * 100
print(f"  G-space LDA:  {g_gap_closed:.0f}%")
print(f"  FC-lag LDA:   {fl_gap_closed:.0f}%")
print("\nDone.")
