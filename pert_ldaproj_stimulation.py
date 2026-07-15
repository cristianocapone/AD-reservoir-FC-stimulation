"""
pert_ldaproj_stimulation.py
============================
Compares two perturbation strategies in G-space:

  (A) CURRENT — W_int = (1-a)*W_AD + a*W_CC_mean  (top-5/full-W/top-1)
      Projects the full W_CC-W_AD displacement; most of the G-space
      displacement may be PERPENDICULAR to the LDA direction.

  (B) LDA-PROJECTED — move each AD patient directly along the LDA
      discriminant axis w_ in G-space:
        g_perturbed = g_i + a * delta * w_
      where delta = (CC_mean_score - AD_patient_score) is the LDA-score
      gap for that patient.  This is the minimal perturbation needed to
      cross the boundary, and wastes no energy on perpendicular dimensions.

Key diagnostic: alignment angle between the actual G-space displacement
  dg = G(W_CC_mean) - G(W_AD)  and  the LDA direction w_lda.
  cos(theta) = (dg . w_) / |dg| — if small, the current stimulation is
  inefficient.

Also does the same analysis for FC-lag LDA (lags 0-2, K=25).

Saves:
  pert_ldaproj_data.npz
  pert_ldaproj_figure.png   (paper-ready 300 DPI)
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
MAX_LAG_FL  = 2
TS_ROOT     = "./timeseries"
LAG0_D      = N_SITES * (N_SITES - 1) // 2
LAGK_D      = N_SITES ** 2

# Alpha grid for comparison (top-5 focus)
ALPHAS_T5   = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0])
ALPHAS_FW   = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
ALPHAS_T1   = np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0])
# LDA-projected alpha: 0..1 → 0% to 100% of the AD→CC gap along w_lda
ALPHAS_PROJ = np.linspace(0, 1.0, 21)

PHYSIO  = 4.0
CC_COL  = "#1565C0"
AD_COL  = "#C62828"
T5_COL  = "#E65100"
GCOL    = "#37474F"
FLCOL   = "#0097A7"
PROJ_G  = "#1B5E20"
PROJ_FL = "#6A1B9A"

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE SETUP (same as pert_fclag_stimulation.py)
# ══════════════════════════════════════════════════════════════════════════════
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
n_ad = len(ad_pids); n_cc = len(cc_pids)
print(f"  {N_patients} patients ({n_cc} CC, {n_ad} AD)")

print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals_pca, evecs_pca = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs_pca[:, np.argsort(evals_pca)[::-1]][:, :N_PC_MODEL]

print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

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

print("W fitting ...")
rng_w       = np.random.default_rng(RNG_SEED + 1)
first_idx   = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single = {pid: sess_X[first_idx[pid]] for pid in unique_pids}
patY_single = {pid: sess_Y[first_idx[pid]] for pid in unique_pids}

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc
W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

# ── LDA helper ─────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.mu0_ = mu0; self.mu1_ = mu1
        self.thr_ = 0.5*(mu0@w + mu1@w)
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i, n, replace=False),
                          rng2.choice(c1i, n, replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

# ══════════════════════════════════════════════════════════════════════════════
# G-SPACE
# ══════════════════════════════════════════════════════════════════════════════
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

Xlda_g, ylda_g = _balance(G_B[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_g = _LDA().fit(Xlda_g, ylda_g)
Z_g   = lda_g.transform(G_B[:, :K_LDA])
if Z_g[patient_labels==0].mean() > Z_g[patient_labels==1].mean():
    lda_g.w_ *= -1; lda_g.thr_ *= -1
    Z_g = lda_g.transform(G_B[:, :K_LDA])
cc_lda_g = Z_g[patient_labels==0]
ad_lda_g = Z_g[patient_labels==1]
thr_g    = 0.5*(cc_lda_g.mean() + ad_lda_g.mean())
w_g      = lda_g.w_    # (K_LDA,) — LDA direction in G-space
print(f"  G-space: CC={cc_lda_g.mean():.3f}  AD={ad_lda_g.mean():.3f}  thr={thr_g:.3f}")

def pid_to_g(pid):
    """G-space coordinates (K_LDA) for a patient (using their unperturbed W)."""
    wp = project_W(pat_W[pid], pid)
    return ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]

def W_to_gvec(W, pid):
    """G-space coordinates (K_LDA) for an arbitrary W matrix."""
    wp = project_W(W, pid)
    return ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]

def W_to_glda(W, pid):
    g = W_to_gvec(W, pid)
    return float(lda_g.transform(g.reshape(1,-1))[0])

# G-space coordinates for all patients
G_all = G_B[:, :K_LDA]   # (N_patients, K_LDA)
g_cc_mean = G_all[patient_labels==0].mean(0)   # CC centroid in G-space

# ══════════════════════════════════════════════════════════════════════════════
# FC-LAG SPACE
# ══════════════════════════════════════════════════════════════════════════════
def lagged_corrcoef(S, lag):
    if lag == 0:
        return np.corrcoef(S.T)
    T = S.shape[0]
    A = S[:T-lag].astype(np.float64); B = S[lag:].astype(np.float64)
    A -= A.mean(0); B -= B.mean(0)
    A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - lag)

def compute_fclag_feat(W, pid):
    Xc = patX_single[pid].astype(np.float64)
    Y  = W.T.astype(np.float64) @ Xc.T
    S  = Y.T
    feats = []
    for lag in range(MAX_LAG_FL + 1):
        fc = lagged_corrcoef(S, lag)
        fc = np.nan_to_num(fc, nan=0., posinf=0., neginf=0.)
        if lag == 0:
            feats.append(fc[np.triu_indices(N_SITES, k=1)])
        else:
            feats.append(fc.flatten())
    return np.concatenate(feats)

print("Computing baseline FC-lag features ...")
fclag_base = np.array([compute_fclag_feat(pat_W[pid], pid)
                        for pid in tqdm(unique_pids, desc="  FC-lag", leave=False)])
fclag_mean = fclag_base.mean(0)
fclag_c    = fclag_base - fclag_mean
C_fl       = fclag_c @ fclag_c.T
ev_fl, evec_fl = np.linalg.eigh(C_fl)
ord_fl     = np.argsort(ev_fl)[::-1]
ev_fl      = np.maximum(ev_fl[ord_fl], 0); evec_fl = evec_fl[:, ord_fl]
G_fl       = evec_fl * np.sqrt(ev_fl)

Xlda_fl, ylda_fl = _balance(G_fl[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_fl = _LDA().fit(Xlda_fl, ylda_fl)
Z_fl   = lda_fl.transform(G_fl[:, :K_LDA])
if Z_fl[patient_labels==0].mean() > Z_fl[patient_labels==1].mean():
    lda_fl.w_ *= -1; lda_fl.thr_ *= -1
    Z_fl = lda_fl.transform(G_fl[:, :K_LDA])
cc_lda_fl = Z_fl[patient_labels==0]
ad_lda_fl = Z_fl[patient_labels==1]
thr_fl    = 0.5*(cc_lda_fl.mean() + ad_lda_fl.mean())
w_fl      = lda_fl.w_    # LDA direction in FC-lag G-space
print(f"  FC-lag: CC={cc_lda_fl.mean():.3f}  AD={ad_lda_fl.mean():.3f}  thr={thr_fl:.3f}")

G_fl_all    = G_fl[:, :K_LDA]
g_fl_cc_mean = G_fl_all[patient_labels==0].mean(0)

def W_to_flvec(W, pid):
    """FC-lag G-space coordinates (K_LDA) for an arbitrary W matrix."""
    feat   = compute_fclag_feat(W, pid)
    feat_c = feat - fclag_mean
    cross  = feat_c @ fclag_c.T
    g      = (cross @ evec_fl) / (np.sqrt(ev_fl) + 1e-12)
    return g[:K_LDA]

def W_to_fllda(W, pid):
    g = W_to_flvec(W, pid)
    return float(lda_fl.transform(g.reshape(1,-1))[0])

# ══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT ANALYSIS — how much of the G-space displacement is along w_lda?
#   Computed for each perturbation type (full_w, top5, top1) at alpha=1
# ══════════════════════════════════════════════════════════════════════════════
print("\n--- Alignment analysis (alpha=1 displacement vs LDA direction) ---")
ALIGN_TYPES = ["full_w", "top5", "top1"]
cos_g_dict  = {k: [] for k in ALIGN_TYPES}
cos_fl_dict = {k: [] for k in ALIGN_TYPES}

for pid in ad_pids:
    g_ad  = pid_to_g(pid)
    idx   = np.where(np.array(list(unique_pids)) == pid)[0][0]
    fl_ad = G_fl_all[idx]
    Wp    = pat_W[pid]
    dW    = W_cc_mean - Wp
    norms = np.linalg.norm(dW, axis=0)   # (N_SITES,) — per brain site

    for pt in ALIGN_TYPES:
        if pt == "full_w":
            W_int = W_cc_mean.copy()           # alpha=1: full replacement
        elif pt == "top5":
            top_k = np.argsort(norms)[::-1][:5]
            W_int = Wp.copy()
            W_int[:, top_k] = W_cc_mean[:, top_k]
        else:  # top1
            top_k = np.argsort(norms)[::-1][:1]
            W_int = Wp.copy()
            W_int[:, top_k] = W_cc_mean[:, top_k]

        # G-space alignment
        dg    = W_to_gvec(W_int, pid) - g_ad
        dg_n  = np.linalg.norm(dg)
        cos_g_dict[pt].append(float((dg @ w_g) / (dg_n + 1e-12)))

        # FC-lag space alignment
        dfl   = W_to_flvec(W_int, pid) - fl_ad
        dfl_n = np.linalg.norm(dfl)
        cos_fl_dict[pt].append(float((dfl @ w_fl) / (dfl_n + 1e-12)))

# Convert to arrays and print summary
cos_g_arrs  = {k: np.array(v) for k, v in cos_g_dict.items()}
cos_fl_arrs = {k: np.array(v) for k, v in cos_fl_dict.items()}

print(f"\n  {'Pert':8s}  {'G cos':>8s}  {'G angle':>8s}  {'G ||%':>6s}  |  "
      f"{'FL cos':>8s}  {'FL angle':>9s}  {'FL ||%':>6s}")
print("  " + "-"*70)
for pt in ALIGN_TYPES:
    cg  = cos_g_arrs[pt];  cfl = cos_fl_arrs[pt]
    print(f"  {pt:8s}  {cg.mean():+8.3f}  "
          f"{np.degrees(np.arccos(np.clip(cg.mean(),-1,1))):7.1f}°  "
          f"{(cg**2).mean()*100:6.1f}%  |  "
          f"{cfl.mean():+8.3f}  "
          f"{np.degrees(np.arccos(np.clip(cfl.mean(),-1,1))):8.1f}°  "
          f"{(cfl**2).mean()*100:6.1f}%")

# Keep backward-compat names (full_w used in figure below)
cos_g_arr  = cos_g_arrs["full_w"]
cos_fl_arr = cos_fl_arrs["full_w"]

# ══════════════════════════════════════════════════════════════════════════════
# PERTURBATION LOOP — current (top-5 + full_w + top1) + LDA-projected
# ══════════════════════════════════════════════════════════════════════════════
print("\nPerturbation loop ...")

# ── (A) Current perturbations ─────────────────────────────────────────────────
res_current = {}
for pert_type, alphas in [("top5",   ALPHAS_T5),
                           ("full_w", ALPHAS_FW),
                           ("top1",   ALPHAS_T1)]:
    print(f"  [{pert_type}]")
    res_current[pert_type] = {}
    for ai, alpha in enumerate(alphas):
        g_scores  = np.zeros(n_ad)
        fl_scores = np.zeros(n_ad)
        for pi, pid in enumerate(ad_pids):
            Wp    = pat_W[pid]
            dW    = W_cc_mean - Wp
            norms = np.linalg.norm(dW, axis=0)
            if pert_type == "full_w":
                W_int = (1-alpha)*Wp + alpha*W_cc_mean
            elif pert_type == "top5":
                top_k = np.argsort(norms)[::-1][:5]
                W_int = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]
            else:
                top_k = np.argsort(norms)[::-1][:1]
                W_int = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]
            g_scores[pi]  = W_to_glda(W_int, pid)
            fl_scores[pi] = W_to_fllda(W_int, pid)
        res_current[pert_type][ai] = {"g": g_scores.copy(), "fl": fl_scores.copy()}
        print(f"    alpha={alpha:5.2f}  G={g_scores.mean():+.3f}  FL={fl_scores.mean():+.3f}",
              flush=True)

# ── (B) LDA-projected perturbation ────────────────────────────────────────────
print("  [lda_proj]")
# G-space: g_perturbed = g_i + alpha * (z_CC - z_i) * w_g
# FC-lag:  same in FL G-space

# Per-patient: gap to fill and individual trajectories
g_scores_proj  = np.zeros((len(ALPHAS_PROJ), n_ad))
fl_scores_proj = np.zeros((len(ALPHAS_PROJ), n_ad))

for pi, pid in enumerate(ad_pids):
    pid_idx = int(np.where(np.array(list(unique_pids)) == pid)[0][0])

    # G-space
    g_i   = G_all[pid_idx]          # (K_LDA,) current G-space position
    z_i   = float(g_i @ w_g)        # current LDA score
    z_cc  = float(g_cc_mean @ w_g)  # CC centroid LDA score
    dz    = z_cc - z_i               # gap to fill

    # FC-lag
    fl_i  = G_fl_all[pid_idx]
    zfl_i = float(fl_i @ w_fl)
    zfl_cc= float(g_fl_cc_mean @ w_fl)
    dzfl  = zfl_cc - zfl_i

    for ai, alpha in enumerate(ALPHAS_PROJ):
        # G: move alpha * dz along w_g
        g_pert = g_i + alpha * dz * w_g
        g_scores_proj[ai, pi] = float(lda_g.transform(g_pert.reshape(1,-1))[0])
        # FL: move alpha * dzfl along w_fl
        fl_pert = fl_i + alpha * dzfl * w_fl
        fl_scores_proj[ai, pi] = float(lda_fl.transform(fl_pert.reshape(1,-1))[0])

print(f"  LDA-proj: G at alpha=1 → {g_scores_proj[-1].mean():+.3f}  "
      f"FL at alpha=1 → {fl_scores_proj[-1].mean():+.3f}")

# alpha to cross boundary (per patient, LDA-projected):
def alpha_to_cross(scores_by_alpha, alphas, thr):
    """First alpha where score crosses thr (per patient column)."""
    n = scores_by_alpha.shape[1]
    out = np.full(n, np.nan)
    for pi in range(n):
        for ai, a in enumerate(alphas):
            if scores_by_alpha[ai, pi] < thr:
                out[pi] = a
                break
    return out

cross_g_proj  = alpha_to_cross(g_scores_proj,  ALPHAS_PROJ, thr_g)
cross_fl_proj = alpha_to_cross(fl_scores_proj, ALPHAS_PROJ, thr_fl)
cross_g_t5    = alpha_to_cross(
    np.array([res_current["top5"][ai]["g"] for ai in range(len(ALPHAS_T5))]),
    ALPHAS_T5, thr_g)
cross_fl_t5   = alpha_to_cross(
    np.array([res_current["top5"][ai]["fl"] for ai in range(len(ALPHAS_T5))]),
    ALPHAS_T5, thr_fl)

print(f"\n  Reclassified at physio limit (top-5, alpha=4):")
safe_idx = np.where(ALPHAS_T5 <= PHYSIO)[0][-1]
g_t5_physio  = np.array([res_current["top5"][safe_idx]["g"]])
fl_t5_physio = np.array([res_current["top5"][safe_idx]["fl"]])
pct_g_t5  = (g_t5_physio  < thr_g ).mean()*100
pct_fl_t5 = (fl_t5_physio < thr_fl).mean()*100
print(f"    G-space  current  top5: {pct_g_t5:.0f}%")
print(f"    FC-lag   current  top5: {pct_fl_t5:.0f}%")

# LDA-proj alpha needed for 50% reclassification
safe_idx_p = np.where(ALPHAS_PROJ <= 0.5)[0][-1]
pct_g_proj  = (g_scores_proj[safe_idx_p]  < thr_g ).mean()*100
pct_fl_proj = (fl_scores_proj[safe_idx_p] < thr_fl).mean()*100
print(f"    G-space  LDA-proj alpha=0.5: {pct_g_proj:.0f}%")
print(f"    FC-lag   LDA-proj alpha=0.5: {pct_fl_proj:.0f}%")

# ── save ───────────────────────────────────────────────────────────────────────
np.savez("pert_ldaproj_data.npz",
         cos_g_fullw=cos_g_arrs["full_w"],  cos_fl_fullw=cos_fl_arrs["full_w"],
         cos_g_top5 =cos_g_arrs["top5"],    cos_fl_top5 =cos_fl_arrs["top5"],
         cos_g_top1 =cos_g_arrs["top1"],    cos_fl_top1 =cos_fl_arrs["top1"],
         # backward compat
         cos_g=cos_g_arr, cos_fl=cos_fl_arr,
         g_scores_proj=g_scores_proj, fl_scores_proj=fl_scores_proj,
         alphas_proj=ALPHAS_PROJ,
         cc_lda_g=cc_lda_g, ad_lda_g=ad_lda_g, thr_g=np.array(thr_g),
         cc_lda_fl=cc_lda_fl, ad_lda_fl=ad_lda_fl, thr_fl=np.array(thr_fl))
print("\nSaved pert_ldaproj_data.npz")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

fig = plt.figure(figsize=(16, 12), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.42)

# ── A: alignment histogram — one panel per space, three pert types overlaid ──
PCOLS   = {"full_w": "#37474F", "top5": "#E65100", "top1": "#6A1B9A"}
PLABELS = {"full_w": "full-W", "top5": "top-5 sites", "top1": "top-1 site"}

ax_a = fig.add_subplot(gs[0, 0])
bins = np.linspace(-1, 1, 25)
for pt in ALIGN_TYPES:
    ax_a.hist(cos_g_arrs[pt], bins=bins, color=PCOLS[pt], alpha=0.55,
              label=f"{PLABELS[pt]} (μ={cos_g_arrs[pt].mean():.2f})")
    ax_a.axvline(cos_g_arrs[pt].mean(), color=PCOLS[pt], lw=1.8, ls="--")
ax_a.axvline(0, color="gray", lw=1, ls=":", alpha=0.6)
ax_a.set_xlabel("cos(θ)  [Δg · w_lda / |Δg|]")
ax_a.set_ylabel("Count (AD patients)")
ax_a.set_title("G-space: alignment of\nperturbation with LDA direction")
ax_a.legend(frameon=False, fontsize=7)

# ── B: LDA-parallel energy fraction — grouped boxplot ────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
# Build 6 groups: [full_w_G, full_w_FL, top5_G, top5_FL, top1_G, top1_FL]
bp_data   = []
bp_labels = []
bp_colors = []
for pt in ALIGN_TYPES:
    bp_data.append(cos_g_arrs[pt]**2 * 100)
    bp_labels.append(f"{PLABELS[pt]}\nG-space")
    bp_colors.append(PCOLS[pt])
    bp_data.append(cos_fl_arrs[pt]**2 * 100)
    bp_labels.append(f"{PLABELS[pt]}\nFC-lag")
    bp_colors.append(PCOLS[pt])

bp = ax_b.boxplot(bp_data, patch_artist=True,
                  medianprops=dict(color="white", lw=2))
for patch, col in zip(bp["boxes"], bp_colors):
    patch.set_facecolor(col); patch.set_alpha(0.7)
# Hatch FC-lag boxes to distinguish from G-space
for i in [1, 3, 5]:
    bp["boxes"][i].set_hatch("///")

ax_b.set_xticks(range(1, 7))
ax_b.set_xticklabels(bp_labels, fontsize=6.5)
ax_b.set_ylabel("LDA-parallel energy  cos²(θ)×100 (%)")
ax_b.set_title("Fraction of displacement energy\nalong the LDA discriminant")
ax_b.set_ylim(0, 100)
ax_b.axhline(100, color="gray", ls="--", lw=1, alpha=0.4)
for i, vals in enumerate(bp_data):
    ax_b.text(i+1, min(vals.mean()+5, 95), f"{vals.mean():.0f}%",
              ha="center", fontsize=7.5, color=bp_colors[i], fontweight="bold")

# ── C: G-space — current top5 vs LDA-projected ───────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
ax_c.axhspan(cc_lda_g.mean()-cc_lda_g.std(), cc_lda_g.mean()+cc_lda_g.std(),
             alpha=0.12, color=CC_COL)
ax_c.axhline(cc_lda_g.mean(), color=CC_COL, lw=1.5, ls="--", alpha=0.9,
             label="CC mean ±1σ")
ax_c.axhline(thr_g, color="gray", lw=0.8, ls="-.", alpha=0.6,
             label=f"Boundary = {thr_g:.2f}")

# current top5
t5_g_mat = np.array([res_current["top5"][ai]["g"] for ai in range(len(ALPHAS_T5))])
m5 = t5_g_mat.mean(1); s5 = t5_g_mat.std(1)
ax_c.fill_between(ALPHAS_T5, m5-s5, m5+s5, alpha=0.18, color=T5_COL)
ax_c.plot(ALPHAS_T5, m5, "-o", ms=4, lw=2.0, color=T5_COL, label="Current top-5")

# LDA-projected
m_proj = g_scores_proj.mean(1); s_proj = g_scores_proj.std(1)
# Scale ALPHAS_PROJ to the same x-axis feel: alpha=1 means reaching CC mean
# Map onto the same x-axis by scaling to ALPHAS_T5 range
ax_c_twin = ax_c.twiny()
ax_c_twin.fill_between(ALPHAS_PROJ, m_proj-s_proj, m_proj+s_proj,
                        alpha=0.18, color=PROJ_G)
ax_c_twin.plot(ALPHAS_PROJ, m_proj, "-s", ms=4, lw=2.0, color=PROJ_G,
               label="LDA-proj (own axis →)")
ax_c_twin.set_xlabel("alpha  (LDA-projected, 0→1 = gap to CC mean)", fontsize=7,
                     color=PROJ_G)
ax_c_twin.tick_params(axis='x', labelcolor=PROJ_G, labelsize=7)
ax_c.set_xlabel("alpha  (top-5 perturbation)")
ax_c.set_ylabel("G-space LDA score")
ax_c.set_title("G-space: current vs LDA-projected")
# combine legends
lines_c, labels_c = ax_c.get_legend_handles_labels()
lines_t, labels_t = ax_c_twin.get_legend_handles_labels()
ax_c.legend(lines_c+lines_t, labels_c+labels_t, frameon=False, fontsize=7)

# ── D: G-space reclassification — current all strategies + LDA-proj ───────────
ax_d = fig.add_subplot(gs[1, 0])
for pert_type, alphas, col, lbl in [
        ("full_w", ALPHAS_FW, "#1B5E20", "Full-W"),
        ("top5",   ALPHAS_T5, T5_COL,    "Top-5"),
        ("top1",   ALPHAS_T1, "#6A1B9A", "Top-1"),
]:
    mat  = np.array([res_current[pert_type][ai]["g"] for ai in range(len(alphas))])
    frac = (mat < thr_g).mean(1) * 100
    ax_d.plot(alphas, frac, "-o", ms=4, lw=1.8, color=col, label=lbl)
# LDA-projected (twin x)
ax_d_t = ax_d.twiny()
frac_proj = (g_scores_proj < thr_g).mean(1) * 100
ax_d_t.plot(ALPHAS_PROJ, frac_proj, "-s", ms=5, lw=2.2, color=PROJ_G,
            label="LDA-proj")
ax_d_t.set_xlabel("alpha (LDA-proj)", fontsize=7, color=PROJ_G)
ax_d_t.tick_params(axis='x', labelcolor=PROJ_G, labelsize=7)
ax_d.axvline(PHYSIO, color="#B71C1C", lw=1.0, ls=":", alpha=0.55)
ax_d.axhline(50, color="gray", ls="--", lw=1, alpha=0.5)
ax_d.set_xlabel("alpha (current perturbations)")
ax_d.set_ylabel("AD reclassified as CC (%)")
ax_d.set_title("G-space: reclassification rate\n(current + LDA-projected)")
lines_d, labels_d = ax_d.get_legend_handles_labels()
lines_dt, labels_dt = ax_d_t.get_legend_handles_labels()
ax_d.legend(lines_d+lines_dt, labels_d+labels_dt, frameon=False, fontsize=7)
ax_d.set_ylim(-2, 105)

# ── E: FC-lag score — current top5 vs LDA-proj ───────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
ax_e.axhspan(cc_lda_fl.mean()-cc_lda_fl.std(), cc_lda_fl.mean()+cc_lda_fl.std(),
             alpha=0.12, color=CC_COL)
ax_e.axhline(cc_lda_fl.mean(), color=CC_COL, lw=1.5, ls="--", alpha=0.9,
             label="CC mean ±1σ")
ax_e.axhline(thr_fl, color="gray", lw=0.8, ls="-.", alpha=0.6,
             label=f"Boundary = {thr_fl:.2f}")
t5_fl_mat = np.array([res_current["top5"][ai]["fl"] for ai in range(len(ALPHAS_T5))])
mf = t5_fl_mat.mean(1); sf = t5_fl_mat.std(1)
ax_e.fill_between(ALPHAS_T5, mf-sf, mf+sf, alpha=0.18, color=FLCOL)
ax_e.plot(ALPHAS_T5, mf, "-o", ms=4, lw=2.0, color=FLCOL, label="Current top-5")
ax_e_t = ax_e.twiny()
mp = fl_scores_proj.mean(1); sp = fl_scores_proj.std(1)
ax_e_t.fill_between(ALPHAS_PROJ, mp-sp, mp+sp, alpha=0.18, color=PROJ_FL)
ax_e_t.plot(ALPHAS_PROJ, mp, "-s", ms=4, lw=2.0, color=PROJ_FL,
            label="LDA-proj (own axis)")
ax_e_t.set_xlabel("alpha (LDA-proj)", fontsize=7, color=PROJ_FL)
ax_e_t.tick_params(axis='x', labelcolor=PROJ_FL, labelsize=7)
ax_e.set_xlabel("alpha (top-5 perturbation)")
ax_e.set_ylabel("FC-lag LDA score")
ax_e.set_title("FC-lag: current vs LDA-projected")
lines_e, labels_e = ax_e.get_legend_handles_labels()
lines_et, labels_et = ax_e_t.get_legend_handles_labels()
ax_e.legend(lines_e+lines_et, labels_e+labels_et, frameon=False, fontsize=7)

# ── F: FC-lag reclassification — current all strategies + LDA-proj ────────────
ax_f = fig.add_subplot(gs[1, 2])
for pert_type, alphas, col, lbl in [
        ("full_w", ALPHAS_FW, "#1B5E20", "Full-W"),
        ("top5",   ALPHAS_T5, FLCOL,     "Top-5"),
        ("top1",   ALPHAS_T1, "#6A1B9A", "Top-1"),
]:
    mat  = np.array([res_current[pert_type][ai]["fl"] for ai in range(len(alphas))])
    frac = (mat < thr_fl).mean(1) * 100
    ax_f.plot(alphas, frac, "-o", ms=4, lw=1.8, color=col, label=lbl)
ax_f_t = ax_f.twiny()
frac_fl_proj = (fl_scores_proj < thr_fl).mean(1) * 100
ax_f_t.plot(ALPHAS_PROJ, frac_fl_proj, "-s", ms=5, lw=2.2, color=PROJ_FL,
            label="LDA-proj")
ax_f_t.set_xlabel("alpha (LDA-proj)", fontsize=7, color=PROJ_FL)
ax_f_t.tick_params(axis='x', labelcolor=PROJ_FL, labelsize=7)
ax_f.axhline(50, color="gray", ls="--", lw=1, alpha=0.5)
ax_f.set_xlabel("alpha (current perturbations)")
ax_f.set_ylabel("AD reclassified as CC (%)")
ax_f.set_title("FC-lag: reclassification rate\n(current + LDA-projected)")
lines_f, labels_f = ax_f.get_legend_handles_labels()
lines_ft, labels_ft = ax_f_t.get_legend_handles_labels()
ax_f.legend(lines_f+lines_ft, labels_f+labels_ft, frameon=False, fontsize=7)
ax_f.set_ylim(-2, 105)

# ── G: Summary bar — reclassification at matched perturbation budget ──────────
ax_g = fig.add_subplot(gs[2, 0:2])
# Compare at: top5 α=4, full_W α=1, LDA-proj α=0.5 (reaches ~halfway to CC mean)
comparisons = []
safe_t5  = np.where(ALPHAS_T5 <= 4.0)[0][-1]
safe_fw  = np.where(ALPHAS_FW <= 1.0)[0][-1]
safe_t1  = np.where(ALPHAS_T1 <= 3.0)[0][-1]
safe_p5  = np.where(ALPHAS_PROJ <= 0.5)[0][-1]
safe_p10 = np.where(ALPHAS_PROJ <= 1.0)[0][-1]

methods = [f"Top-5\n(a=4)", f"Full-W\n(a=1)", f"Top-1\n(a=3)",
           f"LDA-proj\n(a=0.5)", f"LDA-proj\n(a=1.0)"]
g_fracs = [
    (np.array([res_current["top5"][safe_t5]["g"]]) < thr_g).mean()*100,
    (np.array([res_current["full_w"][safe_fw]["g"]]) < thr_g).mean()*100,
    (np.array([res_current["top1"][safe_t1]["g"]]) < thr_g).mean()*100,
    (g_scores_proj[safe_p5] < thr_g).mean()*100,
    (g_scores_proj[safe_p10] < thr_g).mean()*100,
]
fl_fracs = [
    (np.array([res_current["top5"][safe_t5]["fl"]]) < thr_fl).mean()*100,
    (np.array([res_current["full_w"][safe_fw]["fl"]]) < thr_fl).mean()*100,
    (np.array([res_current["top1"][safe_t1]["fl"]]) < thr_fl).mean()*100,
    (fl_scores_proj[safe_p5] < thr_fl).mean()*100,
    (fl_scores_proj[safe_p10] < thr_fl).mean()*100,
]
x = np.arange(len(methods)); bw = 0.35
bars_g  = ax_g.bar(x - bw/2, g_fracs,  bw, color=GCOL,  alpha=0.80, label="G-space LDA")
bars_fl = ax_g.bar(x + bw/2, fl_fracs, bw, color=FLCOL, alpha=0.80, label="FC-lag LDA")
for bar, val in zip(list(bars_g)+list(bars_fl), g_fracs+fl_fracs):
    ax_g.text(bar.get_x()+bar.get_width()/2, val+1.5,
              f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
ax_g.axhline(50, color="gray", ls="--", lw=1, alpha=0.5)
# shade LDA-proj columns
for xi in [3, 4]:
    ax_g.axvspan(xi-0.55, xi+0.55, alpha=0.07, color=PROJ_G, zorder=0)
ax_g.set_xticks(x); ax_g.set_xticklabels(methods, fontsize=8)
ax_g.set_ylabel("AD reclassified as CC (%)")
ax_g.set_title("Reclassification: current perturbations vs LDA-projected\n"
               "(shaded = LDA-projected strategy)")
ax_g.set_ylim(0, 110); ax_g.legend(frameon=False)

# ── H: efficiency summary table ───────────────────────────────────────────────
ax_h = fig.add_subplot(gs[2, 2])
ax_h.axis("off")

lines = ["LDA-parallel energy  cos²(θ)×100 (%)\n",
         f"  {'':8s}  {'G-space':>8s}  {'FC-lag':>8s}",
         "  " + "-"*32]
for pt in ALIGN_TYPES:
    pg  = (cos_g_arrs[pt]**2).mean()*100
    pfl = (cos_fl_arrs[pt]**2).mean()*100
    lines.append(f"  {PLABELS[pt]:10s}  {pg:6.1f}%    {pfl:6.1f}%")
lines += ["",
          "LDA-projected (optimal):",
          "  100% by construction.",
          "  alpha=1 → each AD patient",
          "  reaches CC mean LDA score.",
          "",
          f"Reclassified (alpha=0.5):",
          f"  G-space LDA-proj: {pct_g_proj:.0f}%",
          f"  FC-lag  LDA-proj: {pct_fl_proj:.0f}%",
          f"vs current top-5 (alpha=4):",
          f"  G-space: {pct_g_t5:.0f}%   FC-lag: {pct_fl_t5:.0f}%"]

ax_h.text(0.03, 0.99, "\n".join(lines), transform=ax_h.transAxes,
          fontsize=8, va="top", ha="left",
          bbox=dict(boxstyle="round,pad=0.5", fc="#f5f5f5", alpha=0.8),
          family="monospace")

for ax, lbl in zip(list(fig.axes[:8]), list("ABCDEFGH")):
    try:
        ax.text(-0.12, 1.05, lbl, transform=ax.transAxes,
                fontsize=11, fontweight="bold", va="bottom", ha="left")
    except Exception:
        pass

fig.suptitle(
    "W-matrix stimulation: current vs LDA-projected perturbation\n"
    "LDA-projected = move each AD patient directly along the LDA discriminant "
    "w_ in G-space (minimal perturbation to reach CC)",
    fontsize=9, y=1.01)

fig.savefig("pert_ldaproj_figure.png", dpi=300, bbox_inches="tight",
            facecolor="white")
plt.close()
print("Saved pert_ldaproj_figure.png")

fig.savefig(os.path.join("paper_figures", "figure5_ldaproj.png"),
            dpi=300, bbox_inches="tight", facecolor="white")
print("Saved paper_figures/figure5_ldaproj.png")
print("\nDone.")
