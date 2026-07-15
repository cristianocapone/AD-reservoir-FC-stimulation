"""
pert_sites_stimulation.py
=========================
Stimulation experiment: interpolate each AD patient's readout W toward
the CC-mean W, for four perturbation strategies:

  full_W       — all 121 sites simultaneously
  top5         — the 5 sites (out of 121) with the largest ||dW|| column norm
  top1         — the single site with the largest ||dW|| column norm
  top1_geo10   — the top-1 site plus its 10 geometrically closest neighbours
                 in MNI space (11 sites total).
                 A single spatially-uniform stimulation is assumed: all 11
                 sites receive the SAME perturbation vector, equal to the
                 correction of the top-1 site alone:
                   dW_top1 = W_CC[:, top1] − W_AD[:, top1]
                 i.e. the signal you would give to the most-affected site,
                 broadcast uniformly to the whole neighbourhood.

For each strategy and alpha, compute:
  • G-space LDA score  (reservoir-based)
  • FC-lag LDA score   (lagged-FC-based, lags 0-2, K=25)

Reports:
  • LDA score vs alpha trajectories (mean ± std over 40 AD patients)
  • Reclassification rate (% AD scored < threshold) vs alpha
  • Which sites are most commonly targeted (top5 / top1 histograms)

Saves:
  pert_sites_data.npz
  pert_sites_figure.png
  paper_figures/figure4_sites.png
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import trange, tqdm
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── parameters ────────────────────────────────────────────────────────────────
RNG_SEED   = 42
N_CC_SAMP  = 40       # CC patients sampled (matches G-space experiment)
N_SITES    = 121
N_PC_MODEL = 50
K_PC       = 200
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
K_LDA      = 25
SR         = 0.95
MAX_LAG    = 2        # lags 0, 1, 2  → 36 542 features
TS_ROOT    = "./timeseries"

# Alpha grids — choose so each strategy's interesting region is well sampled
ALPHAS_FW  = np.array([0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
ALPHAS_T5  = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7.5, 10,
                        12.5, 15, 17.5, 20])
ALPHAS_T1  = np.array([0, 1, 2, 3, 4, 5, 7.5, 10, 15, 20, 30, 50])
ALPHAS_GEO = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7.5, 10,
                        12.5, 15, 17.5, 20])   # same range as top5

N_GEO_NEIGHBOURS = 10   # geometric neighbours of top-1 site
PHYSIO = 4.0            # physiological amplitude limit (top-5 reference)

# colours
CC_COL   = "#1565C0"
AD_COL   = "#C62828"
FW_COL   = "#2E7D32"   # full-W       — dark green
T5_COL   = "#E65100"   # top-5        — orange
T1_COL   = "#6A1B9A"   # top-1        — purple
GEO_COL  = "#00838F"   # top1+geo10   — teal
G_COL    = "#37474F"   # G-space axis labels
FL_COL   = "#0097A7"   # FC-lag  axis labels

# ══════════════════════════════════════════════════════════════════════════════
# MNI PARCEL COORDINATES  (Schaefer-100 + Harvard-Oxford subcortical 21)
# ══════════════════════════════════════════════════════════════════════════════
def build_parcel_coords():
    """
    Return (121, 3) array of MNI centroids matching the atlas used in
    extract_timeseries.py:  Schaefer-100 (indices 0-99) + HO-sub 21 (100-120).
    Uses nilearn to fetch the same atlases and resamples HO to Schaefer space.
    """
    import nibabel as nib
    from nilearn import datasets, image
    from scipy.ndimage import center_of_mass

    print("Loading Schaefer-100 atlas for parcel coordinates ...")
    sch   = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    sch_img = nib.load(sch.maps)
    sch_data = sch_img.get_fdata()
    affine   = sch_img.affine

    print("Loading Harvard-Oxford subcortical atlas ...")
    ho    = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_img = nib.load(ho.maps) if isinstance(ho.maps, str) else ho.maps
    ho_r  = image.resample_to_img(ho_img, sch_img, interpolation="nearest")
    ho_data = ho_r.get_fdata()

    # Build combined label map (same logic as extract_timeseries.py)
    combined = np.zeros_like(sch_data, dtype=np.int32)
    # Schaefer labels 1-100 → parcel indices 0-99
    combined[sch_data > 0] = sch_data[sch_data > 0].astype(int)
    # HO labels 1-21 → parcel indices 100-120
    n_sch = 100
    for ho_lbl in range(1, 22):
        mask = (ho_data == ho_lbl) & (sch_data == 0)
        combined[mask] = n_sch + ho_lbl   # 101..121 → 0-indexed below

    coords = np.zeros((N_SITES, 3))
    for idx in range(N_SITES):
        lbl_val = idx + 1   # 1-based label in combined map
        vox_coords = np.array(np.where(combined == lbl_val)).T  # (N_vox, 3)
        if len(vox_coords) == 0:
            continue
        vox_cm = vox_coords.mean(0)   # centroid in voxel space
        # vox → MNI (affine is 4×4)
        mni = affine[:3, :3] @ vox_cm + affine[:3, 3]
        coords[idx] = mni

    print(f"  Parcel centroid range:  x=[{coords[:,0].min():.0f},{coords[:,0].max():.0f}]  "
          f"y=[{coords[:,1].min():.0f},{coords[:,1].max():.0f}]  "
          f"z=[{coords[:,2].min():.0f},{coords[:,2].max():.0f}]  mm MNI")
    return coords   # (121, 3)

parcel_coords = build_parcel_coords()   # (N_SITES, 3)

def geo_neighbours(site_idx, n=N_GEO_NEIGHBOURS):
    """Return indices of the n closest parcels to site_idx (excluding itself)."""
    diffs = parcel_coords - parcel_coords[site_idx]
    dists = np.linalg.norm(diffs, axis=1)
    dists[site_idx] = np.inf   # exclude self
    return np.argsort(dists)[:n]

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
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

pid_raw        = np.array(pid_raw)
labels_raw     = np.array(labels_raw)
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_pids = [unique_pids[i] for i in np.where(patient_labels == 0)[0]]
ad_pids = [unique_pids[i] for i in np.where(patient_labels == 1)[0]]
n_ad = len(ad_pids); n_cc = len(cc_pids)
print(f"  {N_patients} patients  ({n_cc} CC, {n_ad} AD)")

# ══════════════════════════════════════════════════════════════════════════════
# RESERVOIR — TF PASS
# ══════════════════════════════════════════════════════════════════════════════
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals_p, evecs_p = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs_p[:, np.argsort(evals_p)[::-1]][:, :N_PC_MODEL]

print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

print("TF pass ...")
sess_X, sess_Y = {}, {}
for idx in trange(len(signals), desc="  TF"):
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

# ── fit readout W per patient ─────────────────────────────────────────────────
print("W fitting ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
first_idx   = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single = {pid: sess_X[first_idx[pid]] for pid in unique_pids}
patY_single = {pid: sess_Y[first_idx[pid]] for pid in unique_pids}

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc    # (T, N_hidden) \ (T, N_sites)

W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)  # (N_hidden, N_sites)

# ══════════════════════════════════════════════════════════════════════════════
# LDA HELPER
# ══════════════════════════════════════════════════════════════════════════════
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.thr_ = 0.5*(mu0@w + mu1@w)
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
# G-SPACE SVD + LDA
# ══════════════════════════════════════════════════════════════════════════════
print("Building G-space + LDA ...")
pat_Vtk = {}
for pid in tqdm(unique_pids, desc="  SVD-G", leave=False):
    Xca = patX_single[pid].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    pat_Vtk[pid] = Vtx[:kk]

def project_W(W, pid):
    Vt = pat_Vtk[pid]
    return (W.T.astype(np.float64) @ Vt.T @ Vt).flatten()

Wproj  = np.array([project_W(pat_W[pid], pid) for pid in unique_pids])
Wmean  = Wproj.mean(0)
Wcent  = Wproj - Wmean
_, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
Meff   = N_patients - 1
G_B    = Wcent @ Vsvd[:Meff].T   # (N_patients, Meff)

Xlda_g, ylda_g = _balance(G_B[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_g = _LDA().fit(Xlda_g, ylda_g)
Z_g   = lda_g.transform(G_B[:, :K_LDA])
if Z_g[patient_labels==0].mean() > Z_g[patient_labels==1].mean():
    lda_g.w_ *= -1; lda_g.thr_ *= -1
    Z_g = lda_g.transform(G_B[:, :K_LDA])
cc_g = Z_g[patient_labels==0]; ad_g = Z_g[patient_labels==1]
thr_g = 0.5*(cc_g.mean() + ad_g.mean())
print(f"  G-space:  CC={cc_g.mean():.3f}  AD={ad_g.mean():.3f}  thr={thr_g:.3f}")

def W_to_glda(W, pid):
    wp = project_W(W, pid)
    g  = ((wp - Wmean) @ Vsvd[:Meff].T)[:K_LDA]
    return float(lda_g.transform(g.reshape(1,-1))[0])

# ══════════════════════════════════════════════════════════════════════════════
# FC-LAG SPACE SVD + LDA
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
    """Lagged-FC features from predicted signals  Y = W.T @ X_res."""
    Xc = patX_single[pid].astype(np.float64)
    S  = (W.T.astype(np.float64) @ Xc.T).T      # (T_eff, N_sites)
    feats = []
    for lag in range(MAX_LAG + 1):
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
ev_fl, evec_fl = np.linalg.eigh(fclag_c @ fclag_c.T)
ord_fl = np.argsort(ev_fl)[::-1]
ev_fl  = np.maximum(ev_fl[ord_fl], 0); evec_fl = evec_fl[:, ord_fl]
G_fl   = evec_fl * np.sqrt(ev_fl)

Xlda_fl, ylda_fl = _balance(G_fl[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_fl = _LDA().fit(Xlda_fl, ylda_fl)
Z_fl   = lda_fl.transform(G_fl[:, :K_LDA])
if Z_fl[patient_labels==0].mean() > Z_fl[patient_labels==1].mean():
    lda_fl.w_ *= -1; lda_fl.thr_ *= -1
    Z_fl = lda_fl.transform(G_fl[:, :K_LDA])
cc_fl = Z_fl[patient_labels==0]; ad_fl = Z_fl[patient_labels==1]
thr_fl = 0.5*(cc_fl.mean() + ad_fl.mean())
print(f"  FC-lag:   CC={cc_fl.mean():.3f}  AD={ad_fl.mean():.3f}  thr={thr_fl:.3f}")

def W_to_fllda(W, pid):
    feat   = compute_fclag_feat(W, pid)
    feat_c = feat - fclag_mean
    g      = (feat_c @ fclag_c.T @ evec_fl) / (np.sqrt(ev_fl) + 1e-12)
    return float(lda_fl.transform(g[:K_LDA].reshape(1,-1))[0])

# ══════════════════════════════════════════════════════════════════════════════
# PHYSIOLOGICAL DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
def compute_physio_metrics(W_base, W_int, pid, stim_sites, best_site=None):
    """
    Signal RMS ratio and FC amplitude at stimulated sites.

    Parameters
    ----------
    W_base     : (N_hidden, N_sites)  baseline readout (this AD patient's W)
    W_int      : (N_hidden, N_sites)  perturbed readout
    pid        : patient ID
    stim_sites : 1-D array of stimulated site indices
    best_site  : primary stimulation site index (top-1 site), or None

    Returns
    -------
    rms_mean  : mean RMS(int)/RMS(base) across all stim_sites
    rms_top1  : RMS ratio at best_site  (np.nan if best_site is None)
    rms_geo   : mean RMS ratio at geo-neighbour sites only  (np.nan if not applicable)
    fc_amp    : mean |FC| at stim-site rows vs all other sites
    """
    Xc    = patX_single[pid].astype(np.float64)          # (T_eff, N_hidden)
    Ybase = (W_base.T.astype(np.float64) @ Xc.T).T       # (T_eff, N_sites)
    Yint  = (W_int.T.astype(np.float64)  @ Xc.T).T       # (T_eff, N_sites)

    rms_base = np.sqrt((Ybase**2).mean(0) + 1e-24)        # (N_sites,)
    rms_int  = np.sqrt((Yint**2).mean(0))                 # (N_sites,)
    ratio    = rms_int / rms_base                         # (N_sites,)

    rms_mean = float(ratio[stim_sites].mean())

    if best_site is not None:
        rms_top1  = float(ratio[best_site])
        geo_sites = np.array([s for s in stim_sites if s != best_site])
        rms_geo   = float(ratio[geo_sites].mean()) if len(geo_sites) > 0 else np.nan
    else:
        rms_top1 = np.nan
        rms_geo  = np.nan

    # Mean |FC| at stimulated rows (off-diagonal)
    fc = np.corrcoef(Yint.T)
    fc = np.nan_to_num(fc, nan=0.)
    fc_vals = []
    for s in stim_sites:
        row = np.abs(fc[s])
        row = np.delete(row, s)          # remove self-correlation
        fc_vals.append(row.mean())
    fc_amp = float(np.mean(fc_vals))

    return rms_mean, rms_top1, rms_geo, fc_amp

# ══════════════════════════════════════════════════════════════════════════════
# PERTURBATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
# Strategy: W_int[:, sites] = (1-a)*W_AD[:, sites] + a*W_CC_mean[:, sites]
#   full_w  → sites = all 121
#   top5    → sites = 5 with largest ||W_CC_mean[:,s] - W_AD[:,s]||
#   top1    → sites = 1 with largest norm
#
# Site selection per patient: dW = W_cc_mean - pat_W[pid]  (N_hidden x N_sites)
#   norm_s = ||dW[:, s]||  for s in 0..120  → pick argsort top-k
# ══════════════════════════════════════════════════════════════════════════════
print("\nPerturbation loop ...")

STRATS = [
    ("full_w",   ALPHAS_FW,  FW_COL,  "Full-W  (121/121 sites)"),
    ("top5",     ALPHAS_T5,  T5_COL,  "Top-5   ( 5/121 sites)"),
    ("top1",     ALPHAS_T1,  T1_COL,  "Top-1   ( 1/121 sites)"),
    ("top1_geo", ALPHAS_GEO, GEO_COL, "Top-1+geo10  (11/121, top-1 signal)"),
]

results = {}  # results[strat][alpha_idx] = {"g": array(n_ad), "fl": array(n_ad)}
phys    = {}  # phys[strat][alpha_idx]   = {"rms","rms_top1","rms_geo","fc"} each array(n_ad)

# Track which sites are selected (alpha-independent, record at ai==0)
top5_site_counts    = np.zeros(N_SITES, dtype=int)
top1_site_counts    = np.zeros(N_SITES, dtype=int)
geo_site_counts     = np.zeros(N_SITES, dtype=int)   # sites selected by top1_geo

for strat, alphas, col, label in STRATS:
    print(f"  [{strat}]")
    results[strat] = {}
    phys[strat]    = {}
    for ai, alpha in enumerate(alphas):
        g_sc          = np.zeros(n_ad)
        fl_sc         = np.zeros(n_ad)
        rms_acc       = np.zeros(n_ad)
        rms_top1_acc  = np.full(n_ad, np.nan)
        rms_geo_acc   = np.full(n_ad, np.nan)
        fc_acc        = np.zeros(n_ad)

        for pi, pid in enumerate(ad_pids):
            Wp    = pat_W[pid]
            dW    = W_cc_mean - Wp              # (N_hidden, N_SITES)
            norms = np.linalg.norm(dW, axis=0)  # (N_SITES,)  — one norm per brain site
            best_site = None

            if strat == "full_w":
                W_int      = (1 - alpha)*Wp + alpha*W_cc_mean
                stim_sites = np.arange(N_SITES)

            elif strat == "top5":
                top_k      = np.argsort(norms)[::-1][:5]   # 5 sites out of 121
                W_int      = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]
                stim_sites = top_k
                if ai == 0:
                    top5_site_counts[top_k] += 1

            elif strat == "top1":
                top_k      = np.argsort(norms)[::-1][:1]   # 1 site out of 121
                W_int      = Wp.copy()
                W_int[:, top_k] = (1-alpha)*Wp[:, top_k] + alpha*W_cc_mean[:, top_k]
                stim_sites = top_k
                best_site  = int(top_k[0])
                if ai == 0:
                    top1_site_counts[top_k] += 1

            else:  # top1_geo: top-1 site + 10 geo neighbours, top-1 signal for all
                best       = int(np.argsort(norms)[::-1][0])
                nbrs       = geo_neighbours(best, n=N_GEO_NEIGHBOURS)
                top_k      = np.unique(np.concatenate([[best], nbrs]))  # 11 sites

                # Single device → same perturbation for all 11 sites:
                # use the correction computed for the top-1 site only
                dW_top1    = dW[:, best]   # (N_hidden,)
                W_int      = Wp.copy()
                W_int[:, top_k] = Wp[:, top_k] + alpha * dW_top1[:, None]
                stim_sites = top_k
                best_site  = best
                if ai == 0:
                    geo_site_counts[top_k] += 1

            g_sc[pi]  = W_to_glda(W_int, pid)
            fl_sc[pi] = W_to_fllda(W_int, pid)

            # ── physiological diagnostics ──────────────────────────────────
            rm, rt, rg, fc = compute_physio_metrics(Wp, W_int, pid,
                                                    stim_sites, best_site)
            rms_acc[pi] = rm
            if not np.isnan(rt):
                rms_top1_acc[pi] = rt
            if not np.isnan(rg):
                rms_geo_acc[pi]  = rg
            fc_acc[pi] = fc

        results[strat][ai] = {"g": g_sc.copy(), "fl": fl_sc.copy()}
        phys[strat][ai]    = {
            "rms":      rms_acc.copy(),
            "rms_top1": rms_top1_acc.copy(),
            "rms_geo":  rms_geo_acc.copy(),
            "fc":       fc_acc.copy(),
        }
        print(f"    alpha={alpha:5.2f}  G={g_sc.mean():+.3f}  FL={fl_sc.mean():+.3f}"
              f"  RMS={rms_acc.mean():.3f}  FC={fc_acc.mean():.4f}",
              flush=True)

# ── per-patient alpha to cross boundary ───────────────────────────────────────
def alpha_cross(strat, alphas, thr, space):
    out = np.full(n_ad, np.nan)
    for pi in range(n_ad):
        for ai, a in enumerate(alphas):
            if results[strat][ai][space][pi] < thr:
                out[pi] = a; break
    return out

cross_g  = {s: alpha_cross(s, a, thr_g,  "g")  for s, a, *_ in STRATS}
cross_fl = {s: alpha_cross(s, a, thr_fl, "fl") for s, a, *_ in STRATS}

# ── summary stats ─────────────────────────────────────────────────────────────
print("\n── Summary ─────────────────────────────────────────────────────────")
print(f"  {'Strategy':30s}  {'G cross (med)':>14s}  {'FL cross (med)':>14s}  "
      f"{'G @physio':>10s}  {'FL @physio':>10s}")
print("  " + "-"*75)
for strat, alphas, col, label in STRATS:
    physio_ai = np.where(alphas <= PHYSIO)[0]
    physio_idx = physio_ai[-1] if len(physio_ai) else 0
    pct_g  = (results[strat][physio_idx]["g"]  < thr_g ).mean()*100
    pct_fl = (results[strat][physio_idx]["fl"] < thr_fl).mean()*100
    med_g  = np.nanmedian(cross_g[strat])
    med_fl = np.nanmedian(cross_fl[strat])
    sg = f"{med_g:.1f}" if not np.isnan(med_g) else ">max"
    sf = f"{med_fl:.1f}" if not np.isnan(med_fl) else ">max"
    print(f"  {label:30s}  {sg:>14s}  {sf:>14s}  "
          f"{pct_g:9.0f}%  {pct_fl:9.0f}%")

print(f"\n── Physiological diagnostics at α≤{PHYSIO:.0f} ─────────────────────────────────────")
print(f"  {'Strategy':30s}  {'RMS (stim)':>10s}  {'RMS top-1':>10s}  {'RMS geo':>9s}  {'FC amp':>8s}")
print("  " + "-"*76)
for strat, alphas, col, label in STRATS:
    physio_ai  = np.where(alphas <= PHYSIO)[0]
    pidx       = physio_ai[-1] if len(physio_ai) else 0
    d          = phys[strat][pidx]
    rms_s   = f"{np.nanmean(d['rms']):.3f}"
    rms_t1  = f"{np.nanmean(d['rms_top1']):.3f}" \
              if not np.all(np.isnan(d['rms_top1'])) else "   ---"
    rms_geo = f"{np.nanmean(d['rms_geo']):.3f}" \
              if not np.all(np.isnan(d['rms_geo']))  else "   ---"
    fc_s    = f"{np.nanmean(d['fc']):.4f}"
    print(f"  {label:30s}  {rms_s:>10s}  {rms_t1:>10s}  {rms_geo:>9s}  {fc_s:>8s}")
print(f"\n  Note: RMS ratio = RMS(perturbed)/RMS(baseline); FC amp = mean|FC| at stim sites.\n"
      f"  Elevated RMS at the directly stimulated top-1 site is expected.")

print(f"\n  Top-5 site selection (most frequent across {n_ad} AD patients):")
for rank, s in enumerate(np.argsort(top5_site_counts)[::-1][:10], 1):
    if top5_site_counts[s] > 0:
        print(f"    {rank:2d}.  site {s:3d}  selected in {top5_site_counts[s]:2d}/{n_ad} AD patients")
print(f"\n  Top-1 site selection:")
for rank, s in enumerate(np.argsort(top1_site_counts)[::-1][:5], 1):
    if top1_site_counts[s] > 0:
        print(f"    {rank:2d}.  site {s:3d}  selected in {top1_site_counts[s]:2d}/{n_ad} AD patients")
print(f"\n  Top-1+geo10 site selection (11 sites/patient, {n_ad} patients):")
for rank, s in enumerate(np.argsort(geo_site_counts)[::-1][:15], 1):
    if geo_site_counts[s] > 0:
        print(f"    {rank:2d}.  site {s:3d}  selected in {geo_site_counts[s]:2d}/{n_ad} AD patients")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez("pert_sites_data.npz",
         cc_g=cc_g, ad_g=ad_g, thr_g=np.array(thr_g),
         cc_fl=cc_fl, ad_fl=ad_fl, thr_fl=np.array(thr_fl),
         alphas_fw=ALPHAS_FW, alphas_t5=ALPHAS_T5,
         alphas_t1=ALPHAS_T1, alphas_geo=ALPHAS_GEO,
         parcel_coords=parcel_coords,
         top5_site_counts=top5_site_counts,
         top1_site_counts=top1_site_counts,
         geo_site_counts=geo_site_counts,
         **{f"{s}_{ai}_g":  results[s][ai]["g"]
            for s, a, *_ in STRATS for ai in range(len(a))},
         **{f"{s}_{ai}_fl": results[s][ai]["fl"]
            for s, a, *_ in STRATS for ai in range(len(a))})
print("\nSaved pert_sites_data.npz")

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

# Try to load the oscillatory-stimulation results (pert_osc_stimulation.py).
# If present, the figure gains a 4th row of sinusoidal-stimulation panels.
try:
    _osc = np.load("pert_osc_data.npz", allow_pickle=True)
    HAVE_OSC = True
    print("Loaded pert_osc_data.npz — adding oscillatory-stimulation row.")
except Exception as _e:
    HAVE_OSC = False
    print(f"pert_osc_data.npz not found ({_e}); oscillatory row skipped.")

N_ROWS = 4 if HAVE_OSC else 3
fig = plt.figure(figsize=(16, 5.3 * N_ROWS), facecolor="white")
gs  = gridspec.GridSpec(N_ROWS, 3, figure=fig, hspace=0.55, wspace=0.42)

K_OSC_COLS = ["#6A1B9A", "#E65100", "#2E7D32"]   # k = 1, 2, 5 sites

# ── helper: draw band + line for one strategy ─────────────────────────────────
def plot_traj(ax, alphas, strat, space, col, lbl, ls="-", ms=4):
    mat = np.array([results[strat][ai][space] for ai in range(len(alphas))])
    m = mat.mean(1); s = mat.std(1)
    ax.fill_between(alphas, m-s, m+s, alpha=0.15, color=col)
    ax.plot(alphas, m, ls+"o", ms=ms, lw=2.0, color=col, label=lbl)

# ── A: G-space score — full-W ─────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axhspan(cc_g.mean()-cc_g.std(), cc_g.mean()+cc_g.std(), alpha=0.1, color=CC_COL)
ax_a.axhline(cc_g.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean ±1σ")
ax_a.axhline(thr_g, color="gray", lw=1, ls="-.", label=f"Boundary")
plot_traj(ax_a, ALPHAS_FW, "full_w", "g", FW_COL, "Full-W")
ax_a.set_xlabel("alpha"); ax_a.set_ylabel("G-space LDA score")
ax_a.set_title("G-space LDA — full-W\n(all 121 sites)")
ax_a.legend(frameon=False, fontsize=7.5)

# ── B: G-space score — top-5 + top-1 + top1+geo10 ───────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axhspan(cc_g.mean()-cc_g.std(), cc_g.mean()+cc_g.std(), alpha=0.1, color=CC_COL)
ax_b.axhline(cc_g.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean ±1σ")
ax_b.axhline(thr_g, color="gray", lw=1, ls="-.", label="Boundary")
plot_traj(ax_b, ALPHAS_T5,  "top5",     "g", T5_COL,  "Top-5 sites")
plot_traj(ax_b, ALPHAS_T1,  "top1",     "g", T1_COL,  "Top-1 site",       ls="--")
plot_traj(ax_b, ALPHAS_GEO, "top1_geo", "g", GEO_COL, "Top-1+geo10 sites",ls="-.")
ax_b.set_xlabel("alpha"); ax_b.set_ylabel("G-space LDA score")
ax_b.set_title("G-space LDA — focal stimulation\n(top-5,  top-1,  top-1+10 geo neighbours)")
ax_b.legend(frameon=False, fontsize=7.5)

# ── C: G-space reclassification rate ─────────────────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
for strat, alphas, col, lbl in STRATS:
    mat  = np.array([results[strat][ai]["g"] for ai in range(len(alphas))])
    frac = (mat < thr_g).mean(1) * 100
    ax_c.plot(alphas, frac, "-o", ms=4, lw=2, color=col,
              label=lbl.split("(")[0].strip())
ax_c.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6,
             label=f"Physio limit α={PHYSIO:.0f}")
ax_c.axhline(50, color="gray", ls="--", lw=1, alpha=0.4)
ax_c.set_xlabel("alpha"); ax_c.set_ylabel("AD reclassified as CC (%)")
ax_c.set_title("G-space LDA — reclassification rate")
ax_c.set_ylim(-2, 105); ax_c.legend(frameon=False, fontsize=7.5)

# ── D: FC-lag score — full-W ──────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
ax_d.axhspan(cc_fl.mean()-cc_fl.std(), cc_fl.mean()+cc_fl.std(), alpha=0.1, color=CC_COL)
ax_d.axhline(cc_fl.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean ±1σ")
ax_d.axhline(thr_fl, color="gray", lw=1, ls="-.", label="Boundary")
plot_traj(ax_d, ALPHAS_FW, "full_w", "fl", FW_COL, "Full-W")
ax_d.set_xlabel("alpha"); ax_d.set_ylabel("FC-lag LDA score")
ax_d.set_title("FC-lag LDA — full-W\n(all 121 sites)")
ax_d.legend(frameon=False, fontsize=7.5)

# ── E: FC-lag score — top-5 + top-1 + top1+geo10 ────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
ax_e.axhspan(cc_fl.mean()-cc_fl.std(), cc_fl.mean()+cc_fl.std(), alpha=0.1, color=CC_COL)
ax_e.axhline(cc_fl.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean ±1σ")
ax_e.axhline(thr_fl, color="gray", lw=1, ls="-.", label="Boundary")
plot_traj(ax_e, ALPHAS_T5,  "top5",     "fl", T5_COL,  "Top-5 sites")
plot_traj(ax_e, ALPHAS_T1,  "top1",     "fl", T1_COL,  "Top-1 site",       ls="--")
plot_traj(ax_e, ALPHAS_GEO, "top1_geo", "fl", GEO_COL, "Top-1+geo10 sites",ls="-.")
ax_e.set_xlabel("alpha"); ax_e.set_ylabel("FC-lag LDA score")
ax_e.set_title("FC-lag LDA — focal stimulation\n(top-5,  top-1,  top-1+10 geo neighbours)")
ax_e.legend(frameon=False, fontsize=7.5)

# ── F: FC-lag reclassification rate ───────────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
for strat, alphas, col, lbl in STRATS:
    mat  = np.array([results[strat][ai]["fl"] for ai in range(len(alphas))])
    frac = (mat < thr_fl).mean(1) * 100
    ax_f.plot(alphas, frac, "-o", ms=4, lw=2, color=col,
              label=lbl.split("(")[0].strip())
ax_f.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6,
             label=f"Physio limit α={PHYSIO:.0f}")
ax_f.axhline(50, color="gray", ls="--", lw=1, alpha=0.4)
ax_f.set_xlabel("alpha"); ax_f.set_ylabel("AD reclassified as CC (%)")
ax_f.set_title("FC-lag LDA — reclassification rate")
ax_f.set_ylim(-2, 105); ax_f.legend(frameon=False, fontsize=7.5)

# ── G: Signal RMS ratio vs alpha ──────────────────────────────────────────────
ax_g = fig.add_subplot(gs[2, 0])
ax_g.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.4, label="baseline (ratio=1)")
for strat, alphas, col, lbl in STRATS:
    mat  = np.array([phys[strat][ai]["rms"] for ai in range(len(alphas))])
    m = mat.mean(1); s = mat.std(1)
    ax_g.fill_between(alphas, m-s, m+s, alpha=0.12, color=col)
    ax_g.plot(alphas, m, "-o", ms=3, lw=2, color=col,
              label=lbl.split("(")[0].strip())

# Separate dashed line: top-1 site alone for top1 and top1_geo
for strat, alphas, col, lbl in STRATS:
    if strat not in ("top1", "top1_geo"):
        continue
    mat_t1 = np.array([phys[strat][ai]["rms_top1"] for ai in range(len(alphas))])
    m_t1   = np.nanmean(mat_t1, axis=1)
    ax_g.plot(alphas, m_t1, "--", lw=1.5, color=col, alpha=0.65,
              label=f"{lbl.split('(')[0].strip()} (top-1 site)")

ax_g.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6,
             label=f"Physio α={PHYSIO:.0f}")
ax_g.set_xlabel("alpha")
ax_g.set_ylabel("RMS(perturbed) / RMS(baseline)")
ax_g.set_title("Signal amplitude ratio\n(stimulated sites vs baseline)")
ax_g.legend(frameon=False, fontsize=7)

# ── H: FC amplitude vs alpha ───────────────────────────────────────────────────
ax_h = fig.add_subplot(gs[2, 1])
for strat, alphas, col, lbl in STRATS:
    mat = np.array([phys[strat][ai]["fc"] for ai in range(len(alphas))])
    m = mat.mean(1); s = mat.std(1)
    ax_h.fill_between(alphas, m-s, m+s, alpha=0.12, color=col)
    ax_h.plot(alphas, m, "-o", ms=3, lw=2, color=col,
              label=lbl.split("(")[0].strip())
ax_h.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6,
             label=f"Physio α={PHYSIO:.0f}")
ax_h.set_xlabel("alpha")
ax_h.set_ylabel("Mean |FC| at stimulated sites")
ax_h.set_title("FC amplitude\n(stimulated sites, all-pairs)")
ax_h.legend(frameon=False, fontsize=7.5)

# ── I: summary bar chart — reclassified at physio limit ───────────────────────
ax_i = fig.add_subplot(gs[2, 2])
bar_labels, bar_g, bar_fl = [], [], []
for strat, alphas, col, lbl in STRATS:
    physio_ai = np.where(alphas <= PHYSIO)[0]
    physio_idx = physio_ai[-1] if len(physio_ai) else 0
    bar_labels.append(lbl.split("(")[0].strip())
    bar_g.append( (results[strat][physio_idx]["g"]  < thr_g ).mean()*100)
    bar_fl.append((results[strat][physio_idx]["fl"] < thr_fl).mean()*100)

xb = np.arange(len(STRATS))
w  = 0.35
bars_g  = ax_i.bar(xb - w/2, bar_g,  w, color=G_COL,  alpha=0.8, label="G-space LDA")
bars_fl = ax_i.bar(xb + w/2, bar_fl, w, color=FL_COL, alpha=0.8, label="FC-lag LDA")
for bar in bars_g:
    ax_i.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
              f"{bar.get_height():.0f}%", ha="center", fontsize=8, color=G_COL)
for bar in bars_fl:
    ax_i.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1.5,
              f"{bar.get_height():.0f}%", ha="center", fontsize=8, color=FL_COL)
ax_i.set_xticks(xb); ax_i.set_xticklabels(bar_labels, fontsize=8)
ax_i.set_ylabel("AD reclassified as CC (%)")
ax_i.set_title(f"Summary: reclassified\nat physiological limit (α={PHYSIO:.0f})")
ax_i.set_ylim(0, 110); ax_i.legend(frameon=False)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4 — OSCILLATORY (SINUSOIDAL) STIMULATION  (from pert_osc_data.npz)
# A·sin(2π f t) injected at the top-k brain sites; k = 1, 2, 5.
# ══════════════════════════════════════════════════════════════════════════════
if HAVE_OSC:
    osc_freqs = _osc["freqs"]
    osc_amps  = _osc["amps"]
    osc_k     = _osc["k_sites"]
    osc_res   = _osc["results_osc"]    # (n_k, n_f, n_a, n_ad)  FC-lag scores
    osc_rms   = _osc["rms_ratio"]      # (n_k, n_f, n_a, n_ad)
    osc_thr   = float(_osc["thr_fl"])
    osc_ccfl  = _osc["cc_fl"]
    f_eig     = float(_osc["f_eig"])
    f_fft     = float(_osc["f_fft"])
    psd_ad    = _osc["psd_ad"]
    freqs_fft = _osc["freqs_fft"]

    # dominant frequency: lowest mean FC-lag score (most CC-like) over k, amp, patients
    dom_fi = int(np.argmin(osc_res.mean((0, 2, 3))))
    fdom   = float(osc_freqs[dom_fi])

    # ── J: FC-lag score vs amplitude — k = 1, 2, 5 (at dominant frequency) ─────
    ax_j = fig.add_subplot(gs[3, 0])
    ax_j.axhspan(osc_ccfl.mean()-osc_ccfl.std(), osc_ccfl.mean()+osc_ccfl.std(),
                 alpha=0.1, color=CC_COL)
    ax_j.axhline(osc_ccfl.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean ±1σ")
    ax_j.axhline(osc_thr, color="gray", lw=1, ls="-.", label="Boundary")
    for ki, k in enumerate(osc_k):
        m = osc_res[ki, dom_fi].mean(1); s = osc_res[ki, dom_fi].std(1)
        ax_j.fill_between(osc_amps, m-s, m+s, alpha=0.13, color=K_OSC_COLS[ki])
        ax_j.plot(osc_amps, m, "-o", ms=4, lw=2, color=K_OSC_COLS[ki],
                  label=f"k={k} site{'s' if k > 1 else ''}")
    ax_j.set_xlabel("oscillation amplitude  A")
    ax_j.set_ylabel("FC-lag LDA score")
    ax_j.set_title(f"Oscillatory stim — FC-lag score\n"
                   f"(sin @ f={fdom:.3f} c/step, top-k sites)")
    ax_j.legend(frameon=False, fontsize=7.5)

    # ── K: reclassification rate vs amplitude — k = 1, 2, 5 ───────────────────
    ax_k = fig.add_subplot(gs[3, 1])
    for ki, k in enumerate(osc_k):
        frac = (osc_res[ki, dom_fi] < osc_thr).mean(1) * 100
        ax_k.plot(osc_amps, frac, "-o", ms=4, lw=2, color=K_OSC_COLS[ki],
                  label=f"k={k} site{'s' if k > 1 else ''}")
    ax_k.axhline(50, color="gray", ls="--", lw=1, alpha=0.4)
    ax_k.set_xlabel("oscillation amplitude  A")
    ax_k.set_ylabel("AD reclassified as CC (%)")
    ax_k.set_title("Oscillatory stim — reclassification rate\n(FC-lag, top-k sites)")
    ax_k.set_ylim(-2, 105); ax_k.legend(frameon=False, fontsize=7.5)

    # ── L: reservoir power spectrum with dominant modes marked ────────────────
    ax_l = fig.add_subplot(gs[3, 2])
    ax_l.semilogy(freqs_fft[1:], psd_ad[1:], color="#37474F", lw=1.5)
    ax_l.axvline(f_eig, color="#C62828", ls="--", lw=1.5,
                 label=f"J eigenmode  f={f_eig:.3f}")
    ax_l.axvline(f_fft, color="#E65100", ls="--", lw=1.5,
                 label=f"FFT peak  f={f_fft:.3f}")
    ax_l.axvline(fdom, color="#2E7D32", ls=":", lw=1.5,
                 label=f"used  f={fdom:.3f}")
    ax_l.set_xlabel("frequency (cycles/step)")
    ax_l.set_ylabel("mean PSD (log)")
    ax_l.set_title("Reservoir state spectrum\n(AD, dominant oscillatory modes)")
    ax_l.legend(frameon=False, fontsize=7)

fig.suptitle("Stimulation experiment: site-selective W perturbation"
             + ("  +  oscillatory (sinusoidal) stimulation" if HAVE_OSC else "")
             + f"\n(N={n_ad} AD patients, K={K_LDA}, lags 0-{MAX_LAG})",
             fontsize=12, fontweight="bold", y=1.01)

# ── panel labels (A, B, C, ...) in creation order ─────────────────────────────
for _ax, _lbl in zip(fig.axes, "ABCDEFGHIJKL"):
    _ax.text(-0.12, 1.08, _lbl, transform=_ax.transAxes,
             fontsize=13, fontweight="bold", va="top", ha="left")

os.makedirs("paper_figures", exist_ok=True)
fig.savefig("pert_sites_figure.png", bbox_inches="tight")
fig.savefig("paper_figures/figure4_sites.png", bbox_inches="tight")
fig.savefig("paper_figures/figure4_sites.pdf", bbox_inches="tight")
plt.close(fig)
print("Saved pert_sites_figure.png")
print("Saved paper_figures/figure4_sites.png")
print("Saved paper_figures/figure4_sites.pdf")

print("\nDone.")
