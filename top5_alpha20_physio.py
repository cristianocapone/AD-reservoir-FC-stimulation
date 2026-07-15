"""
top5_alpha20_physio.py
======================
Extends top-5 perturbation to α=20 and adds physiological sanity checks
on the reconstructed signal and FC at each alpha step.

Diagnostics per AD patient, per alpha:
  1. LDA score  (already computed)
  2. FC-r       (corr between FC(W_int.T @ X) and CC FC template)
  3. Signal RMS ratio  — RMS(Y_int) / RMS(Y_baseline)  per top-5 site
  4. FC off-diagonal std  — spread of FC entries (should stay ~real data range)
  5. FC extreme fraction  — fraction of |r_ij| > 0.80  (implausibly high)
  6. Signal site-wise variance ratio  — var(Y_int[:, site]) / var(Y_base[:, site])
     averaged over top-5 sites vs all other sites

Saves:
  pertB_direct_data.npz   (overwrites; top5 grid now 0 → 20)
  top5_physio.png         (full diagnostic figure)
  paper_figures/figure4_stimulation.png  (re-rendered)
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
import matplotlib.gridspec as gridspec
from tqdm import trange, tqdm
import warnings as _w; _w.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── params (identical to original) ────────────────────────────────────────────
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
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

# ── extended top-5 grid up to α = 20 ──────────────────────────────────────────
ALPHA_GRIDS = {
    "full_w": np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]),
    "top5":   np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0,
                         5.0, 6.0, 7.5, 10.0, 12.5, 15.0, 17.5, 20.0]),
    "top1":   np.array([0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]),
}

# ── data loading ──────────────────────────────────────────────────────────────
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

pid_raw    = np.array(pid_raw)
labels_raw = np.array(labels_raw)
N_subj     = len(signals)
unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_pids = [unique_pids[i] for i in np.where(patient_labels == 0)[0]]
ad_pids = [unique_pids[i] for i in np.where(patient_labels == 1)[0]]
print(f"  {N_patients} patients ({len(cc_pids)} CC, {len(ad_pids)} AD), {N_subj} sessions")

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── reservoir ─────────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

# ── TF pass ───────────────────────────────────────────────────────────────────
print("TF pass ...")
sess_X, sess_Y, sess_tgt = {}, {}, {}
for idx in trange(N_subj, desc="  TF"):
    s = signals[idx]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    Xraw = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        Xraw.append(res.X.copy())
    Xf = np.array(Xraw)[TIMES_SKIP:]
    sess_X[idx]   = Xf
    sess_Y[idx]   = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T
    sess_tgt[idx] = tgt

# ── W fitting + G-space ───────────────────────────────────────────────────────
print("W fitting + G-space ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
first_idx      = {pid: patient_sids[pid][0] for pid in unique_pids}
patX_single    = {pid: sess_X[first_idx[pid]]   for pid in unique_pids}
patY_single    = {pid: sess_Y[first_idx[pid]]   for pid in unique_pids}
pat_tgt_single = {pid: sess_tgt[first_idx[pid]] for pid in unique_pids}

pat_Vtk = {}
for pid in tqdm(unique_pids, desc="  SVD", leave=False):
    Xca = patX_single[pid].astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk = min(K_PC, int((sx > 1e-8).sum()))
    pat_Vtk[pid] = Vtx[:kk]

def project_W(W, pid):
    Vt_k = pat_Vtk[pid]
    return (W.T.astype(np.float64) @ Vt_k.T @ Vt_k).flatten()

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

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
    n = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i,n,replace=False),
                          rng2.choice(c1i,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

Xlda, ylda = _balance(G_B[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda_B = _LDA().fit(Xlda, ylda)
Z_base = lda_B.transform(G_B[:, :K_LDA])
if Z_base[patient_labels==0].mean() > Z_base[patient_labels==1].mean():
    lda_B.w_ *= -1; lda_B.thr_ *= -1
Z_base = lda_B.transform(G_B[:, :K_LDA])
cc_lda = Z_base[patient_labels==0]
ad_lda = Z_base[patient_labels==1]
print(f"  Baseline: CC={cc_lda.mean():.3f}±{cc_lda.std():.3f}  "
      f"AD={ad_lda.mean():.3f}±{ad_lda.std():.3f}")

def w_to_lda(W, pid):
    wp = project_W(W, pid)
    g  = ((wp - Wmean_B) @ Vsvd_B[:Meff_B].T)[:K_LDA]
    return float(lda_B.transform(g.reshape(1, -1))[0])

# ── reference quantities ───────────────────────────────────────────────────────
W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

fc_cc_list = []
for pid in cc_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    fc_cc_list.append(np.nan_to_num(np.corrcoef(tgt)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)

cc_fc_r = np.array([
    float(np.corrcoef(np.nan_to_num(
        np.corrcoef(pat_tgt_single[pid][:, TIMES_SKIP:])).flatten(),
        FC_cc_mean)[0, 1])
    for pid in cc_pids])

ad_fc_r_base = np.array([
    float(np.corrcoef(np.nan_to_num(
        np.corrcoef(pat_tgt_single[pid][:, TIMES_SKIP:])).flatten(),
        FC_cc_mean)[0, 1])
    for pid in ad_pids])

# Baseline per-patient signal RMS per site  (N_sites,)  for normalisation
ad_sig_rms_base = {}   # pid -> (N_sites,) RMS per site at alpha=0
for pid in ad_pids:
    Xc   = patX_single[pid].astype(np.float64)
    Y0   = (pat_W[pid].T @ Xc.T)          # (N_sites, T_eff)
    ad_sig_rms_base[pid] = np.sqrt(np.mean(Y0**2, axis=1))   # (N_sites,)

# ── extended signal diagnostics helper ────────────────────────────────────────
def signal_diagnostics(W_int, pid, top_k_sites):
    """Returns dict of scalar physio metrics for W_int."""
    Xc = patX_single[pid].astype(np.float64)
    Y  = (W_int.T.astype(np.float64) @ Xc.T)    # (N_sites, T_eff)
    rms_int  = np.sqrt(np.mean(Y**2, axis=1))    # (N_sites,)
    rms_base = ad_sig_rms_base[pid]

    # Ratio: perturbed / baseline RMS, separately for top-5 and remaining sites
    all_sites  = np.arange(N_SITES)
    rest_sites = np.setdiff1d(all_sites, top_k_sites)
    rms_ratio_top5 = (rms_int[top_k_sites] /
                      (rms_base[top_k_sites] + 1e-12)).mean()
    rms_ratio_rest = (rms_int[rest_sites] /
                      (rms_base[rest_sites] + 1e-12)).mean()

    # FC matrix
    fc_mat  = np.nan_to_num(np.corrcoef(Y))          # (N_sites, N_sites)
    off_diag = fc_mat[np.triu_indices(N_SITES, k=1)]
    fc_mean  = off_diag.mean()
    fc_std   = off_diag.std()
    # fraction of off-diagonal |r| > 0.80 (implausibly tight coupling)
    fc_extreme_frac = (np.abs(off_diag) > 0.80).mean()

    # FC-r with CC template
    fc_r = float(np.corrcoef(fc_mat.flatten(), FC_cc_mean)[0, 1])

    return {
        "rms_ratio_top5":   float(rms_ratio_top5),
        "rms_ratio_rest":   float(rms_ratio_rest),
        "fc_r":             fc_r,
        "fc_mean":          float(fc_mean),
        "fc_std":           float(fc_std),
        "fc_extreme_frac":  float(fc_extreme_frac),
    }

def w_fc_r(W, pid):
    """Scalar FC-r for saving to npz."""
    Xc = patX_single[pid].astype(np.float64)
    Y  = (W.T.astype(np.float64) @ Xc.T)
    fc = np.nan_to_num(np.corrcoef(Y)).flatten()
    return float(np.corrcoef(fc, FC_cc_mean)[0, 1])

# ── perturbation loop ──────────────────────────────────────────────────────────
print("\nPerturbation + physio diagnostics ...")
results      = {}
physio_top5  = []    # list of dicts per alpha (only for top5)

for pert_type, alphas in ALPHA_GRIDS.items():
    print(f"\n  [{pert_type}]  α ∈ [{alphas.min():.2g}, {alphas.max():.2g}]"
          f"  ({len(alphas)} points)")
    results[pert_type] = []
    for ai, alpha in enumerate(alphas):
        alpha_res = {}
        physio_accum = {"rms_top5": [], "rms_rest": [], "fc_r": [],
                        "fc_mean": [], "fc_std": [], "fc_extreme": []}
        for pid in ad_pids:
            Wp = pat_W[pid]
            dW = W_cc_mean - Wp
            norms = np.linalg.norm(dW, axis=0)

            if pert_type == "full_w":
                W_int  = (1 - alpha) * Wp + alpha * W_cc_mean
                top_k  = np.arange(N_SITES)
            elif pert_type == "top5":
                top_k  = np.argsort(norms)[::-1][:5]
                W_int  = Wp.copy()
                W_int[:, top_k] = ((1-alpha)*Wp[:, top_k]
                                   + alpha*W_cc_mean[:, top_k])
            else:   # top1
                top_k  = np.argsort(norms)[::-1][:1]
                W_int  = Wp.copy()
                W_int[:, top_k] = ((1-alpha)*Wp[:, top_k]
                                   + alpha*W_cc_mean[:, top_k])

            lda_sc = w_to_lda(W_int, pid)
            fc_r_v = w_fc_r(W_int, pid)
            alpha_res[pid] = (lda_sc, fc_r_v)

            # Extra diagnostics only for top5
            if pert_type == "top5":
                dg = signal_diagnostics(W_int, pid, top_k)
                physio_accum["rms_top5"].append(dg["rms_ratio_top5"])
                physio_accum["rms_rest"].append(dg["rms_ratio_rest"])
                physio_accum["fc_r"].append(dg["fc_r"])
                physio_accum["fc_mean"].append(dg["fc_mean"])
                physio_accum["fc_std"].append(dg["fc_std"])
                physio_accum["fc_extreme"].append(dg["fc_extreme_frac"])

        results[pert_type].append(alpha_res)
        lda_vals = [alpha_res[p][0] for p in ad_pids]
        fcr_vals = [alpha_res[p][1] for p in ad_pids]

        if pert_type == "top5":
            physio_top5.append({k: np.mean(v)
                                for k, v in physio_accum.items()})
            physio_top5[-1]["alpha"] = alpha
            print(f"    α={alpha:5.2f}  LDA={np.mean(lda_vals):+.3f}"
                  f"  FC-r={np.mean(fcr_vals):.3f}"
                  f"  RMS_top5={physio_top5[-1]['rms_top5']:.2f}"
                  f"  RMS_rest={physio_top5[-1]['rms_rest']:.2f}"
                  f"  FC_extreme={physio_top5[-1]['fc_extreme']*100:.1f}%",
                  flush=True)
        else:
            print(f"    α={alpha:5.2f}  LDA={np.mean(lda_vals):+.3f}"
                  f"  FC-r={np.mean(fcr_vals):.3f}", flush=True)

# ── save updated npz ───────────────────────────────────────────────────────────
save_dict = {
    "ad_pids": np.array(ad_pids), "cc_pids": np.array(cc_pids),
    "patient_labels": patient_labels, "Z_base": Z_base,
    "cc_lda": cc_lda, "ad_lda": ad_lda,
    "cc_fc_r": cc_fc_r, "ad_fc_r_base": ad_fc_r_base,
}
for pert_type, alphas in ALPHA_GRIDS.items():
    ad = results[pert_type]
    lda_arr = np.array([[ad[ai][pid][0] for pid in ad_pids]
                         for ai in range(len(alphas))])
    fcr_arr = np.array([[ad[ai][pid][1] for pid in ad_pids]
                         for ai in range(len(alphas))])
    save_dict[f"{pert_type}_alphas"] = alphas
    save_dict[f"{pert_type}_lda"]    = lda_arr
    save_dict[f"{pert_type}_fcr"]    = fcr_arr

np.savez(f"{OUT_DIR}/pertB_direct_data.npz", **save_dict)
print("\nSaved pertB_direct_data.npz")

# ── physio diagnostic figure ───────────────────────────────────────────────────
print("Building physio diagnostic figure ...")
t5_alphas = ALPHA_GRIDS["top5"]
ph = physio_top5   # list of dicts

def _pa(key):
    return np.array([p[key] for p in ph])

rms_top5   = _pa("rms_top5")
rms_rest   = _pa("rms_rest")
fc_r_arr   = _pa("fc_r")
fc_mean_a  = _pa("fc_mean")
fc_std_a   = _pa("fc_std")
fc_ext_a   = _pa("fc_extreme") * 100   # %

# Also get the LDA and per-patient FCr arrays for top5
lda_mat  = np.array([[results["top5"][ai][pid][0] for pid in ad_pids]
                      for ai in range(len(t5_alphas))])
fcr_mat  = np.array([[results["top5"][ai][pid][1] for pid in ad_pids]
                      for ai in range(len(t5_alphas))])

ad_cc_mid = 0.5*(cc_lda.mean() + ad_lda.mean())
T5_COL  = "#E65100"
CC_COL  = "#1565C0"
AD_COL  = "#C62828"

fig = plt.figure(figsize=(16, 12), facecolor="white")
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

# ── 1. LDA score trajectory ────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.axhspan(cc_lda.mean()-cc_lda.std(), cc_lda.mean()+cc_lda.std(),
            alpha=0.15, color=CC_COL, label="CC ±1σ")
ax1.axhline(cc_lda.mean(), color=CC_COL, lw=1.8, ls="--")
ax1.axhline(ad_lda.mean(), color=AD_COL, lw=1.2, ls=":", alpha=0.7)
ax1.axhline(ad_cc_mid, color="gray", lw=0.8, ls="-.", alpha=0.5,
            label=f"Midpoint = {ad_cc_mid:.2f}")
for pi in range(len(ad_pids)):
    ax1.plot(t5_alphas, lda_mat[:, pi], lw=0.5, color=T5_COL, alpha=0.18)
m = lda_mat.mean(1); s = lda_mat.std(1)
ax1.fill_between(t5_alphas, m-s, m+s, alpha=0.22, color=T5_COL)
ax1.plot(t5_alphas, m, "-o", ms=4, lw=2.0, color=T5_COL, label="AD mean ±1σ")
ax1.set_xlabel("α"); ax1.set_ylabel("LDA score  (K=25)")
ax1.set_title("LDA score vs α  (top-5 sites)")
ax1.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax1.spines[sp].set_visible(False)

# ── 2. FC-r with CC template ───────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.axhspan(cc_fc_r.mean()-cc_fc_r.std(), cc_fc_r.mean()+cc_fc_r.std(),
            alpha=0.15, color=CC_COL)
ax2.axhline(cc_fc_r.mean(), color=CC_COL, lw=1.8, ls="--", label="CC FC-r ±1σ")
ax2.axhline(ad_fc_r_base.mean(), color=AD_COL, lw=1.2, ls=":", alpha=0.7,
            label=f"AD baseline = {ad_fc_r_base.mean():.3f}")
for pi in range(len(ad_pids)):
    ax2.plot(t5_alphas, fcr_mat[:, pi], lw=0.5, color=T5_COL, alpha=0.18)
mf = fcr_mat.mean(1); sf = fcr_mat.std(1)
ax2.fill_between(t5_alphas, mf-sf, mf+sf, alpha=0.22, color=T5_COL)
ax2.plot(t5_alphas, mf, "-o", ms=4, lw=2.0, color=T5_COL, label="AD mean ±1σ")
ax2.set_xlabel("α"); ax2.set_ylabel("FC-r (vs CC template)")
ax2.set_title("FC similarity to CC mean")
ax2.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax2.spines[sp].set_visible(False)

# ── 3. Signal RMS ratio ────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
ax3.axhline(1.0, color="gray", lw=1.2, ls="--", alpha=0.7, label="Baseline (=1)")
ax3.plot(t5_alphas, rms_top5, "-o", ms=4, lw=2.0, color=AD_COL,
         label="Top-5 sites")
ax3.plot(t5_alphas, rms_rest, "-s", ms=4, lw=1.5, color="#37474F",
         label="Other 116 sites", alpha=0.85)
# Shade "unphysiological" zone (>3× or <0.33×)
ax3.axhspan(3.0, ax3.get_ylim()[1] if ax3.get_ylim()[1] > 3 else 20,
            alpha=0.10, color="red")
ax3.axhspan(0, 0.33, alpha=0.10, color="red")
ax3.axhline(3.0, color="red", lw=1.0, ls=":", alpha=0.6, label="×3 / ×0.33 threshold")
ax3.axhline(0.33, color="red", lw=1.0, ls=":", alpha=0.6)
ax3.set_xlabel("α"); ax3.set_ylabel("RMS ratio  (W_int vs W_base)")
ax3.set_title("Signal RMS ratio\n(top-5 perturbed sites vs rest)")
ax3.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax3.spines[sp].set_visible(False)

# ── 4. FC off-diagonal std ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 0])
# CC reference: compute fc_std for CC patients
cc_fc_stds = []
for pid in cc_pids:
    tgt = pat_tgt_single[pid][:, TIMES_SKIP:]
    Xc  = patX_single[pid].astype(np.float64)
    Y0  = (pat_W[pid].T @ Xc.T)
    fm  = np.nan_to_num(np.corrcoef(Y0))
    od  = fm[np.triu_indices(N_SITES, k=1)]
    cc_fc_stds.append(od.std())
cc_fc_std_mean = np.mean(cc_fc_stds)
cc_fc_std_sd   = np.std(cc_fc_stds)

ax4.axhspan(cc_fc_std_mean-cc_fc_std_sd, cc_fc_std_mean+cc_fc_std_sd,
            alpha=0.15, color=CC_COL)
ax4.axhline(cc_fc_std_mean, color=CC_COL, lw=1.8, ls="--", label="CC ±1σ")
ax4.plot(t5_alphas, fc_std_a, "-o", ms=4, lw=2.0, color=T5_COL,
         label="AD top-5 (mean)")
ax4.set_xlabel("α"); ax4.set_ylabel("FC off-diagonal std")
ax4.set_title("FC spread (off-diagonal σ)\n—  proxy for overall connectivity structure")
ax4.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax4.spines[sp].set_visible(False)

# ── 5. FC mean (off-diagonal) ─────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 1])
cc_fc_means = []
for pid in cc_pids:
    Xc  = patX_single[pid].astype(np.float64)
    Y0  = (pat_W[pid].T @ Xc.T)
    fm  = np.nan_to_num(np.corrcoef(Y0))
    od  = fm[np.triu_indices(N_SITES, k=1)]
    cc_fc_means.append(od.mean())
cc_fc_mean_mean = np.mean(cc_fc_means)
cc_fc_mean_sd   = np.std(cc_fc_means)

ax5.axhspan(cc_fc_mean_mean-cc_fc_mean_sd,
            cc_fc_mean_mean+cc_fc_mean_sd,
            alpha=0.15, color=CC_COL)
ax5.axhline(cc_fc_mean_mean, color=CC_COL, lw=1.8, ls="--", label="CC ±1σ")
ax5.plot(t5_alphas, fc_mean_a, "-o", ms=4, lw=2.0, color=T5_COL,
         label="AD top-5 (mean)")
ax5.axhline(0.0, color="gray", lw=0.8, ls="--", alpha=0.5)
ax5.set_xlabel("α"); ax5.set_ylabel("FC mean off-diagonal r")
ax5.set_title("Mean FC strength (off-diagonal)\n—  global connectivity level")
ax5.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax5.spines[sp].set_visible(False)

# ── 6. FC extreme fraction ─────────────────────────────────────────────────────
ax6 = fig.add_subplot(gs[1, 2])
cc_fc_ext = []
for pid in cc_pids:
    Xc = patX_single[pid].astype(np.float64)
    Y0 = (pat_W[pid].T @ Xc.T)
    fm = np.nan_to_num(np.corrcoef(Y0))
    od = fm[np.triu_indices(N_SITES, k=1)]
    cc_fc_ext.append((np.abs(od) > 0.80).mean() * 100)
cc_ext_m = np.mean(cc_fc_ext); cc_ext_s = np.std(cc_fc_ext)

ax6.axhspan(cc_ext_m-cc_ext_s, cc_ext_m+cc_ext_s,
            alpha=0.15, color=CC_COL)
ax6.axhline(cc_ext_m, color=CC_COL, lw=1.8, ls="--", label=f"CC ±1σ  ({cc_ext_m:.1f}%)")
ax6.plot(t5_alphas, fc_ext_a, "-o", ms=4, lw=2.0, color=T5_COL,
         label="AD top-5 (mean)")
ax6.set_xlabel("α"); ax6.set_ylabel("% edges with |FC| > 0.80")
ax6.set_title("Extreme FC fraction\n(implausibly strong connections)")
ax6.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax6.spines[sp].set_visible(False)

# ── 7. Reclassification fraction + summary table ───────────────────────────────
ax7 = fig.add_subplot(gs[2, 0:2])
thr_base = ad_cc_mid
frac_base = (ad_lda >= thr_base).mean() * 100
ax7.axhline(frac_base, color=AD_COL, ls=":", lw=1.2, alpha=0.7,
            label=f"Baseline ({frac_base:.0f}%)")
ax7.axhline(50, color="gray", ls="--", lw=1.0, alpha=0.5)

PERT_COLS = {"full_w": "#1B5E20", "top5": "#E65100", "top1": "#6A1B9A"}
LABELS    = {"full_w": "Full-W (121)", "top5": "Top-5", "top1": "Top-1"}
for pt, col in PERT_COLS.items():
    alphas = ALPHA_GRIDS[pt]
    mat    = np.array([[results[pt][ai][pid][0] for pid in ad_pids]
                        for ai in range(len(alphas))])
    frac   = (mat >= thr_base).mean(1) * 100
    ax7.plot(alphas, frac, "-o", ms=4, lw=2.0, color=col, label=LABELS[pt])

ax7.set_xlabel("α"); ax7.set_ylabel("AD patients reclassified as CC (%)")
ax7.set_title("Reclassification rate vs α  (all strategies, extended range)")
ax7.set_ylim(-2, 102)
ax7.legend(frameon=False, fontsize=8)
for sp in ["top","right"]: ax7.spines[sp].set_visible(False)

# ── 8. Summary: physio score = composite index ────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
# Simple composite physiological plausibility score (0=best, 1=worst):
#   penalty for RMS deviation from 1 (for top-5 sites)
#   penalty for FC extreme fraction increase
rms_penalty  = np.abs(rms_top5 - 1.0)    # 0 at baseline
ext_norm     = (fc_ext_a - fc_ext_a[0]) / (fc_ext_a[-1] - fc_ext_a[0] + 1e-6)
physio_score = 1.0 - 0.5*np.clip(rms_penalty / rms_penalty.max(), 0, 1) \
                   - 0.5*np.clip(ext_norm, 0, 1)
ax8.plot(t5_alphas, physio_score, "-o", ms=5, lw=2.2, color="#37474F",
         label="Composite physio score")
ax8.axhline(0.9, color="green", ls="--", lw=1.0, alpha=0.7,
            label="Plausible (>0.9)")
ax8.axhline(0.7, color="orange", ls=":", lw=1.0, alpha=0.7,
            label="Marginal (>0.7)")
ax8.set_xlabel("α"); ax8.set_ylabel("Score  (1=baseline, 0=unphysiological)")
ax8.set_title("Composite physiological plausibility\n(top-5 perturbation)")
ax8.set_ylim(-0.05, 1.10)
ax8.legend(frameon=False, fontsize=7)
for sp in ["top","right"]: ax8.spines[sp].set_visible(False)

fig.suptitle(
    "Top-5 perturbation — α=0→20 — Physiological signal & FC diagnostics\n"
    "Cond. B  (σ=0.05, K_LDA=25, N_hidden=2000, sr=0.95)",
    fontsize=11, fontweight="bold", y=1.005)

fig.savefig(f"{OUT_DIR}/top5_physio.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved top5_physio.png")

# ── also regenerate figure4 ────────────────────────────────────────────────────
print("Regenerating paper figure 4 ...")
import subprocess
subprocess.run(["python", "paper_figures/figure4_stimulation.py"],
               cwd=OUT_DIR, check=True)

print("\nDone.")
