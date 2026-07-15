"""
pert_mode1_stimulation.py
=========================
Compare two site-selection criteria for focal W-interpolation:

  norm    : top-k sites by ||dW[:, k]||  (largest per-site correction)
  mode1   : top-k sites by |W_AD[:, k].T @ v1|  where v1 is the first
            right singular vector of X_res (dominant reservoir eigenmode)

The stimulation signal is the same for both:
    W_int[:, sites] = (1-alpha)*W_AD[:, sites] + alpha*W_CC[:, sites]
i.e.  delta_Y_k(t) = alpha * dW[:, k].T @ X_res(t)  at each selected site.

The hypothesis: stimulating sites that project strongly onto the dominant
reservoir mode may propagate the correction more efficiently through the
recurrent dynamics, even if those sites have smaller ||dW|| individually.

Compares k=1 and k=2 for both criteria (4 strategies total).

Saves:
  pert_mode1_data.npz
  pert_mode1_figure.png
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
import warnings; warnings.filterwarnings("ignore")
from tqdm import trange, tqdm

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── parameters (identical to pert_sites_stimulation.py) ───────────────────────
RNG_SEED   = 42
N_CC_SAMP  = 40
N_SITES    = 121
N_PC_MODEL = 50
K_PC       = 200
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
K_LDA      = 25
SR         = 0.95
MAX_LAG    = 2
TS_ROOT    = "./timeseries"
PHYSIO     = 4.0

ALPHAS = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 7.5, 10, 12.5, 15, 17.5, 20])

# colours
CC_COL  = "#1565C0"
AD_COL  = "#C62828"
# norm-based: warm tones; mode1-based: cool tones
N1_COL  = "#6A1B9A"   # norm  top-1  purple
N2_COL  = "#E65100"   # norm  top-2  orange
N5_COL  = "#B71C1C"   # norm  top-5  dark red
M1_COL  = "#00695C"   # mode1 top-1  teal
M2_COL  = "#0277BD"   # mode1 top-2  blue
M5_COL  = "#1A237E"   # mode1 top-5  indigo
G_COL   = "#37474F"
FL_COL  = "#0097A7"

STRATS = [
    ("norm_top1",  1, N1_COL, "Norm-top1   (k=1, ||dW||)"),
    ("norm_top2",  2, N2_COL, "Norm-top2   (k=2, ||dW||)"),
    ("norm_top5",  5, N5_COL, "Norm-top5   (k=5, ||dW||)"),
    ("mode1_top1", 1, M1_COL, "Mode1-top1  (k=1, v1 proj.)"),
    ("mode1_top2", 2, M2_COL, "Mode1-top2  (k=2, v1 proj.)"),
    ("mode1_top5", 5, M5_COL, "Mode1-top5  (k=5, v1 proj.)"),
]

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
# RESERVOIR
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
first_idx = {pid: patient_sids[pid][0] for pid in unique_pids}
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

print("W fitting ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
patX_single = {pid: sess_X[first_idx[pid]] for pid in unique_pids}
patY_single = {pid: sess_Y[first_idx[pid]] for pid in unique_pids}

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]; Yc = patY_single[pid]
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

# ── per-patient first reservoir eigenmode ─────────────────────────────────────
print("Computing first reservoir eigenmode per AD patient ...")
pat_v1 = {}   # first right singular vector of X_res  (N_hidden,)
for pid in ad_pids:
    Xr = patX_single[pid].astype(np.float64)    # (T_eff, N_hidden)
    _, _, Vt = np.linalg.svd(Xr, full_matrices=False)
    pat_v1[pid] = Vt[0]                          # (N_hidden,)

# ── site selection functions ──────────────────────────────────────────────────
def select_sites_norm(pid, k):
    """Top-k sites by per-site correction norm ||dW[:, k]||."""
    dW    = W_cc_mean - pat_W[pid]
    norms = np.linalg.norm(dW, axis=0)
    return np.argsort(norms)[::-1][:k]

def select_sites_mode1(pid, k):
    """Top-k sites by |W_AD[:, k].T @ v1| — signal in first reservoir eigenmode."""
    v1     = pat_v1[pid]                            # (N_hidden,)
    scores = np.abs(pat_W[pid].T @ v1)              # (N_sites,)
    return np.argsort(scores)[::-1][:k]

# ══════════════════════════════════════════════════════════════════════════════
# LDA HELPERS  (same as pert_sites_stimulation.py)
# ══════════════════════════════════════════════════════════════════════════════
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.thr_ = 0.5*(mu0@w + mu1@w); return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i, n, replace=False),
                          rng2.choice(c1i, n, replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

# ── G-space ───────────────────────────────────────────────────────────────────
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

Wproj = np.array([project_W(pat_W[pid], pid) for pid in unique_pids])
Wmean = Wproj.mean(0); Wcent = Wproj - Wmean
_, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
Meff  = N_patients - 1
G_B   = Wcent @ Vsvd[:Meff].T

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

# ── FC-lag ────────────────────────────────────────────────────────────────────
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
    S  = (W.T.astype(np.float64) @ Xc.T).T
    feats = []
    for lag in range(MAX_LAG + 1):
        fc = lagged_corrcoef(S, lag)
        fc = np.nan_to_num(fc, nan=0., posinf=0., neginf=0.)
        feats.append(fc[np.triu_indices(N_SITES, k=1)] if lag == 0
                     else fc.flatten())
    return np.concatenate(feats)

print("Computing FC-lag LDA ...")
fclag_base = np.array([compute_fclag_feat(pat_W[pid], pid)
                        for pid in tqdm(unique_pids, desc="  FC-lag",
                                        leave=False)])
fclag_mean = fclag_base.mean(0)
fclag_c    = fclag_base - fclag_mean
ev_fl, evec_fl = np.linalg.eigh(fclag_c @ fclag_c.T)
ord_fl   = np.argsort(ev_fl)[::-1]
ev_fl    = np.maximum(ev_fl[ord_fl], 0); evec_fl = evec_fl[:, ord_fl]
G_fl     = evec_fl * np.sqrt(ev_fl)

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
# SHOW SITE OVERLAP: which sites does each criterion pick?
# ══════════════════════════════════════════════════════════════════════════════
print("\nSite selection comparison (across 40 AD patients) ...")
site_counts = {name: np.zeros(N_SITES, int) for name, *_ in STRATS}
for pid in ad_pids:
    for name, k, _, _ in STRATS:
        if name.startswith("norm"):
            sites = select_sites_norm(pid, k)
        else:
            sites = select_sites_mode1(pid, k)
        site_counts[name][sites] += 1

for name, k, _, label in STRATS:
    top = np.argsort(site_counts[name])[::-1][:5]
    top_str = "  ".join([f"s{s}({site_counts[name][s]})" for s in top
                          if site_counts[name][s] > 0])
    print(f"  {label:35s}: {top_str}")

# overlap between norm and mode1 for each k
for k in [1, 2, 5]:
    ov = np.zeros(n_ad)
    for pi, pid in enumerate(ad_pids):
        sn = set(select_sites_norm(pid, k).tolist())
        sm = set(select_sites_mode1(pid, k).tolist())
        ov[pi] = len(sn & sm)
    print(f"  Mean site overlap (norm_top{k} vs mode1_top{k}): "
          f"{ov.mean():.2f}/{k}")

# ══════════════════════════════════════════════════════════════════════════════
# PERTURBATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
print("\nPerturbation loop ...")
results = {}

for name, k, col, label in STRATS:
    print(f"  [{name}]")
    results[name] = {}
    for ai, alpha in enumerate(ALPHAS):
        g_sc  = np.zeros(n_ad)
        fl_sc = np.zeros(n_ad)
        for pi, pid in enumerate(ad_pids):
            Wp = pat_W[pid]
            if name.startswith("norm"):
                sites = select_sites_norm(pid, k)
            else:
                sites = select_sites_mode1(pid, k)
            W_int = Wp.copy()
            W_int[:, sites] = (1-alpha)*Wp[:, sites] + alpha*W_cc_mean[:, sites]
            g_sc[pi]  = W_to_glda(W_int, pid)
            fl_sc[pi] = W_to_fllda(W_int, pid)
        results[name][ai] = {"g": g_sc.copy(), "fl": fl_sc.copy()}
        print(f"    alpha={alpha:5.2f}  G={g_sc.mean():+.3f}  FL={fl_sc.mean():+.3f}",
              flush=True)

# ── crossing alphas ───────────────────────────────────────────────────────────
def alpha_cross(name, thr, space):
    out = np.full(n_ad, np.nan)
    for pi in range(n_ad):
        for ai, a in enumerate(ALPHAS):
            if results[name][ai][space][pi] < thr:
                out[pi] = a; break
    return out

cross_g  = {n: alpha_cross(n, thr_g,  "g")  for n, *_ in STRATS}
cross_fl = {n: alpha_cross(n, thr_fl, "fl") for n, *_ in STRATS}

# ── summary ───────────────────────────────────────────────────────────────────
print(f"\n{'Strategy':35s}  {'G cross(med)':>12s}  {'FL cross(med)':>13s}  "
      f"{'G @physio':>10s}  {'FL @physio':>10s}")
print("-"*90)
for name, k, col, label in STRATS:
    physio_ai  = np.where(ALPHAS <= PHYSIO)[0]
    pidx       = physio_ai[-1] if len(physio_ai) else 0
    pct_g  = (results[name][pidx]["g"]  < thr_g ).mean()*100
    pct_fl = (results[name][pidx]["fl"] < thr_fl).mean()*100
    med_g  = np.nanmedian(cross_g[name])
    med_fl = np.nanmedian(cross_fl[name])
    sg = f"{med_g:.1f}" if not np.isnan(med_g) else ">max"
    sf = f"{med_fl:.1f}" if not np.isnan(med_fl) else ">max"
    print(f"{label:35s}  {sg:>12s}  {sf:>13s}  {pct_g:9.0f}%  {pct_fl:9.0f}%")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez("pert_mode1_data.npz",
         alphas=ALPHAS, cc_g=cc_g, ad_g=ad_g, thr_g=np.array(thr_g),
         cc_fl=cc_fl, ad_fl=ad_fl, thr_fl=np.array(thr_fl),
         **{f"{n}_{ai}_g":  results[n][ai]["g"]  for n, *_ in STRATS
            for ai in range(len(ALPHAS))},
         **{f"{n}_{ai}_fl": results[n][ai]["fl"] for n, *_ in STRATS
            for ai in range(len(ALPHAS))},
         **{f"sites_{n}": site_counts[n] for n, *_ in STRATS})
print("\nSaved pert_mode1_data.npz")

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

def plot_traj(ax, name, space, col, lbl, ls="-"):
    mat = np.array([results[name][ai][space] for ai in range(len(ALPHAS))])
    m = mat.mean(1); s = mat.std(1)
    ax.fill_between(ALPHAS, m-s, m+s, alpha=0.13, color=col)
    ax.plot(ALPHAS, m, ls+"o", ms=4, lw=2, color=col, label=lbl)

fig = plt.figure(figsize=(16, 10), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.42)

# ── A: G-space score ──────────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.axhspan(cc_g.mean()-cc_g.std(), cc_g.mean()+cc_g.std(),
             alpha=0.1, color=CC_COL)
ax_a.axhline(cc_g.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean")
ax_a.axhline(thr_g, color="gray", lw=1, ls="-.", label="Boundary")
for name, k, col, lbl in STRATS:
    ls = "-" if k == 1 else "--"
    plot_traj(ax_a, name, "g", col, lbl.replace("(", "\n("), ls)
ax_a.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6)
ax_a.set_xlabel("alpha"); ax_a.set_ylabel("G-space LDA score")
ax_a.set_title("G-space LDA score\nnorm vs mode1 site selection")
ax_a.legend(frameon=False, fontsize=7)

# ── B: FC-lag score ───────────────────────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
ax_b.axhspan(cc_fl.mean()-cc_fl.std(), cc_fl.mean()+cc_fl.std(),
             alpha=0.1, color=CC_COL)
ax_b.axhline(cc_fl.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean")
ax_b.axhline(thr_fl, color="gray", lw=1, ls="-.", label="Boundary")
for name, k, col, lbl in STRATS:
    ls = "-" if k == 1 else "--"
    plot_traj(ax_b, name, "fl", col, lbl.replace("(", "\n("), ls)
ax_b.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6)
ax_b.set_xlabel("alpha"); ax_b.set_ylabel("FC-lag LDA score")
ax_b.set_title("FC-lag LDA score\nnorm vs mode1 site selection")
ax_b.legend(frameon=False, fontsize=7)

# ── C: Reclassification rates (both classifiers) ──────────────────────────────
ax_c = fig.add_subplot(gs[0, 2])
for name, k, col, lbl in STRATS:
    mat_g  = np.array([results[name][ai]["g"]  for ai in range(len(ALPHAS))])
    mat_fl = np.array([results[name][ai]["fl"] for ai in range(len(ALPHAS))])
    ls = "-" if k == 1 else "--"
    ax_c.plot(ALPHAS, (mat_g  < thr_g ).mean(1)*100, ls+"o",
              ms=3, lw=2, color=col, alpha=0.6)
    ax_c.plot(ALPHAS, (mat_fl < thr_fl).mean(1)*100, ls+"s",
              ms=3, lw=2, color=col, label=lbl.replace("(", "\n("))
ax_c.axvline(PHYSIO, color="#B71C1C", lw=1, ls=":", alpha=0.6,
             label=f"Physio alpha={PHYSIO:.0f}")
ax_c.axhline(50, color="gray", ls="--", lw=1, alpha=0.4)
ax_c.set_xlabel("alpha"); ax_c.set_ylabel("AD reclassified as CC (%)")
ax_c.set_title("Reclassification rate\n(circles=G, squares=FL)")
ax_c.set_ylim(-2, 105); ax_c.legend(frameon=False, fontsize=6.5)

# ── D: Site selection histogram comparison ────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
x = np.arange(N_SITES)
sorted_idx = np.argsort(site_counts["norm_top5"] + site_counts["mode1_top5"])[::-1]
w = 0.14
for i, (name, k, col, lbl) in enumerate(STRATS):
    ax_d.bar(x + (i-2.5)*w, site_counts[name][sorted_idx], w,
             color=col, alpha=0.75, label=lbl.split("(")[0].strip())
ax_d.set_xlabel("Brain site rank (by combined frequency)")
ax_d.set_ylabel("# AD patients selecting this site")
ax_d.set_title("Site selection: norm vs mode1\n(sorted by combined frequency)")
ax_d.legend(frameon=False, fontsize=7)
ax_d.set_xlim(-1, 30)   # show only top-30 sites

# ── E: Mode1 score vs Norm score per site (scatter) ──────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
# Use the mean scores across all AD patients for one reference patient
norms_all   = np.zeros(N_SITES)
mode_all    = np.zeros(N_SITES)
n_contrib   = 0
for pid in ad_pids:
    dW    = W_cc_mean - pat_W[pid]
    norms_all += np.linalg.norm(dW, axis=0)
    v1         = pat_v1[pid]
    mode_all  += np.abs(pat_W[pid].T @ v1)
    n_contrib += 1
norms_all /= n_contrib; mode_all /= n_contrib
# normalise to [0,1] for comparability
norms_n = norms_all / norms_all.max()
mode_n  = mode_all  / mode_all.max()

# colour by whether it's a top-2 site in either criterion
top5_norm  = set(np.argsort(norms_n)[::-1][:5].tolist())
top5_mode1 = set(np.argsort(mode_n)[::-1][:5].tolist())
both = top5_norm & top5_mode1
only_norm  = top5_norm  - both
only_mode1 = top5_mode1 - both
sc = ax_e.scatter(norms_n, mode_n, c="lightgray", s=18, zorder=1)
for j, s in enumerate(sorted(both)):
    ax_e.scatter(norms_n[s], mode_n[s], c="black", s=55, zorder=3,
                 label=f"both: s{s}" if j == 0 else f"s{s}")
for j, s in enumerate(sorted(only_norm)):
    ax_e.scatter(norms_n[s], mode_n[s], c=N5_COL, s=55, zorder=3,
                 marker="^", label=f"norm-only: s{s}" if j == 0 else f"s{s}")
for j, s in enumerate(sorted(only_mode1)):
    ax_e.scatter(norms_n[s], mode_n[s], c=M5_COL, s=55, zorder=3,
                 marker="s", label=f"mode1-only: s{s}" if j == 0 else f"s{s}")
ax_e.set_xlabel("Norm score  (||dW||, normalised)")
ax_e.set_ylabel("Mode1 score  (|W.v1|, normalised)")
ax_e.set_title("Per-site selection score\n(mean over AD patients)")
ax_e.legend(frameon=False, fontsize=7.5)

# ── F: Summary bar at physiological alpha ─────────────────────────────────────
ax_f = fig.add_subplot(gs[1, 2])
physio_ai  = np.where(ALPHAS <= PHYSIO)[0]
pidx       = physio_ai[-1] if len(physio_ai) else 0
bar_labels = [lbl.split("(")[0].strip() for _, _, _, lbl in STRATS]
bar_g  = [(results[n][pidx]["g"]  < thr_g ).mean()*100 for n, *_ in STRATS]
bar_fl = [(results[n][pidx]["fl"] < thr_fl).mean()*100 for n, *_ in STRATS]
xb = np.arange(len(STRATS)); w = 0.35
cols = [col for _, _, col, _ in STRATS]
for i, (bg, bfl, col) in enumerate(zip(bar_g, bar_fl, cols)):
    ax_f.bar(i - w/2, bg,  w, color=G_COL,  alpha=0.5 if i < 2 else 0.9)
    ax_f.bar(i + w/2, bfl, w, color=FL_COL, alpha=0.5 if i < 2 else 0.9)
    ax_f.text(i - w/2, bg  + 1, f"{bg:.0f}%",  ha="center", fontsize=7.5,
              color=G_COL)
    ax_f.text(i + w/2, bfl + 1, f"{bfl:.0f}%", ha="center", fontsize=7.5,
              color=FL_COL)
ax_f.bar([-1], [0], color=G_COL,  alpha=0.8, label="G-space")
ax_f.bar([-1], [0], color=FL_COL, alpha=0.8, label="FC-lag")
ax_f.set_xticks(xb); ax_f.set_xticklabels(bar_labels, fontsize=8)
ax_f.set_ylabel("AD reclassified as CC (%)")
ax_f.set_title(f"Summary at physiological alpha={PHYSIO:.0f}\n"
               f"(filled=mode1, faded=norm)")
ax_f.set_ylim(0, 110); ax_f.legend(frameon=False)

fig.suptitle("Site selection: ||dW|| norm vs first reservoir eigenmode projection\n"
             f"(W-interpolation, N={n_ad} AD, k=1,2 sites)",
             fontsize=11, fontweight="bold", y=1.01)

fig.savefig("pert_mode1_figure.png", bbox_inches="tight")
plt.close(fig)
print("Saved pert_mode1_figure.png")
print("\nDone.")
