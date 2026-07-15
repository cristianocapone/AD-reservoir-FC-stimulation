"""
pert_osc_stimulation.py
=======================
Oscillatory stimulation protocol: inject A*sin(2*pi*f*t) simultaneously at
the top-k brain sites (highest ||dW|| norm per patient) during the reservoir
forward pass. The same signal is broadcast to all k sites.

Sweeps:
  k      -- number of sites stimulated  (K_SITES)
  freq   -- stimulation frequency       (dominant eigenmode + data-driven + bands)
  amp    -- stimulation amplitude

G-space LDA is invariant (W unchanged); FC-lag LDA shifts through
changed Y_stim = W_AD.T @ X_res_stim.

Saves:
  pert_osc_data.npz
  pert_osc_figure.png
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
import warnings; warnings.filterwarnings("ignore")
from tqdm import trange, tqdm

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── parameters ────────────────────────────────────────────────────────────────
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

AMPS    = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
K_SITES = np.array([1, 2, 5])   # number of top sites stimulated simultaneously

CC_COL   = "#1565C0"
AD_COL   = "#C62828"
FL_COL   = "#0097A7"
K_COLS   = ["#6A1B9A", "#E65100", "#2E7D32"]   # purple/orange/green for k=1,2,5

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

print("TF pass (baseline) ...")
first_idx = {pid: patient_sids[pid][0] for pid in unique_pids}
sess_X = {}
for idx in trange(len(signals), desc="  TF"):
    s = signals[idx]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    Xraw = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        Xraw.append(res.X.copy())
    sess_X[idx] = np.array(Xraw)[TIMES_SKIP:]

print("W fitting ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
patX_single = {pid: sess_X[first_idx[pid]] for pid in unique_pids}

def get_tgt(pid):
    s = signals[first_idx[pid]]
    return (s.T @ ev50 @ ev50.T).T   # (N_sites, T_s)

pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX_single[pid]
    Yc = get_tgt(pid)[:, TIMES_SKIP:TIMES_SKIP+Xc.shape[0]].T
    noise = rng_w.normal(0, SIGMA, Xc.shape)
    pat_W[pid] = np.linalg.pinv(Xc + noise) @ Yc

W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

# ══════════════════════════════════════════════════════════════════════════════
# DOMINANT OSCILLATORY MODE
# ══════════════════════════════════════════════════════════════════════════════
print("\nFinding dominant oscillatory modes ...")

print("  Eigenspectrum of J ...")
eigs_J = np.linalg.eigvals(res.J)
cmask  = np.abs(eigs_J.imag) > 1e-8
if cmask.any():
    top_lam = eigs_J[cmask][np.argsort(np.abs(eigs_J[cmask]))[::-1][0]]
    f_eig   = float(np.abs(np.angle(top_lam)) / (2 * np.pi))
    print(f"  Leading complex eigenvalue: {top_lam.real:+.4f}{top_lam.imag:+.4f}j")
    print(f"  f_eig = {f_eig:.4f} c/step  (period = {1/f_eig:.1f} steps)")
else:
    f_eig = 0.1
    print("  No complex eigenvalues; f_eig = 0.1")

print("  FFT of AD reservoir states ...")
T_eff_ref = patX_single[ad_pids[0]].shape[0]
psd_ad    = np.zeros(T_eff_ref // 2 + 1)
for pid in ad_pids:
    Xr = patX_single[pid].astype(np.float64)
    fX = np.fft.rfft(Xr - Xr.mean(0), axis=0)
    psd_ad += (np.abs(fX)**2).mean(1)
psd_ad   /= n_ad
freqs_fft = np.fft.rfftfreq(T_eff_ref)
f_fft_idx = 1 + int(np.argmax(psd_ad[1:]))
f_fft     = float(freqs_fft[f_fft_idx])
print(f"  f_fft = {f_fft:.4f} c/step  (period = {1/f_fft:.1f} steps)")

FREQS_CANDS = np.unique(np.round([f_eig, f_fft, 0.05, 0.1, 0.2, 0.4], 4))
FREQS = FREQS_CANDS[(FREQS_CANDS > 0) & (FREQS_CANDS <= 0.5)]

def freq_label(f):
    tags = []
    if abs(f - f_eig) < 5e-4: tags.append("eig")
    if abs(f - f_fft) < 5e-4: tags.append("FFT")
    return f"{f:.4f}" + (f" ({','.join(tags)})" if tags else "")

print(f"  Testing {len(FREQS)} frequencies: {FREQS}")

# ══════════════════════════════════════════════════════════════════════════════
# FC-LAG LDA
# ══════════════════════════════════════════════════════════════════════════════
def lagged_corrcoef(S, lag):
    if lag == 0:
        return np.corrcoef(S.T)
    T = S.shape[0]
    A = S[:T-lag].astype(np.float64); B = S[lag:].astype(np.float64)
    A -= A.mean(0); B -= B.mean(0)
    A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - lag)

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

def compute_fclag_feat_from_Xres(W, X_res):
    S = (W.T.astype(np.float64) @ X_res.T).T
    feats = []
    for lag in range(MAX_LAG + 1):
        fc = lagged_corrcoef(S, lag)
        fc = np.nan_to_num(fc, nan=0., posinf=0., neginf=0.)
        feats.append(fc[np.triu_indices(N_SITES, k=1)] if lag == 0
                     else fc.flatten())
    return np.concatenate(feats)

print("\nBuilding FC-lag LDA ...")
fclag_base = np.array([compute_fclag_feat_from_Xres(pat_W[pid],
                                                     patX_single[pid])
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
print(f"  FC-lag: CC={cc_fl.mean():.3f}  AD={ad_fl.mean():.3f}  thr={thr_fl:.3f}")

def fl_score_from_Xres(W, X_res):
    feat   = compute_fclag_feat_from_Xres(W, X_res)
    feat_c = feat - fclag_mean
    g      = (feat_c @ fclag_c.T @ evec_fl) / (np.sqrt(ev_fl) + 1e-12)
    return float(lda_fl.transform(g[:K_LDA].reshape(1, -1))[0])

# ══════════════════════════════════════════════════════════════════════════════
# OSCILLATORY STIMULATION FORWARD PASS
# ══════════════════════════════════════════════════════════════════════════════
def run_osc_pass(pid, stim_sites, freq_norm, amplitude):
    """
    Reservoir forward pass: same A*sin(2*pi*freq_norm*t) broadcast to all stim_sites.

    stim_sites : array of site indices
    freq_norm  : cycles per reservoir timestep
    amplitude  : stimulation amplitude (brain signal is O(ff=0.1))
    Returns    : X_stim (T_eff, N_hidden)
    """
    s   = signals[first_idx[pid]]
    T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    Xstim = []
    for t in range(T_s - 1):
        inp = ff * tgt[:, t].copy()
        inp[stim_sites] += amplitude * np.sin(2 * np.pi * freq_norm * t)
        res.step_rate(inp, sigma_dyn=0.)
        Xstim.append(res.X.copy())
    return np.array(Xstim)[TIMES_SKIP:]

# ── per-patient top-k site selection ─────────────────────────────────────────
def get_topk(pid, k):
    dW    = W_cc_mean - pat_W[pid]
    norms = np.linalg.norm(dW, axis=0)
    return np.argsort(norms)[::-1][:k]

# ══════════════════════════════════════════════════════════════════════════════
# SWEEP : k × freq × amp × patient
# results_osc[ki, fi, ai, pi] = FC-lag LDA score
# rms_ratio  [ki, fi, ai, pi] = mean RMS(stim)/RMS(base) at stim sites
# ══════════════════════════════════════════════════════════════════════════════
n_k  = len(K_SITES)
n_f  = len(FREQS)
n_a  = len(AMPS)

results_osc = np.zeros((n_k, n_f, n_a, n_ad))
rms_ratio   = np.zeros((n_k, n_f, n_a, n_ad))

fl_base = np.array([fl_score_from_Xres(pat_W[pid], patX_single[pid])
                     for pid in ad_pids])
print(f"\nBaseline FC-lag (AD mean): {fl_base.mean():+.3f}")
print(f"Sweep: {n_k} k-values x {n_f} freqs x {n_a} amps x {n_ad} patients\n")

for ki, k in enumerate(K_SITES):
    print(f"  ── k={k} site(s) ────────────────────────────────")
    for fi, freq in enumerate(FREQS):
        for ai, amp in enumerate(AMPS):
            if amp == 0:
                for pi in range(n_ad):
                    results_osc[ki, fi, ai, pi] = fl_base[pi]
                    rms_ratio[ki, fi, ai, pi]   = 1.0
            else:
                for pi, pid in enumerate(ad_pids):
                    sites  = get_topk(pid, k)
                    X_stim = run_osc_pass(pid, sites, freq, amp)
                    results_osc[ki, fi, ai, pi] = fl_score_from_Xres(pat_W[pid], X_stim)

                    Wb = pat_W[pid].T.astype(np.float64)
                    Yb = (Wb @ patX_single[pid].T).T
                    Ys = (Wb @ X_stim.T).T
                    rms_b = np.sqrt((Yb[:, sites]**2).mean() + 1e-24)
                    rms_s = np.sqrt((Ys[:, sites]**2).mean())
                    rms_ratio[ki, fi, ai, pi] = rms_s / rms_b

            m  = results_osc[ki, fi, ai].mean()
            pct = (results_osc[ki, fi, ai] < thr_fl).mean() * 100
            print(f"    k={k}  f={freq:.4f}  amp={amp:5.2f}  "
                  f"FL={m:+.3f}  reclassified={pct:3.0f}%  "
                  f"RMS={rms_ratio[ki, fi, ai].mean():.3f}",
                  flush=True)

# ── summary ───────────────────────────────────────────────────────────────────
print("\n── Summary (best amp per k,freq) ──────────────────────────────────")
print(f"  {'k':>3s}  {'Freq':>8s}  {'Best FL':>8s}  {'@ amp':>6s}  {'Reclassified':>12s}")
print("  " + "-"*50)
for ki, k in enumerate(K_SITES):
    for fi, freq in enumerate(FREQS):
        best_ai  = np.argmin(results_osc[ki, fi].mean(1))
        best_m   = results_osc[ki, fi, best_ai].mean()
        best_pct = (results_osc[ki, fi, best_ai] < thr_fl).mean() * 100
        print(f"  {k:>3d}  {freq:8.4f}  {best_m:+8.3f}  "
              f"{AMPS[best_ai]:>6.2f}  {best_pct:>11.0f}%")
print(f"\n  Baseline AD FL: {fl_base.mean():+.3f}   threshold: {thr_fl:+.3f}")
print(f"  G-space LDA is invariant to oscillatory stim (W unchanged).")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez("pert_osc_data.npz",
         freqs=FREQS, amps=AMPS, k_sites=K_SITES,
         results_osc=results_osc, rms_ratio=rms_ratio,
         fl_base=fl_base, thr_fl=np.array(thr_fl),
         cc_fl=cc_fl, ad_fl=ad_fl,
         f_eig=np.array(f_eig), f_fft=np.array(f_fft),
         psd_ad=psd_ad, freqs_fft=freqs_fft)
print("Saved pert_osc_data.npz")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE  (2 rows x 3 cols)
# Row 1: k comparison at best frequency
# Row 2: frequency comparison at k=2; heatmap; per-patient scatter
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

# pick dominant frequency for k-comparison panels
dom_fi = int(np.argmin(results_osc.mean((0, 2, 3))))   # freq with lowest mean FL
cmap_f = plt.cm.plasma(np.linspace(0.1, 0.9, len(FREQS)))

fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor="white")
fig.subplots_adjust(hspace=0.5, wspace=0.42)

# ── A: FL score vs amplitude — k comparison at dominant freq ─────────────────
ax = axes[0, 0]
ax.axhspan(cc_fl.mean()-cc_fl.std(), cc_fl.mean()+cc_fl.std(),
           alpha=0.10, color=CC_COL)
ax.axhline(cc_fl.mean(), color=CC_COL, lw=1.5, ls="--", label="CC mean")
ax.axhline(thr_fl, color="gray", lw=1, ls="-.", label="Boundary")
for ki, k in enumerate(K_SITES):
    m = results_osc[ki, dom_fi].mean(1)
    s = results_osc[ki, dom_fi].std(1)
    ax.fill_between(AMPS, m-s, m+s, alpha=0.15, color=K_COLS[ki])
    ax.plot(AMPS, m, "-o", ms=4, lw=2, color=K_COLS[ki],
            label=f"k={k} sites")
ax.set_xlabel("Stimulation amplitude")
ax.set_ylabel("FC-lag LDA score")
ax.set_title(f"FL score vs amplitude\n(f={FREQS[dom_fi]:.4f} c/step, "
             f"dominant freq)")
ax.legend(frameon=False, fontsize=8)

# ── B: Reclassification rate — k comparison ───────────────────────────────────
ax = axes[0, 1]
for ki, k in enumerate(K_SITES):
    frac = (results_osc[ki, dom_fi] < thr_fl).mean(1) * 100
    ax.plot(AMPS, frac, "-o", ms=4, lw=2, color=K_COLS[ki], label=f"k={k}")
ax.axhline(50, color="gray", ls="--", lw=1, alpha=0.4)
ax.set_xlabel("Stimulation amplitude")
ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title(f"Reclassification rate\n(f={FREQS[dom_fi]:.4f}, all k)")
ax.set_ylim(-2, 105)
ax.legend(frameon=False, fontsize=8)

# ── C: RMS ratio — k comparison ───────────────────────────────────────────────
ax = axes[0, 2]
ax.axhline(1.0, color="gray", ls="--", lw=1, alpha=0.5)
for ki, k in enumerate(K_SITES):
    m = rms_ratio[ki, dom_fi].mean(1)
    s = rms_ratio[ki, dom_fi].std(1)
    ax.fill_between(AMPS, m-s, m+s, alpha=0.15, color=K_COLS[ki])
    ax.plot(AMPS, m, "-o", ms=4, lw=2, color=K_COLS[ki], label=f"k={k}")
ax.set_xlabel("Stimulation amplitude")
ax.set_ylabel("RMS(stim) / RMS(baseline)")
ax.set_title("Signal amplitude at stim sites\n(physiological check)")
ax.legend(frameon=False, fontsize=8)

# ── D: Heatmap k x amp at dominant freq ───────────────────────────────────────
ax = axes[1, 0]
heat_k = results_osc[:, dom_fi, :, :].mean(2)    # (n_k, n_amp)
vmin = min(thr_fl - 0.3, heat_k.min())
vmax = max(ad_fl.mean() + 0.1, heat_k.max())
im   = ax.imshow(heat_k, aspect="auto", origin="lower",
                 cmap="RdBu_r", vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=ax, label="FC-lag LDA (mean AD)")
ax.set_xticks(range(n_a))
ax.set_xticklabels([f"{a:.2f}" for a in AMPS], rotation=45, fontsize=7)
ax.set_yticks(range(n_k))
ax.set_yticklabels([f"k={k}" for k in K_SITES], fontsize=8)
ax.set_xlabel("Stimulation amplitude")
ax.set_ylabel("Number of sites (k)")
ax.set_title(f"FC-lag LDA (k x amp)\nf={FREQS[dom_fi]:.4f}, mean over 40 AD")
try:
    ax.contour(heat_k, levels=[thr_fl], colors=["black"], linewidths=[1.5])
except Exception:
    pass

# ── E: Frequency comparison at best k ─────────────────────────────────────────
best_ki = int(np.argmin(results_osc.mean((1, 2, 3))))
ax = axes[1, 1]
ax.axhline(thr_fl, color="gray", lw=1, ls="-.")
ax.axhline(cc_fl.mean(), color=CC_COL, ls="--", lw=1.5, label="CC mean")
for fi, freq in enumerate(FREQS):
    m = results_osc[best_ki, fi].mean(1)
    ax.plot(AMPS, m, "-o", ms=3, lw=2, color=cmap_f[fi],
            label=freq_label(freq))
ax.set_xlabel("Stimulation amplitude")
ax.set_ylabel("FC-lag LDA score")
ax.set_title(f"Frequency comparison at k={K_SITES[best_ki]}\n"
             f"(best-performing k)")
ax.legend(frameon=False, fontsize=7)

# ── F: Per-patient at best (k, freq, amp) ─────────────────────────────────────
flat_idx = np.unravel_index(results_osc.mean(3).argmin(),
                             results_osc.mean(3).shape)
bki, bfi, bai = flat_idx
best_scores = results_osc[bki, bfi, bai]
ax = axes[1, 2]
ax.axhline(thr_fl, color="gray", ls="-.", lw=1, label="Boundary")
ax.axhline(cc_fl.mean(), color=CC_COL, ls="--", lw=1.5, label="CC mean")
ax.scatter(range(n_ad), np.sort(fl_base)[::-1],
           color=AD_COL, s=20, alpha=0.5, label="Baseline AD")
ax.scatter(range(n_ad), np.sort(best_scores)[::-1],
           color=K_COLS[bki], s=20, alpha=0.85, marker="^",
           label=f"Best stim  k={K_SITES[bki]}, "
                 f"f={FREQS[bfi]:.4f}, A={AMPS[bai]:.2f}")
ax.set_xlabel("AD patient rank")
ax.set_ylabel("FC-lag LDA score")
ax.set_title("Per-patient comparison\n(baseline vs best stim condition)")
ax.legend(frameon=False, fontsize=7.5)

fig.suptitle(f"Oscillatory stimulation: top-k site sweep  "
             f"(k = {list(K_SITES)},  N={n_ad} AD patients)\n"
             f"f_eig={f_eig:.4f}  f_fft={f_fft:.4f}",
             fontsize=11, fontweight="bold", y=1.01)

fig.savefig("pert_osc_figure.png", bbox_inches="tight")
plt.close(fig)
print("Saved pert_osc_figure.png")
print("\nDone.")
