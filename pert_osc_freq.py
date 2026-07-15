"""
pert_osc_freq.py
================
Oscillatory-stimulation FREQUENCY experiment (seeded, reproducible).

Injects A*sin(2*pi*f*t) at the top-k most-affected sites and asks WHICH
frequency most efficiently moves the FC-lag classifier toward CC.  The reservoir
J is seeded so its dominant eigenmode frequency f_eig is reproducible; we sweep a
fine frequency grid that includes f_eig (reservoir resonance) and f_fft (spectral
peak of the AD reservoir states).

Outputs (-> paper_figures/):
  figure6_oscfreq.{png,pdf}   freq/amp efficiency + brain top-1 site
  pert_oscfreq_data.npz
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from tqdm import trange, tqdm
import warnings; warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED   = 42
N_CC_SAMP  = 40
N_SITES    = 121
N_PC_MODEL = 50
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
SR         = 0.95
K_LDA      = 25
MAX_LAG    = 2
TS_ROOT    = "./timeseries"
OUT        = "paper_figures"

K_SITES = [1, 5]
AMPS    = np.array([0.0, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0])

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, pid_raw = [], [], []
for sub, lab in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, sub)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if lab == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMP, len(files)),
                                replace=False))
    for fn in files:
        arr = np.load(os.path.join(folder, fn)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T); pid_raw.append(fn.split("_ses-")[0])
            labels_raw.append(0 if lab == "CC" else 1)
pid_raw = np.array(pid_raw); labels_raw = np.array(labels_raw)
unique_pids = np.unique(pid_raw)
patient_sids = {p: np.where(pid_raw == p)[0] for p in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[p][0]] for p in unique_pids])
cc_pids = [unique_pids[i] for i in np.where(patient_labels == 0)[0]]
ad_pids = [unique_pids[i] for i in np.where(patient_labels == 1)[0]]
n_ad = len(ad_pids)
print(f"  {len(unique_pids)} patients ({len(cc_pids)} CC, {n_ad} AD)")

print("Population PCA ...")
all_sig = np.concatenate([s.T for s in signals], axis=0)
evals, evecs = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── seeded reservoir ──────────────────────────────────────────────────────────
print("Reservoir (seeded) ...")
np.random.seed(RNG_SEED)                     # makes J / Jin reproducible
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN, T=139, dt=0.005,
           sigma_input=0.01, shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

first_idx = {p: patient_sids[p][0] for p in unique_pids}
patX = {}
for p in tqdm(unique_pids, desc="  TF"):
    s = signals[first_idx[p]]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    Xr = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.); Xr.append(res.X.copy())
    patX[p] = np.array(Xr)[TIMES_SKIP:]

print("W fitting ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
pat_W = {}
for p in unique_pids:
    s = signals[first_idx[p]]
    tgt = (s.T @ ev50 @ ev50.T).T
    Xc = patX[p]; Yc = tgt[:, TIMES_SKIP:TIMES_SKIP + Xc.shape[0]].T
    pat_W[p] = np.linalg.pinv(Xc + rng_w.normal(0, SIGMA, Xc.shape)) @ Yc
W_cc_mean = np.mean([pat_W[p] for p in cc_pids], axis=0)

# top-1 / top-k sites per AD patient
top_sites = {}
counts1 = np.zeros(N_SITES, dtype=int)
for p in ad_pids:
    norms = np.linalg.norm(W_cc_mean - pat_W[p], axis=0)
    order = np.argsort(norms)[::-1]
    top_sites[p] = order
    counts1[order[0]] += 1

# ── dominant frequencies + eigenmode top site (modal coupling, "without LDA") ──
from scipy.linalg import eig as sla_eig
wv, vl, vr = sla_eig(res.J, left=True, right=True)
pos = np.where(wv.imag > 1e-8)[0]; i1 = pos[np.argsort(np.abs(wv[pos]))[::-1][0]]
f_eig = float(np.abs(np.angle(wv[i1])) / (2*np.pi))
lam = wv[i1]
site_eig = int(np.argmax(np.abs(vl[:, i1].conj() @ res.Jin)))   # max coupling to eigenmode 1
print(f"  eigenmode top site = {site_eig}  (modal-coupling criterion)")
T_eff = patX[ad_pids[0]].shape[0]
psd = np.zeros(T_eff//2 + 1)
for p in ad_pids:
    Xr = patX[p].astype(np.float64)
    psd += (np.abs(np.fft.rfft(Xr - Xr.mean(0), axis=0))**2).mean(1)
psd /= n_ad
freqs_fft = np.fft.rfftfreq(T_eff)
f_fft = float(freqs_fft[1 + int(np.argmax(psd[1:]))])
print(f"  f_eig={f_eig:.4f}  f_fft={f_fft:.4f} c/step")

FREQS = np.unique(np.round(np.concatenate(
    [np.linspace(0.03, 0.45, 11), [f_eig, f_fft]]), 4))
print(f"  {len(FREQS)} freqs: {FREQS}")

# ── FC-lag LDA ────────────────────────────────────────────────────────────────
def lagcorr(S, lag):
    if lag == 0: return np.corrcoef(S.T)
    T = S.shape[0]; A = S[:T-lag].astype(float); B = S[lag:].astype(float)
    A -= A.mean(0); B -= B.mean(0); A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - lag)

def fclag(W, X):
    S = (W.T.astype(float) @ X.T.astype(float)).T
    fs = []
    for lag in range(MAX_LAG+1):
        fc = np.nan_to_num(lagcorr(S, lag))
        fs.append(fc[np.triu_indices(N_SITES, 1)] if lag == 0 else fc.flatten())
    return np.concatenate(fs)

class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y); X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = (X0-mu0).T@(X0-mu0)+(X1-mu1).T@(X1-mu1)+1e-6*np.eye(X0.shape[1])
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w; self.thr_ = 0.5*(mu0@w+mu1@w); return self
    def transform(self, X): return X @ self.w_

def _bal(X, y, seed=0):
    r = np.random.default_rng(seed); c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([r.choice(c0,n,False), r.choice(c1,n,False)])
    r.shuffle(sel); return X[sel], y[sel]

print("Building FC-lag LDA ...")
fb = np.array([fclag(pat_W[p], patX[p]) for p in tqdm(unique_pids, leave=False)])
fmean = fb.mean(0); fc_c = fb - fmean
ev, evec = np.linalg.eigh(fc_c @ fc_c.T); o = np.argsort(ev)[::-1]
ev = np.maximum(ev[o], 0); evec = evec[:, o]
G = evec*np.sqrt(ev)
Xl, yl = _bal(G[:, :K_LDA], patient_labels, RNG_SEED)
lda = _LDA().fit(Xl, yl); Z = lda.transform(G[:, :K_LDA])
if Z[patient_labels==0].mean() > Z[patient_labels==1].mean():
    lda.w_ *= -1; lda.thr_ *= -1; Z = -Z
cc_fl = Z[patient_labels==0]; thr_fl = 0.5*(cc_fl.mean()+Z[patient_labels==1].mean())
print(f"  thr_fl={thr_fl:.3f}")

def fl_score(W, X):
    f = fclag(W, X) - fmean
    g = (f @ fc_c.T @ evec) / (np.sqrt(ev)+1e-12)
    return float(lda.transform(g[:K_LDA].reshape(1,-1))[0])

# ── oscillatory pass ──────────────────────────────────────────────────────────
def osc_pass(pid, sites, fnorm, amp):
    s = signals[first_idx[pid]]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset(); Xs = []
    for t in range(T_s - 1):
        inp = ff * tgt[:, t].copy()
        inp[sites] += amp * np.sin(2*np.pi*fnorm*t)
        res.step_rate(inp, sigma_dyn=0.); Xs.append(res.X.copy())
    return np.array(Xs)[TIMES_SKIP:]

fl_base = np.array([fl_score(pat_W[p], patX[p]) for p in ad_pids])
print(f"  baseline FL (AD mean) = {fl_base.mean():+.3f}")

nF, nA, nK = len(FREQS), len(AMPS), len(K_SITES)
FL = np.zeros((nK, nF, nA, n_ad))   # FC-lag score
print(f"\nSweep {nK} k x {nF} freq x {nA} amp x {n_ad} AD ...")
for ki, k in enumerate(K_SITES):
    for fi, fr in enumerate(FREQS):
        for ai, amp in enumerate(AMPS):
            if amp == 0:
                FL[ki, fi, ai] = fl_base; continue
            for pi, pid in enumerate(ad_pids):
                sites = top_sites[pid][:k]
                FL[ki, fi, ai, pi] = fl_score(pat_W[pid],
                                              osc_pass(pid, sites, fr, amp))
    print(f"  k={k} done", flush=True)

# ── eigenmode-site single-site drive (modal coupling, NOT LDA / NOT Delta-W) ──
print(f"Eigenmode-site (site {site_eig}) resonant sweep ...")
FL_eig = np.zeros((nF, nA, n_ad))
for fi, fr in enumerate(FREQS):
    for ai, amp in enumerate(AMPS):
        if amp == 0:
            FL_eig[fi, ai] = fl_base; continue
        for pi, pid in enumerate(ad_pids):
            FL_eig[fi, ai, pi] = fl_score(pat_W[pid], osc_pass(pid, [site_eig], fr, amp))
print("  eigenmode-site done", flush=True)

np.savez(os.path.join(OUT, "pert_oscfreq_data.npz"),
         freqs=FREQS, amps=AMPS, k_sites=np.array(K_SITES), FL=FL,
         FL_eig=FL_eig, site_eig=np.array(site_eig),
         fl_base=fl_base, thr_fl=np.array(thr_fl), cc_fl=cc_fl,
         f_eig=np.array(f_eig), f_fft=np.array(f_fft),
         psd=psd, freqs_fft=freqs_fft, counts1=counts1)
print("Saved pert_oscfreq_data.npz")

# ── brain (top-1 sites) ───────────────────────────────────────────────────────
parcel_coords = np.load("pert_sites_data.npz", allow_pickle=True)["parcel_coords"]
from nilearn import plotting
sel = counts1 > 0
disp = plotting.plot_markers(counts1[sel].astype(float), parcel_coords[sel],
                             node_size=40 + 14*counts1[sel], node_cmap="autumn_r",
                             node_vmin=0, node_vmax=float(counts1.max()),
                             display_mode="lzry", alpha=0.9, colorbar=True,
                             title="Top-1 site (most-affected) selection frequency")
disp.savefig(os.path.join(OUT, "figure6_oscbrain.png"), dpi=300)
disp.savefig(os.path.join(OUT, "figure6_oscbrain.pdf")); disp.close()

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6
# ══════════════════════════════════════════════════════════════════════════════
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "legend.fontsize": 8, "figure.dpi": 300,
    "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False})

K_COL = {1: "#6A1B9A", 5: "#2E7D32"}
fig = plt.figure(figsize=(15, 4.8), facecolor="white")
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32, width_ratios=[1, 1, 1.05],
                       left=0.06, right=0.98, top=0.86, bottom=0.16)

def _tag(ax, t, x=-0.14):
    ax.text(x, 1.04, t, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")

amp_ref = nA - 1                       # strongest amplitude for the freq slice
fi_eig = int(np.argmin(np.abs(FREQS - f_eig)))

# ── A: FC-lag score vs frequency (resonance) + PSD ────────────────────────────
ax = fig.add_subplot(gs[0, 0])
for k in K_SITES:
    ki = K_SITES.index(k)
    m = FL[ki, :, amp_ref].mean(1); sdev = FL[ki, :, amp_ref].std(1)
    ax.fill_between(FREQS, m-sdev, m+sdev, color=K_COL[k], alpha=0.12)
    ax.plot(FREQS, m, "-o", ms=4, color=K_COL[k], lw=2,
            label=f"k={k} site{'s' if k>1 else ''}")
ax.axhline(thr_fl, color="gray", ls="-.", lw=1, label="boundary")
ax.axvline(f_eig, color="#C62828", ls="--", lw=1.5, label=f"$f_{{eig}}$={f_eig:.3f}")
ax.axvline(f_fft, color="#E65100", ls=":", lw=1.5, label=f"$f_{{FFT}}$={f_fft:.3f}")
ax.set_xlabel("stimulation frequency  (cycles/step)")
ax.set_ylabel("FC-lag LDA score")
ax.set_title(f"Frequency tuning of efficacy\n(amplitude $A$={AMPS[amp_ref]:.0f})")
ax.legend(frameon=False, fontsize=7); _tag(ax, "A")

# ── B: FC-lag score vs amplitude — resonant vs off-resonance (k=5) ────────────
ax = fig.add_subplot(gs[0, 1])
ki = K_SITES.index(5)
fi_off = int(np.argmax(FREQS))         # highest (off-resonance) frequency
ax.axhline(thr_fl, color="gray", ls="-.", lw=1, label="boundary")
m = FL[ki, fi_eig].mean(1); sd = FL[ki, fi_eig].std(1)
ax.fill_between(AMPS, m-sd, m+sd, color="#C62828", alpha=0.12)
ax.plot(AMPS, m, "-o", ms=5, color="#C62828", lw=2,
        label=f"resonant  $f_{{eig}}$={FREQS[fi_eig]:.3f}")
m = FL[ki, fi_off].mean(1); sd = FL[ki, fi_off].std(1)
ax.fill_between(AMPS, m-sd, m+sd, color="#455A64", alpha=0.12)
ax.plot(AMPS, m, "--s", ms=5, color="#455A64", lw=2,
        label=f"off-res.  $f$={FREQS[fi_off]:.3f}")
ax.set_xlabel("stimulation amplitude  $A$")
ax.set_ylabel("FC-lag LDA score")
ax.set_title("Resonant vs off-resonant drive\n(top-5 sites)")
ax.legend(frameon=False, fontsize=7.5); _tag(ax, "B")

# ── C: brain — top-1 site ─────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.imshow(imread(os.path.join(OUT, "figure6_oscbrain.png"))); ax.axis("off")
ax.set_title("Where: top-1 most-affected site", pad=2)
_tag(ax, "C", x=-0.04)

fig.suptitle("The model prescribes both WHERE and HOW to stimulate: "
             "the top site, driven at the reservoir resonance, is most efficient",
             fontsize=11, fontweight="bold", y=0.99)

for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"figure6_oscfreq.{ext}"), dpi=300,
                bbox_inches="tight", facecolor="white")
    print(f"Saved {OUT}/figure6_oscfreq.{ext}")
plt.close(fig)
print("\nDone.")
