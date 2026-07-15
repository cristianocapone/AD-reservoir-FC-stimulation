"""
figure6_fullW_vs_lda.py
=======================
Figure 6 — full-W correction vs single-site LDA-resonant drive:
efficacy (reclassification rate) and two stimulation-cost metrics:

  1. Zero-lag FC distance  = 1 - corr(FC_stim_upper_tri, FC_ref_upper_tri)
                             (existing metric, from free-run closed-loop)

  2. Lagged FC distance    = 1 - corr(lagFC_stim_vec, lagFC_ref_vec)
                             where lagFC_vec = [lag-0 upper-tri | lag-1 full | lag-2 full]
                             from the free-run closed-loop  (NEW metric)

Layout (2 rows × 3 cols):
  A  Reclassification rate vs dose
  B  Zero-lag FC distance vs dose
  C  Lagged FC distance vs dose  (NEW)
  D  Efficacy vs zero-lag cost
  E  Efficacy vs lagged FC cost  (NEW)
  F  Per-patient: zero-lag vs lagged-FC distance (full-W vs LDA-resonant)

Reads cached data where possible; computes missing quantities on the fly.
Saves: figure6_fullW_vs_lda.{png,pdf}
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.path.insert(0, "..")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from res import RESERVOIRE_SIMPLE

# ── params (must match pert_single_compare.py exactly) ──────────────────────
RNG_SEED    = 42
N_CC_SAMP   = 40
N_SITES     = 121
N_PC_MODEL  = 50
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
SIGMA       = 0.05
SR          = 0.95
K_LDA       = 25
MAX_LAG     = 2
DRIVE_STEPS = 5
TS_ROOT     = "../timeseries"
iu          = np.triu_indices(N_SITES, 1)

# ── alpha / amplitude grids ──────────────────────────────────────────────────
d_sc  = np.load("../pert_single_compare_data.npz", allow_pickle=True)
d_cmp = np.load("../pert_compare3_data.npz",       allow_pickle=True)
d_st  = np.load("../pert_sites_data.npz",          allow_pickle=True)

ALPHAS_FW  = d_st["alphas_fw"]                    # full-W alpha grid
AMPS_LDA   = d_sc["amps"]                         # LDA-resonant amplitude grid
thr_fl     = float(d_sc["thr_f"])
f1         = float(d_cmp["f1"])

# existing cached results (from pert_single_compare.py)
S_lp = d_sc["S_lp"]    # (n_amp, n_ad) FC-lag scores for LDA-resonant
D_lp = d_sc["D_lp"]    # (n_amp, n_ad) zero-lag FC dist for LDA-resonant

n_alpha = len(ALPHAS_FW)
n_amp   = len(AMPS_LDA)

# ── load data + reservoir ────────────────────────────────────────────────────
print("Loading data + reservoir ...")
rng = np.random.default_rng(RNG_SEED)
signals, labs, pids = [], [], []
for sub, lb in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, sub)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if lb == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMP, len(files)), replace=False))
    for fn in files:
        a = np.load(os.path.join(folder, fn)).T
        if a.shape[1] == N_SITES and a.shape[0] >= 139:
            signals.append(a.T); pids.append(fn.split("_ses-")[0])
            labs.append(0 if lb == "CC" else 1)
pids = np.array(pids); labs = np.array(labs); upid = np.unique(pids)
psid = {p: np.where(pids == p)[0] for p in upid}
plabel = np.array([labs[psid[p][0]] for p in upid])
first  = {p: psid[p][0] for p in upid}
cc = [upid[i] for i in np.where(plabel == 0)[0]]
ad = [upid[i] for i in np.where(plabel == 1)[0]]
n_ad = len(ad)

all_sig = np.concatenate([s.T for s in signals], 0)
evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]

np.random.seed(RNG_SEED)
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN, T=139, dt=0.005,
           sigma_input=0.01, shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

patX = {}
for p in tqdm(upid, desc="  teacher-force"):
    s = signals[first[p]]; T = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X.append(res.X.copy())
    patX[p] = np.array(X)[TIMES_SKIP:]

rw = np.random.default_rng(RNG_SEED + 1)
patW = {}
for p in upid:
    s = signals[first[p]]; T = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    Xc = patX[p]; Yc = tgt[:, TIMES_SKIP:TIMES_SKIP + Xc.shape[0]].T
    patW[p] = np.linalg.pinv(Xc + rw.normal(0, SIGMA, Xc.shape)) @ Yc

Wcc = np.mean([patW[p] for p in cc], 0)

# personalised LDA-resonant sites
red_full  = d_cmp["red_full"]   # (121, n_ad)
pers_site = {p: int(np.argmax(red_full[:, pi])) for pi, p in enumerate(ad)}

# ── FC-lag LDA (same as pert_single_compare.py) ──────────────────────────────
def lagc(S, l):
    if l == 0: return np.corrcoef(S.T)
    T = S.shape[0]; A = S[:T-l].copy(); B = S[l:].copy()
    A -= A.mean(0); B -= B.mean(0); A /= A.std(0) + 1e-12; B /= B.std(0) + 1e-12
    return (A.T @ B) / (T - l)

def lagged_fc_vec(S):
    """Raw lagged-FC feature vector from a time-series matrix S (T, N_SITES)."""
    fs = []
    for l in range(MAX_LAG + 1):
        fc = np.nan_to_num(lagc(S, l))
        fs.append(fc[iu] if l == 0 else fc.flatten())
    return np.concatenate(fs)

def freerun_output(Wout, p, site=None, amp=0.0):
    """Run DRIVE_STEPS teacher-forced then free-run; return output time series."""
    s = signals[first[p]]; T = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); res.Jout = Wout.T.copy(); Y = []
    for t in range(T - 1):
        fbk = tgt[:, t] if t <= DRIVE_STEPS else res.y
        inp = ff * np.asarray(fbk, dtype=float).copy()
        if site is not None and amp != 0.:
            inp[site] += amp * np.sin(2 * np.pi * f1 * t)
        res.step_rate(inp, sigma_dyn=0.)
        Y.append(np.asarray(res.y, dtype=float).copy())
    return np.array(Y)[TIMES_SKIP - 1:]    # (T', N_SITES) — skip transient

# reference (unstimulated) free-run lagged-FC vectors and zero-lag FC
print("Computing reference free-run metrics for AD patients ...")
ref_lagfc = {}; ref_zlag_fc = {}
for p in tqdm(ad, desc="  ref"):
    S = freerun_output(patW[p], p)
    ref_lagfc[p]   = lagged_fc_vec(S)
    ref_zlag_fc[p] = np.nan_to_num(np.corrcoef(S.T))

def zlag_dist(S, p):
    fc = np.nan_to_num(np.corrcoef(S.T))
    return 1.0 - float(np.corrcoef(fc[iu], ref_zlag_fc[p][iu])[0, 1])

def lag_dist(S, p):
    return 1.0 - float(np.corrcoef(lagged_fc_vec(S), ref_lagfc[p])[0, 1])

# ── full-W: compute all metrics ───────────────────────────────────────────────
print("\nFull-W: computing zero-lag and lagged FC distances ...")
# FC-lag scores: reconstruct from pert_sites_data.npz
S_fw = np.array([d_st[f"full_w_{ai}_fl"] for ai in range(n_alpha)])  # (n_alpha, n_ad)
D_fw_zlag  = np.zeros((n_alpha, n_ad))   # zero-lag FC distance
D_fw_llag  = np.zeros((n_alpha, n_ad))   # lagged FC distance

for ai, alpha in enumerate(tqdm(ALPHAS_FW, desc="  alpha")):
    for pi, p in enumerate(ad):
        W_int = (1 - alpha) * patW[p] + alpha * Wcc
        S_out = freerun_output(W_int, p)
        D_fw_zlag[ai, pi] = zlag_dist(S_out, p)
        D_fw_llag[ai, pi] = lag_dist(S_out, p)

# ── LDA-resonant: compute lagged FC distances (zero-lag already in D_lp) ─────
print("\nLDA-resonant: computing lagged FC distances ...")
D_lp_llag = np.zeros((n_amp, n_ad))   # lagged FC distance

for ai, A in enumerate(tqdm(AMPS_LDA, desc="  amplitude")):
    for pi, p in enumerate(ad):
        sl = pers_site[p]
        S_out = freerun_output(patW[p], p, site=sl, amp=A)
        D_lp_llag[ai, pi] = lag_dist(S_out, p)

# ── save ─────────────────────────────────────────────────────────────────────
np.savez("../pert_fullW_vs_lda_data.npz",
         alphas_fw=ALPHAS_FW, amps_lda=AMPS_LDA,
         S_fw=S_fw, S_lp=S_lp,
         D_fw_zlag=D_fw_zlag, D_fw_llag=D_fw_llag,
         D_lp_zlag=D_lp, D_lp_llag=D_lp_llag,
         thr_fl=thr_fl)
print("Saved pert_fullW_vs_lda_data.npz")

# ── reclassification rates ────────────────────────────────────────────────────
recl_fw  = (S_fw  < thr_fl).mean(1) * 100    # (n_alpha,)
recl_lda = (S_lp  < thr_fl).mean(1) * 100    # (n_amp,)

# normalised dose axes (0 = no intervention, 1 = max tested)
alpha_n = ALPHAS_FW / ALPHAS_FW.max()
amp_n   = AMPS_LDA  / AMPS_LDA.max()

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9.5, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
CFW  = "#1565C0"   # full-W colour
CLDA = "#2E7D32"   # LDA-resonant colour

fig = plt.figure(figsize=(14, 9), facecolor="white")
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38,
                        left=0.07, right=0.97, top=0.91, bottom=0.08)

def tag(ax, s):
    ax.text(-0.15, 1.06, s, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom")

def mean_band(ax, x, mat, col, lbl, ms=4):
    m = mat.mean(1); s = mat.std(1)
    ax.fill_between(x, m - s, m + s, alpha=0.12, color=col)
    ax.plot(x, m, "-o", ms=ms, lw=2, color=col, label=lbl)

# ── A: reclassification ───────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(alpha_n, recl_fw,  "-o", ms=4, lw=2, color=CFW,  label="Full-W (121 sites, α)")
ax.plot(amp_n,   recl_lda, "-s", ms=4, lw=2, color=CLDA, label="LDA-resonant (1 site, A)")
ax.set_xlabel("Normalised dose"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification rate"); ax.set_ylim(-2, 107)
ax.legend(frameon=False); tag(ax, "A")

# ── B: zero-lag FC distance ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
mean_band(ax, alpha_n, D_fw_zlag, CFW,  "Full-W")
mean_band(ax, amp_n,   D_lp,      CLDA, "LDA-resonant")
ax.set_xlabel("Normalised dose")
ax.set_ylabel("Zero-lag FC distance  (1 − corr)")
ax.set_title("Stimulation cost — zero-lag FC")
ax.legend(frameon=False); tag(ax, "B")

# ── C: lagged FC distance (NEW) ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
mean_band(ax, alpha_n, D_fw_llag,  CFW,  "Full-W")
mean_band(ax, amp_n,   D_lp_llag,  CLDA, "LDA-resonant")
ax.set_xlabel("Normalised dose")
ax.set_ylabel("Lagged FC distance  (1 − corr, lags 0–2)")
ax.set_title("Stimulation cost — lagged FC")
ax.legend(frameon=False); tag(ax, "C")

# ── D: efficacy vs zero-lag cost ─────────────────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
ax.plot(D_fw_zlag.mean(1),  recl_fw,  "-o", ms=5, lw=2, color=CFW,  label="Full-W")
ax.plot(D_lp.mean(1),       recl_lda, "-s", ms=5, lw=2, color=CLDA, label="LDA-resonant")
# mark the 100% crossing point for each
for recl, D, col, mk in [(recl_fw, D_fw_zlag, CFW, "o"), (recl_lda, D_lp, CLDA, "s")]:
    idx = np.where(recl >= 95)[0]
    if len(idx):
        i0 = idx[0]
        ax.scatter(D[i0].mean(), recl[i0], s=120, c=col, marker=mk,
                   edgecolors="k", linewidths=1, zorder=6)
ax.set_xlabel("Zero-lag FC distance  (mean over patients)")
ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Efficacy vs zero-lag cost"); ax.set_ylim(-2, 107)
ax.legend(frameon=False); tag(ax, "D")

# ── E: efficacy vs lagged FC cost (NEW) ─────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
ax.plot(D_fw_llag.mean(1),   recl_fw,  "-o", ms=5, lw=2, color=CFW,  label="Full-W")
ax.plot(D_lp_llag.mean(1),   recl_lda, "-s", ms=5, lw=2, color=CLDA, label="LDA-resonant")
for recl, D, col, mk in [(recl_fw, D_fw_llag, CFW, "o"), (recl_lda, D_lp_llag, CLDA, "s")]:
    idx = np.where(recl >= 95)[0]
    if len(idx):
        i0 = idx[0]
        ax.scatter(D[i0].mean(), recl[i0], s=120, c=col, marker=mk,
                   edgecolors="k", linewidths=1, zorder=6)
ax.set_xlabel("Lagged FC distance  (mean over patients, lags 0–2)")
ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Efficacy vs lagged FC cost"); ax.set_ylim(-2, 107)
ax.legend(frameon=False); tag(ax, "E")

# ── F: per-patient scatter — zero-lag vs lagged FC distance at crossing ───────
ax = fig.add_subplot(gs[1, 2])
# find the first alpha/amplitude index where each condition reaches >= 90% recl
idx_fw  = int(np.where(recl_fw  >= 90)[0][0]) if any(recl_fw  >= 90) else -1
idx_lda = int(np.where(recl_lda >= 90)[0][0]) if any(recl_lda >= 90) else -1

if idx_fw >= 0 and idx_lda >= 0:
    # per-patient: zero-lag vs lagged-FC distance at first >=90% crossing
    ax.scatter(D_fw_zlag[idx_fw],  D_fw_llag[idx_fw],
               c=CFW, s=30, alpha=0.75, label=f"Full-W (α={ALPHAS_FW[idx_fw]:.2f})")
    ax.scatter(D_lp[idx_lda],      D_lp_llag[idx_lda],
               c=CLDA, s=30, marker="s", alpha=0.75,
               label=f"LDA-resonant (A={AMPS_LDA[idx_lda]:.1f})")
    lim_max = max(D_fw_zlag[idx_fw].max(), D_lp[idx_lda].max(),
                  D_fw_llag[idx_fw].max(), D_lp_llag[idx_lda].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(0, lim_max); ax.set_ylim(0, lim_max)
else:
    ax.text(0.5, 0.5, "< 90% reclassification\nnot reached in tested range",
            ha="center", va="center", transform=ax.transAxes)

ax.set_xlabel("Zero-lag FC distance  (per patient)")
ax.set_ylabel("Lagged FC distance  (per patient)")
ax.set_title("Per-patient cost: zero-lag vs lagged FC")
ax.legend(frameon=False, fontsize=7.5); tag(ax, "F")

fig.suptitle(
    "Full-W correction vs single-site LDA-resonant drive: efficacy and stimulation cost",
    fontsize=11, fontweight="bold", y=0.97)

for ext in ("png", "pdf"):
    out = f"figure6_fullW_vs_lda.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
plt.close(fig)
print("Done.")
