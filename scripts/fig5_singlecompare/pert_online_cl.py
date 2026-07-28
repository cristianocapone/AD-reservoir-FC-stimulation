"""
pert_online_cl.py
=================
Online closed-loop stimulation: amplitude titrated in real time from an
expanding-window FC-lag biomarker estimate.

Fixed site (per patient): personalized LDA-resonant site — argmax of per-site
resonance reduction from pert_compare3_data.npz.

Controller: ramp-up-and-hold.
  Every EPOCH free-run steps compute FC-lag score from all output so far.
  If score > thr:  amp = min(A_MAX, amp + DELTA)
  If score <= thr: hold (freeze amplitude at current value)
Starting amplitude: 0.

Compares (all fixed site, no site adaptation):
  (1) Open-loop      : fixed A=A_FIX  (from pert_closedloop_data.npz)
  (2) CL offline     : full-sim oracle for minimum crossing amp (same file)
  (3) Online CL      : new computation

Saves: pert_online_cl_data.npz
       paper_figures/figureS11_online_cl.{png,pdf}
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eig as sla_eig
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); from res import RESERVOIRE_SIMPLE

# ── constants (must match pert_closedloop.py exactly) ─────────────────────────
RNG_SEED = 42; N_CC_SAMP = 40; N_SITES = 121; N_PC_MODEL = 50; TIMES_SKIP = 10
ff = 0.1; N_HIDDEN = 2000; SIGMA = 0.05; SR = 0.95; K_LDA = 25; MAX_LAG = 2
DRIVE_STEPS = 5; TS_ROOT = "./timeseries"; OUT = "paper_figures"
iu = np.triu_indices(N_SITES, 1)
A_FIX = 6.0

# online CL hyperparameters
EPOCH    = 20   # free-run steps between biomarker evaluations
DELTA    = 1.0  # amplitude decrement per epoch
A_START  = 6.0  # starting amplitude (known-good — controller reduces from here)

# ── load signals, fit reservoir, compute patX / patW ─────────────────────────
print("Loading data + reservoir ...")
rng = np.random.default_rng(RNG_SEED); signals, labs, pids = [], [], []
for sub, lb in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, sub)
    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if lb == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMP, len(files)), replace=False))
    for fn in files:
        a = np.load(os.path.join(folder, fn)).T
        if a.shape[1] == N_SITES and a.shape[0] >= 139:
            signals.append(a.T); pids.append(fn.split("_ses-")[0])
            labs.append(0 if lb == "CC" else 1)
pids = np.array(pids); labs = np.array(labs); upid = np.unique(pids)
psid  = {p: np.where(pids == p)[0] for p in upid}
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
    s = signals[first[p]]; T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T-1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.); X.append(res.X.copy())
    patX[p] = np.array(X)[TIMES_SKIP:]

rw = np.random.default_rng(RNG_SEED + 1); patW = {}
for p in upid:
    s = signals[first[p]]; tgt = (s.T @ ev50 @ ev50.T).T
    Xc = patX[p]; Yc = tgt[:, TIMES_SKIP:TIMES_SKIP + Xc.shape[0]].T
    patW[p] = np.linalg.pinv(Xc + rw.normal(0, SIGMA, Xc.shape)) @ Yc

# ── feature / LDA ─────────────────────────────────────────────────────────────
class LDA:
    def fit(s, X, y):
        c0, c1 = np.unique(y); X0, X1 = X[y==c0], X[y==c1]; m0, m1 = X0.mean(0), X1.mean(0)
        Sw = (X0-m0).T@(X0-m0) + (X1-m1).T@(X1-m1) + 1e-6*np.eye(X.shape[1])
        w = np.linalg.solve(Sw, m1-m0); w /= np.linalg.norm(w)+1e-12; s.w = w; return s
    def tr(s, X): return X @ s.w

def balm(X, y, sd=0):
    r = np.random.default_rng(sd)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]; n = min(len(c0), len(c1))
    sel = np.concatenate([r.choice(c0, n, 0), r.choice(c1, n, 0)]); r.shuffle(sel)
    return X[sel], y[sel]

def lagc(S, l):
    if l == 0: return np.corrcoef(S.T)
    T = S.shape[0]; A = S[:T-l].astype(float); B = S[l:].astype(float)
    A -= A.mean(0); B -= B.mean(0); A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T-l)

def feat(W, X):
    S = (W.T.astype(float) @ X.T.astype(float)).T; fs = []
    for l in range(MAX_LAG+1):
        fc = np.nan_to_num(lagc(S, l))
        fs.append(fc[np.triu_indices(N_SITES, 1)] if l == 0 else fc.flatten())
    return np.concatenate(fs)

fb = np.array([feat(patW[p], patX[p]) for p in upid])
fm = fb.mean(0); fcc_mat = fb - fm
evf, evecf = np.linalg.eigh(fcc_mat @ fcc_mat.T); o = np.argsort(evf)[::-1]
evf = np.maximum(evf[o], 0); evecf = evecf[:, o]; Gf = evecf * np.sqrt(evf)
Xl, yl = balm(Gf[:, :K_LDA], plabel, RNG_SEED)
lda_f = LDA().fit(Xl, yl); Zf = lda_f.tr(Gf[:, :K_LDA])
if Zf[plabel==0].mean() > Zf[plabel==1].mean(): lda_f.w *= -1; Zf = -Zf
thr_f = 0.5 * (Zf[plabel==0].mean() + Zf[plabel==1].mean())

def fscore(W, X):
    f = feat(W, X) - fm
    g = (f @ fcc_mat.T @ evecf) / (np.sqrt(evf) + 1e-12)
    return float(lda_f.tr(g[:K_LDA].reshape(1,-1))[0])

# ── reservoir resonant frequency ───────────────────────────────────────────────
wv, vl, vr = sla_eig(res.J, left=True, right=True)
pos = np.where(wv.imag > 1e-8)[0]
f1 = float(abs(np.angle(wv[pos[np.argsort(np.abs(wv[pos]))[::-1][0]]]))/(2*np.pi))
print(f"Resonant frequency f1 = {f1:.4f}")

# ── personalised sites (argmax per-patient resonance reduction) ────────────────
red_full = np.load("pert_compare3_data.npz", allow_pickle=True)["red_full"]
pers_site = {p: int(np.argmax(red_full[:, pi])) for pi, p in enumerate(ad)}

# ── reference free-run FC (no stimulation) ────────────────────────────────────
print("\nComputing reference free-run FC for AD patients ...")
fc_ref = {}
for p in tqdm(ad, desc="  ref"):
    s = signals[first[p]]; T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); res.Jout = patW[p].T.copy(); Y = []
    for t in range(T-1):
        fbk = tgt[:, t] if t <= DRIVE_STEPS else res.y
        res.step_rate(ff * np.asarray(fbk, dtype=float), sigma_dyn=0.)
        Y.append(np.asarray(res.y, dtype=float).copy())
    fc_ref[p] = np.nan_to_num(np.corrcoef(np.array(Y).T[:, TIMES_SKIP-1:]))

def fc_dist(Y_arr, p):
    fc = np.nan_to_num(np.corrcoef(Y_arr.T[:, TIMES_SKIP-1:]))
    return 1.0 - float(np.corrcoef(fc[iu], fc_ref[p][iu])[0, 1])

# ── online CL simulation ───────────────────────────────────────────────────────
def run_online_cl(p, site):
    """
    Dose-minimisation closed-loop controller.
    Starts at A_START (guaranteed to reclassify). Every EPOCH free-run steps,
    compute FC-lag score from the last EPOCH output states (sliding window).
    If still reclassified (score < thr): reduce amp by DELTA (dose-sparing).
    If lost reclassification (score > thr): restore amp by DELTA.
    Mean amp ≈ minimum dose that maintains reclassification.

    Returns: reclassified (0/1), final_score, fc_distance, mean_amp, amp_trajectory
    """
    s = signals[first[p]]; T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); res.Jout = patW[p].T.copy()
    X_hist = []; Y_hist = []
    amp = A_START; amp_traj = []

    for t in range(T-1):
        fbk = tgt[:, t] if t <= DRIVE_STEPS else res.y
        inp = ff * np.asarray(fbk, dtype=float).copy()
        if t > DRIVE_STEPS and amp > 0:
            inp[site] += amp * np.sin(2*np.pi*f1*t)
        res.step_rate(inp, sigma_dyn=0.)
        X_hist.append(res.X.copy())
        Y_hist.append(np.asarray(res.y, dtype=float).copy())
        amp_traj.append(amp)

        # Update amplitude every EPOCH free-run steps using sliding window
        n_free = t - DRIVE_STEPS
        if n_free > 0 and n_free % EPOCH == 0:
            # sliding window: last EPOCH states only
            X_window = np.array(X_hist[-(TIMES_SKIP + EPOCH):])
            if len(X_window) >= EPOCH:
                score = fscore(patW[p], X_window[-EPOCH:])
                if score < thr_f:
                    amp = max(0., amp - DELTA)   # reclassified — reduce dose
                else:
                    amp = min(A_START, amp + DELTA)  # lost it — restore

    X_arr = np.array(X_hist)[TIMES_SKIP:]
    Y_arr = np.array(Y_hist)
    final_score = fscore(patW[p], X_arr)
    reclassified = int(final_score < thr_f)
    d = fc_dist(Y_arr, p)
    mean_amp = float(np.mean(amp_traj[DRIVE_STEPS:]))
    return reclassified, final_score, d, mean_amp, np.array(amp_traj)

print(f"\nOnline CL (dose-minimisation): EPOCH={EPOCH}, DELTA={DELTA}, A_START={A_START} ...")
ocl_rec  = np.zeros(n_ad)
ocl_dist = np.full(n_ad, np.nan)
ocl_amp  = np.full(n_ad, np.nan)
ocl_traj = []

for pi, p in enumerate(tqdm(ad, desc="  patients")):
    rec, score, d, mean_amp, traj = run_online_cl(p, pers_site[p])
    ocl_rec[pi]  = rec
    ocl_dist[pi] = d
    ocl_amp[pi]  = mean_amp
    ocl_traj.append(traj)

# ── load reference conditions ─────────────────────────────────────────────────
cl = np.load("pert_closedloop_data.npz", allow_pickle=True)
ol_rec  = cl["ol_rec"];  ol_dist = cl["ol_dist"]
ca_rec  = cl["ca_rec"];  ca_dist = cl["ca_dist"]; ca_amp = cl["ca_amp"]

print()
for name, rec, dist_arr, amp_arr in [
    ("open-loop",  ol_rec,  ol_dist, np.full(n_ad, A_FIX)),
    ("CL offline", ca_rec,  ca_dist, ca_amp),
    ("online CL",  ocl_rec, ocl_dist, ocl_amp),
]:
    m = rec > 0
    print(f"  {name:12s}  reclass {m.mean()*100:5.1f}%  "
          f"dist {np.nanmean(dist_arr[m]):.3f}  amp {np.nanmean(amp_arr[m]):.2f}")

# ── save ──────────────────────────────────────────────────────────────────────
np.savez("pert_online_cl_data.npz",
         ocl_rec=ocl_rec, ocl_dist=ocl_dist, ocl_amp=ocl_amp,
         ol_rec=ol_rec, ol_dist=ol_dist,
         ca_rec=ca_rec, ca_dist=ca_dist, ca_amp=ca_amp,
         f1=f1, thr_f=thr_f, epoch=EPOCH, delta=DELTA, a_start=A_START)
print("Saved pert_online_cl_data.npz")

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.labelsize": 9.5,
    "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
COL_OL  = "#455A64"   # open-loop: slate
COL_OCL = "#E65100"   # online CL: orange
NAMES   = ["Open-loop\n(fixed A=6)", "Online CL\n(dose-min)"]
rng_j   = np.random.default_rng(7)

def paired_panel(ax, vals_ol, vals_ocl, ylabel, title, lbl):
    """Paired dot plot: each patient connected by a line; means ± SE marked."""
    n = len(vals_ol)
    jit = rng_j.uniform(-0.06, 0.06, n)
    for i in range(n):
        col = "#BDBDBD" if vals_ocl[i] <= vals_ol[i] else "#EF9A9A"
        ax.plot([0 + jit[i], 1 + jit[i]], [vals_ol[i], vals_ocl[i]],
                lw=0.6, color=col, alpha=0.6, zorder=1)
    ax.scatter(0 + jit, vals_ol,  s=18, color=COL_OL,  alpha=0.8, zorder=3, edgecolors="none")
    ax.scatter(1 + jit, vals_ocl, s=18, color=COL_OCL, alpha=0.8, zorder=3, edgecolors="none")
    for xi, (vals, col) in enumerate([(vals_ol, COL_OL), (vals_ocl, COL_OCL)]):
        m, se = np.nanmean(vals), np.nanstd(vals) / np.sqrt(np.sum(~np.isnan(vals)))
        ax.plot([xi - 0.18, xi + 0.18], [m, m], lw=2.5, color=col, zorder=4)
        ax.errorbar(xi, m, yerr=se, fmt="none", ecolor=col, capsize=4, lw=1.5, zorder=4)
    # annotate % change
    pct = (np.nanmean(vals_ocl) - np.nanmean(vals_ol)) / np.nanmean(vals_ol) * 100
    sign = "+" if pct > 0 else ""
    ax.text(0.5, 0.97, f"{sign}{pct:.0f}%", transform=ax.transAxes,
            ha="center", va="top", fontsize=9, fontweight="bold",
            color=COL_OCL if pct < 0 else "#C62828")
    ax.set_xticks([0, 1]); ax.set_xticklabels(NAMES)
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.text(-0.16, 1.04, lbl, transform=ax.transAxes, fontsize=13, fontweight="bold")

fig = plt.figure(figsize=(13.5, 4.5), facecolor="white")
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                        left=0.07, right=0.985, top=0.87, bottom=0.18)

# A — reclassification rate
ax = fig.add_subplot(gs[0, 0])
bars = ax.bar([0, 1], [ol_rec.mean()*100, ocl_rec.mean()*100],
              color=[COL_OL, COL_OCL], alpha=0.85, width=0.5)
for bar, val in zip(bars, [ol_rec.mean()*100, ocl_rec.mean()*100]):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1.5, f"{val:.0f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(NAMES)
ax.set_ylabel("AD reclassified as CC (%)"); ax.set_ylim(0, 112)
ax.set_title("Reclassification rate")
ax.text(-0.16, 1.04, "A", transform=ax.transAxes, fontsize=13, fontweight="bold")

# B — per-patient mean amplitude (paired)
ol_amp_arr = np.full(n_ad, A_FIX)
paired_panel(fig.add_subplot(gs[0, 1]),
             ol_amp_arr, ocl_amp,
             "Mean stimulation amplitude", "Perturbation size", "B")

# C — per-patient FC distance (paired, only reclassified patients)
m_both = (ol_rec > 0) & (ocl_rec > 0)
paired_panel(fig.add_subplot(gs[0, 2]),
             ol_dist[m_both], ocl_dist[m_both],
             "FC distance from baseline (1 − corr)",
             "FC disruption cost\n(reclassified patients)", "C")

fig.suptitle(
    f"Online closed-loop dose-minimisation  |  fixed LDA-resonant site  |"
    f"  epoch={EPOCH} steps, Δ={DELTA}, A_start={A_START}  |  N={n_ad} AD patients",
    fontsize=10, fontweight="bold", y=0.99)

for ext in ("png", "pdf"):
    out = f"{OUT}/figureS11_online_cl.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
plt.close(fig)
print("Done.")
