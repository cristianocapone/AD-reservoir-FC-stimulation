"""
figureS10_physio.py
===================
Supplementary physiological diagnostics for the read-out interpolation
(supports the Results claim that full-W correction leaves signal amplitude near
baseline whereas focal corrections require large local amplification).

For every strategy and interpolation strength alpha we record, at the stimulated
sites, the signal-amplitude ratio RMS_pert / RMS_base and the mean absolute FC
(as defined in Methods, "Physiological diagnostics").

Cheap: the reconstruction is Y = X @ W_int, so no reservoir re-run is needed
beyond the single teacher-forcing pass.

Saves: figureS10_physio.{png,pdf}
"""
import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from res import RESERVOIRE_SIMPLE

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── parameters (match the stimulation analyses) ───────────────────────────────
RNG_SEED = 42; N_CC_SAMP = 40; N_SITES = 121; N_PC_MODEL = 50; TIMES_SKIP = 10
ff = 0.1; N_HIDDEN = 2000; SIGMA = 0.05; SR = 0.95; TS_ROOT = "../timeseries"

FW_COL, T5_COL, T1_COL, GEO_COL = "#2E7D32", "#E65100", "#6A1B9A", "#00838F"

# ── data ──────────────────────────────────────────────────────────────────────
print("Loading data ...")
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
first = {p: np.where(pids == p)[0][0] for p in upid}
plabel = np.array([labs[first[p]] for p in upid])
cc = [upid[i] for i in np.where(plabel == 0)[0]]
ad = [upid[i] for i in np.where(plabel == 1)[0]]
n_ad = len(ad)
print(f"  {len(upid)} patients ({len(cc)} CC, {n_ad} AD)")

all_sig = np.concatenate([s.T for s in signals], 0)
evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]

np.random.seed(RNG_SEED)
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN, T=139, dt=0.005,
           sigma_input=0.01, shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par); res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

patX, patY = {}, {}
for p in tqdm(upid, desc="  teacher-force"):
    s = signals[first[p]]; T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T-1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.); X.append(res.X.copy())
    Xf = np.array(X)[TIMES_SKIP:]
    patX[p] = Xf; patY[p] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T

rw = np.random.default_rng(RNG_SEED + 1); patW = {}
for p in upid:
    patW[p] = np.linalg.pinv(patX[p] + rw.normal(0, SIGMA, patX[p].shape)) @ patY[p]
Wcc = np.mean([patW[p] for p in cc], axis=0)

# ── alpha grids + parcel coords (reuse the stimulation cache) ─────────────────
d = np.load("../pert_sites_data.npz", allow_pickle=True)
A_FW  = d["alphas_fw"]; A_T5 = d["alphas_t5"]
A_T1  = d["alphas_t1"]; A_GEO = d["alphas_geo"]
coords = d["parcel_coords"]

# ── per-patient site sets ranked by ||dW[:,k]|| ───────────────────────────────
site_sets = {}
for p in ad:
    dW = Wcc - patW[p]
    order = np.argsort(np.linalg.norm(dW, axis=0))[::-1]
    t1 = int(order[0])
    nbrs = np.argsort(np.linalg.norm(coords - coords[t1], axis=1))[:11]  # self + 10
    site_sets[p] = {"full_w": np.arange(N_SITES),
                    "top5":   order[:5].astype(int),
                    "top1":   np.array([t1]),
                    "top1_geo": nbrs.astype(int)}

def diagnostics(p, sites, alpha):
    """RMS_pert/RMS_base and mean |FC| at the stimulated sites."""
    W = patW[p]; Wi = W.copy()
    Wi[:, sites] = (1 - alpha) * W[:, sites] + alpha * Wcc[:, sites]
    Yb = patX[p] @ W          # baseline reconstruction   (T, 121)
    Yp = patX[p] @ Wi         # perturbed reconstruction
    rms_b = np.sqrt((Yb[:, sites] ** 2).mean())
    rms_p = np.sqrt((Yp[:, sites] ** 2).mean())
    fc = np.nan_to_num(np.corrcoef(Yp.T))
    m = np.abs(fc[np.ix_(sites, np.arange(N_SITES))])
    return rms_p / (rms_b + 1e-12), float(m.mean())

STRATS = [("full_w", A_FW,  FW_COL,  "Full-W (121 sites)"),
          ("top5",   A_T5,  T5_COL,  "Top-5 sites"),
          ("top1",   A_T1,  T1_COL,  "Top-1 site"),
          ("top1_geo", A_GEO, GEO_COL, "Top-1 + 10 geo-nbrs")]

print("\nComputing physiological diagnostics ...")
R = {}
for name, alphas, col, lbl in STRATS:
    ratio = np.zeros((len(alphas), n_ad)); mfc = np.zeros((len(alphas), n_ad))
    for ai, al in enumerate(tqdm(alphas, desc=f"  {name}", leave=False)):
        for pi, p in enumerate(ad):
            ratio[ai, pi], mfc[ai, pi] = diagnostics(p, site_sets[p][name], al)
    R[name] = dict(alphas=alphas, ratio=ratio, mfc=mfc, col=col, lbl=lbl)
    print(f"  {lbl:22s} RMS ratio at max alpha = {ratio[-1].mean():6.2f}")

np.savez("../physio_diag_data.npz",
         **{f"{k}_{q}": R[k][q] for k in R for q in ("alphas", "ratio", "mfc")})

# ── figure ────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4.0), facecolor="white")
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                       left=0.06, right=0.985, top=0.94, bottom=0.16)

def band(ax, key):
    r = R[key]; m = r["ratio"].mean(1); s = r["ratio"].std(1)
    ax.fill_between(r["alphas"], m-s, m+s, alpha=0.15, color=r["col"])
    ax.plot(r["alphas"], m, "-o", ms=3.5, lw=1.8, color=r["col"], label=r["lbl"])

ax = fig.add_subplot(gs[0, 0])
band(ax, "full_w")
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6, label="baseline amplitude")
ax.set_xlabel(r"Perturbation strength $\alpha$")
ax.set_ylabel(r"$\mathrm{RMS}_{\mathrm{pert}}/\mathrm{RMS}_{\mathrm{base}}$")
ax.set_title(r"Full-W: amplitude within $\sim$2$\times$ of baseline")
ax.legend(frameon=False, fontsize=7.5)

ax = fig.add_subplot(gs[0, 1])
for k in ("top5", "top1", "top1_geo"): band(ax, k)
ax.axhline(1.0, color="k", ls="--", lw=1, alpha=0.6, label="baseline amplitude")
ax.set_xlabel(r"Perturbation strength $\alpha$")
ax.set_ylabel(r"$\mathrm{RMS}_{\mathrm{pert}}/\mathrm{RMS}_{\mathrm{base}}$")
ax.set_title("Focal: large local amplification required")
ax.legend(frameon=False, fontsize=7.5)

ax = fig.add_subplot(gs[0, 2])
for k in ("full_w", "top5", "top1", "top1_geo"):
    r = R[k]; ax.plot(r["alphas"] / r["alphas"].max(), r["mfc"].mean(1),
                      "-o", ms=3.5, lw=1.8, color=r["col"], label=r["lbl"])
ax.set_xlabel("relative dose")
ax.set_ylabel(r"mean $|FC|$ at stimulated sites")
ax.set_title("Connectivity at the stimulated sites")
ax.legend(frameon=False, fontsize=7.5)

for ax, lbl in zip(fig.axes, ["A", "B", "C"]):
    ax.text(-0.14, 1.04, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")

for ext in ("png", "pdf"):
    fig.savefig(f"figureS10_physio.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figureS10_physio.{ext}")
plt.close(fig); print("Done.")
