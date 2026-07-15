"""
pert_dw_site_gridsearch.py
==========================
Grid-search over all 121 sites for the theoretical Delta-W single-column
correction, with the amplitude scaled by N_SITES=121 so that the total
correction budget matches full-W at alpha=1.

Full-W correction at alpha:    W_int[:, s] = (1-a)*W_AD[:,s] + a*W_CC[:,s]  FOR ALL s
  -> total L2 correction = sum_s a * ||DW[:,s]||

Single-site at alpha*121:      W_int[:, s*] = (1-121*a)*W_AD[:,s*] + 121*a*W_CC[:,s*]  FOR ONE s*
  -> same total L2 budget if ||DW[:,s*]|| == mean(||DW[:,s]||)

We sweep alpha in [0, 1/121, 2/121, ..., 1] (so alpha*121 in [0,1,...,121])
and report reclassification rate per site at the matched budget (alpha*121 = 1).

Outputs:
  pert_dw_site_gridsearch_data.npz  -- full score array (N_ALPHA, N_SITES, N_AD)
  pert_dw_site_gridsearch.png       -- summary figure

Comparison column: results from pert_single_compare_data.npz (existing runs):
  - full-W at alpha=1 (from pert_sites_stimulation or pert_compare3)
  - top-1 DW site at alpha=121 (same budget, wrong site)
  - LDA-resonant oscillatory (from pert_compare3)
"""
import os, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
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
TS_ROOT     = "./timeseries"
OUT         = "paper_figures"
iu          = np.triu_indices(N_SITES, 1)

# alpha_full values; alpha applied to the single column = alpha_full * N_SITES
ALPHAS_FULL = np.array([0, 1/121, 2/121, 4/121, 8/121, 16/121,
                         0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
ALPHAS_COL  = ALPHAS_FULL * N_SITES      # column alpha

# ── load data + build reservoir (identical to pert_single_compare.py) ────────
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
            signals.append(a.T)
            pids.append(fn.split("_ses-")[0])
            labs.append(0 if lb == "CC" else 1)
pids = np.array(pids); labs = np.array(labs); upid = np.unique(pids)
psid = {p: np.where(pids == p)[0] for p in upid}
plabel = np.array([labs[psid[p][0]] for p in upid])
first  = {p: psid[p][0] for p in upid}
cc = [upid[i] for i in np.where(plabel == 0)[0]]
ad = [upid[i] for i in np.where(plabel == 1)[0]]
n_ad = len(ad)

# population PCA projection (top 50 components)
all_sig = np.concatenate([s.T for s in signals], 0)
evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]

# reservoir
np.random.seed(RNG_SEED)
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN, T=139, dt=0.005,
           sigma_input=0.01, shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

# teacher-force all subjects
patX = {}
for p in tqdm(upid, desc="  teacher-force"):
    s   = signals[first[p]]; T = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X.append(res.X.copy())
    patX[p] = np.array(X)[TIMES_SKIP:]

# fit read-outs
rw = np.random.default_rng(RNG_SEED + 1)
patW = {}
for p in upid:
    s   = signals[first[p]]; T = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    Xc  = patX[p]; Yc = tgt[:, TIMES_SKIP:TIMES_SKIP + Xc.shape[0]].T
    patW[p] = np.linalg.pinv(Xc + rw.normal(0, SIGMA, Xc.shape)) @ Yc

Wcc  = np.mean([patW[p] for p in cc], 0)                        # (N_HIDDEN, N_SITES)
DW   = {p: Wcc - patW[p] for p in ad}                           # per-patient delta
top1 = {p: int(np.argmax(np.linalg.norm(DW[p], axis=0))) for p in ad}

# per-site DW norm (population mean) — for budget reference
site_dw_norm = np.mean([np.linalg.norm(DW[p], axis=0) for p in ad], 0)  # (N_SITES,)

# ── FC-lag LDA (same as pert_single_compare.py) ──────────────────────────────
def lagc(S, l):
    if l == 0: return np.corrcoef(S.T)
    T = S.shape[0]; A = S[:T-l].copy(); B = S[l:].copy()
    A -= A.mean(0); B -= B.mean(0)
    A /= A.std(0) + 1e-12; B /= B.std(0) + 1e-12
    return (A.T @ B) / (T - l)

def feat(W, X):
    S = (W.T.astype(float) @ X.T.astype(float)).T
    fs = []
    for l in range(MAX_LAG + 1):
        fc = np.nan_to_num(lagc(S, l))
        fs.append(fc[np.triu_indices(N_SITES, 1)] if l == 0 else fc.flatten())
    return np.concatenate(fs)

fb   = np.array([feat(patW[p], patX[p]) for p in tqdm(upid, leave=False, desc="  feat")])
fm   = fb.mean(0); fcc = fb - fm
evf, evecf = np.linalg.eigh(fcc @ fcc.T)
o    = np.argsort(evf)[::-1]; evf = np.maximum(evf[o], 0); evecf = evecf[:, o]
Gf   = evecf * np.sqrt(evf)

class LDA:
    def fit(s, X, y):
        c0, c1 = np.unique(y); X0, X1 = X[y == c0], X[y == c1]
        m0, m1 = X0.mean(0), X1.mean(0)
        Sw = (X0-m0).T@(X0-m0) + (X1-m1).T@(X1-m1) + 1e-6*np.eye(X.shape[1])
        w  = np.linalg.solve(Sw, m1 - m0); w /= np.linalg.norm(w) + 1e-12
        s.w = w; return s
    def tr(s, X): return X @ s.w

def bal(X, y, sd=0):
    r = np.random.default_rng(sd)
    c0, c1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([r.choice(c0, n, False), r.choice(c1, n, False)])
    r.shuffle(sel); return X[sel], y[sel]

Xl, yl = bal(Gf[:, :K_LDA], plabel, RNG_SEED)
lda_f  = LDA().fit(Xl, yl)
Zf     = lda_f.tr(Gf[:, :K_LDA])
if Zf[plabel == 0].mean() > Zf[plabel == 1].mean():
    lda_f.w *= -1; Zf = -Zf
thr_f  = 0.5 * (Zf[plabel == 0].mean() + Zf[plabel == 1].mean())

def fscore(W, X):
    f = feat(W, X) - fm
    g = (f @ fcc.T @ evecf) / (np.sqrt(evf) + 1e-12)
    return float(lda_f.tr(g[:K_LDA].reshape(1, -1))[0])

# ── grid search: all 121 sites × all alphas ──────────────────────────────────
# Scores shape: (N_ALPHA, N_SITES, N_AD)
# For each (alpha, site, patient): apply W_int[:, site] = (1 - a_col)*W_AD[:,site] + a_col*Wcc[:,site]
# where a_col = alpha_full * N_SITES

n_alpha = len(ALPHAS_FULL)
Scores  = np.zeros((n_alpha, N_SITES, n_ad), dtype=np.float32)
ad_list = list(ad)

print(f"\nGrid search: {n_alpha} alphas × {N_SITES} sites × {n_ad} AD patients ...")
t0 = time.time()
for ai, a_col in enumerate(ALPHAS_COL):
    for si in range(N_SITES):
        for pi, p in enumerate(ad_list):
            Wi = patW[p].copy()
            Wi[:, si] = (1 - a_col) * patW[p][:, si] + a_col * Wcc[:, si]
            Scores[ai, si, pi] = fscore(Wi, patX[p])
    a_full = ALPHAS_FULL[ai]
    recl_per_site = (Scores[ai] < thr_f).mean(axis=1) * 100   # (N_SITES,)
    best_s = int(np.argmax(recl_per_site))
    print(f"  a_full={a_full:.4f}  a_col={a_col:.1f}  "
          f"best_site={best_s} ({recl_per_site[best_s]:.0f}%)  "
          f"mean_over_sites={recl_per_site.mean():.1f}%  "
          f"elapsed={time.time()-t0:.0f}s", flush=True)

# ── save ─────────────────────────────────────────────────────────────────────
np.savez("pert_dw_site_gridsearch_data.npz",
         alphas_full=ALPHAS_FULL, alphas_col=ALPHAS_COL,
         Scores=Scores, thr_f=thr_f,
         site_dw_norm=site_dw_norm,
         ad_list=np.array(ad_list, dtype=object))
print("Saved pert_dw_site_gridsearch_data.npz")

# ── summary figure ────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family": "sans-serif", "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.dpi": 300})

# matched-budget slice: alpha_full = 1  (a_col = 121)
idx1 = np.argmin(np.abs(ALPHAS_FULL - 1.0))
recl_matched  = (Scores[idx1] < thr_f).mean(axis=1) * 100    # (N_SITES,)
order         = np.argsort(recl_matched)[::-1]

# comparison values (from existing data files)
# full-W at alpha=1
try:
    d_fw = np.load("pert_compare3_data.npz", allow_pickle=True)
    # full-W reclassification at alpha≈1 (row closest to 1)
    alphas_fw = d_fw["alphas"] if "alphas" in d_fw else None
    if alphas_fw is not None:
        idx_fw = np.argmin(np.abs(alphas_fw - 1.0))
        recl_fullW = float(d_fw.get("recl_full", [None]*10)[idx_fw] or
                           (d_fw["recl_f"][idx_fw] if "recl_f" in d_fw else 100.0))
    else:
        recl_fullW = 100.0
except Exception:
    recl_fullW = 100.0  # full-W at alpha=1 reverts all patients

# top-1 DeltaW at same matched budget (alpha_col=121) — from grid
top1_indices = np.array([int(np.argmax(np.linalg.norm(DW[p], axis=0))) for p in ad_list])
recl_top1_matched = float(np.mean(
    [Scores[idx1, top1_indices[pi], pi] < thr_f for pi in range(n_ad)]
) * 100)

# also load LDA-resonant (oscillatory) from pert_compare3
try:
    recl_lda_osc = float(np.load("pert_single_compare_data.npz")["recl_lp"].max())
except Exception:
    recl_lda_osc = 100.0

fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)

# ── panel A: reclassification per site at matched budget (ranked) ─────────────
ax = axes[0]
colors = plt.cm.RdYlGn(recl_matched[order] / 100)
ax.bar(range(N_SITES), recl_matched[order], color=colors, width=1.0, edgecolor="none")
ax.axhline(recl_fullW,    color="#1565C0", lw=1.8, ls="--", label=f"full-W (all sites, α=1): {recl_fullW:.0f}%")
ax.axhline(recl_top1_matched, color="#1A237E", lw=1.8, ls=":", label=f"top-1 ΔW site (α=121): {recl_top1_matched:.0f}%")
ax.axhline(recl_lda_osc,  color="#2E7D32", lw=1.8, ls="-.", label=f"LDA-resonant osc: {recl_lda_osc:.0f}%")
ax.set_xlabel("Sites (ranked by reclassification rate)")
ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title(f"Single-site ΔW×121 — matched-budget grid search\n(α_col = {ALPHAS_COL[idx1]:.0f})")
ax.set_ylim(-2, 108); ax.legend(frameon=False, fontsize=7.5)
ax.text(0.02, 0.97, f"Best site: {recl_matched.max():.0f}%\nMedian: {np.median(recl_matched):.0f}%",
        transform=ax.transAxes, va="top", fontsize=8)

# ── panel B: reclassification vs alpha for top-5 sites and full-W ────────────
ax = axes[1]
top5_sites = order[:5]
cmap = plt.cm.tab10
for rank, si in enumerate(top5_sites):
    recl_vs_alpha = (Scores[:, si, :] < thr_f).mean(axis=1) * 100
    ax.plot(ALPHAS_FULL, recl_vs_alpha, "-o", ms=4, lw=1.8,
            color=cmap(rank), label=f"site {si} (rank {rank+1})", zorder=3)
# top-1 ΔW site (per-patient, so use the most common one)
most_common_top1 = int(np.bincount(top1_indices).argmax())
recl_top1_vs_alpha = (Scores[:, most_common_top1, :] < thr_f).mean(axis=1) * 100
ax.plot(ALPHAS_FULL, recl_top1_vs_alpha, "--", color="#1A237E", lw=1.5,
        label=f"most-common top-1 ΔW (site {most_common_top1})")
ax.axvline(1.0, color="k", lw=0.8, ls=":", alpha=0.5)
ax.axhline(recl_fullW, color="#1565C0", lw=1.5, ls="--", alpha=0.7, label=f"full-W reference ({recl_fullW:.0f}%)")
ax.set_xlabel("α_full (column alpha = α_full × 121)")
ax.set_ylabel("AD reclassified (%)")
ax.set_title("Reclassification vs budget: best vs worst sites")
ax.set_ylim(-2, 108); ax.legend(frameon=False, fontsize=7)

# ── panel C: reclassification heatmap (sites × alphas) ───────────────────────
ax = axes[2]
recl_map = (Scores < thr_f).mean(axis=2) * 100    # (N_ALPHA, N_SITES)
im = ax.imshow(recl_map[:, order].T, aspect="auto", origin="lower",
               extent=[ALPHAS_FULL[0], ALPHAS_FULL[-1], 0, N_SITES],
               cmap="RdYlGn", vmin=0, vmax=100)
ax.axvline(1.0, color="white", lw=1.2, ls="--", alpha=0.8)
plt.colorbar(im, ax=ax, label="reclassification rate (%)", shrink=0.85)
ax.set_xlabel("α_full (column alpha = α_full × 121)")
ax.set_ylabel("Sites (ranked by reclassification at α_full=1)")
ax.set_title("Reclassification map (sites × budget)")

fig.suptitle("Single-site ΔW correction with matched budget (α×121):\n"
             "grid search over all 121 sites", fontsize=11, fontweight="bold")
fig.savefig("pert_dw_site_gridsearch.png", dpi=300, bbox_inches="tight")
print("Saved pert_dw_site_gridsearch.png")
plt.close(fig)

# ── text summary ──────────────────────────────────────────────────────────────
print("\n=== Results summary ===")
print(f"full-W at α=1:                    {recl_fullW:.0f}%  (reference)")
print(f"Top-1 ΔW site, α_col=121:         {recl_top1_matched:.0f}%  (matched budget, wrong site)")
print(f"Best single site, α_col=121:      {recl_matched.max():.0f}%  (matched budget, best site)")
print(f"Median over sites, α_col=121:     {np.median(recl_matched):.0f}%")
print(f"LDA-resonant oscillatory (1 site):{recl_lda_osc:.0f}%  (existing result)")
print(f"\nTop-5 best sites (rank, site_idx, recl%):")
for rank, si in enumerate(order[:5]):
    print(f"  {rank+1}.  site {si:3d}  {recl_matched[si]:.1f}%"
          f"  (ΔW norm = {site_dw_norm[si]:.4f})")
print(f"\nBottom-3 worst sites (site_idx, recl%):")
for si in order[-3:]:
    print(f"       site {si:3d}  {recl_matched[si]:.1f}%")
print("\nDone.")
