"""
figure2_model.py
================
Figure 2 — Reservoir model: fit quality, FC structure, identifiability.

Runs the teacher-forced + closed-loop reservoir to obtain, per session:
  Y_emp  : PCA-projected empirical target            (N_sites, T)
  Y_sim  : closed-loop simulated reconstruction      (N_sites, T)

Panels (3 rows x 4 cols):
  A  Mean FC — CC data       B  Mean FC — AD data
  C  Mean FC — CC model      D  Mean FC — AD model
  E  4x4 FC correlation matrix between the four group-mean FCs
  F  Cross-subject FC matrix  sim_i vs emp_j  [nb cell 29]
  G  Same- vs cross-subject FC match histogram  [nb cell 29]
  H  Sample subject: empirical vs simulated FC scatter
  I  Delayed-FC fit: median ± IQR   [nb cell 28]  (wide)
  J  FC reconstruction quality vs noise  (cached)   (wide)

Saves: figure2_model.png  figure2_model.pdf
"""
import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from tqdm import trange
import warnings; warnings.filterwarnings("ignore")

from res import RESERVOIRE_SIMPLE

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "lines.linewidth": 1.8,
    "axes.linewidth": 0.8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
CC_COL = "#2196F3"; AD_COL = "#E91E63"

TS_ROOT      = "../timeseries"
RNG_SEED     = 42
N_CC_SAMP    = 40
N_SITES      = 121
TR           = 3.0
trial_dur    = 139
N_PC_MODEL   = 50
TIMES_SKIP   = 10
ff           = 0.1
N_HIDDEN     = 2000
SPECTRAL_RAD = 0.95
noise_size   = 0.025
MAX_LAG      = 20

def _tag(ax, ltr, x=-0.14, y=1.08):
    ax.text(x, y, ltr, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="top", ha="left")

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def delayed_fc(data, delay):
    """data: (regions, time).  Returns delayed FC block (regions, regions)."""
    if delay == 0:
        return np.nan_to_num(np.corrcoef(data))
    if data.shape[1] <= delay:
        return None
    lead = data[:, :-delay]; lag = data[:, delay:]
    corr = np.corrcoef(lead, lag)
    return np.nan_to_num(corr[data.shape[0]:, :data.shape[0]])

# ══════════════════════════════════════════════════════════════════════════════
# SIGNALS
# ══════════════════════════════════════════════════════════════════════════════
print("Loading signals ...")
rng = np.random.default_rng(RNG_SEED)
signals_raw, labels_l = [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    if not os.path.isdir(folder):
        print(f"  WARNING: {folder} not found"); continue
    files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMP, len(files)),
                                replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= trial_dur:
            signals_raw.append(arr)
            labels_l.append(0 if label == "CC" else 1)

labels   = np.array(labels_l)
ctrl_idx = np.where(labels == 0)[0]
ad_idx_  = np.where(labels == 1)[0]
sigs     = [s.T for s in signals_raw]
n_cc     = len(ctrl_idx); n_ad = len(ad_idx_)
P        = len(sigs)
print(f"  {P} sessions  (CC={n_cc}, AD={n_ad})")

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in sigs], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ══════════════════════════════════════════════════════════════════════════════
# RESERVOIR  (teacher-forced fit + closed-loop simulation)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading pickled reservoir (same J/Jin as original notebook) ...")
import pickle
with open("../network_reservoire.pkl", "rb") as f:
    res = pickle.load(f)
res.J = res.J * SPECTRAL_RAD          # exactly as Fig1DEF_MC_data.ipynb cell 20

# Original closed-loop protocol (train_test): reset, teacher-force the first
# DRIVE_STEPS steps with real data, then free-run with the model's own output.
DRIVE_STEPS = 5
rng_r = np.random.default_rng(RNG_SEED)
Y_emp, Y_sim = [], []
for idx in trange(P, desc="  reservoir"):
    s = sigs[idx]; T_s = s.shape[1]
    target = (s.T @ ev50 @ ev50.T).T            # (N_sites, T)
    Y_emp.append(target)

    # ── fit pass: fully teacher-forced (train_test_pinv) ──────────────────────
    res.T = T_s; res.reset()
    X_raw = []
    for t in range(T_s - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    X_fit = np.array(X_raw)[TIMES_SKIP:]
    Y_fit = target[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T
    noise = rng_r.normal(0, noise_size, X_fit.shape)
    W_out = np.linalg.pinv(X_fit + noise) @ Y_fit

    # ── sim pass: reset, drive DRIVE_STEPS, then free-run (train_test) ─────────
    res.reset(); res.Jout = W_out.T.copy()
    Ysim = []
    for t in range(T_s - 1):
        feedback = target[:, t] if t <= DRIVE_STEPS else res.y
        res.step_rate(ff * feedback, sigma_dyn=0.)
        Ysim.append(res.y.copy())
    Y_sim.append(np.array(Ysim).T)              # (N_sites, T-1)

# ── per-session FC (delay-0) ──────────────────────────────────────────────────
print("FC matrices (empirical & simulated) ...")
FC_emp, FC_sim = [], []
fc_emp_flat, fc_sim_flat = [], []
for i in range(P):
    emp = Y_emp[i][:, TIMES_SKIP:]
    sim = Y_sim[i][:, TIMES_SKIP-1:] if Y_sim[i].shape[1] > TIMES_SKIP else Y_sim[i]
    Tm  = min(emp.shape[1], sim.shape[1])
    emp = emp[:, :Tm]; sim = sim[:, :Tm]
    fce = np.nan_to_num(np.corrcoef(emp))
    fcs = np.nan_to_num(np.corrcoef(sim))
    FC_emp.append(fce); FC_sim.append(fcs)
    fc_emp_flat.append(fce.flatten()); fc_sim_flat.append(fcs.flatten())

# ── group-mean FCs:  CC/AD x data/model ───────────────────────────────────────
fc_cc_emp = np.mean([FC_emp[i] for i in ctrl_idx], axis=0)
fc_ad_emp = np.mean([FC_emp[i] for i in ad_idx_],  axis=0)
fc_cc_sim = np.mean([FC_sim[i] for i in ctrl_idx], axis=0)
fc_ad_sim = np.mean([FC_sim[i] for i in ad_idx_],  axis=0)

# ── 4x4 FC correlation matrix between group means (Pearson r, upper triangle) ─
four_fcs  = [fc_cc_emp, fc_ad_emp, fc_cc_sim, fc_ad_sim]
four_lbls = ["CC\ndata", "AD\ndata", "CC\nmodel", "AD\nmodel"]
iu        = np.triu_indices(N_SITES, k=1)
four_vecs = [f[iu] for f in four_fcs]
Cmat = np.zeros((4, 4))
for i in range(4):
    for j in range(4):
        Cmat[i, j] = float(np.corrcoef(four_vecs[i], four_vecs[j])[0, 1])
print("  4x4 FC correlation matrix:")
for i in range(4):
    print("   ", "  ".join(f"{Cmat[i,j]:.3f}" for j in range(4)))

# ── cross-subject FC identifiability matrix [nb cell 29] ──────────────────────
print("Cross-subject FC matrix ...")
S = np.array(fc_sim_flat); E = np.array(fc_emp_flat)
Sz = (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)
Ez = (E - E.mean(1, keepdims=True)) / (E.std(1, keepdims=True) + 1e-12)
cc_matrix    = (Sz @ Ez.T) / S.shape[1]
diag_vals    = np.diag(cc_matrix)
offdiag_vals = cc_matrix[~np.eye(P, dtype=bool)]
print(f"  same-subject med={np.median(diag_vals):.3f}  "
      f"cross-subject med={np.median(offdiag_vals):.3f}")

# representative subject (good fit, 75th pct)
target_q = np.quantile(diag_vals, 0.75)
rep_subj = int(np.argmin(np.abs(diag_vals - target_q)))
print(f"  representative subject: #{rep_subj}  (FC fit r={diag_vals[rep_subj]:.3f})")

# ── delayed-FC fit curves [nb cell 28] ────────────────────────────────────────
print(f"Delayed-FC fit (lag 0..{MAX_LAG}) ...")
all_es, all_e0, all_tr = [], [], []
for i in range(P):
    emp = Y_emp[i][:, TIMES_SKIP:]
    sim = Y_sim[i][:, TIMES_SKIP-1:] if Y_sim[i].shape[1] > TIMES_SKIP else Y_sim[i]
    Tm  = min(emp.shape[1], sim.shape[1])
    emp = emp[:, :Tm]; sim = sim[:, :Tm]
    if Tm < MAX_LAG + 4:
        continue
    even = emp[:, ::2]; odd = emp[:, 1::2]
    fc0  = delayed_fc(emp, 0)
    r_es, r_e0, r_tr = [], [], []
    for d in range(MAX_LAG + 1):
        fe = delayed_fc(emp, d); fs = delayed_fc(sim, d)
        if fe is None or fs is None:
            r_es.append(np.nan); r_e0.append(np.nan); r_tr.append(np.nan); continue
        r_es.append(np.corrcoef(fe.ravel(), fs.ravel())[0, 1])
        r_e0.append(np.corrcoef(fe.ravel(), fc0.ravel())[0, 1])
        fv = delayed_fc(even, d); fo = delayed_fc(odd, d)
        r_tr.append(np.nan if (fv is None or fo is None)
                    else np.corrcoef(fv.ravel(), fo.ravel())[0, 1])
    all_es.append(r_es); all_e0.append(r_e0); all_tr.append(r_tr)
all_es = np.array(all_es); all_e0 = np.array(all_e0); all_tr = np.array(all_tr)
delays_s = np.arange(MAX_LAG + 1) * TR

# ── atlas ordering ────────────────────────────────────────────────────────────
print("Atlas ...")
try:
    from nilearn import datasets as nl_ds
    sch        = nl_ds.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7,
                                                  resolution_mm=2)
    atlas_lbls = [l.decode() if isinstance(l, bytes) else l for l in sch.labels]
    NET_ORDER  = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
    NET_COLORS = dict(Vis="#1f77b4", SomMot="#ff7f0e", DorsAttn="#2ca02c",
                      SalVentAttn="#d62728", Limbic="#9467bd",
                      Cont="#8c564b", Default="#e377c2")
    def _net(lbl):
        for n in NET_ORDER:
            if n in lbl: return n
        return "Other"
    net_assign  = [_net(l) for l in atlas_lbls]
    sorted_idx  = sorted(range(100),
                         key=lambda i: (NET_ORDER.index(net_assign[i])
                                        if net_assign[i] in NET_ORDER else 7, i))
    sorted_nets = [net_assign[i] for i in sorted_idx]
    net_bounds  = [k - 0.5 for k in range(1, 100)
                   if sorted_nets[k] != sorted_nets[k-1]]
    have_atlas  = True
except Exception as e:
    print(f"  No atlas: {e}")
    sorted_idx = list(range(100)); net_bounds = []
    NET_COLORS = {}; have_atlas = False

# ── cached npz ────────────────────────────────────────────────────────────────
nd  = np.load("../fc_recon_noise_sweep.npz", allow_pickle=True)
noise_vals = nd["noise_vals"]; noise_means = nd["noise_means"]
noise_stds = nd["noise_stds"]; r_sess = nd["r_sess"]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering ...")
fig = plt.figure(figsize=(19, 12.8), facecolor="white")
gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.50, wspace=0.42,
                        top=0.985, bottom=0.05, left=0.06, right=0.97)

def _fc_panel(ax, fc_mat, title, title_col, draw_nets=True):
    fc_s = fc_mat[:100, :100][np.ix_(sorted_idx, sorted_idx)]
    im   = ax.imshow(fc_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal")
    cb   = plt.colorbar(im, ax=ax, shrink=0.80, pad=0.03, fraction=0.046)
    cb.set_label("Pearson r", fontsize=8); cb.ax.tick_params(labelsize=7)
    if draw_nets:
        for b in net_bounds:
            ax.axhline(b, color="white", lw=0.4)
            ax.axvline(b, color="white", lw=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Brain region"); ax.set_ylabel("Brain region")
    ax.set_title(title, color=title_col, pad=4)

# ── Row 1: four group-mean FCs (data + model) ─────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
_fc_panel(ax_a, fc_cc_emp, f"Mean FC — CC data  (n={n_cc})", CC_COL)
_tag(ax_a, "A")
if have_atlas:
    hn = [Patch(color=c, label=n, linewidth=0) for n, c in NET_COLORS.items()]
    ax_a.legend(handles=hn, loc="lower right", fontsize=5.2,
                frameon=True, framealpha=0.85, edgecolor="#ccc",
                ncol=1, handlelength=0.9, handleheight=0.7)

ax_b = fig.add_subplot(gs[0, 1])
_fc_panel(ax_b, fc_ad_emp, f"Mean FC — AD data  (n={n_ad})", AD_COL)
_tag(ax_b, "B")

ax_c = fig.add_subplot(gs[0, 2])
_fc_panel(ax_c, fc_cc_sim, "Mean FC — CC model", CC_COL)
_tag(ax_c, "C")

ax_d = fig.add_subplot(gs[0, 3])
_fc_panel(ax_d, fc_ad_sim, "Mean FC — AD model", AD_COL)
_tag(ax_d, "D")

# ── Row 2: E corr-matrix, F cross-subj matrix, G histogram, H sample-FC scatter
ax_e = fig.add_subplot(gs[1, 0])
cmin = float(Cmat[~np.eye(4, dtype=bool)].min())
im_e = ax_e.imshow(Cmat, cmap="viridis", vmin=cmin - 0.01, vmax=1.0)
ax_e.set_xticks(range(4)); ax_e.set_yticks(range(4))
ax_e.set_xticklabels(four_lbls, fontsize=7.5)
ax_e.set_yticklabels(four_lbls, fontsize=7.5)
for i in range(4):
    for j in range(4):
        val = Cmat[i, j]
        ax_e.text(j, i, f"{val:.3f}", ha="center", va="center",
                  fontsize=8,
                  color="black" if val > (cmin + 1.0) / 2 else "white")
cb_e = plt.colorbar(im_e, ax=ax_e, shrink=0.80, pad=0.03, fraction=0.046)
cb_e.set_label("Pearson r", fontsize=8); cb_e.ax.tick_params(labelsize=7)
ax_e.set_title("FC correlation matrix\nbetween group means", pad=4)
_tag(ax_e, "E")

ax_f = fig.add_subplot(gs[1, 1])
im_f = ax_f.imshow(cc_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
cb_f = plt.colorbar(im_f, ax=ax_f, shrink=0.80, pad=0.03, fraction=0.046)
cb_f.set_label("Pearson r", fontsize=8); cb_f.ax.tick_params(labelsize=7)
ax_f.set_xlabel("Empirical subject j"); ax_f.set_ylabel("Simulated subject i")
ax_f.set_title("Cross-subject FC match\n(diagonal = same subject)", pad=4)
_tag(ax_f, "F")

ax_g = fig.add_subplot(gs[1, 2])
ax_g.hist(offdiag_vals, bins=40, alpha=0.6, color="#9E9E9E",
          density=True, label="cross-subject")
ax_g.hist(diag_vals, bins=20, alpha=0.8, color="#1565C0",
          density=True, label="same subject")
ax_g.axvline(np.median(offdiag_vals), color="#616161", ls="--", lw=1.5)
ax_g.axvline(np.median(diag_vals),    color="#0D47A1", ls="--", lw=1.5)
ax_g.text(0.03, 0.97,
          f"same med = {np.median(diag_vals):.2f}\n"
          f"cross med = {np.median(offdiag_vals):.2f}",
          transform=ax_g.transAxes, va="top", fontsize=7.5,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax_g.set_xlabel("FC reconstruction r"); ax_g.set_ylabel("Density")
ax_g.set_title("Same- vs cross-subject\nFC identifiability", pad=4)
ax_g.legend(frameon=False, fontsize=7.5, loc="center right")
_tag(ax_g, "G"); _clean(ax_g)

ax_h = fig.add_subplot(gs[1, 3])
ve = FC_emp[rep_subj][iu]; vs = FC_sim[rep_subj][iu]
ax_h.scatter(ve, vs, s=4, alpha=0.25, color="#6A1B9A", edgecolors="none")
lims = [-0.85, 0.85]
ax_h.plot(lims, lims, "r--", lw=1, label="y = x")
r_subj = np.corrcoef(ve, vs)[0, 1]
ax_h.text(0.04, 0.96, f"r = {r_subj:.2f}", transform=ax_h.transAxes,
          va="top", fontsize=9,
          bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
ax_h.set_xlim(lims); ax_h.set_ylim(lims)
ax_h.set_xlabel("Empirical FC  (Pearson r)")
ax_h.set_ylabel("Simulated FC  (Pearson r)")
ax_h.set_title(f"Sample FC: data vs. model\n(subj #{rep_subj}, per region-pair)", pad=4)
ax_h.legend(frameon=False, fontsize=7.5, loc="lower right")
_tag(ax_h, "H"); _clean(ax_h)

# ── Row 3: I delayed-FC fit (wide),  J FC-reconstruction vs noise (wide) ───────
ax_i = fig.add_subplot(gs[2, 0:2])
for arr, col, lbl in [
        (all_tr, "#212121", "Even/odd test-retest (ceiling)"),
        (all_es, "#1565C0", "Empirical vs. simulated FC"),
        (all_e0, "#9E9E9E", "Empirical vs. zero-lag FC"),
]:
    med = np.nanmedian(arr, axis=0)
    p25 = np.nanpercentile(arr, 25, axis=0)
    p75 = np.nanpercentile(arr, 75, axis=0)
    ax_i.plot(delays_s, med, color=col, lw=2.0, label=lbl)
    ax_i.fill_between(delays_s, p25, p75, color=col, alpha=0.16)
ax_i.axhline(0, color="k", ls=":", lw=0.8, alpha=0.5)
ax_i.set_xlabel("Delay (s)"); ax_i.set_ylabel("Pearson r  (delayed FC)")
ax_i.set_title("Model fit across temporal delays  (median ± IQR, all sessions)", pad=4)
ax_i.set_ylim(-0.12, 1.05); ax_i.set_xlim(0, MAX_LAG * TR)
ax_i.set_xticks(np.arange(0, MAX_LAG * TR + 1, TR * 5))
ax_i.legend(frameon=False, fontsize=8, loc="upper right")
_tag(ax_i, "I", x=-0.06); _clean(ax_i)

ax_j = fig.add_subplot(gs[2, 2:4])
ax_j.errorbar(range(len(noise_vals)), noise_means, yerr=noise_stds,
              marker="o", ms=5, color="#2E7D32", lw=2, capsize=4,
              label="Per-patient W  (concat.)")
ax_j.axhline(r_sess.mean(), color="#E65100", lw=1.8, ls="--",
             label=f"Per-session W: {r_sess.mean():.3f}")
ax_j.fill_between(range(len(noise_vals)),
                  r_sess.mean() - r_sess.std(), r_sess.mean() + r_sess.std(),
                  alpha=0.15, color="#E65100")
best_i = int(np.argmax(noise_means))
ax_j.scatter([best_i], [noise_means[best_i]], s=80, color="#FF6F00",
             zorder=5, edgecolors="k", lw=0.8,
             label=f"Optimal σ={noise_vals[best_i]:.4f}")
ax_j.set_xticks(range(len(noise_vals)))
ax_j.set_xticklabels([f"{v:.3f}" for v in noise_vals],
                      rotation=40, ha="right", fontsize=7)
ax_j.set_xlabel("Noise σ"); ax_j.set_ylabel("FC reconstruction r")
ax_j.set_title("FC reconstruction quality vs. regularisation noise", pad=4)
ax_j.legend(frameon=False, fontsize=7.5, loc="lower right")
_tag(ax_j, "J", x=-0.06); _clean(ax_j)

for ext in ("png", "pdf"):
    out = f"figure2_model.{ext}"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved {out}")
plt.close()
