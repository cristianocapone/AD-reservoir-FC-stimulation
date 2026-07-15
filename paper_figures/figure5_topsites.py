"""
figure5_topsites.py
===================
Figure 5 — per-session most-affected read-out sites, and their brain location.

For each AD patient, the per-site correction magnitude is the column norm of
dW = W_CC_mean - W_AD.  We show:
  A  per-session top-5 site membership matrix (AD patients x 121 sites)
  B  site-selection frequency (how many AD patients have each site in top-5/top-1)
  C  glass-brain markers at the selected parcels, sized/coloured by frequency

Reuses the same reservoir / W-fit (seeds, sigma) as figure4_sites so the sites
match.  MNI parcel coordinates are loaded from ../pert_sites_data.npz.

Saves: figure5_topsites.{png,pdf}  and  figure5_brain.{png,pdf}
"""
import os, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
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

# ── parameters (match pert_sites_stimulation.py) ──────────────────────────────
RNG_SEED   = 42
N_CC_SAMP  = 40
N_SITES    = 121
N_PC_MODEL = 50
TIMES_SKIP = 10
ff         = 0.1
N_HIDDEN   = 2000
SIGMA      = 0.05
SR         = 0.95
TS_ROOT    = "../timeseries"
TOPK       = 5

# ── parcel labels ─────────────────────────────────────────────────────────────
def load_labels(path="../timeseries/parcel_labels.txt"):
    lab = {}
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                lab[int(parts[0]) - 1] = parts[1].strip()
    return lab

def short_label(name):
    if name is None:
        return "?"
    # subcortical Harvard-Oxford names: "Left Putamen" -> "L Putamen"
    name = name.replace("Left ", "L ").replace("Right ", "R ")
    # Schaefer: "7Networks_LH_Vis_1" -> "LH Vis1"
    if name.startswith("7Networks_"):
        p = name.replace("7Networks_", "").split("_")
        return (p[0] + " " + "".join(p[1:])) if len(p) > 1 else p[0]
    return name

labels = load_labels()

# ── data loading (identical sampling to pert_sites) ───────────────────────────
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

pid_raw    = np.array(pid_raw)
labels_raw = np.array(labels_raw)
unique_pids    = np.unique(pid_raw)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_pids = [unique_pids[i] for i in np.where(patient_labels == 0)[0]]
ad_pids = [unique_pids[i] for i in np.where(patient_labels == 1)[0]]
n_ad = len(ad_pids); n_cc = len(cc_pids)
print(f"  {len(unique_pids)} patients ({n_cc} CC, {n_ad} AD)")

# ── PCA + reservoir ───────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

print("Reservoir TF pass ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

first_idx = {pid: patient_sids[pid][0] for pid in unique_pids}
patX, patY = {}, {}
for pid in tqdm(unique_pids, desc="  TF"):
    s = signals[first_idx[pid]]; T_s = s.shape[1]
    tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    X_raw = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    Xf = np.array(X_raw)[TIMES_SKIP:]
    patX[pid] = Xf
    patY[pid] = tgt[:, TIMES_SKIP:TIMES_SKIP + len(Xf)].T

print("W fitting ...")
rng_w = np.random.default_rng(RNG_SEED + 1)
pat_W = {}
for pid in tqdm(unique_pids, desc="  W-fit", leave=False):
    Xc = patX[pid]; Yc = patY[pid]
    pat_W[pid] = np.linalg.pinv(Xc + rng_w.normal(0, SIGMA, Xc.shape)) @ Yc
W_cc_mean = np.mean([pat_W[pid] for pid in cc_pids], axis=0)

# ── per-AD-patient site norms  ||dW[:, k]||  ──────────────────────────────────
norm_mat = np.zeros((n_ad, N_SITES))           # (n_ad, 121)
for pi, pid in enumerate(ad_pids):
    dW = W_cc_mean - pat_W[pid]
    norm_mat[pi] = np.linalg.norm(dW, axis=0)

# top-5 / top-1 membership per patient
top5_mask = np.zeros((n_ad, N_SITES), dtype=bool)
counts5   = np.zeros(N_SITES, dtype=int)
counts1   = np.zeros(N_SITES, dtype=int)
for pi in range(n_ad):
    order = np.argsort(norm_mat[pi])[::-1]
    top5_mask[pi, order[:TOPK]] = True
    counts5[order[:TOPK]] += 1
    counts1[order[0]]     += 1

order_freq = np.argsort(counts5)[::-1]          # sites by selection frequency
sel_sites  = order_freq[counts5[order_freq] > 0]
print(f"\nMost-selected sites (top-5 across {n_ad} AD patients):")
for s in order_freq[:8]:
    print(f"  site {s:3d}  {short_label(labels.get(s)):20s}  "
          f"top5={counts5[s]:2d}  top1={counts1[s]:2d}")

# ── MNI parcel coordinates (saved by pert_sites_stimulation.py) ───────────────
parcel_coords = np.load("../pert_sites_data.npz",
                        allow_pickle=True)["parcel_coords"]   # (121,3)

# ══════════════════════════════════════════════════════════════════════════════
# BRAIN RENDER (nilearn glass brain) — standalone, then embedded in panel C
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering glass brain ...")
from nilearn import plotting
sel = counts5 > 0
disp = plotting.plot_markers(
    node_values   = counts5[sel].astype(float),
    node_coords   = parcel_coords[sel],
    node_size     = 25 + 10 * counts5[sel],
    node_cmap     = "YlOrRd",
    node_vmin     = 0, node_vmax = float(counts5.max()),
    display_mode  = "lzry", alpha = 0.85, colorbar = True,
    title         = f"Top-5 ||dW|| selection frequency (n={n_ad} AD)")
disp.savefig("figure5_brain.png", dpi=300)
disp.savefig("figure5_brain.pdf")
disp.close()
print("Saved figure5_brain.png / .pdf")

# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE FIGURE 5
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering composite figure ...")
fig = plt.figure(figsize=(15, 8.5), facecolor="white")
gs  = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.25, 1.0],
                        height_ratios=[1.0, 1.0], hspace=0.42, wspace=0.28,
                        left=0.07, right=0.97, top=0.90, bottom=0.10)

def _tag(ax, t, x=-0.10, y=1.04):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=13,
            fontweight="bold", va="bottom", ha="left")

# ── A: per-session top-5 membership matrix (sites sorted by frequency) ────────
ax_a = fig.add_subplot(gs[:, 0])
M = top5_mask[:, sel_sites].astype(float)        # (n_ad, n_selected)
ax_a.imshow(M, aspect="auto", cmap="Purples", interpolation="nearest",
            vmin=0, vmax=1)
ax_a.set_xlabel("Brain site (sorted by selection frequency)")
ax_a.set_ylabel("AD patient (session)")
ax_a.set_title(f"Per-session top-{TOPK} most-affected sites")
# label the leading site columns with anatomy
n_lab = min(12, len(sel_sites))
ax_a.set_xticks(range(n_lab))
ax_a.set_xticklabels([f"{short_label(labels.get(s))}"
                      for s in sel_sites[:n_lab]],
                     rotation=55, ha="right", fontsize=6.5)
ax_a.set_yticks([0, n_ad // 2, n_ad - 1])
_tag(ax_a, "A")

# ── B: selection-frequency bars (top-5 and top-1) ─────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
nb = min(15, len(sel_sites))
xs = np.arange(nb)
ax_b.bar(xs - 0.2, counts5[sel_sites[:nb]], 0.4, color="#E65100",
         alpha=0.85, label="in top-5")
ax_b.bar(xs + 0.2, counts1[sel_sites[:nb]], 0.4, color="#6A1B9A",
         alpha=0.85, label="top-1")
ax_b.set_xticks(xs)
ax_b.set_xticklabels([short_label(labels.get(s)) for s in sel_sites[:nb]],
                     rotation=55, ha="right", fontsize=6.5)
ax_b.set_ylabel(f"# AD patients (of {n_ad})")
ax_b.set_title("Site-selection frequency")
ax_b.legend(frameon=False, fontsize=8)
_tag(ax_b, "B", x=-0.12)

# ── C: glass brain (embedded) ─────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])
ax_c.imshow(imread("figure5_brain.png"))
ax_c.axis("off")
ax_c.set_title("Anatomical location of selected sites", pad=2)
_tag(ax_c, "C", x=-0.04, y=1.02)

fig.suptitle("Most-affected read-out sites per AD session and their brain "
             "location\n"
             r"(per-site correction magnitude $\|\Delta W\|$, "
             f"top-{TOPK} per patient; N={n_ad} AD)",
             fontsize=12, fontweight="bold", y=0.99)

for ext in ("png", "pdf"):
    fig.savefig(f"figure5_topsites.{ext}", dpi=300, bbox_inches="tight",
                facecolor="white")
    print(f"Saved figure5_topsites.{ext}")
plt.close(fig)
print("\nDone.")
