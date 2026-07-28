#!/usr/bin/env python3
"""
Sanity check + visual report for one fmriprep subject.

Outputs (saved next to this script):
  qc_motion.png          – FD trace + distribution
  qc_tsnr.png            – tSNR per Schaefer-100 parcel
  qc_temporal_traces.png – BOLD traces for 12 representative ROIs
  qc_fc_matrix.png       – 100x100 FC matrix (Pearson r)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from nilearn import datasets, maskers, signal, image

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).parent / "fmriprep_output"
SUBJECT    = "sub-006"
SESSION    = "ses-20120514"
RUN        = "run-01"
TR         = 3.0          # seconds
HP_CUTOFF  = 0.01         # Hz  (high-pass)
LP_CUTOFF  = 0.1          # Hz  (low-pass)
OUT_DIR    = Path(__file__).parent
# ─────────────────────────────────────────────────────────────────────────────

func_dir = ROOT / SUBJECT / SESSION / "func"
tag = f"{SUBJECT}_{SESSION}_task-rest_{RUN}"

bold_path      = func_dir / f"{tag}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
mask_path      = func_dir / f"{tag}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
confounds_path = func_dir / f"{tag}_desc-confounds_timeseries.tsv"

print(f"BOLD  : {bold_path.name}  exists={bold_path.exists()}")
print(f"Mask  : {mask_path.name}  exists={mask_path.exists()}")
print(f"Conf  : {confounds_path.name}  exists={confounds_path.exists()}")

# ── 1. LOAD CONFOUNDS ─────────────────────────────────────────────────────────
conf_df = pd.read_csv(confounds_path, sep="\t")

motion_cols = [c for c in conf_df.columns if c.startswith(("trans_", "rot_"))]
nuisance_cols = motion_cols + ["csf", "white_matter",
                               "csf_derivative1", "white_matter_derivative1"]
nuisance_cols = [c for c in nuisance_cols if c in conf_df.columns]

# Fill first-row NaNs (derivatives) with 0
conf_clean = conf_df[nuisance_cols].fillna(0).values

fd = conf_df["framewise_displacement"].fillna(0).values
n_vols = len(fd)
t_axis = np.arange(n_vols) * TR

# ── 2. FETCH SCHAEFER-100 ATLAS ───────────────────────────────────────────────
print("Fetching Schaefer-100 atlas …")
schaefer = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
atlas_img   = schaefer.maps
roi_labels  = [l.decode() if isinstance(l, bytes) else l for l in schaefer.labels]

# ── 3. EXTRACT PARCEL TIME SERIES ─────────────────────────────────────────────
print("Extracting parcel time series …")
masker = maskers.NiftiLabelsMasker(
    labels_img=atlas_img,
    mask_img=mask_path,
    standardize=True,
    detrend=True,
    t_r=TR,
    high_pass=HP_CUTOFF,
    low_pass=LP_CUTOFF,
    resampling_target="labels",
    memory_level=0,
)
ts = masker.fit_transform(str(bold_path), confounds=conf_clean)
# ts shape: (n_vols, 100)
print(f"  Time series shape: {ts.shape}")

# ── 4. COMPUTE FC MATRIX ──────────────────────────────────────────────────────
fc = np.corrcoef(ts.T)          # 100 × 100

# ── 5. tSNR SANITY CHECK (voxel-level, whole brain) ──────────────────────────
print("Computing tSNR …")
bold_img = nib.load(str(bold_path))
bold_data = bold_img.get_fdata(dtype=np.float32)
mask_data = nib.load(str(mask_path)).get_fdata() > 0.5
ts_mean = bold_data[mask_data].mean(axis=1)
ts_std  = bold_data[mask_data].std(axis=1)
tsnr_vals = np.where(ts_std > 0, ts_mean / ts_std, 0)
print(f"  Median tSNR (brain): {np.median(tsnr_vals):.1f}")

# Per-parcel tSNR using masker on non-cleaned signal
masker_raw = maskers.NiftiLabelsMasker(
    labels_img=atlas_img,
    mask_img=mask_path,
    standardize=False,
    detrend=False,
    resampling_target="labels",
    memory_level=0,
)
ts_raw = masker_raw.fit_transform(str(bold_path))
parcel_mean = ts_raw.mean(axis=0)
parcel_std  = ts_raw.std(axis=0)
parcel_tsnr = np.where(parcel_std > 0, parcel_mean / parcel_std, 0)

# ── 6. PLOT: MOTION QC ────────────────────────────────────────────────────────
print("Plotting motion QC …")
fig, axes = plt.subplots(2, 1, figsize=(14, 6), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle(f"Motion QC — {SUBJECT} {SESSION} {RUN}", fontsize=13, fontweight="bold")

ax = axes[0]
ax.plot(t_axis, fd, color="#d62728", lw=0.9, label="FD (mm)")
ax.axhline(0.5, color="k", ls="--", lw=0.8, label="FD=0.5 mm threshold")
ax.fill_between(t_axis, 0, fd, where=fd > 0.5, color="#d62728", alpha=0.25)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Framewise displacement (mm)")
ax.legend(fontsize=9)
n_outliers = (fd > 0.5).sum()
ax.set_title(f"FD trace  |  mean={fd.mean():.3f} mm  |  outliers>{0.5} mm: {n_outliers}/{n_vols} ({100*n_outliers/n_vols:.1f}%)",
             fontsize=10)

ax2 = axes[1]
ax2.hist(fd, bins=40, color="#1f77b4", edgecolor="white", lw=0.4)
ax2.axvline(0.5, color="k", ls="--", lw=0.8)
ax2.set_xlabel("FD (mm)")
ax2.set_ylabel("Count")

plt.tight_layout()
out_path = OUT_DIR / "qc_motion.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved {out_path}")

# ── 7. PLOT: tSNR PER PARCEL ──────────────────────────────────────────────────
print("Plotting tSNR per parcel …")
fig, ax = plt.subplots(figsize=(14, 5))
fig.suptitle(f"tSNR per Schaefer-100 parcel — {SUBJECT} {SESSION}", fontsize=13, fontweight="bold")

sorted_idx = np.argsort(parcel_tsnr)[::-1]
colors = ["#2ca02c" if parcel_tsnr[i] >= 50 else "#ff7f0e" if parcel_tsnr[i] >= 30 else "#d62728"
          for i in sorted_idx]
ax.bar(range(100), parcel_tsnr[sorted_idx], color=colors, edgecolor="none")
ax.axhline(50, color="k", ls="--", lw=0.8, label="tSNR=50 (good)")
ax.axhline(30, color="gray", ls=":", lw=0.8, label="tSNR=30 (acceptable)")
ax.set_xlabel("Parcel rank (sorted by tSNR)")
ax.set_ylabel("tSNR")
ax.legend(fontsize=9)
ax.set_title(f"Median={np.median(parcel_tsnr):.1f}  |  Min={parcel_tsnr.min():.1f}  |  Max={parcel_tsnr.max():.1f}",
             fontsize=10)

plt.tight_layout()
out_path = OUT_DIR / "qc_tsnr.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved {out_path}")

# ── 8. PLOT: TEMPORAL TRACES (12 ROIs) ────────────────────────────────────────
print("Plotting temporal traces …")

# Pick 12 representative ROIs spanning networks
# Schaefer labels follow format: 7Networks_LH_Vis_1, 7Networks_LH_SomMot_1, etc.
networks = {"Vis": "#1f77b4", "SomMot": "#ff7f0e", "DorsAttn": "#2ca02c",
            "SalVentAttn": "#d62728", "Limbic": "#9467bd", "Cont": "#8c564b",
            "Default": "#e377c2"}

selected = []
for net in networks:
    idxs = [i for i, lbl in enumerate(roi_labels) if net in lbl]
    if idxs:
        selected.append((idxs[0], net))
    if len(selected) >= 12:
        break
# pad if needed
if len(selected) < 12:
    remaining = [i for i in range(100) if i not in [s[0] for s in selected]]
    for i in remaining[:12 - len(selected)]:
        selected.append((i, ""))

fig = plt.figure(figsize=(16, 14))
fig.suptitle(f"BOLD temporal traces (Schaefer-100, cleaned) — {SUBJECT} {SESSION}",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.3)

for plot_i, (roi_i, net) in enumerate(selected[:12]):
    row, col = divmod(plot_i, 2)
    ax = fig.add_subplot(gs[row, col])
    color = networks.get(net, "#7f7f7f")
    ax.plot(t_axis, ts[:, roi_i], color=color, lw=0.7)
    ax.axhline(0, color="k", lw=0.4, ls="--")
    short_lbl = roi_labels[roi_i].replace("7Networks_", "").replace("LH_", "L-").replace("RH_", "R-")
    ax.set_title(f"ROI {roi_i+1}: {short_lbl}", fontsize=8, pad=2)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("BOLD (z)", fontsize=7)
    ax.tick_params(labelsize=6)

out_path = OUT_DIR / "qc_temporal_traces.png"
plt.savefig(out_path, dpi=150)
plt.close()
print(f"  Saved {out_path}")

# ── 9. PLOT: FC MATRIX ────────────────────────────────────────────────────────
print("Plotting FC matrix …")

# Sort parcels by network (Schaefer 7-network order)
net_order = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
net_colors = {"Vis": "#1f77b4", "SomMot": "#ff7f0e", "DorsAttn": "#2ca02c",
              "SalVentAttn": "#d62728", "Limbic": "#9467bd",
              "Cont": "#8c564b", "Default": "#e377c2"}

def get_network(lbl):
    for n in net_order:
        if n in lbl:
            return n
    return "Other"

net_assignments = [get_network(l) for l in roi_labels]
sort_key = lambda i: (net_order.index(net_assignments[i]) if net_assignments[i] in net_order else 99, i)
sorted_rois = sorted(range(100), key=sort_key)
fc_sorted = fc[np.ix_(sorted_rois, sorted_rois)]
sorted_nets = [net_assignments[i] for i in sorted_rois]

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle(f"Functional Connectivity (Pearson r) — Schaefer-100\n{SUBJECT} {SESSION} {RUN}",
             fontsize=13, fontweight="bold")

im = ax.imshow(fc_sorted, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto",
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

# Network boundary lines + colour bars
boundaries = []
prev_net = sorted_nets[0]
for i, n in enumerate(sorted_nets):
    if n != prev_net:
        boundaries.append(i - 0.5)
        prev_net = n
for b in boundaries:
    ax.axhline(b, color="white", lw=0.8)
    ax.axvline(b, color="white", lw=0.8)

# Network labels on diagonal
prev_b = 0
for b in boundaries + [100]:
    mid = (prev_b + b) / 2
    net = sorted_nets[int(mid)]
    ax.text(mid, mid, net.replace("SalVentAttn", "SVAttn"),
            ha="center", va="center", fontsize=6.5, fontweight="bold",
            color="white",
            bbox=dict(fc=net_colors.get(net, "gray"), ec="none", alpha=0.7, pad=1))
    prev_b = b

ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("ROI (sorted by network)", fontsize=10)
ax.set_ylabel("ROI (sorted by network)", fontsize=10)

# Colour strip on sides
strip_w = 3
left_strip = np.array([net_order.index(n) if n in net_order else 7 for n in sorted_nets]).reshape(-1, 1)
ax_strip_left  = fig.add_axes([ax.get_position().x0 - 0.03, ax.get_position().y0,
                                0.012, ax.get_position().height])
ax_strip_top   = fig.add_axes([ax.get_position().x0, ax.get_position().y1,
                                ax.get_position().width, 0.012])
cmap_net = matplotlib.colors.ListedColormap(list(net_colors.values()))
ax_strip_left.imshow(left_strip, aspect="auto", cmap=cmap_net,
                     vmin=0, vmax=len(net_order), origin="upper")
ax_strip_top.imshow(left_strip.T, aspect="auto", cmap=cmap_net,
                    vmin=0, vmax=len(net_order))
ax_strip_left.axis("off")
ax_strip_top.axis("off")

out_path = OUT_DIR / "qc_fc_matrix.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {out_path}")

# ── 10. PRINT SUMMARY STATS ───────────────────────────────────────────────────
print("\n" + "="*60)
print(f"SANITY CHECK SUMMARY  —  {SUBJECT} {SESSION} {RUN}")
print("="*60)
print(f"  Volumes             : {n_vols}  (TR={TR}s, {n_vols*TR/60:.1f} min)")
print(f"  Mean FD             : {fd.mean():.3f} mm")
print(f"  Outlier volumes     : {n_outliers}/{n_vols} ({100*n_outliers/n_vols:.1f}%)")
print(f"  Median parcel tSNR  : {np.median(parcel_tsnr):.1f}")
print(f"  FC mean (off-diag)  : {fc[np.triu_indices(100,1)].mean():.3f}")
print(f"  FC std  (off-diag)  : {fc[np.triu_indices(100,1)].std():.3f}")
within_net_fc, between_net_fc = [], []
for i in range(100):
    for j in range(i+1, 100):
        if net_assignments[i] == net_assignments[j]:
            within_net_fc.append(fc[i, j])
        else:
            between_net_fc.append(fc[i, j])
print(f"  Within-network FC   : {np.mean(within_net_fc):.3f}")
print(f"  Between-network FC  : {np.mean(between_net_fc):.3f}")
print("="*60)
print("Done. Figures saved to:", OUT_DIR)
