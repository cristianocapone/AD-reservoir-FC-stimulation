"""
visualize_esempio.py
====================
Visualise two fMRIPrep-preprocessed resting-state examples from 'esempio/'.

For each of the two subjects (one session each) the script produces a
3-panel figure:

  Panel A  – Functional connectivity matrix (Schaefer-100, network-sorted)
  Panel B  – First 5 PC timeseries (raw + smoothed) + explained-variance bar
  Panel C  – Power spectrum of the 5 PCs (semi-log)

Atlas : Schaefer 2018, 100 parcels, 7 networks, MNI 2009c space
Denoising: 24-motion + WM + CSF (no global-signal regression) + bandpass 0.01-0.1 Hz

Output : esempio_visualization.png  (next to this script)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

import nibabel as nib
from nilearn import datasets, image
from nilearn.maskers import NiftiLabelsMasker
from sklearn.decomposition import PCA
from scipy.signal import welch, savgol_filter

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection")
ESEMPIO = BASE / "esempio"
OUTDIR  = BASE / "esempio_out"
OUTDIR.mkdir(exist_ok=True)

# ── Atlas: Schaefer-100, 7 networks, MNI 2009c ───────────────────────────────
print("Fetching Schaefer-100 atlas …")
schaefer = datasets.fetch_atlas_schaefer_2018(
    n_rois=100, yeo_networks=7, resolution_mm=2
)
atlas_img  = schaefer.maps       # 4-D label image
atlas_lbls = list(schaefer.labels)
# Decode bytes if needed
atlas_lbls = [l.decode() if isinstance(l, bytes) else l for l in atlas_lbls]

# ── Network colour map ────────────────────────────────────────────────────────
NET_ORDER  = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_COLORS = {
    "Vis":        "#1f77b4",
    "SomMot":     "#ff7f0e",
    "DorsAttn":   "#2ca02c",
    "SalVentAttn":"#d62728",
    "Limbic":     "#9467bd",
    "Cont":       "#8c564b",
    "Default":    "#e377c2",
}

def get_net(label):
    for n in NET_ORDER:
        if n in label:
            return n
    return NET_ORDER[-1]

net_assign = [get_net(l) for l in atlas_lbls]   # list[str], length 100
sort_key   = lambda i: (NET_ORDER.index(net_assign[i]), i)
sorted_idx = sorted(range(100), key=sort_key)

# Network boundaries after sorting
sorted_nets = [net_assign[i] for i in sorted_idx]
net_bounds  = []
for i in range(1, 100):
    if sorted_nets[i] != sorted_nets[i - 1]:
        net_bounds.append(i - 0.5)

# ── Select one session per subject ───────────────────────────────────────────
def first_session(sub_dir: Path):
    """Return (bold_nii, confounds_tsv, tr, label) for the earliest session."""
    sessions = sorted(sub_dir.glob("ses-*"))
    for ses in sessions:
        bold = list((ses / "func").glob(
            "*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))
        conf = list((ses / "func").glob("*desc-confounds_timeseries.tsv"))
        jsn  = list((ses / "func").glob(
            "*space-MNI152NLin2009cAsym_desc-preproc_bold.json"))
        if bold and conf:
            tr = 3.0
            if jsn:
                meta = json.loads(jsn[0].read_text())
                tr   = float(meta.get("RepetitionTime", 3.0))
            label = f"{sub_dir.name}\n{ses.name}"
            return bold[0], conf[0], tr, label
    raise FileNotFoundError(f"No valid session found in {sub_dir}")

subjects = sorted(ESEMPIO.glob("sub-*"))
print(f"Found {len(subjects)} subject(s): {[s.name for s in subjects]}")

examples = []
for sub in subjects[:2]:            # take at most 2
    bold, conf, tr, label = first_session(sub)
    print(f"  {label.replace(chr(10), '  ')}  TR={tr}s")
    examples.append(dict(bold=bold, conf=conf, tr=tr, label=label))

# ── Confound selection ────────────────────────────────────────────────────────
MOTION_COLS = [
    "trans_x", "trans_y", "trans_z",
    "rot_x",   "rot_y",   "rot_z",
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1",
    "trans_x_power2", "trans_y_power2", "trans_z_power2",
    "rot_x_power2",   "rot_y_power2",   "rot_z_power2",
    "trans_x_derivative1_power2", "trans_y_derivative1_power2",
    "trans_z_derivative1_power2",
    "rot_x_derivative1_power2",   "rot_y_derivative1_power2",
    "rot_z_derivative1_power2",
]
PHYSIO_COLS = ["csf", "white_matter",
               "csf_derivative1", "white_matter_derivative1"]

def load_confounds(tsv_path):
    df   = pd.read_csv(tsv_path, sep="\t")
    keep = [c for c in MOTION_COLS + PHYSIO_COLS if c in df.columns]
    sub  = df[keep].copy()
    sub.fillna(0, inplace=True)
    return sub.values.astype(float)

# ── Extract parcel timeseries ─────────────────────────────────────────────────
def extract_ts(bold_path, confounds_arr, tr):
    masker = NiftiLabelsMasker(
        labels_img   = atlas_img,
        standardize  = "zscore_sample",
        detrend      = True,
        low_pass     = 0.10,
        high_pass    = 0.01,
        t_r          = tr,
        resampling_target = "data",
        memory       = str(OUTDIR / "_nilearn_cache"),
        memory_level = 1,
        verbose      = 0,
    )
    ts = masker.fit_transform(str(bold_path), confounds=confounds_arr)
    return ts   # shape (T, 100)

# ── Functional connectivity ───────────────────────────────────────────────────
def fc_from_ts(ts):
    fc = np.corrcoef(ts.T)          # 100×100
    np.fill_diagonal(fc, 0.0)
    return fc

# ── Main extraction ───────────────────────────────────────────────────────────
for ex in examples:
    print(f"\nProcessing {ex['label'].replace(chr(10),' ')} …")
    conf_arr = load_confounds(ex["conf"])
    ts       = extract_ts(ex["bold"], conf_arr, ex["tr"])
    ex["ts"] = ts
    ex["fc"] = fc_from_ts(ts)
    print(f"  timeseries shape: {ts.shape}   "
          f"FC range [{ex['fc'].min():.3f}, {ex['fc'].max():.3f}]")

# ── Figure ────────────────────────────────────────────────────────────────────
N_SUB    = len(examples)
SMOOTH_W = max(11, int(11 / examples[0]["tr"]) | 1)   # ~33 s window, odd
N_PC     = 5

fig = plt.figure(figsize=(22, 10 * N_SUB), constrained_layout=False)
fig.patch.set_facecolor("#f7f7f7")

outer = gridspec.GridSpec(N_SUB, 1, figure=fig, hspace=0.45)

for si, ex in enumerate(examples):
    ts  = ex["ts"]          # (T, 100)
    fc  = ex["fc"]          # (100, 100)
    tr  = ex["tr"]
    T   = ts.shape[0]
    t   = np.arange(T) * tr

    # PCA on timeseries (fit on parcels, project to time)
    pca    = PCA(n_components=N_PC)
    scores = pca.fit_transform(ts)          # (T, N_PC)
    evr    = pca.explained_variance_ratio_  # (N_PC,)

    # ── inner 3-column layout ────────────────────────────────────────────────
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[si],
        width_ratios=[1.0, 1.4, 0.9],
        wspace=0.35,
    )

    # ── title band ───────────────────────────────────────────────────────────
    fig.text(
        0.5, (outer[si].get_position(fig).y1 + 0.01),
        f"Subject: {ex['label'].replace(chr(10), '  |  ')}   "
        f"T={T} vols  TR={tr:.3f}s  "
        f"(Schaefer-100, bandpass 0.01-0.10 Hz)",
        ha="center", va="bottom", fontsize=12, fontweight="bold",
        transform=fig.transFigure,
    )

    # ── Panel A: FC matrix ────────────────────────────────────────────────────
    ax_fc = fig.add_subplot(inner[0, 0])
    fc_s  = fc[np.ix_(sorted_idx, sorted_idx)]
    im    = ax_fc.imshow(fc_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8,
                         aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax_fc, shrink=0.75, label="Pearson r", pad=0.02)
    for b in net_bounds:
        ax_fc.axhline(b, color="white", lw=0.8)
        ax_fc.axvline(b, color="white", lw=0.8)

    # Network labels on diagonal
    prev = 0
    for b_end in net_bounds + [100]:
        mid = (prev + int(b_end + 0.5)) / 2
        net = sorted_nets[int(mid)]
        ax_fc.text(mid, mid, net[:6], ha="center", va="center",
                   fontsize=6, fontweight="bold", color="white",
                   bbox=dict(fc=NET_COLORS.get(net, "gray"),
                             ec="none", alpha=0.75, pad=1.5))
        prev = int(b_end + 0.5)

    ax_fc.set_xticks([]); ax_fc.set_yticks([])
    ax_fc.set_title("Functional Connectivity\n(Schaefer-100, network-sorted)",
                    fontsize=10)

    # FC stats annotation
    idx_u  = np.triu_indices(100, k=1)
    fc_ut  = fc[idx_u]
    ax_fc.set_xlabel(
        f"mean r={fc_ut.mean():.3f}  std={fc_ut.std():.3f}",
        fontsize=8, labelpad=4
    )

    # ── Panel B: 5 PCA timeseries + explained-variance bar ───────────────────
    inner_b = gridspec.GridSpecFromSubplotSpec(
        N_PC + 1, 1, subplot_spec=inner[0, 1], hspace=0.15
    )

    # variance bar at top
    ax_bar = fig.add_subplot(inner_b[0, 0])
    ax_bar.bar(range(1, N_PC + 1), evr * 100,
               color=[f"C{i}" for i in range(N_PC)],
               edgecolor="white", linewidth=0.5)
    ax_bar.set_ylabel("% var", fontsize=7)
    ax_bar.set_xticks(range(1, N_PC + 1))
    ax_bar.tick_params(labelsize=6)
    ax_bar.set_title(f"First {N_PC} PCs ({evr.sum()*100:.1f}% total var)",
                     fontsize=10)

    for pc in range(N_PC):
        ax_pc = fig.add_subplot(inner_b[pc + 1, 0])
        raw    = scores[:, pc]
        smooth = savgol_filter(raw, window_length=SMOOTH_W, polyorder=3)
        ax_pc.plot(t, raw,    lw=0.5, color="silver", alpha=0.85)
        ax_pc.plot(t, smooth, lw=1.4, color=f"C{pc}")
        ax_pc.axhline(0, color="k", lw=0.3)
        ax_pc.set_ylabel(f"PC{pc+1}\n({evr[pc]*100:.1f}%)",
                         fontsize=7, rotation=0, labelpad=42)
        ax_pc.set_xlim(0, t[-1])
        ax_pc.tick_params(labelsize=5)
        ax_pc.set_yticks([])
        if pc < N_PC - 1:
            ax_pc.set_xticks([])
        else:
            ax_pc.set_xlabel("Time (s)", fontsize=8)

    # ── Panel C: Power spectrum of 5 PCs ─────────────────────────────────────
    ax_ps = fig.add_subplot(inner[0, 2])
    nyq   = 1 / (2 * tr)
    nperseg = max(16, T // 4)
    for pc in range(N_PC):
        f_hz, psd = welch(scores[:, pc], fs=1/tr, nperseg=nperseg)
        ax_ps.semilogy(f_hz, psd, lw=1.2, color=f"C{pc}",
                       label=f"PC{pc+1} ({evr[pc]*100:.1f}%)")
    ax_ps.axvspan(0.01, 0.10, alpha=0.12, color="green",
                  label="0.01–0.10 Hz band")
    ax_ps.axvline(nyq, color="red", lw=1.0, ls=":", label=f"Nyquist {nyq:.3f} Hz")
    ax_ps.set_xlabel("Frequency (Hz)", fontsize=9)
    ax_ps.set_ylabel("PSD (a.u.)", fontsize=9)
    ax_ps.set_xlim(0, nyq + 0.01)
    ax_ps.tick_params(labelsize=7)
    ax_ps.legend(fontsize=7, loc="upper right", framealpha=0.8)
    ax_ps.set_title("Power Spectrum — 5 PCs", fontsize=10)
    ax_ps.grid(True, which="both", alpha=0.2)

# ── Global network legend ─────────────────────────────────────────────────────
handles = [Patch(color=c, label=n) for n, c in NET_COLORS.items()]
fig.legend(
    handles=handles, loc="lower center", ncol=7, fontsize=8,
    title="Schaefer-100 Networks", title_fontsize=9,
    bbox_to_anchor=(0.5, -0.01), framealpha=0.9
)

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = BASE / "esempio_visualization.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print(f"\n✓  Saved: {out_path}")
print(f"   Figure size ≈ {22}×{10 * N_SUB} inches at 150 dpi")
