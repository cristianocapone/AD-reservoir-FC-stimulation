"""
compare_gsr.py  (3-way)
=======================
Three-way comparison for two example sessions:

  A) no-GSR pre   – timeseries/AD/*.npy          (Schaefer-100 + HO-21, no GSR)
  B) no-GSR live  – extracted here from NIfTI     (Schaefer-100 only, 24-motion+WM/CSF)
  C) GSR pre      – timeseries_GSR/AD/*.npy       (same as A + global_signal regression)

All use Schaefer-100 parcels (first 100 rows of the .npy files).
TR ≈ 3 s, bandpass 0.01-0.10 Hz.

Figure layout
─────────────
Top (per subject row):
  [FC A]  [FC B]  [FC C]  [ΔFC = C−A]  [explained variance bar × 3  +  PSD 3-way overlay]

Bottom (per subject, 3-strip block):
  PC1-5 timeseries for A, B, C side by side

Output: compare_gsr.png
"""

import sys, json, warnings
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.lines as mlines

from nilearn import datasets
from nilearn.maskers import NiftiLabelsMasker
from sklearn.decomposition import PCA
from scipy.signal import welch, savgol_filter

# ── Paths ────────────────────────────────────────────────────────────────────
BASE     = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection")
ESEMPIO  = BASE / "esempio"
TS_DIR   = BASE / "timeseries"    / "AD"
GSR_DIR  = BASE / "timeseries_GSR" / "AD"
CACHE    = BASE / "esempio_out" / "_nilearn_cache"

# ── Atlas ────────────────────────────────────────────────────────────────────
print("Loading Schaefer-100 atlas …")
schaefer   = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
atlas_img  = schaefer.maps
atlas_lbls = [l.decode() if isinstance(l, bytes) else l for l in schaefer.labels]

NET_ORDER  = ["Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"]
NET_COLORS = {
    "Vis": "#1f77b4", "SomMot": "#ff7f0e", "DorsAttn": "#2ca02c",
    "SalVentAttn": "#d62728", "Limbic": "#9467bd",
    "Cont": "#8c564b", "Default": "#e377c2",
}
COND_COLORS = {"A": "#2ca02c", "B": "#ff7f0e", "C": "#d62728"}
COND_LS     = {"A": "-",       "B": "--",      "C": ":"}
COND_LABELS = {
    "A": "no-GSR pre\n(timeseries/)",
    "B": "no-GSR live\n(NIfTI → masker)",
    "C": "GSR pre\n(timeseries_GSR/)",
}

def get_net(label):
    for n in NET_ORDER:
        if n in label:
            return n
    return NET_ORDER[-1]

net_assign  = [get_net(l) for l in atlas_lbls]
sort_key    = lambda i: (NET_ORDER.index(net_assign[i]), i)
sorted_idx  = sorted(range(100), key=sort_key)
sorted_nets = [net_assign[i] for i in sorted_idx]
net_bounds  = [i - 0.5 for i in range(1, 100) if sorted_nets[i] != sorted_nets[i-1]]

# ── Confound helpers (live extraction) ───────────────────────────────────────
MOTION_COLS = [
    "trans_x","trans_y","trans_z","rot_x","rot_y","rot_z",
    "trans_x_derivative1","trans_y_derivative1","trans_z_derivative1",
    "rot_x_derivative1","rot_y_derivative1","rot_z_derivative1",
    "trans_x_power2","trans_y_power2","trans_z_power2",
    "rot_x_power2","rot_y_power2","rot_z_power2",
    "trans_x_derivative1_power2","trans_y_derivative1_power2","trans_z_derivative1_power2",
    "rot_x_derivative1_power2","rot_y_derivative1_power2","rot_z_derivative1_power2",
]
PHYSIO_COLS = ["csf","white_matter","csf_derivative1","white_matter_derivative1"]

def load_confounds(tsv):
    df   = pd.read_csv(tsv, sep="\t")
    keep = [c for c in MOTION_COLS + PHYSIO_COLS if c in df.columns]
    return df[keep].fillna(0).values.astype(float)

def extract_live(bold_path, conf_arr, tr):
    masker = NiftiLabelsMasker(
        labels_img=atlas_img, standardize="zscore_sample",
        detrend=True, low_pass=0.10, high_pass=0.01, t_r=tr,
        resampling_target="data",
        memory=str(CACHE), memory_level=1, verbose=0,
    )
    return masker.fit_transform(str(bold_path), confounds=conf_arr)  # (T, 100)

def npy_to_ts(path):
    """Load .npy (121,T), keep first 100 rows (Schaefer), transpose → (T,100)."""
    arr = np.load(path)
    return arr[:100, :].T

def fc(ts):
    m = np.corrcoef(ts.T); np.fill_diagonal(m, 0); return m

def pca5(ts):
    p = PCA(n_components=5)
    sc = p.fit_transform(ts)  # (T,5)
    return sc, p.explained_variance_ratio_

# ── Session definitions ───────────────────────────────────────────────────────
SESSIONS = [
    dict(
        sub="sub-002S5018", ses="ses-20121108",
        npy_A = TS_DIR  / "sub-002_S_5018_ses-20121108_task-rest_run-01_bold_timeseries.npy",
        npy_C = GSR_DIR / "sub-002_S_5018_ses-20121108_task-rest_run-01_bold_timeseries.npy",
    ),
    dict(
        sub="sub-006S4192", ses="ses-20110927",
        npy_A = TS_DIR  / "sub-006_S_4192_ses-20110927_task-rest_run-01_bold_timeseries.npy",
        npy_C = GSR_DIR / "sub-006_S_4192_ses-20110927_task-rest_run-01_bold_timeseries.npy",
    ),
]

# ── Load & process ────────────────────────────────────────────────────────────
N_PC     = 5
SMOOTH_W = 11   # Savitzky-Golay vols (~33 s at TR 3 s)

for s in SESSIONS:
    sub_dir = ESEMPIO / s["sub"] / s["ses"] / "func"
    bold  = next(sub_dir.glob("*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))
    conf  = next(sub_dir.glob("*desc-confounds_timeseries.tsv"))
    jsn   = next(sub_dir.glob("*space-MNI152NLin2009cAsym_desc-preproc_bold.json"))
    s["tr"] = float(json.loads(jsn.read_text()).get("RepetitionTime", 3.0))

    s["ts_A"] = npy_to_ts(s["npy_A"])
    s["ts_B"] = extract_live(bold, load_confounds(conf), s["tr"])
    s["ts_C"] = npy_to_ts(s["npy_C"])

    for k in ("A","B","C"):
        s[f"fc_{k}"]  = fc(s[f"ts_{k}"])
        s[f"sc_{k}"], s[f"evr_{k}"] = pca5(s[f"ts_{k}"])

    s["fc_diff"] = s["fc_C"] - s["fc_A"]   # GSR − no-GSR_pre
    s["T"] = s["ts_A"].shape[0]
    s["t"] = np.arange(s["T"]) * s["tr"]

    idx_u = np.triu_indices(100, k=1)
    print(f"{s['sub']} {s['ses']}  "
          f"FC mean:  A(noGSR-pre)={s['fc_A'][idx_u].mean():+.4f}  "
          f"B(noGSR-live)={s['fc_B'][idx_u].mean():+.4f}  "
          f"C(GSR)={s['fc_C'][idx_u].mean():+.4f}")

# ── Helpers ───────────────────────────────────────────────────────────────────
def draw_fc(ax, fc_mat, vmin, vmax, cmap, mean_r=None):
    fc_s = fc_mat[np.ix_(sorted_idx, sorted_idx)]
    im   = ax.imshow(fc_s, cmap=cmap, vmin=vmin, vmax=vmax,
                     aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, shrink=0.78, label="r", pad=0.02)
    for b in net_bounds:
        ax.axhline(b, color="white", lw=0.7)
        ax.axvline(b, color="white", lw=0.7)
    prev = 0
    for b_end in net_bounds + [100]:
        mid = (prev + int(b_end + 0.5)) / 2
        net = sorted_nets[int(mid)]
        ax.text(mid, mid, net[:4], ha="center", va="center",
                fontsize=5, fontweight="bold", color="white",
                bbox=dict(fc=NET_COLORS.get(net,"gray"), ec="none", alpha=0.7, pad=1))
        prev = int(b_end + 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    if mean_r is not None:
        ax.set_xlabel(f"mean r = {mean_r:+.3f}", fontsize=7, labelpad=2)

# ── Figure ────────────────────────────────────────────────────────────────────
N_SUB = len(SESSIONS)
fig   = plt.figure(figsize=(30, 11 * N_SUB), facecolor="#f4f4f4")
fig.suptitle(
    "3-way comparison — no-GSR pre (timeseries/)  |  no-GSR live (NIfTI)  |  GSR pre (timeseries_GSR/)\n"
    "Schaefer-100 parcellation, bandpass 0.01–0.10 Hz",
    fontsize=13, fontweight="bold", y=0.998,
)

outer = gridspec.GridSpec(N_SUB, 1, figure=fig, hspace=0.18,
                          top=0.960, bottom=0.04)

for si, s in enumerate(SESSIONS):
    label = f"{s['sub']}  {s['ses']}   TR={s['tr']:.3f}s"
    idx_u = np.triu_indices(100, k=1)

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[si],
        height_ratios=[1.6, 1.0], hspace=0.38,
    )

    # ── TOP BAND: 5 columns ──────────────────────────────────────────────────
    top = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=inner[0],
        wspace=0.28,
        width_ratios=[1, 1, 1, 1, 1.15],
    )

    col_titles = [
        f"(A) no-GSR pre\n(timeseries/)",
        f"(B) no-GSR live\n(NIfTI → masker)",
        f"(C) GSR pre\n(timeseries_GSR/)",
        "ΔFC = C − A\n(GSR effect)",
        "Explained var & PSD",
    ]

    # FC panels A, B, C
    for ci, k in enumerate(("A","B","C")):
        ax = fig.add_subplot(top[0, ci])
        draw_fc(ax, s[f"fc_{k}"], -0.8, 0.8, "RdBu_r",
                mean_r=s[f"fc_{k}"][idx_u].mean())
        ax.set_title(col_titles[ci], fontsize=9, pad=4,
                     color=COND_COLORS[k])
        if ci == 0:
            ax.set_ylabel(label, fontsize=9, labelpad=4, fontweight="bold")

    # ΔFC
    ax_d = fig.add_subplot(top[0, 3])
    vd   = max(0.25, round(np.abs(s["fc_diff"]).max(), 1))
    draw_fc(ax_d, s["fc_diff"], -vd, vd, "PuOr")
    ax_d.set_title(col_titles[3], fontsize=9, pad=4)
    diff_mean = s["fc_diff"][idx_u].mean()
    ax_d.set_xlabel(f"mean Δr = {diff_mean:+.3f}", fontsize=7, labelpad=2)

    # Variance bars + PSD
    right = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=top[0, 4], hspace=0.50,
        height_ratios=[1, 1.6],
    )

    ax_var = fig.add_subplot(right[0])
    x = np.arange(1, N_PC + 1)
    w = 0.26
    for ki, (k, off) in enumerate(zip(("A","B","C"), (-w, 0, w))):
        ax_var.bar(x + off, s[f"evr_{k}"] * 100, width=w,
                   color=COND_COLORS[k], alpha=0.85,
                   label=["no-GSR pre","no-GSR live","GSR pre"][ki])
    ax_var.set_xticks(x)
    ax_var.set_xticklabels([f"PC{i}" for i in x], fontsize=7)
    ax_var.set_ylabel("% var", fontsize=7)
    ax_var.tick_params(labelsize=6)
    ax_var.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax_var.set_title("Explained variance", fontsize=8)
    # annotate totals
    for ki, k in enumerate(("A","B","C")):
        ax_var.text(N_PC + 0.55, s[f"evr_{k}"][0]*100 - ki*2,
                    f"Σ={s[f'evr_{k}'].sum()*100:.0f}%",
                    fontsize=6, color=COND_COLORS[k], va="top")

    ax_psd = fig.add_subplot(right[1])
    nyq     = 1 / (2 * s["tr"])
    nperseg = max(16, s["T"] // 4)
    for k in ("A","B","C"):
        for pc in range(N_PC):
            f_hz, psd = welch(s[f"sc_{k}"][:, pc], fs=1/s["tr"], nperseg=nperseg)
            ax_psd.semilogy(f_hz, psd, lw=1.0,
                            color=COND_COLORS[k], ls=COND_LS[k], alpha=0.7)
    ax_psd.axvspan(0.01, 0.10, alpha=0.10, color="green")
    ax_psd.axvline(nyq, color="red", lw=0.8, ls=":", alpha=0.6)
    ax_psd.set_xlim(0, nyq + 0.01)
    ax_psd.set_xlabel("Hz", fontsize=7)
    ax_psd.set_ylabel("PSD", fontsize=7)
    ax_psd.tick_params(labelsize=6)
    ax_psd.set_title("PSD — 5 PCs (each condition)", fontsize=8)
    ax_psd.grid(True, which="both", alpha=0.2)
    ax_psd.set_title(col_titles[4], fontsize=9, pad=4)

    # ── BOTTOM BAND: 3 timeseries strips ────────────────────────────────────
    bot = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=inner[1], wspace=0.06,
    )

    for ci, k in enumerate(("A","B","C")):
        ts_gs = gridspec.GridSpecFromSubplotSpec(
            N_PC, 1, subplot_spec=bot[0, ci], hspace=0.08,
        )
        evr = s[f"evr_{k}"]
        sc  = s[f"sc_{k}"]
        for pc in range(N_PC):
            ax = fig.add_subplot(ts_gs[pc])
            raw    = sc[:, pc]
            smooth = savgol_filter(raw, window_length=SMOOTH_W, polyorder=3)
            ax.plot(s["t"], raw,    lw=0.5, color="silver", alpha=0.85)
            ax.plot(s["t"], smooth, lw=1.3, color=f"C{pc}")
            ax.axhline(0, color="k", lw=0.25)
            ax.set_xlim(0, s["t"][-1])
            ax.set_yticks([])
            ax.tick_params(labelsize=5)
            ax.set_ylabel(f"PC{pc+1}\n({evr[pc]*100:.1f}%)",
                          fontsize=6, rotation=0, labelpad=42)
            if pc < N_PC - 1:
                ax.set_xticks([])
            else:
                ax.set_xlabel("Time (s)", fontsize=7)
            if pc == 0:
                ax.set_title(
                    f"{['(A) no-GSR pre','(B) no-GSR live','(C) GSR pre'][ci]}\n"
                    f"Σ5PC = {evr.sum()*100:.1f}%",
                    fontsize=8, pad=3,
                    color=COND_COLORS[k],
                )

# ── Global legend ─────────────────────────────────────────────────────────────
net_handles = [Patch(color=c, label=n) for n, c in NET_COLORS.items()]
cond_handles = [
    mlines.Line2D([],[],color=COND_COLORS[k], lw=2,
                  ls=COND_LS[k], label=COND_LABELS[k].replace("\n"," "))
    for k in ("A","B","C")
]
fig.legend(
    handles=net_handles + cond_handles,
    loc="lower center", ncol=10, fontsize=7.5,
    bbox_to_anchor=(0.5, 0.004), framealpha=0.92,
    title="Schaefer-100 networks  |  conditions",
    title_fontsize=8,
)

# ── Save ──────────────────────────────────────────────────────────────────────
out = BASE / "compare_gsr.png"
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"\n✓  Saved: {out}")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n── Mean FC (upper triangle) ──────────────────────────────────────────")
hdr = f"{'Subject/session':<40} {'A no-GSR-pre':>14} {'B no-GSR-live':>15} {'C GSR':>10} {'C−A':>8}"
print(hdr); print("─"*len(hdr))
for s in SESSIONS:
    idx_u = np.triu_indices(100, k=1)
    a = s["fc_A"][idx_u].mean(); b = s["fc_B"][idx_u].mean(); c = s["fc_C"][idx_u].mean()
    print(f"{s['sub']} {s['ses']}   {a:+.4f}         {b:+.4f}    {c:+.4f}  {c-a:+.4f}")

print("\n── PCA total var (5 PCs) ─────────────────────────────────────────────")
for s in SESSIONS:
    print(f"{s['sub']} {s['ses']}")
    for k in ("A","B","C"):
        lbl = COND_LABELS[k].replace("\n"," ")
        print(f"  {k} {lbl}: PC1={s[f'evr_{k}'][0]*100:.1f}%  "
              f"total={s[f'evr_{k}'].sum()*100:.1f}%")
