"""Visualize one sample timeseries: raw signals, PCA, ICA, and FC."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA, FastICA
from scipy.signal import savgol_filter, welch
import json

BASE   = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection")
TS     = BASE / "timeseries"
FMRI   = BASE / "fmriprep_output"
OUTDIR = BASE / "qc_figures"
OUTDIR.mkdir(exist_ok=True)

# ── Pick one representative file (a CN subject, good shape) ───────────────────
good = []
for grp in ("CN", "AD", "MCI"):
    for f in sorted((TS / grp).glob("*.npy")):
        arr = np.load(f)
        if arr.shape == (121, 140):
            good.append((grp, f, arr))
    if good:
        break

grp, fpath, ts = good[0]   # shape (121, 140): parcels x time
print(f"Sample file: {fpath.name}  shape={ts.shape}  group={grp}")

# ── Get actual TR from fmriprep JSON ─────────────────────────────────────────
TR = 3.0
jsons = list(FMRI.rglob("*desc-preproc_bold.json"))
if jsons:
    with open(jsons[0]) as f:
        TR = float(json.load(f).get("RepetitionTime", 3.0))
print(f"TR = {TR} s  |  Nyquist = {1/(2*TR):.4f} Hz")

T   = ts.shape[1]          # 140 volumes
t   = np.arange(T) * TR    # 0 … 417 s

# ── 1. RAW SIGNALS + SMOOTHED OVERLAY (4 parcels, full width) ─────────────────
# Savitzky-Golay: window=15 TRs (45 s), order=3  →  ~0.022 Hz low-pass display
SMOOTH_WIN = 15

fig, axes = plt.subplots(4, 1, figsize=(14, 8), sharex=True)
fig.suptitle(
    f"Parcel timeseries — {fpath.name[:55]}\n"
    f"TR={TR}s  |  bandpass 0.01–0.1 Hz  |  grey=raw, colour=SavGol smoothed",
    fontsize=9
)
for i, ax in enumerate(axes):
    smooth = savgol_filter(ts[i], window_length=SMOOTH_WIN, polyorder=3)
    ax.plot(t, ts[i], lw=0.6, color="silver", alpha=0.9)
    ax.plot(t, smooth, lw=1.5, color=f"C{i}")
    ax.set_ylabel(f"P{i+1}", fontsize=8, rotation=0, labelpad=24)
    ax.axhline(0, color="k", lw=0.3)
    ax.tick_params(labelsize=7)
    ax.set_ylim(-4, 4)
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(OUTDIR / "01_raw_signals.png", dpi=150)
plt.close()
print("Saved 01_raw_signals.png")

# ── 2. PCA — first 5 components + explained variance ─────────────────────────
pca  = PCA(n_components=5)
Xpca = pca.fit_transform(ts.T)   # (140, 5)

fig, axes = plt.subplots(6, 1, figsize=(14, 10))
ax_var = axes[0]
ax_var.bar(range(1, 6), pca.explained_variance_ratio_ * 100, color="steelblue")
ax_var.set_ylabel("% var")
ax_var.set_xlabel("PC")
ax_var.set_title("PCA explained variance")
ax_var.set_xticks(range(1, 6))

for i, ax in enumerate(axes[1:]):
    raw = Xpca[:, i]
    smooth = savgol_filter(raw, window_length=SMOOTH_WIN, polyorder=3)
    ax.plot(t, raw, lw=0.6, color="silver", alpha=0.9)
    ax.plot(t, smooth, lw=1.5, color=f"C{i}")
    ax.set_ylabel(f"PC{i+1}", rotation=0, labelpad=28, fontsize=8)
    ax.axhline(0, color="k", lw=0.3)
    ax.tick_params(labelsize=6)
axes[-1].set_xlabel("Time (s)")
fig.suptitle("PCA components  (grey=raw, colour=smoothed)", fontsize=10)
plt.tight_layout()
fig.savefig(OUTDIR / "02_pca.png", dpi=150)
plt.close()
print("Saved 02_pca.png")

# ── 3. ICA — 5 independent components ────────────────────────────────────────
ica  = FastICA(n_components=5, random_state=42, max_iter=500)
Xica = ica.fit_transform(ts.T)   # (140, 5)

fig, axes = plt.subplots(5, 1, figsize=(14, 8), sharex=True)
fig.suptitle("ICA components — FastICA 5  (grey=raw, colour=smoothed)", fontsize=10)
for i, ax in enumerate(axes):
    raw = Xica[:, i]
    smooth = savgol_filter(raw, window_length=SMOOTH_WIN, polyorder=3)
    ax.plot(t, raw, lw=0.6, color="silver", alpha=0.9)
    ax.plot(t, smooth, lw=1.5, color=f"C{i+5}")
    ax.set_ylabel(f"IC{i+1}", rotation=0, labelpad=28, fontsize=8)
    ax.axhline(0, color="k", lw=0.3)
    ax.tick_params(labelsize=6)
axes[-1].set_xlabel("Time (s)")
plt.tight_layout()
fig.savefig(OUTDIR / "03_ica.png", dpi=150)
plt.close()
print("Saved 03_ica.png")

# ── 4. FUNCTIONAL CONNECTIVITY MATRIX ────────────────────────────────────────
fc = np.corrcoef(ts)
np.fill_diagonal(fc, 0)

boundary = 100
fig, ax = plt.subplots(figsize=(11, 9))
im = ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
               interpolation="nearest")
plt.colorbar(im, ax=ax, label="Pearson r")
ax.axhline(boundary - 0.5, color="k", lw=1.5, ls="--")
ax.axvline(boundary - 0.5, color="k", lw=1.5, ls="--")
ax.text(50, -3, "Cortical (Schaefer-100)", ha="center", va="top", fontsize=8)
ax.text(109, -3, "Subcort\n(HO-21)", ha="center", va="top", fontsize=7)
ticks = list(range(0, 100, 10)) + list(range(100, 121, 5))
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels([str(tk+1) for tk in ticks], fontsize=6, rotation=90)
ax.set_yticklabels([str(tk+1) for tk in ticks], fontsize=6)
ax.set_title(f"FC matrix — {fpath.name[:60]}\n(121 parcels; diagonal=0)")
plt.tight_layout()
fig.savefig(OUTDIR / "04_fc_matrix.png", dpi=150)
plt.close()
print("Saved 04_fc_matrix.png")

# ── 5. FC DISTRIBUTION ────────────────────────────────────────────────────────
idx   = np.triu_indices(121, k=1)
fc_ut = fc[idx]

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(fc_ut, bins=80, color="steelblue", edgecolor="none", alpha=0.85)
ax.axvline(fc_ut.mean(), color="red", lw=1.5, ls="--",
           label=f"mean={fc_ut.mean():.3f}")
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Pearson r")
ax.set_ylabel("Count")
ax.set_title("FC distribution (upper triangle, 7260 pairs)")
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(OUTDIR / "05_fc_distribution.png", dpi=150)
plt.close()
print("Saved 05_fc_distribution.png")

# ── 6. POWER SPECTRUM ─────────────────────────────────────────────────────────
nyquist = 1 / (2 * TR)
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(6):
    f_hz, psd = welch(ts[i], fs=1/TR, nperseg=T//2)
    ax.semilogy(f_hz, psd, lw=0.9, label=f"P{i+1}")
ax.axvspan(0.01, 0.1, alpha=0.15, color="green", label="bandpass 0.01–0.1 Hz")
ax.axvline(nyquist, color="red", lw=1.2, ls=":", label=f"Nyquist {nyquist:.3f} Hz")
ax.axvline(0.1, color="orange", lw=1.0, ls="--", label="LP cutoff 0.1 Hz")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("PSD")
ax.set_title(f"Power spectral density — first 6 parcels  (TR={TR}s, Nyquist={nyquist:.3f} Hz)")
ax.legend(fontsize=7, ncol=2)
ax.set_xlim(0, nyquist + 0.01)
plt.tight_layout()
fig.savefig(OUTDIR / "06_power_spectrum.png", dpi=150)
plt.close()
print("Saved 06_power_spectrum.png")

print(f"\nAll figures saved to: {OUTDIR}")
print(f"  FC mean={fc_ut.mean():.4f}  std={fc_ut.std():.4f}  "
      f"range=[{fc_ut.min():.3f}, {fc_ut.max():.3f}]")
print(f"  PCA 5-PC variance: {pca.explained_variance_ratio_.sum()*100:.1f}%")
