"""
Download ABIDE I preprocessed ROI time series via nilearn.

Pipeline : CPAC  (most widely used, closest to fMRIPrep conventions)
Atlas    : Harvard-Oxford (rois_ho, 111 cortical+subcortical regions)
           — closest available to the 121-parcel Schaefer+subcortical scheme
           used in the AD analysis; atlas mismatch should be kept in mind
           when comparing results directly.
Filtering: band-pass 0.01-0.10 Hz (matches AD preprocessing)
GSR      : False (matches AD preprocessing)
QC       : quality_checked=True  (removes ~20% of sessions flagged by the
           ABIDE QC team for excessive motion / scanner artefacts)

Output
------
ABIDE/
  cpac_nofilt_noglobal/   <- nilearn cache (raw S3 files)
  abide_timeseries.npz    <- dict: 'timeseries' (list), 'phenotypic' (array)
  abide_summary.txt       <- quick cohort summary
"""

import numpy as np
from nilearn.datasets import fetch_abide_pcp
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "ABIDE")
os.makedirs(DATA_DIR, exist_ok=True)

print("Fetching ABIDE I  —  CPAC | Harvard-Oxford atlas | BPF | no GSR ...")
print("(downloads from S3; first run may take 10-30 min depending on bandwidth)\n")

abide = fetch_abide_pcp(
    data_dir=DATA_DIR,
    pipeline="cpac",
    band_pass_filtering=True,
    global_signal_regression=False,
    derivatives=["rois_ho"],   # Harvard-Oxford ROI time series
    quality_checked=True,
    verbose=1,
)

# ── phenotype ──────────────────────────────────────────────────────────────
pheno = abide.phenotypic
# DX_GROUP: 1 = ASD, 2 = control
dx    = pheno["DX_GROUP"]
n_asd  = int(np.sum(dx == 1))
n_ctrl = int(np.sum(dx == 2))
n_tot  = len(dx)

# ── time series ────────────────────────────────────────────────────────────
ts_list = abide.rois_ho          # list of (T, 111) arrays, one per subject

# sanity-check shapes
shapes = [ts.shape for ts in ts_list]
trs    = [s[0] for s in shapes]
n_roi  = shapes[0][1] if shapes else 0

print(f"\n{'='*55}")
print(f"  Subjects downloaded : {n_tot}")
print(f"    ASD               : {n_asd}")
print(f"    Controls          : {n_ctrl}")
print(f"  Parcels (HO atlas)  : {n_roi}")
print(f"  Volumes per session : min={min(trs)}, max={max(trs)}, "
      f"median={int(np.median(trs))}")
print(f"  Sites               : {len(np.unique(pheno['SITE_ID']))}")
print(f"{'='*55}\n")

# ── save ───────────────────────────────────────────────────────────────────
out_npz = os.path.join(DATA_DIR, "abide_timeseries.npz")
np.savez(out_npz,
         timeseries=np.array(ts_list, dtype=object),
         phenotypic=pheno)
print(f"Saved time series + phenotype -> {out_npz}")

summary_lines = [
    "ABIDE I  —  CPAC | Harvard-Oxford (111 ROIs) | BPF | no GSR | QC-passed",
    f"Total subjects : {n_tot}",
    f"  ASD          : {n_asd}",
    f"  Controls     : {n_ctrl}",
    f"Parcels        : {n_roi}",
    f"Volume range   : {min(trs)} – {max(trs)} (median {int(np.median(trs))})",
    f"Sites          : {len(np.unique(pheno['SITE_ID']))}",
    "",
    "Atlas note: Harvard-Oxford (111 regions) differs from the 121-parcel",
    "Schaefer-100+subcortical scheme used in the AD analysis.",
    "Direct cross-study comparison of region-level features should account",
    "for this mismatch.",
]
out_txt = os.path.join(DATA_DIR, "abide_summary.txt")
with open(out_txt, "w") as f:
    f.write("\n".join(summary_lines))
print(f"Saved summary       -> {out_txt}")
