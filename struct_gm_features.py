"""
struct_gm_features.py
=====================
Structural-MRI counterpart to the FC pipeline: parcel-wise grey-matter volume in
the SAME 121-parcel atlas (Schaefer-100 cortical + 21 Harvard-Oxford subcortical)
used for the functional analyses, extracted from the fmriprep anatomical
derivatives that already exist in fmriprep_output/.

For each subject with a
  sub-*/anat/*space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz
the GM probability map is resampled to the atlas grid and integrated within each
parcel, giving a GM volume in mm^3 per parcel. Two normalisations are saved:
raw volume, and volume divided by the subject's total GM (a head-size proxy,
standing in for TIV, which would need FreeSurfer).

Output: struct_gm_features.npz  (vol [n,121], labels, subjects, parcel names)
"""
import os, glob
import numpy as np
import nibabel as nib
from nilearn import datasets, image
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")

FMRIPREP = "fmriprep_output"
OUT = "struct_gm_features.npz"


def build_combined_atlas():
    """Schaefer-100 + all 21 HO subcortical labels = 121 parcels (as extract_timeseries.py)."""
    sch = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    sch_img = nib.load(sch.maps)
    sch_lab = [l.decode() if isinstance(l, bytes) else str(l) for l in sch.labels]

    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_img = image.resample_to_img(ho.maps, sch_img, interpolation="nearest",
                                   force_resample=True, copy_header=True)
    ho_dat = np.asarray(ho_img.dataobj).astype(int)
    sch_dat = np.asarray(sch_img.dataobj).astype(int)

    comb = sch_dat.copy()
    for i in range(1, 22):                       # 21 HO subcortical labels
        comb[(ho_dat == i) & (comb == 0)] = 100 + i
    labels = sch_lab[:100] + [f"HO-{i}" for i in range(1, 22)]
    return nib.Nifti1Image(comb.astype(np.int16), sch_img.affine), labels


def parcel_gm_volume(gm_path, atlas_dat, atlas_img, voxel_mm3):
    """Integrate the GM probability map inside each parcel -> volume in mm^3."""
    gm = image.resample_to_img(gm_path, atlas_img, interpolation="continuous",
                               force_resample=True, copy_header=True)
    g = np.asarray(gm.dataobj, dtype=np.float32)
    out = np.zeros(121, dtype=np.float64)
    for k in range(1, 122):
        m = atlas_dat == k
        if m.any():
            out[k - 1] = g[m].sum() * voxel_mm3
    return out, float(g.sum() * voxel_mm3)       # parcel volumes, total GM


def collect_subjects():
    """Map normalised subject id -> GM probseg path."""
    found = {}
    for d in sorted(os.listdir(FMRIPREP)):
        p = os.path.join(FMRIPREP, d, "anat")
        if not (d.startswith("sub-") and os.path.isdir(p)):
            continue
        g = sorted(glob.glob(os.path.join(
            p, "*space-MNI152NLin2009cAsym_label-GM_probseg.nii.gz")))
        if g:
            found[d.replace("_S_", "S")] = g[0]   # first (session-less or earliest)
    return found


def cohort_labels():
    """Diagnosis from the BIDS download trees. CN_bids is a SUPERSET that also
    contains every AD_bids and MCI_bids subject, so 'control' must be defined by
    subtraction, not by membership in CN_bids."""
    def subs(d):
        return set(x.replace("_S_", "S") for x in os.listdir(d) if x.startswith("sub-"))
    cn_all, ad, mci = subs("CN_bids"), subs("AD_bids"), subs("MCI_bids")
    cn_clean = cn_all - ad - mci
    return cn_clean, ad, mci


if __name__ == "__main__":
    print("Building 121-parcel atlas ...")
    atlas_img, labels = build_combined_atlas()
    atlas_dat = np.asarray(atlas_img.dataobj).astype(int)
    voxel_mm3 = float(np.abs(np.linalg.det(atlas_img.affine[:3, :3])))
    print(f"  atlas {atlas_dat.shape}, voxel {voxel_mm3:.2f} mm^3, "
          f"{len(np.unique(atlas_dat))-1} parcels")

    found = collect_subjects()
    cn_clean, ad, mci = cohort_labels()
    print(f"BIDS trees: CN_bids-minus-AD/MCI={len(cn_clean)}, AD={len(ad)}, MCI={len(mci)}")

    rows, y, sid = [], [], []
    tot_gm = []
    for s, path in tqdm(sorted(found.items()), desc="GM volumes"):
        if s in ad:      lb = 1
        elif s in mci:   lb = 2
        elif s in cn_clean: lb = 0
        else:            continue
        v, tg = parcel_gm_volume(path, atlas_dat, atlas_img, voxel_mm3)
        rows.append(v); tot_gm.append(tg); y.append(lb); sid.append(s)

    V = np.array(rows); y = np.array(y); sid = np.array(sid); tot_gm = np.array(tot_gm)
    print(f"\nExtracted {V.shape} : CN {(y==0).sum()}, AD {(y==1).sum()}, MCI {(y==2).sum()}")
    np.savez(OUT, vol=V, total_gm=tot_gm, labels=y, subjects=sid,
             parcels=np.array(labels))
    print(f"Saved {OUT}")
