#!/usr/bin/env python3
"""
Extract parcel timeseries for all AD and MCI subjects.

Atlas: Schaefer-100 (cortical) + Harvard-Oxford subcortical all 21 labels
Total parcels: 100 + 21 = 121

Output: N x T numpy arrays (.npy) saved per session
        timeseries/AD/<subject>_<session>_<run>_timeseries.npy
        timeseries/MCI/<subject>_<session>_<run>_timeseries.npy

Also saves a shared labels file: timeseries/parcel_labels.txt
"""

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
from nilearn import datasets, maskers, image

# ── Paths ─────────────────────────────────────────────────────────────────────
FMRIPREP  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\fmriprep_output")
OUT_BASE  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries")
PYTHON    = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\bids_env\python.exe")

# Group membership (from participants.tsv files)
AD_BIDS   = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\AD_bids")
MCI_BIDS  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\MCI_bids")
CN_BIDS   = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\CN_bids")

# ── Signal processing parameters ─────────────────────────────────────────────
HP_CUTOFF = 0.01   # Hz
LP_CUTOFF = 0.1    # Hz

# ── Confound columns to use ───────────────────────────────────────────────────
MOTION_PREFIXES = ("trans_x", "trans_y", "trans_z",
                   "rot_x",   "rot_y",   "rot_z")
PHYSIO_COLS     = ("csf", "white_matter",
                   "csf_derivative1", "white_matter_derivative1")


def build_combined_atlas():
    """Return (atlas_img, labels) for Schaefer-100 + all 21 HO subcortical labels (=121 total)."""
    print("Loading Schaefer-100 atlas ...")
    schaefer   = datasets.fetch_atlas_schaefer_2018(n_rois=100, resolution_mm=2)
    sch_img    = nib.load(schaefer.maps)
    sch_labels = [l.decode() if isinstance(l, bytes) else l
                  for l in schaefer.labels
                  if (l.decode() if isinstance(l, bytes) else l) != "Background"]

    print("Loading Harvard-Oxford subcortical atlas ...")
    ho = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr25-2mm")
    ho_img_orig = ho.maps if isinstance(ho.maps, nib.Nifti1Image) else nib.load(ho.maps)

    # Resample HO to Schaefer space
    ho_img = image.resample_to_img(ho_img_orig, sch_img,
                                   interpolation="nearest")

    ho_labels_all = [l.decode() if isinstance(l, bytes) else l
                     for l in ho.labels]

    # Keep ALL 21 HO labels (skip index 0 = "Background")
    # voxel value == list index (0=Background, 1=Left Cerebral White Matter, ..., 21=Right Accumbens)
    # This matches the original 121-parcel pipeline: 100 Schaefer + 21 HO
    keep_idx = {i: name
                for i, name in enumerate(ho_labels_all)
                if i != 0}

    print(f"  Keeping all {len(keep_idx)} HO subcortical regions: "
          f"{list(keep_idx.values())}")

    # Build combined atlas data
    sch_data = np.asarray(sch_img.dataobj, dtype=np.int32)
    ho_data  = np.asarray(ho_img.dataobj, dtype=np.int32)
    combined = sch_data.copy()

    sub_labels = []
    new_label  = 100
    for ho_val, ho_name in sorted(keep_idx.items()):  # sorted by HO value = original order
        new_label += 1
        mask = (ho_data == ho_val) & (sch_data == 0)
        combined[mask] = new_label
        sub_labels.append(ho_name)

    combined_img = nib.Nifti1Image(combined, sch_img.affine, sch_img.header)
    all_labels   = sch_labels + sub_labels
    print(f"  Combined atlas: {len(all_labels)} parcels "
          f"({len(sch_labels)} cortical + {len(sub_labels)} subcortical)")
    return combined_img, all_labels


def get_confounds(conf_path: Path) -> np.ndarray:
    """Load and clean confound matrix (motion 6+deriv+sq+sq_deriv, CSF/WM+deriv)."""
    df = pd.read_csv(conf_path, sep="\t")

    cols = []
    # 24 HMP: 6 params + 6 deriv + 6 power2 + 6 power2_deriv
    for pfx in MOTION_PREFIXES:
        for suffix in ("", "_derivative1", "_power2", "_derivative1_power2"):
            c = pfx + suffix
            if c in df.columns:
                cols.append(c)
    # CSF / WM
    for c in PHYSIO_COLS:
        if c in df.columns:
            cols.append(c)

    return df[cols].fillna(0).values


def build_ad_mapping():
    """Return {(site, ses, run): patient_id} from AD_bids directory structure.
    Returns (mapping, ambiguous_keys) where ambiguous keys map to None."""
    mapping = {}
    ambiguous = set()

    for pat_dir in sorted(AD_BIDS.iterdir()):
        if not pat_dir.is_dir() or "_S_" not in pat_dir.name:
            continue
        pat_id = pat_dir.name                           # sub-006_S_4546
        site   = pat_id.split("_S_")[0]                # sub-006

        for ses_dir in pat_dir.iterdir():
            if not (ses_dir.is_dir() and ses_dir.name.startswith("ses-")):
                continue
            ses  = ses_dir.name
            func = ses_dir / "func"
            if not func.exists():
                continue
            runs = set()
            for f in func.iterdir():
                m = re.search(r"(run-\d+)", f.name)
                runs.add(m.group(1) if m else "run-01")
            for run in runs:
                key = (site, ses, run)
                if key in mapping:
                    ambiguous.add(key)
                    mapping[key] = None          # collision → ambiguous
                else:
                    mapping[key] = pat_id

    n_known = sum(1 for v in mapping.values() if v is not None)
    print(f"  AD mapping: {n_known} unambiguous, {len(ambiguous)} ambiguous, "
          f"{len(mapping)-n_known-len(ambiguous)} unmappable")
    return mapping, ambiguous


def resolve_ad_tag(tag: str, site: str, ad_mapping: dict) -> str:
    """Replace site prefix in tag with full patient ID if known."""
    parts    = tag.split("_")
    ses_part = next((p for p in parts if p.startswith("ses-")), None)
    run_part = next((p for p in parts if p.startswith("run-")), "run-01")
    key      = (site, ses_part, run_part)
    pat_id   = ad_mapping.get(key, "UNKNOWN")
    if pat_id is None:
        pat_id = "AMBIGUOUS"
    if pat_id not in ("UNKNOWN", "AMBIGUOUS"):
        return tag.replace(site + "_", pat_id + "_", 1)
    return tag   # keep site-only name for unknowns


def get_tr(bold_json: Path) -> float:
    with open(bold_json) as f:
        return float(json.load(f)["RepetitionTime"])


def process_bold(bold_path: Path, atlas_img, masker_cache: dict,
                 out_path: Path):
    """Extract and save N x T timeseries for one BOLD file."""
    stem    = bold_path.stem.replace(".nii", "")
    mask_p  = bold_path.parent / bold_path.name.replace(
                  "desc-preproc_bold", "desc-brain_mask")
    conf_p  = bold_path.parent / bold_path.name.replace(
                  "_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
                  "_desc-confounds_timeseries.tsv")
    json_p  = bold_path.with_suffix("").with_suffix(".json")

    missing = [p for p in (mask_p, conf_p, json_p) if not p.exists()]
    if missing:
        print(f"  SKIP {bold_path.name} — missing: {[p.name for p in missing]}")
        return

    tr      = get_tr(json_p)
    conf    = get_confounds(conf_p)

    masker_key = (str(mask_p), tr)
    if masker_key not in masker_cache:
        masker_cache[masker_key] = maskers.NiftiLabelsMasker(
            labels_img=atlas_img,
            mask_img=str(mask_p),
            standardize="zscore_sample",
            detrend=True,
            t_r=tr,
            high_pass=HP_CUTOFF,
            low_pass=LP_CUTOFF,
            resampling_target="labels",
            memory_level=0,
        )
    m = masker_cache[masker_key]

    ts = m.fit_transform(str(bold_path), confounds=conf)  # T x N
    ts_out = ts.T                                          # N x T

    np.save(out_path, ts_out)
    print(f"  Saved {out_path.name}  shape={ts_out.shape}")


def main():
    # ── Prepare output directories ────────────────────────────────────────────
    for grp in ("AD", "MCI", "CN"):
        (OUT_BASE / grp).mkdir(parents=True, exist_ok=True)

    # ── Build atlas ───────────────────────────────────────────────────────────
    atlas_img, all_labels = build_combined_atlas()

    # Save shared label file
    label_file = OUT_BASE / "parcel_labels.txt"
    label_file.write_text("\n".join(
        f"{i+1:03d}  {lbl}" for i, lbl in enumerate(all_labels)
    ))
    print(f"Labels saved to {label_file}  ({len(all_labels)} parcels)")

    # ── Identify groups ───────────────────────────────────────────────────────
    mci_ids = set(pd.read_csv(MCI_BIDS / "participants.tsv", sep="\t")
                  ["participant_id"].tolist())
    cn_ids  = set(pd.read_csv(CN_BIDS  / "participants.tsv", sep="\t")
                  ["participant_id"].tolist())

    print("Building AD session->patient mapping from AD_bids ...")
    ad_mapping, _ = build_ad_mapping()

    # ── Process all subjects ──────────────────────────────────────────────────
    ad_re = re.compile(r"^sub-\d+$")

    masker_cache = {}
    stats = {"AD": 0, "MCI": 0, "CN": 0, "skipped": 0}

    for sub_dir in sorted(FMRIPREP.iterdir()):
        if not sub_dir.is_dir() or not sub_dir.name.startswith("sub-"):
            continue

        sub = sub_dir.name
        if ad_re.match(sub):
            group = "AD"
        elif sub in mci_ids:
            group = "MCI"
        elif sub in cn_ids:
            group = "CN"
        else:
            continue   # unrecognised subject — skip

        print(f"\n[{group}] {sub}")

        bold_files = sorted(sub_dir.rglob(
            "*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))

        for bold_path in bold_files:
            # Build output filename from BOLD stem
            name = bold_path.name
            # strip space/desc tags to keep sub_ses_run
            tag = "_".join(p for p in name.replace(".nii.gz", "").split("_")
                           if not p.startswith(("space-", "desc-", "res-")))
            # For AD: replace site-level sub ID with full patient ID if known
            if group == "AD":
                tag = resolve_ad_tag(tag, sub, ad_mapping)
            out_path = OUT_BASE / group / f"{tag}_timeseries.npy"

            if out_path.exists():
                print(f"  EXISTS {out_path.name} — skip")
                stats[group] += 1
                continue

            try:
                process_bold(bold_path, atlas_img, masker_cache, out_path)
                stats[group] += 1
            except Exception as e:
                print(f"  ERROR {bold_path.name}: {e}")
                stats["skipped"] += 1

    print("\n" + "=" * 55)
    print(f"Done.  AD={stats['AD']}  MCI={stats['MCI']}  CN={stats['CN']}  "
          f"skipped={stats['skipped']}")
    print(f"Output: {OUT_BASE}")
    print("=" * 55)


if __name__ == "__main__":
    main()
