"""
Convert ADNI CN DICOM data to BIDS format.
Only processes sessions where the SAME DATE has both:
  - Anatomical T1w  (MPRAGE / MP-RAGE / Accelerated Sagittal MPRAGE)
  - Functional BOLD (Resting State fMRI / Extended Resting State fMRI)

Outputs to D:\ADNI_BIDS\
Uses dcm2niix for DICOM->NIfTI conversion; no dcm2bids dependency needed.
Resume-safe: skips sessions already converted.
"""

import os
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

# ── Paths ─────────────────────────────────────────────────────────────────────
ADNI_DIR  = Path(r"D:\ADNI_definitivo_2_CN\ADNI")
BIDS_DIR  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\ADNI_BIDS")
TMP_DIR   = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\ADNI_BIDS_tmp")
LOG_PATH  = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection\adni_to_bids.log")
DCM2NIIX  = r"C:\Users\user\anaconda3\Library\bin\dcm2niix.exe"

# ── Scan-type classification ───────────────────────────────────────────────────
ANAT_FOLDER_NAMES = {
    "MPRAGE", "MP-RAGE", "MP-RAGE_REPEAT", "MP-RAGE_Repeat", "MP-RAGE_repeat",
    "MPRAGE_REPEAT", "MPRAGE_Repeat", "MPRAGE_repeat",
    "Accelerated_Sagittal_MPRAGE",
}
FUNC_FOLDER_NAMES = {
    "Resting_State_fMRI",
    "Extended_Resting_State_fMRI",
}
# Repeat/secondary anatomical → kept but tagged with acq-repeat
ANAT_REPEAT_NAMES = {
    "MP-RAGE_REPEAT", "MP-RAGE_Repeat", "MP-RAGE_repeat",
    "MPRAGE_REPEAT", "MPRAGE_Repeat", "MPRAGE_repeat",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def bids_sub(adni_id: str) -> str:
    """002_S_0295 -> 002S0295"""
    return adni_id.replace("_", "").replace("-", "")


def bids_ses(date_time_folder: str) -> str:
    """2011-06-02_07_58_50.0 -> 20110602"""
    return date_time_folder.split("_")[0].replace("-", "")


MIN_BOLD_VOLS = 50  # reject incomplete BOLD runs shorter than this


def _nifti_nvols(nii_path: Path) -> int:
    """Return number of volumes in a NIfTI file without loading all data."""
    import struct, gzip
    try:
        opener = gzip.open if nii_path.suffix == ".gz" else open
        with opener(nii_path, "rb") as fh:
            fh.read(40)           # skip to dim array offset
            # NIfTI-1: dim[0..7] are int16 at byte 40
            raw = fh.read(16)     # 8 × int16
        dims = struct.unpack("<8h", raw)
        ndim = dims[0]
        return dims[4] if ndim >= 4 else 1
    except Exception:
        return 0


def classify_nifti(json_path: Path, nii_path: Path = None):
    """
    Return (suffix, acq_tag, meta_dict) where:
      suffix   : 'T1w' | 'bold' | None
      acq_tag  : '' | 'accelerated' | 'extended' | 'repeat'
    """
    try:
        with open(json_path) as f:
            meta = json.load(f)
    except Exception:
        return None, "", {}

    sd  = meta.get("SeriesDescription", "").lower()
    acq = meta.get("MRAcquisitionType", "")

    # Anatomical
    if acq == "3D":
        if "mprage" in sd or "mp-rage" in sd or "mp_rage" in sd:
            if "accelerated" in sd:
                return "T1w", "accelerated", meta
            if "repeat" in sd:
                return "T1w", "repeat", meta
            return "T1w", "", meta

    # Functional — reject very short / incomplete runs
    if "fmri" in sd or ("resting" in sd and acq == "2D"):
        if nii_path and _nifti_nvols(nii_path) < MIN_BOLD_VOLS:
            return None, "", meta   # incomplete scan
        if "extended" in sd:
            return "bold", "extended", meta
        return "bold", "", meta

    return None, "", meta


def run_dcm2niix(dcm_dirs: list, out_dir: Path, log_fh):
    """Run dcm2niix on each DICOM directory; all outputs go to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for dcm_dir in dcm_dirs:
        cmd = [
            DCM2NIIX,
            "-b", "y",      # BIDS sidecar JSON
            "-ba", "n",     # no anonymise
            "-z", "y",      # gzip
            "-f", "%3s_%d", # SeriesNumber_SeriesDate → unique names
            "-o", str(out_dir),
            str(dcm_dir),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"    dcm2niix WARN on {dcm_dir.name}: {result.stderr[:200]}"
            print(msg)
            log_fh.write(msg + "\n")


def write_to_bids(nii: Path, json_path: Path, bids_sub_dir: Path,
                  sub: str, ses: str, suffix: str, acq: str,
                  run_num: int, meta: dict, log_fh):
    """Copy NIfTI + sidecar JSON to correct BIDS location."""
    datatype = "anat" if suffix == "T1w" else "func"
    out_dir  = bids_sub_dir / f"ses-{ses}" / datatype
    out_dir.mkdir(parents=True, exist_ok=True)

    acq_str = f"_acq-{acq}" if acq else ""
    run_str = f"_run-{run_num:02d}" if run_num > 1 else ""

    if suffix == "T1w":
        base = f"sub-{sub}_ses-{ses}{acq_str}{run_str}_T1w"
    else:
        task_str = "_task-rest"
        base = f"sub-{sub}_ses-{ses}{task_str}{acq_str}{run_str}_bold"
        meta["TaskName"] = "rest"

    ext = ".nii.gz" if nii.suffix == ".gz" else ".nii"
    shutil.copy2(nii,  out_dir / f"{base}{ext}")
    with open(out_dir / f"{base}.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    msg = f"      -> {datatype}/{base}{ext}"
    print(msg)
    log_fh.write(msg + "\n")


# ── Session discovery ─────────────────────────────────────────────────────────

def find_qualifying_sessions(subj_dir: Path):
    """
    Returns dict:  date_str -> {'anat': [dcm_dirs], 'func': [dcm_dirs]}
    Only dates that have AT LEAST ONE anat AND ONE func dir.
    """
    per_date = defaultdict(lambda: {"anat": [], "func": []})

    for scan_type_dir in subj_dir.iterdir():
        if not scan_type_dir.is_dir():
            continue
        name = scan_type_dir.name
        if name not in ANAT_FOLDER_NAMES and name not in FUNC_FOLDER_NAMES:
            continue
        modality = "anat" if name in ANAT_FOLDER_NAMES else "func"

        for date_time_dir in scan_type_dir.iterdir():
            if not date_time_dir.is_dir():
                continue
            date = bids_ses(date_time_dir.name)
            for sess_dir in date_time_dir.iterdir():
                if sess_dir.is_dir() and any(sess_dir.glob("*.dcm")):
                    per_date[date][modality].append(sess_dir)

    return {d: v for d, v in per_date.items()
            if v["anat"] and v["func"]}


# ── BIDS metadata files ───────────────────────────────────────────────────────

def write_bids_root(bids_dir: Path, participants: list):
    with open(bids_dir / "dataset_description.json", "w") as f:
        json.dump({
            "Name": "ADNI CN Resting-State fMRI",
            "BIDSVersion": "1.9.0",
            "DatasetType": "raw",
            "Authors": ["Alzheimer's Disease Neuroimaging Initiative (ADNI)"],
            "Acknowledgements": (
                "Data were obtained from the ADNI database (adni.loni.usc.edu). "
                "ADNI is funded by NIA grant U01 AG024904."
            ),
        }, f, indent=2)

    with open(bids_dir / "task-rest_bold.json", "w") as f:
        json.dump({
            "TaskName": "rest",
            "Instructions": "Lie still and rest with eyes open, fixating on a crosshair.",
        }, f, indent=2)

    with open(bids_dir / "participants.tsv", "w") as f:
        f.write("participant_id\tadni_id\n")
        for adni_id, sub in sorted(participants, key=lambda x: x[1]):
            f.write(f"sub-{sub}\t{adni_id}\n")

    with open(bids_dir / "participants.json", "w") as f:
        json.dump({
            "participant_id": {"Description": "BIDS subject identifier"},
            "adni_id": {"Description": "Original ADNI subject ID (e.g. 002_S_0295)"},
        }, f, indent=2)

    with open(bids_dir / "README", "w") as f:
        f.write(
            "ADNI Cognitively Normal (CN) subjects — resting-state fMRI + T1w.\n"
            "Only sessions with BOTH anatomical and functional data are included.\n"
            "Generated by adni_to_bids.py.\n"
        )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    BIDS_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    log_fh = open(LOG_PATH, "a", buffering=1, encoding="utf-8")
    log_fh.write(f"\n=== Run started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")

    subjects = sorted(p for p in ADNI_DIR.iterdir() if p.is_dir())
    func_subjects = [
        s for s in subjects
        if any((s / ft).is_dir() for ft in FUNC_FOLDER_NAMES)
    ]

    print(f"Subjects with fMRI   : {len(func_subjects)}")
    print(f"BIDS output          : {BIDS_DIR}")
    print()

    participants = []
    total_sessions = 0
    skipped_sessions = 0
    errors = 0

    for i, subj_dir in enumerate(func_subjects, 1):
        adni_id = subj_dir.name
        sub     = bids_sub(adni_id)

        sessions = find_qualifying_sessions(subj_dir)
        if not sessions:
            continue

        print(f"[{i:3d}/{len(func_subjects)}] sub-{sub}  ({len(sessions)} session(s))")
        log_fh.write(f"sub-{sub}: {len(sessions)} session(s)\n")

        bids_sub_dir = BIDS_DIR / f"sub-{sub}"

        for ses, modalities in sorted(sessions.items()):
            total_sessions += 1
            ses_anat_dir = bids_sub_dir / f"ses-{ses}" / "anat"
            ses_func_dir = bids_sub_dir / f"ses-{ses}" / "func"

            # Resume: skip if both anat and func dirs already exist and are non-empty
            if (ses_anat_dir.exists() and any(ses_anat_dir.iterdir()) and
                    ses_func_dir.exists() and any(ses_func_dir.iterdir())):
                print(f"  ses-{ses} already converted, skipping.")
                skipped_sessions += 1
                if adni_id not in [p[0] for p in participants]:
                    participants.append((adni_id, sub))
                continue

            print(f"  ses-{ses}  "
                  f"({len(modalities['anat'])} anat dirs, "
                  f"{len(modalities['func'])} func dirs)")

            tmp_ses = TMP_DIR / f"sub-{sub}_ses-{ses}"
            try:
                # Convert all DICOMs for this session
                all_dcm_dirs = modalities["anat"] + modalities["func"]
                run_dcm2niix(all_dcm_dirs, tmp_ses, log_fh)

                # Sort resulting NIfTIs into BIDS
                nii_files = list(tmp_ses.glob("*.nii.gz")) + list(tmp_ses.glob("*.nii"))
                t1w_counts  = defaultdict(int)  # acq_tag -> count
                bold_counts = defaultdict(int)

                for nii in sorted(nii_files):
                    json_p = nii.with_suffix("").with_suffix(".json")
                    if nii.suffix == ".nii":
                        json_p = nii.with_suffix(".json")
                    if not json_p.exists():
                        continue

                    suffix, acq, meta = classify_nifti(json_p, nii)
                    if suffix is None:
                        continue

                    if suffix == "T1w":
                        t1w_counts[acq] += 1
                        run_num = t1w_counts[acq]
                    else:
                        bold_counts[acq] += 1
                        run_num = bold_counts[acq]

                    write_to_bids(nii, json_p, bids_sub_dir,
                                  sub, ses, suffix, acq, run_num, meta, log_fh)

                if adni_id not in [p[0] for p in participants]:
                    participants.append((adni_id, sub))

            except Exception as e:
                errors += 1
                msg = f"  ERROR ses-{ses}: {e}"
                print(msg)
                log_fh.write(msg + "\n")
            finally:
                shutil.rmtree(tmp_ses, ignore_errors=True)

    # Write root-level BIDS files
    write_bids_root(BIDS_DIR, participants)

    summary = (
        f"\n=== Done ===\n"
        f"  Subjects converted : {len(participants)}\n"
        f"  Sessions converted : {total_sessions - skipped_sessions}\n"
        f"  Sessions skipped   : {skipped_sessions}\n"
        f"  Errors             : {errors}\n"
        f"  BIDS output        : {BIDS_DIR}\n"
    )
    print(summary)
    log_fh.write(summary)
    log_fh.close()


if __name__ == "__main__":
    main()
