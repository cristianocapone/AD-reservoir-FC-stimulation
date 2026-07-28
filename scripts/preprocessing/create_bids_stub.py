#!/usr/bin/env python3
"""Create a minimal stub BIDS dataset from the existing fmriprep output structure.

fmriprep --reports-only needs a BIDS input to validate participants and read
bold series filenames. This script creates tiny but structurally correct stub
files so reports can be regenerated without re-doing the expensive preprocessing.
"""

import gzip
import json
import struct
from pathlib import Path
import re

FMRIPREP_OUT = Path(__file__).parent / "fmriprep_output"
STUB_DIR = Path(__file__).parent / "bids_stub"
SUBJECTS = ["006", "013", "018", "053", "100", "130", "136"]


def minimal_nifti_gz() -> bytes:
    """Return a gzip-compressed minimal valid NIfTI-1 header (no voxel data)."""
    hdr = bytearray(348)
    struct.pack_into("<i", hdr, 0, 348)       # sizeof_hdr
    struct.pack_into("<h", hdr, 40, 3)        # dim[0] = 3 (3-D)
    struct.pack_into("<h", hdr, 42, 1)        # dim[1]
    struct.pack_into("<h", hdr, 44, 1)        # dim[2]
    struct.pack_into("<h", hdr, 46, 1)        # dim[3]
    struct.pack_into("<f", hdr, 76, 1.0)      # pixdim[1]
    struct.pack_into("<f", hdr, 80, 1.0)      # pixdim[2]
    struct.pack_into("<f", hdr, 84, 1.0)      # pixdim[3]
    struct.pack_into("<f", hdr, 108, 352.0)   # vox_offset (header + extension)
    struct.pack_into("<h", hdr, 70, 16)       # datatype = float32
    struct.pack_into("<h", hdr, 72, 32)       # bitpix
    hdr[344:348] = b"n+1\0"                   # magic
    # 4-byte empty extension block
    buf = bytes(hdr) + b"\x00\x00\x00\x00"
    return gzip.compress(buf, compresslevel=1)


def bold_sidecar(tr: float = 2.0) -> dict:
    return {
        "TaskName": "rest",
        "RepetitionTime": tr,
        "MagneticFieldStrength": 3,
        "PhaseEncodingDirection": "j-",
    }


def t1w_sidecar() -> dict:
    return {"MagneticFieldStrength": 3}


def get_func_runs(sub: str):
    """Return list of (session, run_label) pairs found in fmriprep output."""
    runs = []
    sub_dir = FMRIPREP_OUT / f"sub-{sub}"
    for ses_dir in sorted(sub_dir.iterdir()):
        if not ses_dir.is_dir() or ses_dir.name.startswith("ses-multi"):
            continue
        ses = ses_dir.name  # e.g. ses-20120514
        func_dir = ses_dir / "func"
        if not func_dir.exists():
            continue
        for f in sorted(func_dir.glob("*_desc-preproc_bold.nii.gz")):
            m = re.search(r"_(run-\d+)_", f.name)
            if m:
                runs.append((ses, m.group(1)))
    return runs


def get_t1w_sessions(sub: str):
    """Return list of session labels that have anat output."""
    sessions = []
    sub_dir = FMRIPREP_OUT / f"sub-{sub}"
    for ses_dir in sorted(sub_dir.iterdir()):
        if not ses_dir.is_dir() or ses_dir.name.startswith("ses-multi"):
            continue
        if (ses_dir / "anat").exists():
            sessions.append(ses_dir.name)
    return sessions


def main():
    STUB_DIR.mkdir(exist_ok=True)
    stub_nii = minimal_nifti_gz()

    participants = []

    for sub in SUBJECTS:
        func_runs = get_func_runs(sub)
        t1w_sessions = get_t1w_sessions(sub)

        if not func_runs:
            print(f"sub-{sub}: no func sessions found in fmriprep output, skipping")
            continue

        # Only use sessions that have func data — extra anat-only sessions cause
        # a KeyError in nireports when it tries to format {session} for a func report
        func_sessions = sorted({ses for ses, _ in func_runs})
        print(f"sub-{sub}: {len(func_runs)} func run(s) across {len(func_sessions)} session(s)")
        participants.append(f"sub-{sub}")

        for ses in func_sessions:
            # Anat stub (one T1w per func session so fmriprep has a reference)
            anat_dir = STUB_DIR / f"sub-{sub}" / ses / "anat"
            anat_dir.mkdir(parents=True, exist_ok=True)
            nii_path = anat_dir / f"sub-{sub}_{ses}_T1w.nii.gz"
            if not nii_path.exists():
                nii_path.write_bytes(stub_nii)
            json_path = anat_dir / f"sub-{sub}_{ses}_T1w.json"
            if not json_path.exists():
                json_path.write_text(json.dumps(t1w_sidecar(), indent=2))

        # Func stubs
        for ses, run in func_runs:
            func_dir = STUB_DIR / f"sub-{sub}" / ses / "func"
            func_dir.mkdir(parents=True, exist_ok=True)
            base = f"sub-{sub}_{ses}_task-rest_{run}_bold"
            nii_path = func_dir / f"{base}.nii.gz"
            if not nii_path.exists():
                nii_path.write_bytes(stub_nii)
            json_path = func_dir / f"{base}.json"
            if not json_path.exists():
                json_path.write_text(json.dumps(bold_sidecar(), indent=2))

    # dataset_description.json
    desc = {
        "Name": "AD fmriprep stub",
        "BIDSVersion": "1.8.0",
    }
    (STUB_DIR / "dataset_description.json").write_text(json.dumps(desc, indent=2))

    # participants.tsv
    tsv = "participant_id\n" + "\n".join(participants) + "\n"
    (STUB_DIR / "participants.tsv").write_text(tsv)

    print(f"\nStub BIDS dataset created at: {STUB_DIR}")
    print(f"Participants: {', '.join(participants)}")


if __name__ == "__main__":
    main()
