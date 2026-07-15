#!/usr/bin/env python3
"""Convert AD/ADNI DICOM folder structure into a BIDS-formatted dataset."""

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

SERIES_MAP = {
    "MPRAGE": ("anat", "T1w"),
    "MP-RAGE": ("anat", "T1w"),
    "MP-RAGE_REPEAT": ("anat", "T1w"),
    "MPRAGE_REPEAT": ("anat", "T1w"),
    "Accelerated_Sagittal_MPRAGE": ("anat", "T1w"),
    "Resting_State_fMRI": ("func", "bold"),
    "Extended_Resting_State_fMRI": ("func", "bold"),
}


def get_series_info(series_name: str):
    if series_name in SERIES_MAP:
        return SERIES_MAP[series_name]
    normalized = series_name.replace("-", "_").lower()
    if "mprage" in normalized or "mp_rage" in normalized:
        return "anat", "T1w"
    if "rest" in normalized:
        return "func", "bold"
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert your AD/ADNI DICOM tree into BIDS format."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Root source folder containing subjects (e.g. AD/ADNI).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="BIDS root output folder.",
    )
    parser.add_argument(
        "--dcm2niix",
        default="dcm2niix",
        help="Path to the dcm2niix executable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversion operations without running dcm2niix.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing BIDS files if they already exist.",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source DICOM folders after successful conversion.",
    )
    return parser.parse_args()


def parse_session_label(folder_name: str) -> str:
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", folder_name)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"
    return folder_name.replace(" ", "_").replace("/", "_")


def make_dataset_description(bids_root: Path):
    description = {
        "Name": "AD/ADNI converted dataset",
        "BIDSVersion": "1.8.0",
        "License": "CC0",
        "GeneratedBy": [
            {
                "Name": "convert_to_bids.py",
                "Version": "1.0",
                "Description": "Custom DICOM-to-BIDS conversion helper",
            }
        ],
    }
    path = bids_root / "dataset_description.json"
    bids_root.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(description, fp, indent=2)
    print(f"Written {path}")


def make_participants_file(bids_root: Path, subjects):
    participants_path = bids_root / "participants.tsv"
    with participants_path.open("w", encoding="utf-8") as fp:
        fp.write("participant_id\n")
        for sub in sorted(subjects):
            fp.write(f"sub-{sub}\n")
    print(f"Written {participants_path}")


def run_dcm2niix(dcm2niix_cmd, source_dir: Path, out_dir: Path, filename: str, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        for existing in out_dir.glob(f"{filename}.*"):
            try:
                existing.unlink()
            except OSError:
                print(f"Warning: could not remove existing file {existing}")

    command = [
        dcm2niix_cmd,
        "-z",
        "y",
        "-o",
        str(out_dir),
        "-f",
        filename,
        str(source_dir),
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, check=True)


def delete_source_folder(source_dir: Path):
    try:
        shutil.rmtree(source_dir)
        print(f"Deleted source DICOM folder {source_dir}")
    except OSError as exc:
        print(f"Warning: could not delete source folder {source_dir}: {exc}")


def convert_series(series_path: Path, bids_root: Path, sub_label: str, session_label: str, series_name: str, dcm2niix_path: str, dry_run: bool, force: bool, run_index: int):
    series_info = get_series_info(series_name)
    if series_info is None:
        print(f"Skipping unknown series name: {series_name}")
        return None
    modality, suffix = series_info
    source_files = list(series_path.rglob("*.dcm"))
    if not source_files:
        print(f"Warning: no DICOM files found in {series_path}")
        return None

    dest_dir = bids_root / f"sub-{sub_label}" / f"ses-{session_label}" / modality
    name_base = f"sub-{sub_label}_ses-{session_label}"
    if modality == "func":
        run_label = f"run-{run_index:02d}"
        name_base = f"{name_base}_task-rest_{run_label}"
    output_filename = f"{name_base}_{suffix}"

    nii_path = dest_dir / f"{output_filename}.nii.gz"
    json_path = dest_dir / f"{output_filename}.json"
    if nii_path.exists() and json_path.exists() and not force:
        print(f"Skipping already converted series: {nii_path}")
        return output_filename

    if dry_run:
        print(f"DRY RUN: convert {series_path} -> {dest_dir}/{output_filename}.*")
        return output_filename

    run_dcm2niix(dcm2niix_path, series_path, dest_dir, output_filename, force)
    return output_filename


def main():
    args = parse_args()
    source_root = Path(args.source)
    bids_root = Path(args.output)

    if not source_root.exists():
        raise FileNotFoundError(f"Source folder not found: {source_root}")

    subjects = []
    for sub_dir in sorted(source_root.iterdir()):
        if not sub_dir.is_dir():
            continue
        print(f"Processing subject {sub_dir.name}")
        subjects.append(sub_dir.name)
        for series_name in sorted(sub_dir.iterdir()):
            if not series_name.is_dir():
                continue
            print(f"  series {series_name.name}")
            if series_name.name not in SERIES_MAP:
                print(f"Skipping unknown series folder: {series_name}")
                continue
            func_run = 1
            for date_dir in sorted(series_name.iterdir()):
                if not date_dir.is_dir():
                    continue
                session_label = parse_session_label(date_dir.name)
                dest_dir = bids_root / f"sub-{sub_dir.name}" / f"ses-{session_label}" / SERIES_MAP[series_name.name][0]
                if args.dry_run:
                    print(f"Would convert {date_dir} to BIDS session {session_label}")
                if series_name.name.startswith("Resting"):
                    converted = convert_series(date_dir, bids_root, sub_dir.name, session_label, series_name.name, args.dcm2niix, args.dry_run, args.force, func_run)
                    func_run += 1
                else:
                    converted = convert_series(date_dir, bids_root, sub_dir.name, session_label, series_name.name, args.dcm2niix, args.dry_run, args.force, func_run)
                if converted and args.delete_source and not args.dry_run:
                    delete_source_folder(date_dir)

    make_dataset_description(bids_root)
    make_participants_file(bids_root, subjects)
    print("BIDS conversion finished.")


if __name__ == "__main__":
    main()
