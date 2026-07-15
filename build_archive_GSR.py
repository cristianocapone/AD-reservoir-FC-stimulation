"""Build parcellated_data_MC_GSR.zip — timeseries with Global Signal Regression."""
import json, zipfile, sys, numpy as np
from pathlib import Path
from datetime import date

sys.stdout.reconfigure(encoding='utf-8')

BASE = Path(r"C:\Users\user\Desktop\2026.AD_MotionCorrection")
TS   = BASE / "timeseries_GSR"
OUT  = BASE / "parcellated_data_MC_GSR.zip"

# ── Inventory ─────────────────────────────────────────────────────────────────
groups = {}
for grp in ("AD", "MCI", "CN"):
    folder = TS / grp
    files  = sorted(folder.glob("*.npy"))
    good   = [f for f in files if np.load(f).shape == (121, 140)]
    groups[grp] = {"all": files, "good": good}

ad_full = [f for f in groups["AD"]["good"] if "_S_" in f.name]
ad_site = [f for f in groups["AD"]["good"] if "_S_" not in f.name]
ad_pats = sorted(set(f.name.split("_ses-")[0] for f in ad_full))

# ── Dataset description ───────────────────────────────────────────────────────
desc = {
    "Name": "AD Motion-Corrected fMRI Parcel Timeseries (with GSR)",
    "Date": str(date.today()),
    "Description": (
        "Resting-state fMRI parcel timeseries extracted from fmriprep 25.2.5 "
        "motion-corrected BOLD data (ADNI dataset). "
        "Atlas: Schaefer-100 (7-network, 100 cortical parcels) + "
        "Harvard-Oxford subcortical (all 21 labels). "
        "Total parcels: 121. "
        "Global Signal Regression (GSR) applied: global_signal and "
        "global_signal_derivative1 included as confounds."
    ),
    "Atlas": {
        "cortical": "Schaefer-2018, 100 ROIs, 7 Networks, 2mm",
        "subcortical": "Harvard-Oxford sub-maxprob-thr25-2mm, all 21 labels",
        "total_parcels": 121,
        "label_file": "timeseries/parcel_labels.txt"
    },
    "Preprocessing": {
        "software": "fmriprep 25.2.5",
        "confounds": (
            "24 HMP (6 params + derivatives + quadratics) + "
            "CSF + WM + derivatives + "
            "global_signal + global_signal_derivative1 (GSR)"
        ),
        "bandpass_Hz": [0.01, 0.1],
        "standardize": "zscore_sample",
        "detrend": True,
        "GSR": True
    },
    "FileFormat": {
        "shape": "(N_parcels=121, T=140)",
        "dtype": "float32",
        "note": "Files with T!=140 or N!=121 excluded (extended runs or incomplete parcellation)"
    },
    "Groups": {
        "AD": {
            "total_files": len(groups["AD"]["all"]),
            "usable_121x140": len(groups["AD"]["good"]),
            "identified_patients": len(ad_pats),
            "identified_patient_ids": ad_pats,
            "site_only_sessions": len(ad_site),
            "site_only_note": "Sessions from AD patients not in AD_bids; need ADNIMERGE.csv to resolve full IDs"
        },
        "MCI": {
            "total_files": len(groups["MCI"]["all"]),
            "usable_121x140": len(groups["MCI"]["good"]),
        },
        "CN": {
            "total_files": len(groups["CN"]["all"]),
            "usable_121x140": len(groups["CN"]["good"]),
        }
    },
    "TotalUsableSessions": sum(len(g["good"]) for g in groups.values()),
    "FilenameConvention": {
        "AD_identified": "sub-{SITE}_S_{ID}_ses-{DATE}_task-rest[_run-{N}]_bold_timeseries.npy",
        "AD_unresolved": "sub-{SITE}_ses-{DATE}_task-rest[_run-{N}]_bold_timeseries.npy",
        "MCI_CN": "sub-{ADNI_ID}_ses-{DATE}_task-rest[_run-{N}]_bold_timeseries.npy"
    },
    "CompareWith": "parcellated_data_MC.zip — identical pipeline without GSR"
}

# ── Build zip ─────────────────────────────────────────────────────────────────
total = 0
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
    zf.writestr("dataset_description.json",
                json.dumps(desc, indent=2, ensure_ascii=False))
    print("  + dataset_description.json")

    zf.write(TS / "parcel_labels.txt", "timeseries/parcel_labels.txt")
    print("  + timeseries/parcel_labels.txt")

    for grp in ("AD", "MCI", "CN"):
        for f in groups[grp]["all"]:
            zf.write(f, f"timeseries/{grp}/{f.name}")
            total += 1

    print(f"  + {total} timeseries files")

size_mb = OUT.stat().st_size / 1e6
print(f"\nSaved: {OUT.name}  ({size_mb:.1f} MB)")
print(f"Usable (121x140): {sum(len(g['good']) for g in groups.values())} sessions")
print(f"  AD={len(groups['AD']['good'])}  MCI={len(groups['MCI']['good'])}  CN={len(groups['CN']['good'])}")
