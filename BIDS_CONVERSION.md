# Convert AD/ADNI DICOM to BIDS

This workspace contains a helper script to convert the AD dataset structure into a BIDS dataset.

## What it supports

- `MPRAGE` → `anat/sub-<label>_ses-<date>_T1w.nii.gz`
- `Resting_State_fMRI` and `Extended_Resting_State_fMRI` → `func/sub-<label>_ses-<date>_task-rest_run-<nn>_bold.nii.gz`

## Requirements

- A working Python environment
- A local conda environment named `bids_env` created in the workspace
- `dcm2niix` installed in `bids_env`

## Usage

From the repository root, run the launcher script:

```powershell
run_convert_to_bids.bat --source .\AD\ADNI --output .\AD_bids
```

This wrapper uses the local environment if it exists:

- `.\bids_env\Library\bin\dcm2niix.exe` on Windows
- otherwise it falls back to the global Python interpreter

For a dry run:

```powershell
run_convert_to_bids.bat --source .\AD\ADNI --output .\AD_bids --dry-run
```

To continue conversion without reprocessing already converted sessions:

```powershell
run_convert_to_bids.bat --source .\AD\ADNI --output .\AD_bids
```

To delete DICOM source folders after successful conversion:

```powershell
run_convert_to_bids.bat --source .\AD\ADNI --output .\AD_bids --delete-source
```

To run directly with the local environment:

```powershell
.\bids_env\python.exe convert_to_bids.py --source .\AD\ADNI --output .\AD_bids --dcm2niix .\bids_env\Library\bin\dcm2niix.exe
```

## Notes

- The script uses the folder names to create `sub-` and `ses-` labels.
- A `dataset_description.json` and `participants.tsv` file are created automatically.
- After conversion, validate with the BIDS Validator:

```powershell
bids-validator .\AD_bids
```
