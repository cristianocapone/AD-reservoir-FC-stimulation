# Scripts

Analysis and figure-generation code, organised by the manuscript figure each
script feeds. Compiled figure PDFs live in `../paper_figures/` (referenced by the
LaTeX); only the generating `.py` scripts live here.

## Layout

| Folder | Contents |
|--------|----------|
| `common/` | Shared modules imported by other scripts: `res.py` (reservoir core), `external_oasis_validate.py`, `pert_compare3.py`. |
| `fig1_data/` | Dataset composition, FC group structure, reconstruction quality. |
| `fig2_model/` | Reservoir fit, read-out geometry, FC reconstruction sweeps. |
| `fig3_classification/` | AD-vs-control classification, learning curves, K/σ sweeps. |
| `fig4_stimulation/` | Full-W vs focal stimulation, dose-response, oscillatory drives. |
| `fig5_singlecompare/` | Single-site read-out correction vs resonant drive, closed-loop control. |
| `fig6_topsites/` | Pathology (ΔW) sites vs discriminant-aligned therapy sites. |
| `fig_supplement/` | Supplementary figures (scaling, physiology, frequency scans, etc.). |
| `reanalysis/` | Cohort rebuild from authoritative ADNI labels, ΔW null test, structural-MRI and fusion classifiers, LOSO/batch checks. |
| `preprocessing/` | BIDS conversion, timeseries extraction, filename/ID fixes. |
| `misc/` | Exploratory / one-off inspection and QC scripts. |

## Running

Run every script **from the repository root**, e.g.

```bash
python scripts/reanalysis/struct_gm_classify.py
```

Two reasons:

1. Scripts read and write their `.npz`/`.pkl` cache files by paths relative to the
   working directory, which is assumed to be the repo root.
2. Scripts that import a shared module (`res`, `external_oasis_validate`,
   `pert_compare3`) carry a small auto-added path bootstrap at the top that puts
   `scripts/common/` on `sys.path`; it locates the folder relative to the file, so
   it works from any subfolder as long as the repo layout is intact.

## Data

Imaging data (ADNI) and all derived caches (`*.npz`, `*.npy`, `*.pkl`, BIDS trees)
are **not** in the repository — they are governed by the ADNI Data Use Agreement.
Obtain them from [adni.loni.usc.edu](https://adni.loni.usc.edu) and regenerate the
caches with the `preprocessing/` and analysis scripts.
