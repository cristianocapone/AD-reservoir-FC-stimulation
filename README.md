# From read-out geometry to in-silico stimulation

Analysis and figure-generation code for the manuscript:

> **From read-out geometry to in-silico stimulation: a distributed
> functional-connectivity signature of Alzheimer's disease.**
> C. Capone, E. Cece, A. Ciardiello, G. Gigante, M. Mattia, E. Cisbani.

We fit subject-specific reservoir-computing models to resting-state fMRI from
Alzheimer's disease (AD) patients and cognitively unimpaired controls, use them
to classify AD from controls via two complementary read-outs (read-out geometry
and reconstructed lagged functional connectivity), and use the same models as an
in-silico testbed for focal and closed-loop stimulation.

## Data availability

The imaging data were downloaded from the **Alzheimer's Disease Neuroimaging
Initiative (ADNI)** database (<https://adni.loni.usc.edu>). ADNI data are
available to qualified researchers upon registration and acceptance of the ADNI
Data Use Agreement. **In accordance with that agreement, no raw or derived
imaging data are redistributed in this repository** — only code. Qualified
researchers can obtain the same data directly from ADNI and reproduce the
derived time series with the preprocessing scripts included here.

The data directories referenced by the scripts (BIDS trees, parcellated time
series, caches, and large `.npz`/`.pkl`/`.zip` artefacts) are intentionally
excluded via `.gitignore`.

## Repository layout

- `paper/` — LaTeX manuscript (`main.tex`), bibliography (`refs.bib`), and the
  compiled `main.pdf`.
- `paper_figures/` — the figure PDFs used by the manuscript.
- `*.py` — analysis and figure-generation scripts, grouped by theme:
  - `adni_to_bids.py`, `convert_to_bids.py`, `motion_correct.py`,
    `extract_timeseries*.py` — data conversion and preprocessing.
  - `Fig1DEF_*`, `classification_comparison.py`, `cc_vs_ad_classifier.py`,
    `tangent_fc_classifier.py` — dataset, model fit, and classification.
  - `pert*.py`, `perturbation_*.py` — in-silico stimulation and closed-loop
    control experiments.
- `*.ipynb` — exploratory / summary notebooks.

## Requirements

Python 3 with NumPy, SciPy, scikit-learn, nilearn, and Matplotlib. Preprocessing
additionally uses a BIDS/fMRIPrep toolchain (see `BIDS_CONVERSION.md`).

## Citation

If you use this code, please cite the manuscript above. Data used in preparation
of this work were obtained from the ADNI database (adni.loni.usc.edu); the ADNI
investigators contributed to the design and implementation of ADNI and/or
provided data but did not participate in the analysis or writing of this report.
