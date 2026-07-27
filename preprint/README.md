# From read-out geometry to in-silico stimulation: a distributed functional-connectivity signature of Alzheimer's disease

Preprint source and compiled PDF.

**Status:** Preprint — not yet peer reviewed.

## Contents

| File | Description |
|------|-------------|
| `main.tex` | Manuscript source (LaTeX, `article` class) |
| `main.pdf` | Compiled preprint (35 pp., incl. Supplementary Information) |
| `refs.bib` | Bibliography (BibTeX) |
| `naturemag.bst` | Nature-house reference style |
| `main.bbl` | Pre-built bibliography (lets `main.tex` compile with `pdflatex` alone) |
| `figures/` | The 16 figure PDFs referenced by the manuscript |

## Building

Self-contained. With a TeX distribution (TeX Live / MiKTeX):

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

`main.bbl` is included, so a single `pdflatex main` also suffices if you don't
want to run BibTeX.

## Data

Data were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI,
[adni.loni.usc.edu](https://adni.loni.usc.edu)); see the manuscript
Acknowledgments. ADNI data are subject to ADNI's data-use agreement and are not
redistributed here.
