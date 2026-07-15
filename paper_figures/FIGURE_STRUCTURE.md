# Paper Figure Structure
_Last updated: 2026-06-03_

## Overview
This folder contains the four main paper figures + supporting information figure.
All figures are 300 DPI, consistent font (sans-serif, 9–10 pt labels), no top/right spines.
Panel labels use large bold uppercase letters (A, B, C, D / E, F).

---

## Main Figures

### Figure 1 — Data: Dataset Overview
**File:** `figure1_data.png`  
**Script:** `figure1_data.py`  
**Reads:** `../g_space_cache.npz`, `../fc_recon_noise_sweep.npz`, `../analysis/fig4_fc_group_means.png`, `../timeseries/CN/`, `../timeseries/AD/`

| Panel | Content |
|-------|---------|
| A | Dataset composition: CC (n=143 patients, 561 sessions) vs AD (n=40, 151 sessions) |
| B | Sessions-per-patient histogram (CC vs AD, with median lines) |
| C | Group-mean FC matrix comparison (embedded from `analysis/fig4_fc_group_means.png`) |
| D | W-matrix FC reconstruction quality vs noise level σ (from `fc_recon_noise_sweep.npz`) |

**Key stats:** 183 total patients, 712 total sessions. Optimal noise σ=0.5, mean FC-r=0.68.

---

### Figure 2 — Model: G-space Representation
**File:** `figure2_model.png`  
**Script:** `figure2_model.py`  
**Reads:** `../g_space_cache.npz`

| Panel | Content |
|-------|---------|
| A | G-space PC1 vs PC2 scatter — all 183 patients (CC=blue, AD=red, class means=X) |
| B | Cumulative variance explained by G-space PCs; vertical marker at K=15 (optimal) |
| C | PC1 score distribution per class (histogram + KDE + t-test annotation) |
| D | G-space PC1 vs PC3 scatter (complementary view) |

**Key stats:** K=15 PCs explain 11.1% variance (patient G-space). G-space derived from 702-session Gram matrix (2000-unit reservoir, SR=0.95, ff=0.1).

---

### Figure 3 — Classification: LDA G-space Trajectory
**File:** `figure3_classification.png`  
**Script:** `figure3_classification.py`  
**Reads:** `../g_space_cache.npz`, `../rf_results.npz`

| Panel | Content |
|-------|---------|
| A–D | G-space PC1/PC2 at N=10, 20, 30, 40 patients per class (background=all, foreground=sampled) |
|     | LDA direction arrow + approx. decision boundary (dashed), LOPO BAL-ACC/AUROC in corner |
| E | LDA LOPO BAL-ACC learning curve (mean ± std, 30 reps) with orange markers at displayed N |
| F | LDA LOPO AUROC learning curve (same) |

**Key stats at N=40:** BAL-ACC=0.675±0.052, AUROC=0.764±0.047.  
Best N: ~30–40 (plateau region). LDA outperforms RF (see Supplementary).  
LOPO CV: leave-one-patient-out; orientation fix applied (re-evaluate z_tr after w flip).

---

### Figure 4 — Stimulation: In-silico W-matrix Intervention
**File:** `figure4_stimulation.png`  
**Script:** `figure4_stimulation.py`  
**Reads:** `../pertB_direct_data.npz`

| Panel | Content |
|-------|---------|
| A | Baseline LDA score distributions — CC (n=36) vs AD (n=40), violin + jittered points, MWU test |
| B | Dose-response: mean LDA score vs alpha for all 3 strategies (Full-W, Top-5, Top-1) |
| C | Per-patient AD trajectories — Top-5 perturbation |
| D | Fraction AD patients reclassified as CC vs alpha (all 3 strategies) |

**Perturbation model:** `W_int = (1-α)·W_AD + α·W̄_CC`  
**Strategies:**  
- `full_w`: all 121 brain parcels perturbed (alphas 0–2)  
- `top5`: 5 most-deviant parcels per patient (alphas 0–5)  
- `top1`: single most-deviant parcel per patient (alphas 0–10)  

**Key result:** Full-W at α=1 shifts mean AD LDA score to ~CC territory. Top-5 requires larger α but is more targeted.

---

## Supporting Information

### Supp. Figure S — Classification Details
**File:** `suppfig_curves.png`  
**Script:** `suppfig_curves.py`  
**Reads:** `../rf_results.npz`, `../g_space_cache.npz`

| Panel | Content |
|-------|---------|
| S1a | LDA BAL-ACC learning curve (detailed, all N with point annotations) |
| S1b | LDA AUROC learning curve |
| S2a | K-sweep BAL-ACC — LDA vs RF-100 at N=40, with % variance annotations, optimal K marker |
| S2b | K-sweep AUROC — same |
| S3a | LDA vs RF-100 vs RF-reg BAL-ACC comparison across all N |
| S3b | LDA vs RF-100 vs RF-reg AUROC comparison across all N |

**Key result:** LDA consistently outperforms RF (at N=40: LDA BAL-ACC=0.640 vs RF=0.585). G-space is linearly structured; non-linear classifiers offer no benefit. Optimal K=15.

---

## Data Flow Summary

```
timeseries/CN/*.npy + timeseries/AD/*.npy  (712 sessions)
    │
    ├─ Population PCA (50 PCs, N_SITES=121)
    ├─ Reservoir (N=2000, SR=0.95, ff=0.1, T=139)
    ├─ W-matrix per session (T×N_sites ridge regression, σ=0.05)
    ├─ Session SVD projection (K_PC=200)
    ├─ 702×702 Gram matrix → eigenvectors
    └─ G_pat_full (183×38)  ──→  g_space_cache.npz
                │
                ├─── Figure 2: G-space visualization
                ├─── Figure 3: LOPO-LDA classification (K=15)
                └─── rf_results.npz  ──→  Figure 3 learning curve + Supp S

Perturbation:  W_int = (1-α)·W_AD + α·W̄_CC
    ──→  pertB_direct_data.npz  ──→  Figure 4

FC reconstruction:  W_int.T @ X  vs  CC template
    ──→  fc_recon_noise_sweep.npz  ──→  Figure 1D
```

---

## Regeneration Commands
From the `paper_figures/` directory:
```bash
python figure1_data.py
python figure2_model.py
python figure3_classification.py
python figure4_stimulation.py
python suppfig_curves.py
```

All scripts run in < 30 s (no reservoir pass needed; cache files must exist).  
To rebuild the cache: run `../rf_lc.py` (takes ~30–60 min for full TF pass).

---

## Color Palette
| Group | Hex |
|-------|-----|
| CC (Cognitively Unimpaired) | `#1565C0` (deep blue) |
| AD (Alzheimer's Disease) | `#C62828` (deep red) |
| LDA performance | `#1565C0` |
| RF-100 performance | `#2E7D32` (dark green) |
| RF-reg performance | `#7B1FA2` (dark purple) |
| Full-W perturbation | `#1B5E20` (forest green) |
| Top-5 perturbation | `#E65100` (burnt orange) |
| Top-1 perturbation | `#6A1B9A` (deep purple) |
| Optimal K marker | `#E65100` |
