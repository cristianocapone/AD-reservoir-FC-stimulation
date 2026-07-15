# Parcellated fMRI Timeseries — ADNI Motion-Corrected Dataset
Generated: 2026-05-12

## Overview
Resting-state fMRI BOLD timeseries parcellated with:
  - Schaefer-100 atlas (7-network, 100 cortical parcels)
  - Harvard-Oxford subcortical atlas (14 subcortical regions)
  Total: 114 parcels

Motion correction: fMRIPrep 25.2.5 (ICA-AROMA + confound regression)
Timeseries extraction: Nilearn NiftiLabelsMasker, standardized

## File format
Each .npy file: NumPy array, shape = (114, 140)
  - Axis 0: parcels (see parcel_labels.txt for names)
  - Axis 1: TRs  (TR = 2.5 s, 140 volumes = 350 s acquisition)

Files with non-standard T (200 TRs, extended acquisitions) are excluded.
Files with incomplete parcellations (N < 114) are excluded.

## Subjects
  AD (Alzheimer's Disease):  151 sessions
  MCI (Mild Cognitive Impairment): 121 sessions
  CN (Cognitively Normal):   561 sessions
  Total:                     833 sessions

## File naming
  <SubjectID>_<SessionID>.npy
  SubjectID format: sub-NNNSNNNNN  (ADNI standard)

## Parcels
See parcel_labels.txt for the ordered list of 114 parcel names.
Parcels 1-100: Schaefer-100 cortical (7-network parcellation)
Parcels 101-114: Harvard-Oxford subcortical

## Analysis notebook
metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML___MC.ipynb
  Reservoir computing + LDA/SVM classification (AD vs CN, two-class)
  Runs on the timeseries in this archive.
