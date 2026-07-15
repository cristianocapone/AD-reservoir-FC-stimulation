"""
Rename timeseries/AD/*.npy files from site-level IDs (sub-006_ses-...)
to full ADNI patient IDs (sub-006_S_4546_ses-...).

Mapping built from AD_bids/ directory structure:
  AD_bids/sub-006_S_4546/ses-20120601/func/sub-006_S_4546_ses-20120601_task-rest_run-02_bold.json
  -> (site=sub-006, ses=ses-20120601, run=run-02) -> sub-006_S_4546

The 2 truly ambiguous (site, ses, run) triples are flagged.
"""
import sys, re, shutil
from pathlib import Path
from collections import defaultdict
sys.stdout.reconfigure(encoding='utf-8')

AD_BIDS = Path('AD_bids')
TS_AD   = Path('timeseries/AD')

# ── Build (site, ses, run) -> patient_id ──────────────────────────────────────
mapping = {}   # (site, ses, run) -> patient_id  (None = ambiguous)
ambiguous = set()

for pat_dir in sorted(AD_BIDS.iterdir()):
    if not pat_dir.is_dir() or '_S_' not in pat_dir.name:
        continue
    pat_id = pat_dir.name           # sub-006_S_4546
    site   = 'sub-' + pat_id.split('_S_')[0].replace('sub-', '')  # sub-006

    for ses_dir in pat_dir.iterdir():
        if not (ses_dir.is_dir() and ses_dir.name.startswith('ses-')):
            continue
        ses = ses_dir.name
        func = ses_dir / 'func'
        if not func.exists():
            continue
        # Collect run numbers from BOLD json/nii files
        runs = set()
        for f in func.iterdir():
            m = re.search(r'(run-\d+)', f.name)
            runs.add(m.group(1) if m else 'run-01')

        for run in runs:
            key = (site, ses, run)
            if key in mapping:
                # Collision
                ambiguous.add(key)
                mapping[key] = None
            else:
                mapping[key] = pat_id

print(f'Mapping entries: {len(mapping)}')
print(f'Ambiguous keys:  {len(ambiguous)}')
for k in sorted(ambiguous):
    print(f'  AMBIGUOUS: {k}')

# ── Rename timeseries/AD files ────────────────────────────────────────────────
# Filename pattern: sub-006_ses-20120601_task-rest_run-02_bold_timeseries.npy
# OR (no run):      sub-006_ses-20120601_task-rest_bold_timeseries.npy

renamed = skipped = ambig_count = 0

for npy in sorted(TS_AD.glob('*.npy')):
    name = npy.stem  # e.g. sub-006_ses-20120601_task-rest_run-02_bold_timeseries
    parts = name.split('_')

    site = parts[0]              # sub-006
    # find ses-
    ses  = next((p for p in parts if p.startswith('ses-')), None)
    run_m = next((p for p in parts if p.startswith('run-')), 'run-01')

    key = (site, ses, run_m)
    pat_id = mapping.get(key)

    if pat_id is None:
        if key in ambiguous:
            new_name = npy.name.replace(site + '_', site + '_S_AMBIGUOUS_', 1)
            npy.rename(TS_AD / new_name)
            print(f'  AMBIGUOUS: {npy.name} -> {new_name}')
            ambig_count += 1
        else:
            print(f'  UNMAPPED:  {npy.name}  key={key}')
            skipped += 1
    else:
        # Replace site prefix with full patient ID
        new_name = npy.name.replace(site + '_', pat_id + '_', 1)
        if new_name != npy.name:
            npy.rename(TS_AD / new_name)
            renamed += 1

print(f'\nDone: renamed={renamed}  ambiguous={ambig_count}  unmapped={skipped}')
