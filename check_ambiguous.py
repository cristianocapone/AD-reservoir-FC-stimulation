import sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ad_bids = Path('AD_bids')
fmri    = Path('fmriprep_output')

cases = [
    ('sub-130', 'ses-20121017', 'run-01', ['sub-130_S_4971','sub-130_S_4984']),
    ('sub-130', 'ses-20130125', 'run-02', ['sub-130_S_4971','sub-130_S_4982']),
]

for site, ses, run, pats in cases:
    print(f'\n{ses} {run}:')
    # fmriprep sidecar JSON
    fp_func = fmri / site / ses / 'func'
    fp_jsons = list(fp_func.glob(f'*{run}*bold.json')) if fp_func.exists() else []
    if fp_jsons:
        with open(fp_jsons[0]) as f:
            fp_meta = json.load(f)
        print(f'  fmriprep AcquisitionTime={fp_meta.get("AcquisitionTime","?")}')
    else:
        print(f'  fmriprep: no JSON in {fp_func}')

    # BIDS sidecars
    for pat in pats:
        func = ad_bids / pat / ses / 'func'
        jsons = list(func.glob(f'*{run}*bold.json'))
        if jsons:
            with open(jsons[0]) as f:
                meta = json.load(f)
            print(f'  {pat}: AcquisitionTime={meta.get("AcquisitionTime","?")}')
