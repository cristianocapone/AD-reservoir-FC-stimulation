"""
build_adni_date_map.py
======================
Build (site, YYYYMMDD) -> ADNI subject id from the authoritative
ADNI_DEFINITIVE_SELECTION download tree, so that timeseries files whose BIDS
subject id was truncated to the site code (sub-006, sub-013, ...) can be traced
back to the patient they actually came from.

ADNI layout:  ADNI/<SUBJECT>/<SeriesDescription>/<YYYY-MM-DD_HH_MM_SS.0>/<uid>/
Caches to adni_date_map.json so the slow directory walk runs once.
"""
import os, json, re

TREES = {"AD": r"D:/2025.AD/ADNI_DEFINITIVE_SELECTION/AD/ADNI",
         "CN": r"D:/2025.AD/ADNI_DEFINITIVE_SELECTION/CN/ADNI"}
OUT = "adni_date_map.json"

date_map = {}          # "site|YYYYMMDD" -> {ADNI ids}
label_of = {}          # ADNI id -> "AD" / "CN"

for group, root in TREES.items():
    subs = sorted(s for s in os.listdir(root) if "_S_" in s)
    print(f"{group}: scanning {len(subs)} subjects ...")
    for i, sub in enumerate(subs):
        label_of[sub] = group
        site = sub.split("_S_")[0]
        sd = os.path.join(root, sub)
        try:
            series = os.listdir(sd)
        except OSError:
            continue
        for ser in series:
            if "fmri" not in ser.lower() and "rest" not in ser.lower():
                continue
            try:
                dates = os.listdir(os.path.join(sd, ser))
            except OSError:
                continue
            for dt in dates:
                m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", dt)
                if m:
                    date_map.setdefault(f"{site}|{''.join(m.groups())}", set()).add(sub)
        if (i + 1) % 50 == 0:
            print(f"   {i+1}/{len(subs)}")

amb = sum(1 for v in date_map.values() if len(v) > 1)
print(f"\n{len(date_map)} (site, date) keys; {amb} ambiguous (>1 subject)")
json.dump({"date_map": {k: sorted(v) for k, v in date_map.items()},
           "label_of": label_of}, open(OUT, "w"))
print(f"Saved {OUT}")
