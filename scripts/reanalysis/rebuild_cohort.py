"""
rebuild_cohort.py
=================
Rebuild the CN/AD cohort with authoritative diagnostic labels and a single
preprocessing batch per subject.

Why this is needed
------------------
timeseries/CN was extracted from the download D:/ADNI_definitivo_2_CN, whose name
is misleading: it holds 2071 subjects and is not filtered by diagnosis. It
contains 62 of the 73 AD subjects, so 17 AD patients (those passing the volume
filter) ended up in the control group under a different filename convention,
carrying both labels at identical session dates.

The authoritative labels come from D:/2025.AD/ADNI_DEFINITIVE_SELECTION, whose
AD (73) and CN (377) subject lists are disjoint - these are ADNI's own
diagnosis-filtered group queries.

Two independent defects are fixed here:
  1. LABELS  - every session is relabelled from the DEFINITIVE lists; sessions
     whose subject is in neither list are dropped rather than guessed.
  2. BATCH   - timeseries/CN and timeseries/AD are two preprocessing batches of
     overlapping scans. Diagnosis was perfectly confounded with batch. Each
     subject is now assigned ONE batch, preferring the CN batch (larger and
     shared by both groups) so that batch no longer predicts diagnosis.

Also recovers the 7 truncated BIDS ids (sub-006, sub-013, ...) that collapsed
several patients from one site into a single pseudo-subject, by matching each
file's session date against AD_bids.

Writes cohort_clean.npz: one row per retained session, with subject, adni id,
label, site, batch and source path. No timeseries are copied or modified.
"""
import os, re, json
import numpy as np

DEF_AD = r"D:/2025.AD/ADNI_DEFINITIVE_SELECTION/AD/ADNI"
DEF_CN = r"D:/2025.AD/ADNI_DEFINITIVE_SELECTION/CN/ADNI"
MIN_VOL, N_PARC = 139, 121
OUT = "cohort_clean.npz"


def adni_ids(path):
    return set(x for x in os.listdir(path) if "_S_" in x)


def to_adni(sub):
    """'sub-002S0295' or 'sub-002_S_5018' -> '002_S_0295' / '002_S_5018'."""
    s = sub.replace("sub-", "")
    if "_S_" in s:
        return s
    m = re.match(r"^(\d+)S(\d+)$", s)
    return f"{m.group(1)}_S_{m.group(2)}" if m else None


def session_of(fn):
    m = re.search(r"_ses-(\d+)", fn)
    return m.group(1) if m else None


# ── authoritative labels ──────────────────────────────────────────────────────
AD_IDS, CN_IDS = adni_ids(DEF_AD), adni_ids(DEF_CN)
assert not (AD_IDS & CN_IDS), "DEFINITIVE lists overlap - labels not trustworthy"
print(f"authoritative labels: {len(CN_IDS)} CN, {len(AD_IDS)} AD (disjoint)")


# ── recover truncated ids by (site, scan date) against the ADNI tree ──────────
# adni_date_map.json is built by build_adni_date_map.py from the authoritative
# ADNI_DEFINITIVE_SELECTION download; AD_bids cannot be used for this because
# only 2 of the 47 truncated sessions appear in it.
_DM = json.load(open("adni_date_map.json"))
SES_MAP = {k: set(v) for k, v in _DM["date_map"].items()}


def resolve(sub, fn):
    """Return the ADNI id for a timeseries file, recovering truncated ids.
    A truncated id is only accepted when the (site, date) key is unambiguous."""
    aid = to_adni(sub)
    if aid:
        return aid, False
    site = sub.replace("sub-", "")            # truncated: site code only
    cand = SES_MAP.get(f"{site}|{session_of(fn)}", set())
    return (sorted(cand)[0], True) if len(cand) == 1 else (None, True)


# ── scan both batches ─────────────────────────────────────────────────────────
rows, unresolved, unlabelled = [], 0, 0
for batch, folder in [("CN", "timeseries/CN"), ("AD", "timeseries/AD")]:
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".npy"):
            continue
        sub = fn.split("_ses-")[0]
        aid, was_trunc = resolve(sub, fn)
        if aid is None:
            unresolved += 1
            continue
        if aid in AD_IDS:
            lab = 1
        elif aid in CN_IDS:
            lab = 0
        else:
            unlabelled += 1
            continue
        p = os.path.join(folder, fn)
        a = np.load(p, mmap_mode="r")
        nvol = a.shape[1] if a.shape[0] == N_PARC else a.shape[0]
        if (a.shape[0] != N_PARC and a.shape[1] != N_PARC) or nvol < MIN_VOL:
            continue
        rows.append((aid, lab, aid.split("_S_")[0], batch, p,
                     session_of(fn), nvol, was_trunc))

print(f"scanned sessions: {len(rows)} retained, "
      f"{unresolved} unresolvable truncated ids, "
      f"{unlabelled} not in either authoritative list")

aid = np.array([r[0] for r in rows]); lab = np.array([r[1] for r in rows])
site = np.array([r[2] for r in rows]); batch = np.array([r[3] for r in rows])
path = np.array([r[4] for r in rows]); ses = np.array([r[5] for r in rows])
nvol = np.array([r[6] for r in rows]); trunc = np.array([r[7] for r in rows])

# ── how the old labelling compares ────────────────────────────────────────────
old = np.where(batch == "AD", 1, 0)
print(f"\nrelabelled sessions (old folder label != authoritative): "
      f"{(old != lab).sum()} of {len(lab)}")
print(f"  sessions in the CN folder that are actually AD: "
      f"{((batch=='CN') & (lab==1)).sum()}")
print(f"  recovered truncated-id sessions: {trunc.sum()}")

# ── one batch per subject: prefer CN batch, so batch cannot predict label ─────
keep = np.zeros(len(rows), bool)
for a in np.unique(aid):
    m = aid == a
    pref = "CN" if (m & (batch == "CN")).any() else "AD"
    keep |= m & (batch == pref)

print(f"\nafter one-batch-per-subject: {keep.sum()} sessions")
for lb, nm in [(0, "CN"), (1, "AD")]:
    m = keep & (lab == lb)
    frac = (batch[m] == "CN").mean()
    print(f"  {nm}: {m.sum():4d} sessions, {len(np.unique(aid[m])):3d} subjects, "
          f"{frac*100:.0f}% from the CN preprocessing batch")

np.savez(OUT, adni_id=aid[keep], label=lab[keep], site=site[keep],
         batch=batch[keep], path=path[keep], session=ses[keep],
         nvol=nvol[keep], recovered=trunc[keep])
# the unfiltered table keeps both batches per subject, which the batch-isolation
# test needs in order to score the same patient through each preprocessing batch
np.savez("cohort_clean_allbatches.npz", adni_id=aid, label=lab, site=site,
         batch=batch, path=path, session=ses, nvol=nvol, recovered=trunc)
print(f"\nSaved {OUT} and cohort_clean_allbatches.npz")
