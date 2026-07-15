import numpy as np, os
from collections import defaultdict

N_SITES = 121
TS_ROOT = "./timeseries"

pid_sessions = defaultdict(list)
for grp in ["CN","AD"]:
    folder = os.path.join(TS_ROOT, grp)
    for f in sorted(os.listdir(folder)):
        if f.endswith(".npy"):
            pid = f.split("_ses-")[0]
            arr = np.load(os.path.join(folder,f)).T   # (T, N_sites)
            if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
                pid_sessions[pid].append((arr, grp))

# sample one subject, show session stats
pids = sorted(pid_sessions.keys())
sess_counts = [len(pid_sessions[p]) for p in pids]
print(f"Total patients: {len(pids)}")
print(f"Sessions per patient: min={min(sess_counts)} max={max(sess_counts)} mean={np.mean(sess_counts):.1f}")
# show one session shape
s0, g0 = pid_sessions[pids[0]][0]
print(f"Example session shape (T, N_sites): {s0.shape}  group={g0}")
print(f"T range: {min(s[0].shape[0] for ss in pid_sessions.values() for s in ss)} - {max(s[0].shape[0] for ss in pid_sessions.values() for s in ss)}")
