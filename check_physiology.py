"""
check_physiology.py
===================
For the two "extreme" perturbed W conditions:
  A) Full-W interpolation   alpha=1.0  (W = W_cc_mean)
  B) Top-5-site             alpha=10.0 (5 cols pushed 10x past W_cc_mean)

For every AD patient compute:
  - Signal: mean, std, max |amplitude|  (divergence check)
  - Simulated FC-r with CC-mean FC        (template match)
  - Simulated FC-r with patient's OWN empirical FC  (self-match)
  - Simulated FC-r with the ORIGINAL alpha=0 simulated FC  (how much it changed)

Also saves a figure with 4 example FC matrices per condition (one AD patient).
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── Config (must match perturbation_perpatient.py) ─────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
NOISE_SIZE  = 0.025
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

# ── Load data ──────────────────────────────────────────────────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, patient_ids_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            patient_ids_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

patient_ids_raw = np.array(patient_ids_raw)
labels_raw      = np.array(labels_raw)
unique_pids     = np.unique(patient_ids_raw)
N_patients      = len(unique_pids)
patient_sids    = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
patient_labels  = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
cc_idx          = np.where(patient_labels == 0)[0]
ad_idx          = np.where(patient_labels == 1)[0]
N_subj          = len(signals)
print(f"  Patients: {N_patients}  (CC={len(cc_idx)}, AD={len(ad_idx)})")

# ── Population PCA ─────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Reservoir ─────────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

# ── Teacher-forced pass ────────────────────────────────────────────────────────
print("Teacher-forced pass ...")
sess_X, sess_Y, sess_target = {}, {}, {}
for idx in trange(N_subj, desc="  TF"):
    s      = signals[idx]
    T_s    = s.shape[1]
    tgt    = (s.T @ ev50 @ ev50.T).T
    res.T  = T_s; res.reset()
    X_raw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    X_fit = np.array(X_raw)[TIMES_SKIP:]
    Y_fit = tgt[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T
    sess_X[idx]      = X_fit
    sess_Y[idx]      = Y_fit
    sess_target[idx] = tgt

pat_target = {pid: np.concatenate([sess_target[i] for i in patient_sids[pid]], axis=1)
              for pid in unique_pids}

# Per-patient empirical FC (from PCA-projected signal)
pat_fc_emp = {}
for pid in unique_pids:
    tc = np.concatenate([sess_target[i] for i in patient_sids[pid]], axis=1)
    pat_fc_emp[pid] = np.nan_to_num(np.corrcoef(tc[:, TIMES_SKIP:]))

# ── Per-patient W ──────────────────────────────────────────────────────────────
print(f"Per-patient W (noise={NOISE_SIZE}) ...")
rng_p  = np.random.default_rng(RNG_SEED + 1)
pat_W  = {}
pat_X  = {}
for pid in tqdm(unique_pids, desc="  Fitting W"):
    idxs   = patient_sids[pid]
    X_coll = np.vstack([sess_X[i] for i in idxs])
    Y_coll = np.vstack([sess_Y[i] for i in idxs])
    noise  = rng_p.normal(0, NOISE_SIZE, X_coll.shape)
    pat_W[pid] = np.linalg.pinv(X_coll + noise) @ Y_coll
    pat_X[pid] = X_coll

# CC mean W
W_cc_mean = np.mean([pat_W[unique_pids[i]] for i in cc_idx], axis=0)

# CC mean FC (closed-loop)
IU = np.triu_indices(N_SITES, k=1)
fc_cc_list = []
for i in tqdm(cc_idx, desc="CC closed-loop FC", leave=False):
    pid = unique_pids[i]
    tgt = pat_target[pid]; T = tgt.shape[1]
    res.T = T; res.reset()
    for t in range(T - 1): res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
    res.Jout = pat_W[pid].T.copy(); res.y = res.Jout @ res.X
    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())
    Y_sim = np.array(Y_sim)[TIMES_SKIP:].T
    fc_cc_list.append(np.nan_to_num(np.corrcoef(Y_sim)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)
FC_cc_mean_mat = FC_cc_mean.reshape(N_SITES, N_SITES)

# ── Helper: run closed-loop, return signal + FC ────────────────────────────────
def run_closed_loop(W, pid):
    tgt = pat_target[pid]; T = tgt.shape[1]
    res.T = T; res.reset()
    for t in range(T - 1): res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
    res.Jout = W.T.copy(); res.y = res.Jout @ res.X
    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())
    Y_sim = np.array(Y_sim)[TIMES_SKIP:].T   # (N_sites, T_eff)
    FC    = np.nan_to_num(np.corrcoef(Y_sim))
    return Y_sim, FC

# ── Conditions to test ─────────────────────────────────────────────────────────
def site_importance(pid):
    dW = W_cc_mean - pat_W[pid]
    return np.linalg.norm(dW, axis=0)

def W_fullW(pid, alpha):
    return (1 - alpha) * pat_W[pid] + alpha * W_cc_mean

def W_top5(pid, alpha):
    W = pat_W[pid].copy()
    top5 = np.argsort(site_importance(pid))[::-1][:5]
    W[:, top5] = (1 - alpha) * pat_W[pid][:, top5] + alpha * W_cc_mean[:, top5]
    return W

CONDITIONS = [
    ("Baseline (α=0)",       lambda pid: pat_W[pid]),
    ("Full-W  α=1",          lambda pid: W_fullW(pid, 1.0)),
    ("Top-5   α=10",         lambda pid: W_top5(pid, 10.0)),
]

# ── Compute metrics for all AD patients ────────────────────────────────────────
print("\n[Physiology check] Running AD patients through each condition ...")

records = {name: dict(sig_std=[], sig_max=[], fc_r_cc=[], fc_r_emp=[], fc_r_base=[])
           for name, _ in CONDITIONS}

base_FC = {}   # pid -> FC at alpha=0, for relative change

for pid_i in tqdm(ad_idx, desc="AD patients"):
    pid = unique_pids[pid_i]

    # baseline first (needed for fc_r_base)
    _, FC0 = run_closed_loop(pat_W[pid], pid)
    base_FC[pid] = FC0

    for name, W_fn in CONDITIONS:
        W   = W_fn(pid)
        Y_sim, FC = run_closed_loop(W, pid)

        sig_std = float(Y_sim.std())
        sig_max = float(np.abs(Y_sim).max())
        fc_r_cc  = float(np.corrcoef(FC.flatten(), FC_cc_mean)[0, 1])
        fc_r_emp = float(np.corrcoef(FC[IU], pat_fc_emp[pid][IU])[0, 1])
        fc_r_base = float(np.corrcoef(FC.flatten(), base_FC[pid].flatten())[0, 1])

        records[name]["sig_std"].append(sig_std)
        records[name]["sig_max"].append(sig_max)
        records[name]["fc_r_cc"].append(fc_r_cc)
        records[name]["fc_r_emp"].append(fc_r_emp)
        records[name]["fc_r_base"].append(fc_r_base)

# ── Print summary table ────────────────────────────────────────────────────────
print()
print("=" * 82)
print(f"{'Condition':<22}  {'sig_std':>9}  {'sig_max':>9}  "
      f"{'FC-r/CC':>9}  {'FC-r/emp':>9}  {'FC-r/base':>10}")
print("-" * 82)
for name, _ in CONDITIONS:
    r = records[name]
    print(f"  {name:<20}  "
          f"{np.mean(r['sig_std']):>8.4f}  "
          f"{np.mean(r['sig_max']):>8.4f}  "
          f"{np.mean(r['fc_r_cc']):>8.4f}  "
          f"{np.mean(r['fc_r_emp']):>8.4f}  "
          f"{np.mean(r['fc_r_base']):>9.4f}")
print("=" * 82)
print()
print("  sig_std  : mean signal std across sites (amplitude scale)")
print("  sig_max  : max |signal| (divergence indicator; empirical ~0.2-2.0)")
print("  FC-r/CC  : simulated FC vs CC-mean FC  (structure match to template)")
print("  FC-r/emp : simulated FC vs patient own empirical FC")
print("  FC-r/base: simulated FC vs baseline (alpha=0) simulated FC")

# ── Plot example FC matrices (first AD patient) ────────────────────────────────
pid_ex  = unique_pids[ad_idx[0]]
fig, axes = plt.subplots(2, len(CONDITIONS) + 1, figsize=(5*(len(CONDITIONS)+1), 9),
                         facecolor="white")

# Top row: FC matrices
titles_top = ["Empirical FC\n(own signal)"] + [name for name, _ in CONDITIONS]
mats = [pat_fc_emp[pid_ex]] + [run_closed_loop(W_fn(pid_ex), pid_ex)[1]
                                for _, W_fn in CONDITIONS]

for ax, mat, title in zip(axes[0], mats, titles_top):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.75, fraction=0.046)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)

# Bottom row: signal traces (first 5 sites)
titles_bot = ["Empirical signal\n(PCA-projected)"] + [name for name, _ in CONDITIONS]
emp_sig = np.concatenate([sess_target[i] for i in patient_sids[pid_ex]], axis=1)
emp_sig = emp_sig[:, TIMES_SKIP:]
sigs_plot = [emp_sig] + [run_closed_loop(W_fn(pid_ex), pid_ex)[0]
                         for _, W_fn in CONDITIONS]

t_ax = np.arange(emp_sig.shape[1])
for ax, sig, title in zip(axes[1], sigs_plot, titles_bot):
    for si in range(5):
        ax.plot(t_ax[:sig.shape[1]], sig[si, :len(t_ax)], lw=0.8, alpha=0.7)
    ax.set_xlabel("Time step"); ax.set_ylabel("Signal")
    ax.set_title(title, fontsize=9)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.02, 0.97,
            f"std={sig.std():.3f}\nmax|·|={np.abs(sig).max():.3f}",
            transform=ax.transAxes, fontsize=7.5, va="top")

fig.suptitle(f"Physiology check — example AD patient: {pid_ex}\n"
             f"FC matrices (top) and signal traces (bottom, first 5 sites)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/physiology_check.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved -> physiology_check.png")
