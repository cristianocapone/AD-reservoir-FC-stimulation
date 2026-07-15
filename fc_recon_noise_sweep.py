"""
fc_recon_noise_sweep.py
=======================
Evaluate FC reconstruction quality from per-patient W models.

Key change vs per_patient_W.py:
  - Fit W on ALL sessions of a patient (same as before)
  - Evaluate on CONCATENATED signal of that patient (not per-session)
    → empirical FC from concatenated signal
    → simulated FC from closed-loop on concatenated signal
  - Sweep noise_size to find the best regularisation level

Noise sweep: [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.10, 0.20, 0.50, 1.00]
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

# ── Config ────────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"
OUT_DIR     = "./summary_out"
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_VALS = [0.0001, 0.0005, 0.001, 0.005, 0.010, 0.025, 0.050, 0.10, 0.20, 0.50, 1.00]

# ── Load data ─────────────────────────────────────────────────────────────────
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
            signals.append(arr.T)                          # (N_sites, T)
            patient_ids_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

patient_ids_raw = np.array(patient_ids_raw)
labels_raw      = np.array(labels_raw)
N_subj          = len(signals)

unique_pids   = np.unique(patient_ids_raw)
N_patients    = len(unique_pids)
patient_sids  = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])

print(f"  Sessions: {N_subj}  (CC={(labels_raw==0).sum()}, AD={(labels_raw==1).sum()})")
print(f"  Patients: {N_patients}  (CC={(patient_labels==0).sum()}, AD={(patient_labels==1).sum()})")

# ── Population PCA ────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Pre-project all signals through PCA ───────────────────────────────────────
print("Pre-projecting signals ...")
sess_target = {}   # idx → (N_sites, T)  — PCA-smoothed signal
for idx in range(N_subj):
    s  = signals[idx]                        # (N_sites, T)
    pc = s.T @ ev50                          # (T, 50)
    sess_target[idx] = (pc @ ev50.T).T       # (N_sites, T)

# Per-patient concatenated target (empirical) ─────────────────────────────────
pat_target_concat = {}   # pid → (N_sites, T_total)
pat_fc_emp        = {}   # pid → (N_sites, N_sites)  empirical FC from concat signal
for pid in unique_pids:
    idxs = patient_sids[pid]
    tc   = np.concatenate([sess_target[i] for i in idxs], axis=1)  # (N_sites, T_total)
    pat_target_concat[pid] = tc
    pat_fc_emp[pid] = np.nan_to_num(np.corrcoef(tc[:, TIMES_SKIP:]))

# ── Reservoir init ────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr
print(f"  Spectral radius set to 0.95  (was {sr:.4f})")

# ── Helper: closed-loop FC-r on a CONCATENATED signal ─────────────────────────
def closed_loop_fc_r_concat(res, target_concat, W, ff, skip):
    """
    target_concat : (N_sites, T_total) — concatenated PCA-projected signal
    W             : (N_hidden, N_sites) — output weight matrix

    Teacher-forces the reservoir through the ENTIRE concatenated signal to build
    a representative warm state, then runs closed-loop for the same duration and
    computes FC-r between empirical and simulated FC.
    """
    T = target_concat.shape[1]
    res.T = T; res.reset()

    # Teacher-forced warm-up (full concatenated signal)
    for t in range(T - 1):
        res.step_rate(ff * target_concat[:, t], sigma_dyn=0.)

    # Switch to closed loop
    res.Jout = W.T.copy()
    res.y    = res.Jout @ res.X

    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())

    Y_sim  = np.array(Y_sim)[skip:].T          # (N_sites, T_eff)

    FC_sim = np.nan_to_num(np.corrcoef(Y_sim))
    FC_emp = np.nan_to_num(np.corrcoef(target_concat[:, skip:]))
    iu     = np.triu_indices(N_SITES, k=1)
    r      = np.corrcoef(FC_emp[iu], FC_sim[iu])[0, 1]
    return float(r)


# ── BASELINE: per-session W evaluated per-session (noise=0.025) ───────────────
print("\n[BASELINE] Per-session W, per-session FC-r  (noise=0.025) ...")
rng_b  = np.random.default_rng(RNG_SEED)
r_sess_list = []

for idx in trange(N_subj, desc="  Per-session W + FC-r"):
    target = sess_target[idx]
    T_s    = target.shape[1]
    res.T  = T_s; res.reset()
    X_raw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    X_fit = np.array(X_raw)[TIMES_SKIP:]
    Y_fit = target[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T
    noise = rng_b.normal(0, 0.025, X_fit.shape)
    W_s   = np.linalg.pinv(X_fit + noise) @ Y_fit

    r = closed_loop_fc_r_concat(res, target, W_s, ff, TIMES_SKIP)
    r_sess_list.append(r)

r_sess = np.array(r_sess_list)
print(f"  Per-session W (eval per-session):  "
      f"mean FC-r = {r_sess.mean():.4f} ± {r_sess.std():.4f}")

# ── SWEEP: per-patient W evaluated on concatenated signal ─────────────────────
print("\n[SWEEP] Per-patient W (eval on concat signal), varying noise ...")
print(f"  {'noise':>10}  {'mean FC-r':>10}  {'std':>7}  {'min':>7}  {'max':>7}")
print(f"  {'-'*10}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}")

noise_results = {}   # noise_val → array of FC-r per patient

for noise_size in NOISE_VALS:
    rng_p  = np.random.default_rng(RNG_SEED + 1)
    r_vals = []

    for pid in tqdm(unique_pids, desc=f"  noise={noise_size:.4f}", leave=False):
        idxs   = patient_sids[pid]
        # Re-collect X,Y for this patient with the current noise level
        X_parts, Y_parts = [], []
        for i in idxs:
            target_i = sess_target[i]
            T_i = target_i.shape[1]
            res.T = T_i; res.reset()
            X_raw_i = []
            for t in range(T_i - 1):
                res.step_rate(ff * target_i[:, t], sigma_dyn=0.)
                X_raw_i.append(res.X.copy())
            X_fit_i = np.array(X_raw_i)[TIMES_SKIP:]
            Y_fit_i = target_i[:, TIMES_SKIP:TIMES_SKIP + len(X_fit_i)].T
            X_parts.append(X_fit_i)
            Y_parts.append(Y_fit_i)

        X_coll = np.vstack(X_parts)
        Y_coll = np.vstack(Y_parts)

        noise = rng_p.normal(0, noise_size, X_coll.shape)
        W_pat = np.linalg.pinv(X_coll + noise) @ Y_coll

        # Evaluate on concatenated signal
        target_cat = pat_target_concat[pid]
        r = closed_loop_fc_r_concat(res, target_cat, W_pat, ff, TIMES_SKIP)
        r_vals.append(r)

    r_arr = np.array(r_vals)
    noise_results[noise_size] = r_arr
    print(f"  {noise_size:>10.4f}  {r_arr.mean():>10.4f}  "
          f"{r_arr.std():>7.4f}  {r_arr.min():>7.4f}  {r_arr.max():>7.4f}")

# ── Summary table ─────────────────────────────────────────────────────────────
best_noise = max(noise_results, key=lambda n: noise_results[n].mean())
best_r     = noise_results[best_noise].mean()

print("\n" + "=" * 65)
print("FC-r Summary")
print("=" * 65)
print(f"  Baseline: per-session W (sess eval)  "
      f"mean FC-r = {r_sess.mean():.4f}")
print()
print(f"  {'noise':>10}  {'mean FC-r':>10}  {'std':>7}")
for n in NOISE_VALS:
    r_arr = noise_results[n]
    marker = "  <- best" if n == best_noise else ""
    print(f"  {n:>10.4f}  {r_arr.mean():>10.4f}  {r_arr.std():>7.4f}{marker}")
print(f"\n  Best noise level: {best_noise}  (mean FC-r = {best_r:.4f})")
print("=" * 65)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor="white")

# Left: noise sweep curve
ax = axes[0]
means = [noise_results[n].mean() for n in NOISE_VALS]
stds  = [noise_results[n].std()  for n in NOISE_VALS]
ax.errorbar(range(len(NOISE_VALS)), means, yerr=stds,
            marker="o", color="#1565C0", lw=2, capsize=4, label="Per-patient W (concat eval)")
ax.axhline(r_sess.mean(), color="#E64A19", lw=1.8, ls="--",
           label=f"Baseline per-session W = {r_sess.mean():.3f}")
ax.axhline(r_sess.mean() + r_sess.std(), color="#E64A19", lw=0.8, ls=":")
ax.axhline(r_sess.mean() - r_sess.std(), color="#E64A19", lw=0.8, ls=":")
ax.set_xticks(range(len(NOISE_VALS)))
ax.set_xticklabels([str(n) for n in NOISE_VALS], rotation=30, ha="right", fontsize=8)
ax.set_xlabel("Noise regularisation (noise_size)")
ax.set_ylabel("FC-r (Pearson r, empirical vs simulated FC)")
ax.set_title("Per-patient W: FC-r vs noise level\n(evaluated on concatenated signal)")
ax.legend(fontsize=8)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Right: FC-r distributions for best noise vs baseline
ax = axes[1]
r_best = noise_results[best_noise]
bp = ax.boxplot([r_sess, r_best],
                labels=[f"Per-session W\n(sess eval)\nn={len(r_sess)}",
                        f"Per-patient W\nnoise={best_noise}\n(concat eval)"],
                patch_artist=True, widths=0.5)
colors = ["#FFAB91", "#1565C0"]
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.8)
for median in bp["medians"]:
    median.set_color("black"); median.set_linewidth(2)

# Overlay individual points
rng_jit = np.random.default_rng(0)
for xi, r_arr in zip([1, 2], [r_sess, r_best]):
    jit = rng_jit.normal(0, 0.06, len(r_arr))
    ax.scatter(xi + jit, r_arr, alpha=0.4, s=18, color="k", zorder=3)

ax.set_ylabel("FC-r")
ax.set_title("FC-r distributions\n(per-session vs best per-patient W)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
out_path = f"{OUT_DIR}/fc_recon_noise_sweep.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved -> {out_path}")

# ── Save arrays for supplementary figure in notebook ─────────────────────────
np.savez("fc_recon_noise_sweep.npz",
         noise_vals     = np.array(NOISE_VALS),
         noise_means    = np.array([noise_results[n].mean() for n in NOISE_VALS]),
         noise_stds     = np.array([noise_results[n].std()  for n in NOISE_VALS]),
         r_sess         = r_sess,                    # per-session W FC-r (per-session eval)
         r_pat_best     = noise_results[best_noise], # per-patient W FC-r at best noise (per-patient eval)
         best_noise     = np.float64(best_noise),
         patient_labels = patient_labels,
         patient_ids_raw = patient_ids_raw,
         labels_raw     = labels_raw,
         )
print("Arrays saved -> fc_recon_noise_sweep.npz")
