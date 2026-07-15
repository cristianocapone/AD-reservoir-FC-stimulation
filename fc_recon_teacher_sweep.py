"""
fc_recon_teacher_sweep.py
=========================
Evaluate FC reconstruction quality of per-patient W using TEACHER-FORCED
reconstruction (not closed-loop), consistent with the 85% classification.

Pipeline:
  - Teacher-force each session through the reservoir (reset between sessions)
  - Collect X (reservoir states) and Y (PCA-smoothed target) — same as 85% pipeline
  - For per-patient W: fit W_pat = pinv(X_coll + noise) @ Y_coll  (all sessions pooled)
  - For per-session W: fit W_sess = pinv(X_sess + noise) @ Y_sess  (one session only)

FC-r evaluation (TEACHER-FORCED, not closed-loop):
  - Y_hat = X_sess @ W          (teacher-forced reconstruction, open-loop readout)
  - FC_emp = corrcoef(Y_sess.T)  empirical FC from PCA-smoothed target
  - FC_hat = corrcoef(Y_hat.T)   reconstructed FC from W applied to teacher-forced X
  - r = corrcoef(FC_emp_iu, FC_hat_iu)

Noise sweep: [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.025, 0.05, 0.10, 0.20]

Saves:
  - fc_recon_teacher_sweep.npz   (arrays for supplementary figure)
  - summary_out/fc_recon_teacher_sweep.png
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

# ── Config ─────────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"
OUT_DIR     = "./summary_out"
os.makedirs(OUT_DIR, exist_ok=True)

NOISE_VALS = [1e-4, 5e-4, 1e-3, 5e-3, 0.01, 0.025, 0.05, 0.10, 0.20]

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
N_subj          = len(signals)

unique_pids    = np.unique(patient_ids_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])

print(f"  Sessions: {N_subj}  (CC={(labels_raw==0).sum()}, AD={(labels_raw==1).sum()})")
print(f"  Patients: {N_patients}  (CC={(patient_labels==0).sum()}, AD={(patient_labels==1).sum()})")

# ── Population PCA ─────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Reservoir init ─────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr
print(f"  Spectral radius = 0.95  (original: {sr:.4f})")

# ── Pre-compute teacher-forced X, Y for all sessions ─────────────────────────
#  (same as 85% classification pipeline)
print("Teacher-forced pass (all sessions) ...")
sess_X, sess_Y = {}, {}   # idx -> X (T_eff, N_hidden), Y (T_eff, N_sites)

for idx in trange(N_subj, desc="  Teacher-force"):
    s      = signals[idx]                         # (N_sites, T)
    T_s    = s.shape[1]
    target = (s.T @ ev50 @ ev50.T).T             # (N_sites, T)  50-PC reconstruction
    res.T  = T_s; res.reset()
    X_raw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    X_fit = np.array(X_raw)[TIMES_SKIP:]          # (T_eff, N_hidden)
    Y_fit = target[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T   # (T_eff, N_sites)
    sess_X[idx] = X_fit
    sess_Y[idx] = Y_fit

# ── Helper: FC-r from teacher-forced reconstruction ────────────────────────────
IU = np.triu_indices(N_SITES, k=1)

def tf_fc_r(X, Y, W):
    """FC-r: corrcoef(FC_emp, FC_hat) where Y_hat = X @ W."""
    Y_hat  = X @ W                    # (T_eff, N_sites)
    FC_emp = np.nan_to_num(np.corrcoef(Y.T))
    FC_hat = np.nan_to_num(np.corrcoef(Y_hat.T))
    r = np.corrcoef(FC_emp[IU], FC_hat[IU])[0, 1]
    return float(r)

# ── BASELINE A: per-session W, noise=0.025 ────────────────────────────────────
print("\n[BASELINE-A] Per-session W  (noise=0.025, teacher-forced eval) ...")
rng_b   = np.random.default_rng(RNG_SEED)
r_s025  = []
for idx in range(N_subj):
    X, Y  = sess_X[idx], sess_Y[idx]
    noise = rng_b.normal(0, 0.025, X.shape)
    W_s   = np.linalg.pinv(X + noise) @ Y
    r_s025.append(tf_fc_r(X, Y, W_s))
r_s025 = np.array(r_s025)
print(f"  Mean FC-r = {r_s025.mean():.4f} +/- {r_s025.std():.4f}")

# ── BASELINE B: per-session W, no noise ───────────────────────────────────────
print("[BASELINE-B] Per-session W  (noise=0, perfect pinv) ...")
r_s0  = []
for idx in range(N_subj):
    X, Y = sess_X[idx], sess_Y[idx]
    W_s  = np.linalg.pinv(X) @ Y
    r_s0.append(tf_fc_r(X, Y, W_s))
r_s0 = np.array(r_s0)
print(f"  Mean FC-r = {r_s0.mean():.4f} +/- {r_s0.std():.4f}")
print("  (upper bound: per-session perfect fit)")

# ── SWEEP: per-patient W, teacher-forced eval ─────────────────────────────────
print("\n[SWEEP] Per-patient W (teacher-forced eval), varying noise ...")
print(f"  {'noise':>8}  {'mean FC-r':>10}  {'std':>7}  {'min':>7}  {'max':>7}")
print(f"  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*7}  {'-'*7}")

noise_results   = {}   # noise_val -> array of shape (N_subj,)  — one FC-r per session

for noise_size in NOISE_VALS:
    rng_p = np.random.default_rng(RNG_SEED + 1)
    r_all = []   # one FC-r per session (using that patient's W)

    for pid in tqdm(unique_pids, desc=f"  noise={noise_size:.4f}", leave=False):
        idxs = patient_sids[pid]
        # Fit per-patient W from pooled (X, Y)
        X_coll = np.vstack([sess_X[i] for i in idxs])
        Y_coll = np.vstack([sess_Y[i] for i in idxs])
        noise  = rng_p.normal(0, noise_size, X_coll.shape)
        W_pat  = np.linalg.pinv(X_coll + noise) @ Y_coll

        # Evaluate per session
        for i in idxs:
            r_all.append(tf_fc_r(sess_X[i], sess_Y[i], W_pat))

    r_arr = np.array(r_all)
    noise_results[noise_size] = r_arr
    print(f"  {noise_size:>8.4f}  {r_arr.mean():>10.4f}  "
          f"{r_arr.std():>7.4f}  {r_arr.min():>7.4f}  {r_arr.max():>7.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
best_noise = max(noise_results, key=lambda n: noise_results[n].mean())
best_r     = noise_results[best_noise].mean()

print("\n" + "=" * 70)
print("FC-r Summary  (teacher-forced reconstruction)")
print("=" * 70)
print(f"  Per-session W, noise=0.025  : {r_s025.mean():.4f} +/- {r_s025.std():.4f}")
print(f"  Per-session W, noise=0      : {r_s0.mean():.4f} +/- {r_s0.std():.4f}  (upper bound)")
print()
print(f"  Per-patient W (teacher-forced eval):")
for n in NOISE_VALS:
    r = noise_results[n]
    marker = "  <- best" if n == best_noise else ""
    print(f"    noise={n:<7.4f}  {r.mean():.4f} +/- {r.std():.4f}{marker}")
print(f"\n  Best noise for per-patient W: {best_noise}  (mean FC-r = {best_r:.4f})")
print("=" * 70)

# ── Save arrays for supplementary figure ──────────────────────────────────────
np.savez("fc_recon_teacher_sweep.npz",
         noise_vals   = np.array(NOISE_VALS),
         r_sess_025   = r_s025,
         r_sess_0     = r_s0,
         noise_means  = np.array([noise_results[n].mean() for n in NOISE_VALS]),
         noise_stds   = np.array([noise_results[n].std()  for n in NOISE_VALS]),
         r_pat_best   = noise_results[best_noise],
         best_noise   = np.float64(best_noise),
         patient_labels = patient_labels,
         patient_ids_raw = patient_ids_raw,
         labels_raw   = labels_raw,
         )
print("Arrays saved -> fc_recon_teacher_sweep.npz")

# ── Per-patient average for CC vs AD comparison ───────────────────────────────
r_best_pat = []
for pid in unique_pids:
    idxs = patient_sids[pid]
    r_best_pat.append(np.mean([noise_results[best_noise][list(range(N_subj)).index(i)
                                                          if False else i]
                               for i in idxs]))

# Rebuild per-patient FC-r properly
print("\nPer-patient FC-r (best noise) ...")
rng_final = np.random.default_rng(RNG_SEED + 1)
r_pat_cc, r_pat_ad = [], []
pat_r_by_label = {0: [], 1: []}
for pid_i, pid in enumerate(unique_pids):
    idxs = patient_sids[pid]
    X_coll = np.vstack([sess_X[i] for i in idxs])
    Y_coll = np.vstack([sess_Y[i] for i in idxs])
    noise  = rng_final.normal(0, best_noise, X_coll.shape)
    W_pat  = np.linalg.pinv(X_coll + noise) @ Y_coll
    r_mean = np.mean([tf_fc_r(sess_X[i], sess_Y[i], W_pat) for i in idxs])
    pat_r_by_label[patient_labels[pid_i]].append(r_mean)

print(f"  CC patients:  mean FC-r = {np.mean(pat_r_by_label[0]):.4f} "
      f"+/- {np.std(pat_r_by_label[0]):.4f}")
print(f"  AD patients:  mean FC-r = {np.mean(pat_r_by_label[1]):.4f} "
      f"+/- {np.std(pat_r_by_label[1]):.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
BLUE = "#1565C0"; ORANGE = "#E64A19"

# --- Panel A: noise sweep ---
ax = axes[0]
means = [noise_results[n].mean() for n in NOISE_VALS]
stds  = [noise_results[n].std()  for n in NOISE_VALS]
ax.errorbar(range(len(NOISE_VALS)), means, yerr=stds,
            marker="o", color=BLUE, lw=2, capsize=4, zorder=3,
            label="Per-patient W (teacher-forced eval)")
ax.axhline(r_s025.mean(), color=ORANGE, lw=1.8, ls="--",
           label=f"Per-session W (noise=0.025): {r_s025.mean():.3f}")
ax.axhline(r_s025.mean() + r_s025.std(), color=ORANGE, lw=0.8, ls=":")
ax.axhline(r_s025.mean() - r_s025.std(), color=ORANGE, lw=0.8, ls=":")
ax.axhline(r_s0.mean(),   color="gray",  lw=1.2, ls=":",
           label=f"Per-session W (noise=0, upper bound): {r_s0.mean():.3f}")
ax.set_xticks(range(len(NOISE_VALS)))
ax.set_xticklabels([str(n) for n in NOISE_VALS], rotation=40, ha="right", fontsize=8)
ax.set_xlabel("Noise regularisation (noise_size)")
ax.set_ylabel("FC-r (Pearson r)")
ax.set_title("Per-patient W: FC-r vs noise\n(teacher-forced reconstruction)")
ax.legend(fontsize=7.5, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel B: boxplot comparison ---
ax = axes[1]
r_pat_best_all = noise_results[best_noise]
bp = ax.boxplot([r_s025, r_pat_best_all],
                labels=[f"Per-session W\n(noise=0.025)\nn={len(r_s025)}",
                        f"Per-patient W\n(noise={best_noise})\nn={len(r_pat_best_all)}"],
                patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], [ORANGE, BLUE]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("black"); median.set_linewidth(2)
rng_jit = np.random.default_rng(0)
for xi, r_arr in zip([1, 2], [r_s025, r_pat_best_all]):
    jit = rng_jit.normal(0, 0.06, len(r_arr))
    ax.scatter(xi + jit, r_arr, alpha=0.3, s=14, color="k", zorder=3)
ax.set_ylabel("FC-r (teacher-forced reconstruction)")
ax.set_title("FC-r: per-session vs per-patient W\n(teacher-forced eval)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# --- Panel C: CC vs AD FC-r at best noise ---
ax = axes[2]
cc_r = np.array(pat_r_by_label[0])
ad_r = np.array(pat_r_by_label[1])
bp2  = ax.boxplot([cc_r, ad_r], labels=["CC", "AD"],
                  patch_artist=True, widths=0.5)
for patch, c in zip(bp2["boxes"], ["#42A5F5", "#EF5350"]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
for median in bp2["medians"]:
    median.set_color("black"); median.set_linewidth(2)
for xi, r_arr in zip([1, 2], [cc_r, ad_r]):
    jit = rng_jit.normal(0, 0.07, len(r_arr))
    ax.scatter(xi + jit, r_arr, alpha=0.4, s=18, color="k", zorder=3)
ax.set_ylabel("FC-r (per-patient W, teacher-forced)")
ax.set_title(f"CC vs AD FC reconstruction quality\n(noise={best_noise})")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

from scipy import stats
t, p = stats.ttest_ind(cc_r, ad_r)
ax.text(0.5, 0.97, f"p={p:.3f}", transform=ax.transAxes,
        ha="center", va="top", fontsize=9)

plt.tight_layout()
out_path = f"{OUT_DIR}/fc_recon_teacher_sweep.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nPlot saved -> {out_path}")
