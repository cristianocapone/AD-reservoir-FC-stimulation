"""
perturbation_sim_refit.py
=========================
CORRECT perturbation approach (matching Fig1DEF_MC_data.ipynb):

For each alpha and each AD patient:
  1. Teacher-forced pass on original signal  →  warm reservoir state
  2. Run autonomous (closed-loop) simulation with W_interp as Jout
  3. If signal is stable: re-fit W_fitted from simulated signal (Y_sim)
     Else:                fall back to projecting W_interp directly (static)
  4. W_fitted → G-score (projected into original X-space basis of that patient)
  5. LOPO LDA on G-scores  →  AUROC

CC patients keep their original W / G-score at every alpha (no perturbation).

Two perturbation types:
  A) Full-W therapeutic:  W_int = (1-alpha)*W_pat + alpha*W_cc_mean  [AD only]
  B) Top-5-site:          5 most-divergent columns interpolated        [AD only]

Stability criterion: |Y_sim|_max < SIM_MAX_STABLE (default 100).
If the simulation diverges the script falls back to the static projection so
the dose-response curve is always complete; diverged entries are flagged.
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import roc_auc_score
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

# Stability threshold: if max|Y_sim| exceeds this, treat simulation as diverged
SIM_MAX_STABLE = 100.0

ALPHA_THERAPEUTIC = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                     6.0, 7.0, 8.0, 9.0, 10.0]
ALPHA_SINGLESITE  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                     6.0, 7.0, 8.0, 9.0, 10.0]

K_LDA = 2    # best k from noise_ba_sweep.py (AUROC=0.8514 at sigma=0.025)

col_cc = "#2196F3"   # blue
col_ad = "#E91E63"   # pink

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

cc_idx = np.where(patient_labels == 0)[0]
ad_idx = np.where(patient_labels == 1)[0]

print(f"  Sessions: {N_subj}   Patients: {N_patients}  "
      f"(CC={len(cc_idx)}, AD={len(ad_idx)})")

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

# ── Teacher-forced pass (all sessions) ────────────────────────────────────────
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

# ── G-scores via population SVD ────────────────────────────────────────────────
print("G-scores (SVD of W stack) ...")
W_proj_list = []
for pid in unique_pids:
    W      = pat_W[pid]
    X_coll = pat_X[pid]
    W_T    = W.T.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(X_coll.astype(np.float64), full_matrices=False)
    k_pc   = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k   = Vtx[:k_pc]
    W_proj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

W_stack  = np.array(W_proj_list)
W_mean   = W_stack.mean(0)
Wcent    = W_stack - W_mean
_, _, Vt_svd = np.linalg.svd(Wcent, full_matrices=False)
M_eff    = N_patients - 1
G_pat    = Wcent @ Vt_svd[:M_eff].T   # (N_patients, M_eff)
print(f"  G_pat shape: {G_pat.shape}")

# ── LDA on ALL patients (fixed — used for every alpha) ─────────────────────────
print(f"LDA on G-scores (k={K_LDA}) ...")

class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = (X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1) + 1e-6*np.eye(X0.shape[1])
        w  = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_   = w
        self.thr_ = 0.5*((X0@w).mean()+(X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_
    def predict(self, X):
        return np.where(self.transform(X) >= self.thr_,
                        self.classes_[1], self.classes_[0])

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([rng2.choice(c0,n,replace=False), rng2.choice(c1,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

X_lda_all, y_lda_all = _balance(G_pat[:, :K_LDA], patient_labels, seed=RNG_SEED)
lda = _LDA().fit(X_lda_all, y_lda_all)

Z_all = lda.transform(G_pat[:, :K_LDA])
if Z_all[patient_labels == 0].mean() > Z_all[patient_labels == 1].mean():
    lda.w_ *= -1; lda.thr_ *= -1
Z_all = lda.transform(G_pat[:, :K_LDA])

y_lda     = patient_labels
train_acc = np.mean(lda.predict(G_pat[:, :K_LDA]) == patient_labels)
print(f"  LDA train accuracy (all data): {train_acc*100:.1f}%")
print(f"  CC mean LDA: {Z_all[y_lda==0].mean():.3f}   "
      f"AD mean LDA: {Z_all[y_lda==1].mean():.3f}")

# ── CC mean W ──────────────────────────────────────────────────────────────────
W_cc_mean = np.mean([pat_W[unique_pids[i]] for i in cc_idx], axis=0)

# ── CC mean FC (closed-loop) ───────────────────────────────────────────────────
print("Computing CC mean FC (closed-loop) ...")
IU = np.triu_indices(N_SITES, k=1)
fc_cc_list = []
for i in tqdm(cc_idx, desc="  CC closed-loop", leave=False):
    pid   = unique_pids[i]
    tgt   = pat_target[pid]; T = tgt.shape[1]
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def w_to_g(W, X_coll):
    """Project W into G-space using X_coll as X-space basis."""
    W_T  = W.T.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(X_coll.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    w_proj = (W_T @ Vt_k.T @ Vt_k).flatten()
    g      = (w_proj - W_mean) @ Vt_svd[:M_eff].T
    return g[:K_LDA]

rng_sim = np.random.default_rng(RNG_SEED + 2)

def run_sim_refit(W_int, pid):
    """
    Run autonomous simulation with W_int, then re-fit W from the simulated signal.

    Steps:
      1. Warm up reservoir with teacher-forced original signal of pid
      2. Run autonomous (closed-loop) simulation with W_int
      3. If signal is stable: re-fit W_fitted from Y_sim, return G-score + FC
      4. If signal diverges: return (None, None) — caller falls back to static G

    Returns
    -------
    g_new : np.ndarray of shape (K_LDA,), or None if diverged
    fc_r  : float (FC-r vs CC mean), or None if diverged
    stable: bool
    """
    tgt = pat_target[pid]
    T   = tgt.shape[1]

    # ── 1. Warm-up: teacher-forced pass on original signal ──
    res.T = T; res.reset()
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)

    # ── 2. Autonomous simulation with W_int ────────────────
    res.Jout = W_int.T.copy()
    res.y    = res.Jout @ res.X
    Y_sim_list = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim_list.append(res.y.copy())
    Y_sim = np.array(Y_sim_list).T   # (N_sites, T-1)

    # ── 3. Divergence check ────────────────────────────────
    if not np.isfinite(Y_sim).all() or np.abs(Y_sim).max() > SIM_MAX_STABLE:
        return None, None, False

    # FC from simulated signal
    Y_eff  = Y_sim[:, TIMES_SKIP:]
    FC_sim = np.nan_to_num(np.corrcoef(Y_eff))
    fc_r   = float(np.corrcoef(FC_sim.flatten(), FC_cc_mean)[0, 1])

    # ── 4. Teacher-force on Y_sim → collect X_aut, Y_aut ──
    T2    = Y_sim.shape[1]
    res.T = T2; res.reset()
    X_aut_list = []
    for t in range(T2 - 1):
        res.step_rate(ff * Y_sim[:, t], sigma_dyn=0.)
        X_aut_list.append(res.X.copy())
    X_aut = np.array(X_aut_list)[TIMES_SKIP:]                  # (T_eff, N_hidden)
    Y_aut = Y_sim[:, TIMES_SKIP:TIMES_SKIP + len(X_aut)].T     # (T_eff, N_sites)

    # ── 5. Re-fit W from simulated signal ─────────────────
    noise    = rng_sim.normal(0, NOISE_SIZE, X_aut.shape)
    W_fitted = np.linalg.pinv(X_aut + noise) @ Y_aut    # (N_hidden, N_sites)

    # ── 6. G-score: project W_fitted using ORIGINAL pat_X ─
    #    (ensures comparability with baseline G_pat)
    g_new = w_to_g(W_fitted, pat_X[pid])

    return g_new, fc_r, True


def closed_loop_fc_r(W, pid):
    """Quick FC-r vs CC mean for a given W (no re-fit, for CC patients)."""
    tgt = pat_target[pid]; T = tgt.shape[1]
    res.T = T; res.reset()
    for t in range(T - 1): res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
    res.Jout = W.T.copy(); res.y = res.Jout @ res.X
    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())
    Y_sim = np.array(Y_sim)[TIMES_SKIP:].T
    FC = np.nan_to_num(np.corrcoef(Y_sim))
    return float(np.corrcoef(FC.flatten(), FC_cc_mean)[0, 1])


# ══════════════════════════════════════════════════════════════════════════════
# PART A — Therapeutic offline interpolation  (Full-W, simulation + re-fit)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Therapeutic Full-W interpolation — simulation + re-fit ...")

res_ther = {a: dict(g=[], fc_corr=[], z=None, pred=None, auc=None, n_stable=0)
            for a in ALPHA_THERAPEUTIC}

# Pre-compute baseline FC-r for ALL patients once (used for alpha=0 and CC)
print("Pre-computing baseline closed-loop FC-r for all patients ...")
pat_fc_r_base = {}
for pid in tqdm(unique_pids, desc="  baseline FC-r", leave=False):
    pat_fc_r_base[pid] = closed_loop_fc_r(pat_W[pid], pid)

for alpha in ALPHA_THERAPEUTIC:
    print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
    g_list, fc_list, n_stable = [], [], 0

    for pid_i, pid in enumerate(unique_pids):
        is_ad = (patient_labels[pid_i] == 1)

        if not is_ad:
            # CC: original G-score and FC (unchanged at every alpha)
            g_list.append(G_pat[pid_i, :K_LDA])
            fc_list.append(pat_fc_r_base[pid])
        else:
            # AD: interpolate W, run simulation, re-fit
            W_pat = pat_W[pid]
            W_int = (1 - alpha) * W_pat + alpha * W_cc_mean

            if alpha == 0.0:
                # Baseline: use original G directly (avoids running diverged simulation)
                g_list.append(G_pat[pid_i, :K_LDA])
                fc_list.append(pat_fc_r_base[pid])
            else:
                g_new, fc_r, stable = run_sim_refit(W_int, pid)
                if stable:
                    g_list.append(g_new)
                    fc_list.append(fc_r)
                    n_stable += 1
                else:
                    # Fallback: static projection of interpolated W
                    # FC is meaningless for a diverged simulation → NaN
                    g_list.append(w_to_g(W_int, pat_X[pid]))
                    fc_list.append(np.nan)

    G_new   = np.array(g_list)
    z_new   = lda.transform(G_new)
    pred    = lda.predict(G_new)
    auc_val = roc_auc_score(patient_labels, z_new)
    res_ther[alpha].update(g=G_new, fc_corr=np.array(fc_list),
                           z=z_new, pred=pred, auc=auc_val, n_stable=n_stable)
    frac = np.mean(pred[y_lda==1] == 0)
    print(f"AD->CC={frac:.2f}  LDA(CC)={z_new[y_lda==0].mean():.2f}  "
          f"LDA(AD)={z_new[y_lda==1].mean():.2f}  AUROC={auc_val:.4f}  "
          f"[stable: {n_stable}/{len(ad_idx)}]")

# ── Plot A1: LDA KDE ridgeline ─────────────────────────────────────────────────
fig, axes = plt.subplots(len(ALPHA_THERAPEUTIC), 1,
                         figsize=(7, 1.6*len(ALPHA_THERAPEUTIC)), sharex=True,
                         facecolor="white")
all_z = np.concatenate([res_ther[a]['z'] for a in ALPHA_THERAPEUTIC])
xs = np.linspace(all_z.min()-0.5, all_z.max()+0.5, 400)
for ax, alpha in zip(axes, ALPHA_THERAPEUTIC):
    z = res_ther[alpha]['z']
    for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
        pts = z[y_lda == cls]
        if len(pts) < 3: continue
        try:    kde = gaussian_kde(pts, bw_method="scott")
        except: kde = gaussian_kde(pts + np.random.normal(0, max(pts.std(), 1e-4)*1e-3, pts.shape))
        ax.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
        ax.plot(xs, kde(xs), color=col, lw=1.5)
        ax.axvline(pts.mean(), color=col, ls="--", lw=1)
    stab = res_ther[alpha]['n_stable']
    ax.set_ylabel(f"a={alpha:.2f}\n({stab} stable)", fontsize=8)
    ax.set_yticks([])
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
axes[-1].set_xlabel("LDA score  (CC  ←  →  AD)", fontsize=11)
fig.suptitle("Full-W interpolation — LDA score distribution\n"
             "(simulation + re-fit, per-patient W, noise=0.025)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefit_ther_lda_kde.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefit_ther_lda_kde.png")

# ── Plot A2: dose-response (4 panels) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4), facecolor="white")

ax = axes[0]
for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
    fracs = [np.mean(res_ther[a]["pred"][y_lda==cls] == 0) for a in ALPHA_THERAPEUTIC]
    ax.plot(ALPHA_THERAPEUTIC, fracs, "-o", color=col, label=name, lw=2)
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("Dose α"); ax.set_ylabel("Fraction predicted as CC")
ax.set_title("Fraction classified as CC"); ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
    ms = [res_ther[a]["z"][y_lda==cls].mean() for a in ALPHA_THERAPEUTIC]
    se = [res_ther[a]["z"][y_lda==cls].std()/np.sqrt((y_lda==cls).sum())
          for a in ALPHA_THERAPEUTIC]
    ax.plot(ALPHA_THERAPEUTIC, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_THERAPEUTIC, np.subtract(ms,se), np.add(ms,se), color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("Mean LDA score")
ax.set_title("Mean LDA score (CC ← → AD)")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[2]
for pidx_list, col, name in [(cc_idx, col_cc, "CC"), (ad_idx, col_ad, "AD")]:
    ms = [np.nanmean(res_ther[a]["fc_corr"][pidx_list]) for a in ALPHA_THERAPEUTIC]
    se = [np.nanstd(res_ther[a]["fc_corr"][pidx_list])/np.sqrt(np.sum(np.isfinite(res_ther[a]["fc_corr"][pidx_list])))
          for a in ALPHA_THERAPEUTIC]
    ax.plot(ALPHA_THERAPEUTIC, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_THERAPEUTIC, np.subtract(ms,se), np.add(ms,se), color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("FC corr with CC mean (stable sims only)")
ax.set_title("FC similarity to CC template")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[3]
aucs_a = [res_ther[a]["auc"] for a in ALPHA_THERAPEUTIC]
stabs  = [res_ther[a]["n_stable"] for a in ALPHA_THERAPEUTIC]
ax.plot(ALPHA_THERAPEUTIC, aucs_a, "-o", color="#6A1B9A", lw=2, label="AUROC")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
# Mark how many patients had stable simulations (as text)
for x, y_val, s in zip(ALPHA_THERAPEUTIC, aucs_a, stabs):
    if s > 0 and s < len(ad_idx):
        ax.annotate(f"{s}", (x, y_val), textcoords="offset points",
                    xytext=(0, 6), fontsize=6, ha="center", color="#6A1B9A")
ax.set_xlabel("Dose α"); ax.set_ylabel("AUROC")
ax.set_title("AD vs CC discriminability\n(AUROC; italic=n stable AD sims)")
ax.set_ylim(0.40, 1.00)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Full-W interpolation dose-response  "
             "(simulation + re-fit, per-patient W, noise=0.025)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefit_ther_doseresponse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefit_ther_doseresponse.png")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Top-5-site perturbation  (simulation + re-fit)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[B] Top-5-site perturbation — simulation + re-fit ...")

res_pert = {a: dict(g=[], fc_corr=[], z=None, pred=None, auc=None, n_stable=0)
            for a in ALPHA_SINGLESITE}

site_counts = np.zeros(N_SITES, dtype=int)

for alpha in ALPHA_SINGLESITE:
    print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
    g_list, fc_list, n_stable = [], [], 0

    for pid_i, pid in enumerate(unique_pids):
        is_ad = (patient_labels[pid_i] == 1)
        W_pat = pat_W[pid]

        if not is_ad:
            g_list.append(G_pat[pid_i, :K_LDA])
            fc_list.append(pat_fc_r_base[pid])
        else:
            dW         = W_cc_mean - W_pat
            site_imp   = np.linalg.norm(dW, axis=0)
            top5_sites = np.argsort(site_imp)[::-1][:5]

            if alpha == ALPHA_SINGLESITE[1]:
                site_counts[top5_sites] += 1

            if alpha == 0.0:
                g_list.append(G_pat[pid_i, :K_LDA])
                fc_list.append(pat_fc_r_base[pid])
            else:
                W_int = W_pat.copy()
                W_int[:, top5_sites] = ((1 - alpha) * W_pat[:, top5_sites]
                                        + alpha * W_cc_mean[:, top5_sites])
                g_new, fc_r, stable = run_sim_refit(W_int, pid)
                if stable:
                    g_list.append(g_new)
                    fc_list.append(fc_r)
                    n_stable += 1
                else:
                    g_list.append(w_to_g(W_int, pat_X[pid]))
                    fc_list.append(np.nan)

    G_new  = np.array(g_list)
    z_new  = lda.transform(G_new)
    pred   = lda.predict(G_new)
    auc_val = roc_auc_score(patient_labels, z_new)
    res_pert[alpha].update(g=G_new, fc_corr=np.array(fc_list),
                           z=z_new, pred=pred, auc=auc_val, n_stable=n_stable)
    frac = np.mean(pred[y_lda==1] == 0)
    print(f"AD->CC={frac:.2f}  LDA(AD)={z_new[y_lda==1].mean():.2f}  "
          f"AUROC={auc_val:.4f}  [stable: {n_stable}/{len(ad_idx)}]")

# ── Plot B1: LDA KDE ridgeline ─────────────────────────────────────────────────
fig, axes = plt.subplots(len(ALPHA_SINGLESITE), 1,
                         figsize=(7, 1.6*len(ALPHA_SINGLESITE)), sharex=True,
                         facecolor="white")
all_z = np.concatenate([res_pert[a]['z'] for a in ALPHA_SINGLESITE])
xs    = np.linspace(all_z.min()-0.5, all_z.max()+0.5, 400)
for ax, alpha in zip(axes, ALPHA_SINGLESITE):
    z = res_pert[alpha]['z']
    for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
        pts = z[y_lda == cls]
        if len(pts) < 3: continue
        try:    kde = gaussian_kde(pts, bw_method="scott")
        except: kde = gaussian_kde(pts + np.random.normal(0, max(pts.std(), 1e-4)*1e-3, pts.shape))
        ax.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
        ax.plot(xs, kde(xs), color=col, lw=1.5)
        ax.axvline(pts.mean(), color=col, ls="--", lw=1)
    stab = res_pert[alpha]['n_stable']
    ax.set_ylabel(f"a={alpha:.1f}\n({stab} stable)", fontsize=8)
    ax.set_yticks([])
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)
axes[0].legend(frameon=False, fontsize=9, loc="upper right")
axes[-1].set_xlabel("LDA score  (CC  ←  →  AD)", fontsize=11)
fig.suptitle("Top-5-site perturbation — LDA score distribution\n"
             "(simulation + re-fit, per-patient W, noise=0.025)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefit_pert_lda_kde.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefit_pert_lda_kde.png")

# ── Plot B2: dose-response (4 panels) ─────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 4), facecolor="white")

ax = axes[0]
for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
    fracs = [np.mean(res_pert[a]["pred"][y_lda==cls] == 0) for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, fracs, "-o", color=col, label=name, lw=2)
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("Dose α"); ax.set_ylabel("Fraction predicted as CC")
ax.set_title("Fraction classified as CC"); ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
    ms = [res_pert[a]["z"][y_lda==cls].mean() for a in ALPHA_SINGLESITE]
    se = [res_pert[a]["z"][y_lda==cls].std()/np.sqrt((y_lda==cls).sum())
          for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_SINGLESITE, np.subtract(ms,se), np.add(ms,se), color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("Mean LDA score")
ax.set_title("Mean LDA score (CC ← → AD)")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[2]
for pidx_list, col, name in [(cc_idx, col_cc, "CC"), (ad_idx, col_ad, "AD")]:
    ms = [np.nanmean(res_pert[a]["fc_corr"][pidx_list]) for a in ALPHA_SINGLESITE]
    se = [np.nanstd(res_pert[a]["fc_corr"][pidx_list])/np.sqrt(np.sum(np.isfinite(res_pert[a]["fc_corr"][pidx_list])))
          for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_SINGLESITE, np.subtract(ms,se), np.add(ms,se), color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("FC corr with CC mean (stable sims only)")
ax.set_title("FC similarity to CC template")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[3]
aucs_b = [res_pert[a]["auc"] for a in ALPHA_SINGLESITE]
ax.plot(ALPHA_SINGLESITE, aucs_b, "-o", color="#6A1B9A", lw=2, label="AUROC")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
for x, y_val, s in zip(ALPHA_SINGLESITE, aucs_b, [res_pert[a]["n_stable"] for a in ALPHA_SINGLESITE]):
    if s > 0 and s < len(ad_idx):
        ax.annotate(f"{s}", (x, y_val), textcoords="offset points",
                    xytext=(0, 6), fontsize=6, ha="center", color="#6A1B9A")
ax.set_xlabel("Dose α"); ax.set_ylabel("AUROC")
ax.set_title("AD vs CC discriminability\n(AUROC; italic=n stable AD sims)")
ax.set_ylim(0.40, 1.00)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Top-5-site perturbation dose-response  "
             "(simulation + re-fit, per-patient W, noise=0.025)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefit_pert_doseresponse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefit_pert_doseresponse.png")

# ══════════════════════════════════════════════════════════════════════════════
# PART C — Overlay comparison: static projection vs simulation+re-fit  (AUROC)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C] Overlay: static projection vs sim+re-fit ...")

# Re-run static approach for comparison (same logic as perturbation_perpatient.py)
res_ther_static = {}
res_pert_static = {}

for alpha in ALPHA_THERAPEUTIC:
    # Static: project interpolated W directly — no simulation needed for AUROC
    g_list = [w_to_g((1-alpha)*pat_W[pid] + alpha*W_cc_mean if patient_labels[pi]==1
                     else pat_W[pid], pat_X[pid])
              for pi, pid in enumerate(unique_pids)]
    G_new = np.array(g_list)
    z_new = lda.transform(G_new)
    res_ther_static[alpha] = dict(auc=roc_auc_score(patient_labels, z_new))

print("  Static therapeutic done.")

for alpha in ALPHA_SINGLESITE:
    g_list = []
    for pi, pid in enumerate(unique_pids):
        W_pat = pat_W[pid]
        if patient_labels[pi] == 1:   # AD
            dW   = W_cc_mean - W_pat
            top5 = np.argsort(np.linalg.norm(dW, axis=0))[::-1][:5]
            W_int = W_pat.copy()
            W_int[:, top5] = (1-alpha)*W_pat[:,top5] + alpha*W_cc_mean[:,top5]
        else:
            W_int = W_pat
        g_list.append(w_to_g(W_int, pat_X[pid]))
    G_new = np.array(g_list)
    z_new = lda.transform(G_new)
    res_pert_static[alpha] = dict(auc=roc_auc_score(patient_labels, z_new))

print("  Static top-5 done.")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")

# Left: Full-W therapeutic
ax = axes[0]
aucs_static = [res_ther_static[a]["auc"] for a in ALPHA_THERAPEUTIC]
aucs_simref = [res_ther[a]["auc"]        for a in ALPHA_THERAPEUTIC]
ax.plot(ALPHA_THERAPEUTIC, aucs_static, "-o", color="#1565C0", lw=2.5,
        label="Static projection (W_interp → G)")
ax.plot(ALPHA_THERAPEUTIC, aucs_simref, "-s", color="#E64A19", lw=2.5,
        label="Sim + re-fit (Y_sim → W_fitted → G)")
ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="chance")
ax.set_xlabel("Dose α", fontsize=11); ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("Full-W therapeutic: static vs sim+re-fit", fontsize=10)
ax.set_ylim(0.0, 1.00)
ax.legend(frameon=True, framealpha=0.9, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Right: Top-5-site
ax = axes[1]
aucs_static5 = [res_pert_static[a]["auc"] for a in ALPHA_SINGLESITE]
aucs_simref5 = [res_pert[a]["auc"]         for a in ALPHA_SINGLESITE]
ax.plot(ALPHA_SINGLESITE, aucs_static5, "-o", color="#1565C0", lw=2.5,
        label="Static projection (W_interp → G)")
ax.plot(ALPHA_SINGLESITE, aucs_simref5, "-s", color="#E64A19", lw=2.5,
        label="Sim + re-fit (Y_sim → W_fitted → G)")
ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="chance")
ax.set_xlabel("Dose α", fontsize=11); ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("Top-5-site: static vs sim+re-fit", fontsize=10)
ax.set_ylim(0.0, 1.00)
ax.legend(frameon=True, framealpha=0.9, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Comparison: static W projection vs simulation + W re-fit\n"
             "(per-patient W, noise=0.025)", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/simrefit_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved simrefit_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print(f"{'Condition':<28}  {'alpha':>6}  {'AD->CC':>6}  "
      f"{'LDA(AD)':>8}  {'AUROC':>7}  {'n_stable':>8}")
print("-" * 80)
for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    if alpha not in res_ther: continue
    r  = res_ther[alpha]
    frac = np.mean(r["pred"][y_lda==1] == 0)
    print(f"  {'Full-W sim+refit':<26}  {alpha:>6.1f}  {frac:>6.2f}  "
          f"{r['z'][y_lda==1].mean():>8.3f}  {r['auc']:>7.4f}  "
          f"{r['n_stable']:>8}/{len(ad_idx)}")
print("-" * 80)
for alpha in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
    if alpha not in res_pert: continue
    r  = res_pert[alpha]
    frac = np.mean(r["pred"][y_lda==1] == 0)
    print(f"  {'Top-5 sim+refit':<26}  {alpha:>6.1f}  {frac:>6.2f}  "
          f"{r['z'][y_lda==1].mean():>8.3f}  {r['auc']:>7.4f}  "
          f"{r['n_stable']:>8}/{len(ad_idx)}")
print("=" * 80)
print(f"\nAll 5 PNGs saved:")
for name in ["simrefit_ther_lda_kde.png", "simrefit_ther_doseresponse.png",
             "simrefit_pert_lda_kde.png", "simrefit_pert_doseresponse.png",
             "simrefit_comparison.png"]:
    print(f"  {name}")
