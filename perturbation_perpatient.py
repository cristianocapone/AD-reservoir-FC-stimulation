"""
perturbation_perpatient.py
==========================
Redo the perturbation experiment (Fig 3) using per-patient W with sigma=0.5.

Two perturbation types:
  A) Therapeutic offline interpolation:
       W_interp = (1-alpha)*W_pat + alpha*W_cc_mean   [AD only]
  B) Top-5-site perturbation:
       The 5 columns of W with largest AD-CC weight difference are replaced

Both evaluated via:
  - G-score projection of W_interp into archetype space
  - LDA score (from LDA trained on ALL patients' original G-scores)
  - Closed-loop FC similarity to CC mean

Saves 6 PNGs (matching Fig3 panels):
  therapeutic_lda_kde.png      therapeutic_doseresponse.png
  pert_lda_kde.png             pert_doseresponse.png
  pert_per_patient.png         pert_site_importance.png
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

# ── Config ─────────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
NOISE_SIZE  = 0.025        # per-patient W noise (same as 85% classification)
K_LDA       = 2            # best k at noise=0.025 (from noise_ba_sweep.py)
TS_ROOT     = "./timeseries"
OUT_DIR     = "."          # save PNGs in the root (same as original notebook)

ALPHA_THERAPEUTIC = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ALPHA_SINGLESITE  = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

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

cc_idx = np.where(patient_labels == 0)[0]   # CC patient indices
ad_idx = np.where(patient_labels == 1)[0]   # AD patient indices

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

# ── Teacher-forced X, Y (all sessions, reset between sessions) ─────────────────
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

# Per-patient concatenated target (for closed-loop evaluation)
pat_target = {pid: np.concatenate([sess_target[i] for i in patient_sids[pid]], axis=1)
              for pid in unique_pids}

# ── Per-patient W (noise=0.5) ──────────────────────────────────────────────────
print(f"Per-patient W (noise={NOISE_SIZE}) ...")
rng_p   = np.random.default_rng(RNG_SEED + 1)
pat_W   = {}   # pid -> W (N_hidden, N_sites)
pat_X   = {}   # pid -> X_coll (T_total, N_hidden)

for pid in tqdm(unique_pids, desc="  Fitting W"):
    idxs   = patient_sids[pid]
    X_coll = np.vstack([sess_X[i] for i in idxs])
    Y_coll = np.vstack([sess_Y[i] for i in idxs])
    noise  = rng_p.normal(0, NOISE_SIZE, X_coll.shape)
    pat_W[pid]  = np.linalg.pinv(X_coll + noise) @ Y_coll
    pat_X[pid]  = X_coll

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
G_pat    = Wcent @ Vt_svd[:M_eff].T       # (N_patients, M_eff)
print(f"  G_pat shape: {G_pat.shape}")

# ── LDA on ALL patients (no LOPO — for perturbation visualisation) ─────────────
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

# Orient: CC on left (negative)
Z_all = lda.transform(G_pat[:, :K_LDA])
if Z_all[patient_labels == 0].mean() > Z_all[patient_labels == 1].mean():
    lda.w_ *= -1; lda.thr_ *= -1
Z_all = lda.transform(G_pat[:, :K_LDA])

y_lda = patient_labels   # 0=CC, 1=AD (all patients)
train_acc = np.mean(lda.predict(G_pat[:, :K_LDA]) == patient_labels)
print(f"  LDA train accuracy (all data): {train_acc*100:.1f}%")
print(f"  CC mean LDA: {Z_all[y_lda==0].mean():.3f}   AD mean LDA: {Z_all[y_lda==1].mean():.3f}")

# ── CC mean W and FC ───────────────────────────────────────────────────────────
W_cc_mean = np.mean([pat_W[unique_pids[i]] for i in cc_idx], axis=0)  # (N_hidden, N_sites)

# CC mean FC (from closed-loop of individual CC patients)
print("Computing CC mean FC (closed-loop) ...")
IU = np.triu_indices(N_SITES, k=1)
fc_cc_list = []
for i in tqdm(cc_idx, desc="  CC closed-loop", leave=False):
    pid    = unique_pids[i]
    tgt    = pat_target[pid]
    W      = pat_W[pid]
    T      = tgt.shape[1]
    res.T  = T; res.reset()
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
    res.Jout = W.T.copy(); res.y = res.Jout @ res.X
    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())
    Y_sim = np.array(Y_sim)[TIMES_SKIP:].T
    fc_cc_list.append(np.nan_to_num(np.corrcoef(Y_sim)).flatten())
FC_cc_mean = np.mean(fc_cc_list, axis=0)

# ── Helper: closed-loop FC ─────────────────────────────────────────────────────
def closed_loop_fc(W, pid):
    """Teacher-force concat signal of pid, then autonomous run with W. Returns FC matrix."""
    tgt = pat_target[pid]
    T   = tgt.shape[1]
    res.T = T; res.reset()
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
    res.Jout = W.T.copy(); res.y = res.Jout @ res.X
    Y_sim = []
    for t in range(T - 1):
        res.step_rate(ff * res.y, sigma_dyn=0.)
        Y_sim.append(res.y.copy())
    Y_sim = np.array(Y_sim)[TIMES_SKIP:].T
    return np.nan_to_num(np.corrcoef(Y_sim))

def w_to_g(W, X_coll):
    """Project W into G-score space (per-patient archetype projection)."""
    W_T  = W.T.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(X_coll.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    w_proj = (W_T @ Vt_k.T @ Vt_k).flatten()
    g      = (w_proj - W_mean) @ Vt_svd[:M_eff].T
    return g[:K_LDA]

# ══════════════════════════════════════════════════════════════════════════════
# PART A — Therapeutic offline interpolation
# ══════════════════════════════════════════════════════════════════════════════
print("\n[A] Therapeutic offline interpolation ...")
print(f"  W_interp = (1-alpha)*W_pat + alpha*W_cc_mean  [AD only]")

res_ther = {a: dict(g=[], fc_corr=[], z=None, pred=None, auc=None) for a in ALPHA_THERAPEUTIC}

for alpha in ALPHA_THERAPEUTIC:
    print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
    g_list, fc_list = [], []

    for pid_i, pid in enumerate(unique_pids):
        is_ad = (patient_labels[pid_i] == 1)
        W_pat = pat_W[pid]
        W_int = (1-alpha)*W_pat + alpha*W_cc_mean if is_ad else W_pat

        FC_sim = closed_loop_fc(W_int, pid)
        fc_r   = np.corrcoef(FC_sim.flatten(), FC_cc_mean)[0, 1]
        g_new  = w_to_g(W_int, pat_X[pid])

        g_list.append(g_new)
        fc_list.append(fc_r)

    G_new   = np.array(g_list)
    z_new   = lda.transform(G_new)
    pred    = lda.predict(G_new)
    auc_val = roc_auc_score(patient_labels, z_new)
    res_ther[alpha]['g']       = G_new
    res_ther[alpha]['fc_corr'] = np.array(fc_list)
    res_ther[alpha]['z']       = z_new
    res_ther[alpha]['pred']    = pred
    res_ther[alpha]['auc']     = auc_val
    frac_ad_cc = np.mean(pred[y_lda==1] == 0)
    print(f"AD->CC={frac_ad_cc:.2f}  LDA(CC)={z_new[y_lda==0].mean():.2f}  LDA(AD)={z_new[y_lda==1].mean():.2f}  AUROC={auc_val:.4f}")

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
        try:
            kde = gaussian_kde(pts, bw_method="scott")
        except Exception:
            s = max(float(pts.std()), 1e-4)
            kde = gaussian_kde(pts + np.random.normal(0, s*1e-3, pts.shape), bw_method="scott")
        ax.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
        ax.plot(xs, kde(xs), color=col, lw=1.5)
        ax.axvline(pts.mean(), color=col, ls="--", lw=1)
    ax.set_ylabel(f"a={alpha:.2f}", fontsize=10)
    ax.set_yticks([])
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)

axes[0].legend(frameon=False, fontsize=9, loc="upper right")
axes[-1].set_xlabel("LDA score  (CC <- -> AD)", fontsize=11)
fig.suptitle("Offline interpolation — LDA score distribution\n(per-patient W, noise=0.025)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/therapeutic_lda_kde.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved therapeutic_lda_kde.png")

# ── Plot A2: dose-response ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(19, 4), facecolor="white")

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
    se = [res_ther[a]["z"][y_lda==cls].std()/np.sqrt((y_lda==cls).sum()) for a in ALPHA_THERAPEUTIC]
    ax.plot(ALPHA_THERAPEUTIC, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_THERAPEUTIC, np.array(ms)-se, np.array(ms)+se, color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("Mean LDA score")
ax.set_title("Mean LDA score (CC ← → AD)")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[2]
for pidx_list, col, name in [(cc_idx, col_cc, "CC"), (ad_idx, col_ad, "AD")]:
    ms = [res_ther[a]["fc_corr"][pidx_list].mean() for a in ALPHA_THERAPEUTIC]
    se = [res_ther[a]["fc_corr"][pidx_list].std()/np.sqrt(len(pidx_list)) for a in ALPHA_THERAPEUTIC]
    ax.plot(ALPHA_THERAPEUTIC, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_THERAPEUTIC, np.array(ms)-se, np.array(ms)+se, color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("FC corr with CC mean")
ax.set_title("FC similarity to CC template")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[3]
aucs_a = [res_ther[a]["auc"] for a in ALPHA_THERAPEUTIC]
ax.plot(ALPHA_THERAPEUTIC, aucs_a, "-o", color="#6A1B9A", lw=2, label="AUROC")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
ax.set_xlabel("Dose α"); ax.set_ylabel("AUROC")
ax.set_title("AD vs CC discriminability\n(AUROC, fixed LDA)")
ax.set_ylim(0.40, 1.00)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Offline interpolation dose-response  (per-patient W, noise=0.025)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/therapeutic_doseresponse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved therapeutic_doseresponse.png")

# ══════════════════════════════════════════════════════════════════════════════
# PART B — Top-5-site perturbation (AD only) — static column replacement
#
# For each AD patient:
#   1. Identify top-5 sites by column norm of dW = W_cc_mean - W_pat
#   2. Replace those columns with a linear mix: W_int[:, top5] = (1-alpha)*W_pat + alpha*W_cc_mean
#      where alpha ∈ [0, 1] is the direct mix fraction
#   3. Project W_int into G-space → LDA score
#   4. Run closed-loop with W_int → compute FC-r vs CC mean
#
# Note: The online-perturbation + re-fit approach (as in the original notebook)
# requires high FC-r for the autonomous dynamics to be stable. At sigma=0.025
# the per-patient FC-r is ~0.46, insufficient for that method.  The static
# approach here is self-consistent and shows a genuine partial dose-response.
# ══════════════════════════════════════════════════════════════════════════════
print("\n[B] Top-5-site perturbation (static column mix) ...")

res_pert = {a: dict(g=[], fc_corr=[], z=None, pred=None, top_site=[], auc=None)
            for a in ALPHA_SINGLESITE}

# For site-importance counting across AD patients
site_counts = np.zeros(N_SITES, dtype=int)

for alpha in ALPHA_SINGLESITE:
    print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
    g_list, fc_list, site_list = [], [], []

    for pid_i, pid in enumerate(unique_pids):
        is_ad = (patient_labels[pid_i] == 1)
        W_pat = pat_W[pid]

        if is_ad:
            dW              = W_cc_mean - W_pat              # (N_hidden, N_sites)
            site_importance = np.linalg.norm(dW, axis=0)    # (N_sites,)
            top5_sites      = np.argsort(site_importance)[::-1][:5]
            site_list.append(top5_sites.tolist())
            if alpha == ALPHA_SINGLESITE[1]:                 # count at first non-zero dose
                site_counts[top5_sites] += 1

            # Static mix at top-5 sites only; all other columns unchanged
            W_int = W_pat.copy()
            W_int[:, top5_sites] = ((1 - alpha) * W_pat[:, top5_sites]
                                    + alpha        * W_cc_mean[:, top5_sites])
        else:
            W_int = W_pat
            site_list.append(-1)

        FC_sim = closed_loop_fc(W_int, pid)
        fc_r   = np.corrcoef(FC_sim.flatten(), FC_cc_mean)[0, 1]
        g_new  = w_to_g(W_int, pat_X[pid])

        g_list.append(g_new)
        fc_list.append(fc_r)

    G_new  = np.array(g_list)
    z_new  = lda.transform(G_new)
    pred   = lda.predict(G_new)
    auc_val = roc_auc_score(patient_labels, z_new)
    res_pert[alpha]['g']        = G_new
    res_pert[alpha]['fc_corr']  = np.array(fc_list)
    res_pert[alpha]['z']        = z_new
    res_pert[alpha]['pred']     = pred
    res_pert[alpha]['top_site'] = site_list
    res_pert[alpha]['auc']      = auc_val
    frac = np.mean(pred[y_lda==1] == 0)
    print(f"AD->CC={frac:.2f}  LDA(AD)={z_new[y_lda==1].mean():.2f}  AUROC={auc_val:.4f}")

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
        try:
            kde = gaussian_kde(pts, bw_method="scott")
        except Exception:
            s = max(float(pts.std()), 1e-4)
            kde = gaussian_kde(pts + np.random.normal(0, s*1e-3, pts.shape), bw_method="scott")
        ax.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
        ax.plot(xs, kde(xs), color=col, lw=1.5)
        ax.axvline(pts.mean(), color=col, ls="--", lw=1)
    ax.set_ylabel(f"a={alpha:.1f}", fontsize=10)
    ax.set_yticks([])
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)

axes[0].legend(frameon=False, fontsize=9, loc="upper right")
axes[-1].set_xlabel("LDA score  (CC <- -> AD)", fontsize=11)
fig.suptitle("Top-5-site perturbation — LDA score distribution\n(per-patient W, noise=0.025)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert_lda_kde.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert_lda_kde.png")

# ── Plot B2: dose-response ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(19, 4), facecolor="white")

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
    se = [res_pert[a]["z"][y_lda==cls].std()/np.sqrt((y_lda==cls).sum()) for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_SINGLESITE, np.array(ms)-se, np.array(ms)+se, color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("Mean LDA score")
ax.set_title("Mean LDA score (CC ← → AD)")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[2]
for pidx_list, col, name in [(cc_idx, col_cc, "CC"), (ad_idx, col_ad, "AD")]:
    ms = [res_pert[a]["fc_corr"][pidx_list].mean() for a in ALPHA_SINGLESITE]
    se = [res_pert[a]["fc_corr"][pidx_list].std()/np.sqrt(len(pidx_list)) for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_SINGLESITE, np.array(ms)-se, np.array(ms)+se, color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("FC corr with CC mean")
ax.set_title("FC similarity to CC template")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[3]
aucs_b = [res_pert[a]["auc"] for a in ALPHA_SINGLESITE]
ax.plot(ALPHA_SINGLESITE, aucs_b, "-o", color="#6A1B9A", lw=2, label="AUROC")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
ax.set_xlabel("Dose α"); ax.set_ylabel("AUROC")
ax.set_title("AD vs CC discriminability\n(AUROC, fixed LDA)")
ax.set_ylim(0.40, 1.00)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Top-5-site perturbation dose-response  (per-patient W, noise=0.025)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert_doseresponse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert_doseresponse.png")

# ── Plot B3: per-patient LDA trajectory (single-site, AD patients) ────────────
fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

z_baseline = res_pert[ALPHA_SINGLESITE[0]]["z"]
z_cc_mean  = z_baseline[y_lda == 0].mean()

for pid_i in ad_idx:
    zs = [res_pert[a]["z"][pid_i] for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, zs, "-o", color=col_ad, alpha=0.25, lw=1, ms=3)

# CC band (baseline)
z_cc_vals = res_pert[ALPHA_SINGLESITE[0]]["z"][y_lda==0]
ax.axhspan(z_cc_vals.mean()-z_cc_vals.std(),
           z_cc_vals.mean()+z_cc_vals.std(),
           color=col_cc, alpha=0.15, label="CC +/-1SD")
ax.axhline(z_cc_vals.mean(), color=col_cc, lw=1.5, ls="--", label="CC mean")

# Mean AD trajectory
z_ad_means = [res_pert[a]["z"][y_lda==1].mean() for a in ALPHA_SINGLESITE]
ax.plot(ALPHA_SINGLESITE, z_ad_means, "-o", color=col_ad, lw=2.5, ms=8,
        label="AD mean", zorder=5)

ax.set_xlabel("Dose a (top-5-site perturbation)", fontsize=11)
ax.set_ylabel("LDA score", fontsize=11)
ax.set_title("Per-patient LDA score trajectories\n(AD patients, top-5-site, per-patient W, noise=0.025)",
             fontsize=10)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert_per_patient.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert_per_patient.png")

# ── Plot B4: site importance map ───────────────────────────────────────────────
# Compute mean dW column norms across all AD patients
mean_site_importance = np.zeros(N_SITES)
for i in ad_idx:
    pid = unique_pids[i]
    dW  = W_cc_mean - pat_W[pid]
    mean_site_importance += np.linalg.norm(dW, axis=0)
mean_site_importance /= len(ad_idx)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")

# Left: site importance bar chart (top 20)
ax = axes[0]
top20 = np.argsort(mean_site_importance)[::-1][:20]
ax.bar(range(20), mean_site_importance[top20], color=col_ad, alpha=0.8)
ax.set_xticks(range(20))
ax.set_xticklabels([str(s) for s in top20], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Brain site index")
ax.set_ylabel("Mean ||dW|| column norm (AD-CC)")
ax.set_title("Top-20 sites: AD-CC weight difference\n(per-patient W, noise=0.025)")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

# Right: site selection frequency (at first non-zero dose)
ax = axes[1]
top_freq = np.argsort(site_counts)[::-1][:15]
freq_vals = site_counts[top_freq]
ax.bar(range(len(top_freq)), freq_vals, color="#7B1FA2", alpha=0.8)
ax.set_xticks(range(len(top_freq)))
ax.set_xticklabels([str(s) for s in top_freq], rotation=60, ha="right", fontsize=8)
ax.set_xlabel("Brain site index")
ax.set_ylabel(f"n AD patients selecting this site (out of {len(ad_idx)})")
ax.set_title("Most frequently selected AD->CC perturbation site")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert_site_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert_site_importance.png")

# ══════════════════════════════════════════════════════════════════════════════
# PART C — Single-site perturbation (top-1 site per AD patient)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[C] Single-site perturbation (top-1 site, static column mix) ...")

res_pert1 = {a: dict(g=[], fc_corr=[], z=None, pred=None, auc=None)
             for a in ALPHA_SINGLESITE}

site1_counts = np.zeros(N_SITES, dtype=int)

for alpha in ALPHA_SINGLESITE:
    print(f"  alpha={alpha:.2f} ...", end=" ", flush=True)
    g_list, fc_list = [], []

    for pid_i, pid in enumerate(unique_pids):
        is_ad = (patient_labels[pid_i] == 1)
        W_pat = pat_W[pid]

        if is_ad:
            dW             = W_cc_mean - W_pat
            site_imp       = np.linalg.norm(dW, axis=0)
            top1_site      = np.array([np.argmax(site_imp)])   # single best site
            if alpha == ALPHA_SINGLESITE[1]:
                site1_counts[top1_site] += 1
            W_int = W_pat.copy()
            W_int[:, top1_site] = ((1 - alpha) * W_pat[:, top1_site]
                                   + alpha        * W_cc_mean[:, top1_site])
        else:
            W_int = W_pat

        FC_sim = closed_loop_fc(W_int, pid)
        fc_r   = np.corrcoef(FC_sim.flatten(), FC_cc_mean)[0, 1]
        g_new  = w_to_g(W_int, pat_X[pid])
        g_list.append(g_new)
        fc_list.append(fc_r)

    G_new  = np.array(g_list)
    z_new  = lda.transform(G_new)
    pred   = lda.predict(G_new)
    auc_val = roc_auc_score(patient_labels, z_new)
    res_pert1[alpha]['g']       = G_new
    res_pert1[alpha]['fc_corr'] = np.array(fc_list)
    res_pert1[alpha]['z']       = z_new
    res_pert1[alpha]['pred']    = pred
    res_pert1[alpha]['auc']     = auc_val
    frac = np.mean(pred[y_lda==1] == 0)
    print(f"AD->CC={frac:.2f}  LDA(AD)={z_new[y_lda==1].mean():.2f}  AUROC={auc_val:.4f}")

# ── Plot C1: LDA KDE ridgeline ─────────────────────────────────────────────────
fig, axes = plt.subplots(len(ALPHA_SINGLESITE), 1,
                         figsize=(7, 1.6*len(ALPHA_SINGLESITE)), sharex=True,
                         facecolor="white")
all_z = np.concatenate([res_pert1[a]['z'] for a in ALPHA_SINGLESITE])
xs    = np.linspace(all_z.min()-0.5, all_z.max()+0.5, 400)

for ax, alpha in zip(axes, ALPHA_SINGLESITE):
    z = res_pert1[alpha]['z']
    for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
        pts = z[y_lda == cls]
        if len(pts) < 3: continue
        try:
            kde = gaussian_kde(pts, bw_method="scott")
        except Exception:
            s = max(float(pts.std()), 1e-4)
            kde = gaussian_kde(pts + np.random.normal(0, s*1e-3, pts.shape), bw_method="scott")
        ax.fill_between(xs, kde(xs), alpha=0.35, color=col, label=name)
        ax.plot(xs, kde(xs), color=col, lw=1.5)
        ax.axvline(pts.mean(), color=col, ls="--", lw=1)
    ax.set_ylabel(f"a={alpha:.1f}", fontsize=10)
    ax.set_yticks([])
    for sp in ["top","right","left"]: ax.spines[sp].set_visible(False)

axes[0].legend(frameon=False, fontsize=9, loc="upper right")
axes[-1].set_xlabel("LDA score  (CC <- -> AD)", fontsize=11)
fig.suptitle("Single-site perturbation — LDA score distribution\n(per-patient W, noise=0.025)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert1_lda_kde.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert1_lda_kde.png")

# ── Plot C2: dose-response ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(19, 4), facecolor="white")

ax = axes[0]
for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
    fracs = [np.mean(res_pert1[a]["pred"][y_lda==cls] == 0) for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, fracs, "-o", color=col, label=name, lw=2)
ax.axhline(0.5, color="gray", ls=":", lw=1)
ax.set_xlabel("Dose α"); ax.set_ylabel("Fraction predicted as CC")
ax.set_title("Fraction classified as CC"); ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[1]
for cls, col, name in [(0, col_cc, "CC"), (1, col_ad, "AD")]:
    ms = [res_pert1[a]["z"][y_lda==cls].mean() for a in ALPHA_SINGLESITE]
    se = [res_pert1[a]["z"][y_lda==cls].std()/np.sqrt((y_lda==cls).sum()) for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_SINGLESITE, np.array(ms)-se, np.array(ms)+se, color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("Mean LDA score")
ax.set_title("Mean LDA score (CC ← → AD)")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[2]
for pidx_list, col, name in [(cc_idx, col_cc, "CC"), (ad_idx, col_ad, "AD")]:
    ms = [res_pert1[a]["fc_corr"][pidx_list].mean() for a in ALPHA_SINGLESITE]
    se = [res_pert1[a]["fc_corr"][pidx_list].std()/np.sqrt(len(pidx_list)) for a in ALPHA_SINGLESITE]
    ax.plot(ALPHA_SINGLESITE, ms, "-o", color=col, label=name, lw=2)
    ax.fill_between(ALPHA_SINGLESITE, np.array(ms)-se, np.array(ms)+se, color=col, alpha=0.2)
ax.set_xlabel("Dose α"); ax.set_ylabel("FC corr with CC mean")
ax.set_title("FC similarity to CC template")
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

ax = axes[3]
aucs_c = [res_pert1[a]["auc"] for a in ALPHA_SINGLESITE]
ax.plot(ALPHA_SINGLESITE, aucs_c, "-o", color="#6A1B9A", lw=2, label="AUROC")
ax.axhline(0.5, color="gray", ls=":", lw=1, label="chance")
ax.set_xlabel("Dose α"); ax.set_ylabel("AUROC")
ax.set_title("AD vs CC discriminability\n(AUROC, fixed LDA)")
ax.set_ylim(0.40, 1.00)
ax.legend(frameon=False, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

fig.suptitle("Single-site perturbation dose-response  (per-patient W, noise=0.025)",
             fontsize=12, fontweight="bold", y=1.01)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert1_doseresponse.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert1_doseresponse.png")

# ── Plot C3: overlay comparison — top-1 vs top-5 vs therapeutic (AUROC) ───────
fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

ax.plot(ALPHA_THERAPEUTIC, [res_ther[a]["auc"] for a in ALPHA_THERAPEUTIC],
        "-o", color="#1565C0", lw=2.5, label="Full-W (therapeutic)")
ax.plot(ALPHA_SINGLESITE,  [res_pert[a]["auc"]  for a in ALPHA_SINGLESITE],
        "-s", color="#E64A19", lw=2.5, label="Top-5-site")
ax.plot(ALPHA_SINGLESITE,  [res_pert1[a]["auc"] for a in ALPHA_SINGLESITE],
        "-^", color="#2E7D32", lw=2.5, label="Single-site (top-1)")
ax.axhline(0.5, color="gray", ls=":", lw=1.2, label="chance")
ax.set_xlabel("Dose α", fontsize=11)
ax.set_ylabel("AUROC", fontsize=11)
ax.set_title("AUROC vs dose: full-W vs top-5 vs single-site\n(AD vs CC discriminability, fixed LDA)",
             fontsize=10)
ax.set_ylim(0.40, 1.00)
ax.legend(frameon=True, framealpha=0.9, fontsize=9)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/pert_auroc_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved pert_auroc_comparison.png")

print("\nAll 9 perturbation PNGs saved.")
print("\nSummary:")
print(f"  Therapeutic    alpha=1.0:  "
      f"AD->CC = {np.mean(res_ther[1.0]['pred'][y_lda==1]==0)*100:.0f}%  "
      f"LDA(AD) = {res_ther[1.0]['z'][y_lda==1].mean():.3f}  "
      f"AUROC = {res_ther[1.0]['auc']:.4f}")
print(f"  Top-5-site     alpha={ALPHA_SINGLESITE[-1]:.1f}: "
      f"AD->CC = {np.mean(res_pert[ALPHA_SINGLESITE[-1]]['pred'][y_lda==1]==0)*100:.0f}%  "
      f"LDA(AD) = {res_pert[ALPHA_SINGLESITE[-1]]['z'][y_lda==1].mean():.3f}  "
      f"AUROC = {res_pert[ALPHA_SINGLESITE[-1]]['auc']:.4f}")
print(f"  Single-site    alpha={ALPHA_SINGLESITE[-1]:.1f}: "
      f"AD->CC = {np.mean(res_pert1[ALPHA_SINGLESITE[-1]]['pred'][y_lda==1]==0)*100:.0f}%  "
      f"LDA(AD) = {res_pert1[ALPHA_SINGLESITE[-1]]['z'][y_lda==1].mean():.3f}  "
      f"AUROC = {res_pert1[ALPHA_SINGLESITE[-1]]['auc']:.4f}")
print(f"  Most perturbed site (top-5): {np.argmax(site_counts)} "
      f"(selected by {site_counts.max()}/{len(ad_idx)} AD patients)")
print(f"  Most perturbed site (top-1): {np.argmax(site1_counts)} "
      f"(selected by {site1_counts.max()}/{len(ad_idx)} AD patients)")
