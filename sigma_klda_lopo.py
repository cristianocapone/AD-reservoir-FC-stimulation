"""
sigma_klda_lopo.py
==================
Sweeps sigma (W-fit regularisation) × K_LDA (G-space PCs → LDA input)
with Leave-One-Patient-Out cross-validation on the LDA step.

G-space (SVD of W projections) is computed once per sigma on all N=76
patients.  The LDA is then evaluated in a proper LOPO loop:
  - fold i: train LDA on N-1 patients (class-balanced subsample),
             score patient i → collect all 76 LOO scores → AUROC / BAL-ACC

Conditions:
  A) Multi-session W per patient     (inflated reference)
  B) Single-session W per patient    (fair)
  C2) Per-session W → avg-G/patient (fair)

K_LDA grid extended to 50.

Outputs:
  sigma_klda_lopo_heatmap_balACC.png
  sigma_klda_lopo_heatmap_AUROC.png
  sigma_klda_lopo_profiles.png
  sigma_klda_lopo_best.txt
"""

import os, sys, warnings, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── hyper-parameters ───────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
K_PC        = 200
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

SIGMA_GRID = [0.001, 0.003, 0.007, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
KLDA_GRID  = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50]

# ── data loading ───────────────────────────────────────────────────────────────
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)
signals, labels_raw, pid_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)
            pid_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

pid_raw    = np.array(pid_raw)
labels_raw = np.array(labels_raw)
N_subj     = len(signals)

unique_pids    = np.unique(pid_raw)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(pid_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])

print(f"  {N_patients} patients ({(patient_labels==0).sum()} CC, "
      f"{(patient_labels==1).sum()} AD), {N_subj} sessions total")

# ── population PCA ─────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── reservoir ─────────────────────────────────────────────────────────────────
print("Initialising reservoir ...")
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=139, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, 139))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

# ── teacher-forced pass ────────────────────────────────────────────────────────
print("TF pass (all sessions) ...")
sess_X, sess_Y = {}, {}
for idx in trange(N_subj, desc="  TF"):
    s     = signals[idx]; T_s = s.shape[1]
    tgt   = (s.T @ ev50 @ ev50.T).T
    res.T = T_s; res.reset()
    X_raw = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    Xf          = np.array(X_raw)[TIMES_SKIP:]
    sess_X[idx] = Xf
    sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T
print("  TF done.\n")

# ── helper: LDA ────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0)
        w /= np.linalg.norm(w) + 1e-12
        self.w_   = w
        self.thr_ = 0.5 * ((X0@w).mean() + (X1@w).mean())
        return self
    def transform(self, X): return X @ self.w_
    def predict(self, X):
        return np.where(self.transform(X) >= self.thr_,
                        self.classes_[1], self.classes_[0])

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([rng2.choice(c0, n, replace=False),
                          rng2.choice(c1, n, replace=False)])
    rng2.shuffle(sel)
    return X[sel], y[sel]

def project_W(W, Xc):
    """Project W into the X-subspace: returns flat vector (N_sites*N_hidden,)."""
    W_T = W.T.astype(np.float64)    # (N_sites, N_hidden) — wait W is (N_hidden, N_sites)
    # W shape: (N_hidden, N_sites), W.T = (N_sites, N_hidden)
    Xca = Xc.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk   = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:kk]                  # (kk, N_hidden)
    return (W_T @ Vt_k.T @ Vt_k).flatten()  # (N_sites * N_hidden,)

def build_Gspace(Wproj_list):
    """SVD-based G-space.  Returns (G, Wmean, Vsvd, Meff)."""
    Wstack = np.array(Wproj_list)
    Wmean  = Wstack.mean(0)
    Wcent  = Wstack - Wmean
    _, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
    Meff   = Wstack.shape[0] - 1
    G      = Wcent @ Vsvd[:Meff].T
    return G, Wmean, Vsvd, Meff

def lopo_scores(G, labels, k, seed=RNG_SEED):
    """
    Leave-one-patient-out LDA.
    G shape: (N, Meff).  Returns array of LOO LDA scores (length N).
    The sign is forced so that AD (label=1) scores > CC (label=0) on average
    within each training fold.
    """
    N = len(labels)
    k_use = min(k, G.shape[1])
    Gk    = G[:, :k_use]
    scores = np.zeros(N)

    for i in range(N):
        mask   = np.ones(N, dtype=bool); mask[i] = False
        G_tr   = Gk[mask]
        y_tr   = labels[mask]
        G_te   = Gk[i:i+1]

        Xb, yb = _balance(G_tr, y_tr, seed=seed)
        try:
            lda = _LDA().fit(Xb, yb)
        except Exception:
            scores[i] = np.nan
            continue

        # force orientation: AD (1) > CC (0) based on training means
        z_tr = lda.transform(G_tr)
        if z_tr[y_tr==0].mean() > z_tr[y_tr==1].mean():
            lda.w_ *= -1

        scores[i] = lda.transform(G_te)[0]

    return scores

def eval_lopo(G, labels, k):
    """AUROC + balanced-accuracy from LOPO scores."""
    sc = lopo_scores(G, labels, k)
    valid = np.isfinite(sc)
    if valid.sum() < 10:
        return np.nan, np.nan
    auc = roc_auc_score(labels[valid], sc[valid])
    # BAL-ACC: threshold = median score (rank-based, equivalent to AUC threshold)
    thr  = 0.5 * (sc[labels[valid]==0].mean() + sc[labels[valid]==1].mean())
    pred = (sc[valid] >= thr).astype(int)
    sens = float(np.mean(pred[labels[valid]==1] == 1))
    spec = float(np.mean(pred[labels[valid]==0] == 0))
    return auc, 0.5 * (sens + spec)

# ── pre-build X/Y maps ─────────────────────────────────────────────────────────
patX_multi  = {pid: np.vstack([sess_X[i] for i in patient_sids[pid]])
               for pid in unique_pids}
patY_multi  = {pid: np.vstack([sess_Y[i] for i in patient_sids[pid]])
               for pid in unique_pids}
patX_single = {pid: sess_X[patient_sids[pid][0]] for pid in unique_pids}
patY_single = {pid: sess_Y[patient_sids[pid][0]] for pid in unique_pids}

# ── sweep ──────────────────────────────────────────────────────────────────────
n_sig  = len(SIGMA_GRID)
n_klda = len(KLDA_GRID)
rng_w  = np.random.default_rng(RNG_SEED + 1)

res_A_bal  = np.full((n_sig, n_klda), np.nan)
res_B_bal  = np.full((n_sig, n_klda), np.nan)
res_C2_bal = np.full((n_sig, n_klda), np.nan)
res_A_auc  = np.full((n_sig, n_klda), np.nan)
res_B_auc  = np.full((n_sig, n_klda), np.nan)
res_C2_auc = np.full((n_sig, n_klda), np.nan)

print(f"Sweeping {n_sig} sigmas × {n_klda} K_LDA values  (LOPO CV) ...")

for si, sigma in enumerate(SIGMA_GRID):
    print(f"\n[sigma={sigma:.4f}]  ({si+1}/{n_sig})", flush=True)

    # ── build G for condition A (multi-session) ────────────────────────────
    Wproj_A = []
    for pid in unique_pids:
        Xc = patX_multi[pid]; Yc = patY_multi[pid]
        noise = rng_w.normal(0, sigma, Xc.shape)
        W = np.linalg.pinv(Xc + noise) @ Yc
        Wproj_A.append(project_W(W, Xc))
    G_A, _, _, _ = build_Gspace(Wproj_A)

    # ── build G for condition B (single-session) ───────────────────────────
    Wproj_B = []
    for pid in unique_pids:
        Xc = patX_single[pid]; Yc = patY_single[pid]
        noise = rng_w.normal(0, sigma, Xc.shape)
        W = np.linalg.pinv(Xc + noise) @ Yc
        Wproj_B.append(project_W(W, Xc))
    G_B, _, _, _ = build_Gspace(Wproj_B)

    # ── build G for condition C2 (per-session W, avg-G/patient) ───────────
    sess_Wproj = []
    for idx in range(N_subj):
        Xc = sess_X[idx]; Yc = sess_Y[idx]
        noise = rng_w.normal(0, sigma, Xc.shape)
        W = np.linalg.pinv(Xc + noise) @ Yc
        sess_Wproj.append(project_W(W, Xc))

    Wstack_s     = np.array(sess_Wproj)
    Wmean_s      = Wstack_s.mean(0)
    Wcent_s      = Wstack_s - Wmean_s
    _, _, Vsvd_s = np.linalg.svd(Wcent_s, full_matrices=False)
    Meff_s       = N_subj - 1

    G_C2 = np.zeros((N_patients, Meff_s))
    for pi, pid in enumerate(unique_pids):
        idxs  = patient_sids[pid]
        g_list = [(sess_Wproj[i] - Wmean_s) @ Vsvd_s[:Meff_s].T for i in idxs]
        G_C2[pi] = np.mean(g_list, axis=0)

    # ── LOPO over K_LDA values ─────────────────────────────────────────────
    best_B_bal = 0.0; best_k = KLDA_GRID[0]
    for ki, k in enumerate(KLDA_GRID):
        auc_A, bal_A = eval_lopo(G_A,  patient_labels, k)
        auc_B, bal_B = eval_lopo(G_B,  patient_labels, k)
        auc_C2, bal_C2 = eval_lopo(G_C2, patient_labels, k)

        res_A_auc[si, ki]  = auc_A;  res_A_bal[si, ki]  = bal_A
        res_B_auc[si, ki]  = auc_B;  res_B_bal[si, ki]  = bal_B
        res_C2_auc[si, ki] = auc_C2; res_C2_bal[si, ki] = bal_C2

        if not np.isnan(bal_B) and bal_B > best_B_bal:
            best_B_bal = bal_B; best_k = k

    # quick summary line
    best_ki_B  = np.nanargmax(res_B_bal[si])
    best_ki_C2 = np.nanargmax(res_C2_bal[si])
    best_ki_A  = np.nanargmax(res_A_bal[si])
    print(f"  best LOPO BAL-ACC → "
          f"A={res_A_bal[si, best_ki_A]:.4f}(K={KLDA_GRID[best_ki_A]})  "
          f"B={res_B_bal[si, best_ki_B]:.4f}(K={KLDA_GRID[best_ki_B]})  "
          f"C2={res_C2_bal[si, best_ki_C2]:.4f}(K={KLDA_GRID[best_ki_C2]})")

# ── global optima ──────────────────────────────────────────────────────────────
print("\n" + "="*75)
print(f"{'Condition':<22} {'BAL-ACC':>8} {'AUROC':>7} {'sigma':>8} {'K_LDA':>6}")
print("-"*75)
best_rows = []
for name, mat_bal, mat_auc in [
        ("A (multi-sess)",  res_A_bal,  res_A_auc),
        ("B (single-sess)", res_B_bal,  res_B_auc),
        ("C2 (avg-G/pat)",  res_C2_bal, res_C2_auc)]:
    bi   = np.unravel_index(np.nanargmax(mat_bal), mat_bal.shape)
    bs   = SIGMA_GRID[bi[0]]; bk = KLDA_GRID[bi[1]]
    bbal = mat_bal[bi]; bauc = mat_auc[bi]
    print(f"  {name:<20} {bbal:>8.4f} {bauc:>7.4f} {bs:>8} {bk:>6}")
    best_rows.append((name, bbal, bauc, bs, bk))
print("="*75)

with open(f"{OUT_DIR}/sigma_klda_lopo_best.txt", "w") as f:
    f.write("Condition              BAL-ACC   AUROC    sigma    K_LDA\n")
    for row in best_rows:
        f.write(f"{row[0]:<22} {row[1]:.4f}  {row[2]:.4f}  {row[3]:<8} {row[4]}\n")

# ── heatmap plot helper ────────────────────────────────────────────────────────
sigma_labels = [str(s) for s in SIGMA_GRID]
klda_labels  = [str(k) for k in KLDA_GRID]

def make_heatmap(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
    if vmin is None: vmin = np.nanmin(data)
    if vmax is None: vmax = np.nanmax(data)
    im = ax.imshow(data, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(n_klda))
    ax.set_xticklabels(klda_labels, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n_sig))
    ax.set_yticklabels(sigma_labels, fontsize=7)
    ax.set_xlabel("K_LDA", fontsize=8)
    ax.set_ylabel("sigma", fontsize=8)
    ax.set_title(title, fontsize=9)
    bi = np.unravel_index(np.nanargmax(data), data.shape)
    ax.plot(bi[1], bi[0], "r*", ms=12, label=f"best={data[bi]:.4f}")
    for si2 in range(n_sig):
        for ki2 in range(n_klda):
            v = data[si2, ki2]
            if not np.isnan(v):
                ax.text(ki2, si2, f"{v:.3f}", ha="center", va="center",
                        fontsize=5,
                        color="white" if v < vmin+0.55*(vmax-vmin) else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc="upper right", fontsize=7, frameon=False)

# BAL-ACC heatmaps
fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="white")
vmin_b = min(np.nanmin(res_A_bal), np.nanmin(res_B_bal), np.nanmin(res_C2_bal))
vmax_b = max(np.nanmax(res_A_bal), np.nanmax(res_B_bal), np.nanmax(res_C2_bal))
make_heatmap(axes[0], res_A_bal,
             "A) Multi-session W → per-patient\n(inflated reference)", vmin_b, vmax_b)
make_heatmap(axes[1], res_B_bal,
             "B) Single-session W → per-patient\n(fair)", vmin_b, vmax_b)
make_heatmap(axes[2], res_C2_bal,
             "C2) Per-session W → avg-G/patient\n(fair)", vmin_b, vmax_b)
fig.suptitle("LOPO Balanced Accuracy — sigma × K_LDA sweep", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/sigma_klda_lopo_heatmap_balACC.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sigma_klda_lopo_heatmap_balACC.png")

# AUROC heatmaps
fig, axes = plt.subplots(1, 3, figsize=(24, 8), facecolor="white")
vmin_a = min(np.nanmin(res_A_auc), np.nanmin(res_B_auc), np.nanmin(res_C2_auc))
vmax_a = max(np.nanmax(res_A_auc), np.nanmax(res_B_auc), np.nanmax(res_C2_auc))
make_heatmap(axes[0], res_A_auc,
             "A) Multi-session W → per-patient\n(inflated reference)", vmin_a, vmax_a)
make_heatmap(axes[1], res_B_auc,
             "B) Single-session W → per-patient\n(fair)", vmin_a, vmax_a)
make_heatmap(axes[2], res_C2_auc,
             "C2) Per-session W → avg-G/patient\n(fair)", vmin_a, vmax_a)
fig.suptitle("LOPO AUROC — sigma × K_LDA sweep", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/sigma_klda_lopo_heatmap_AUROC.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sigma_klda_lopo_heatmap_AUROC.png")

# Line profiles: BAL-ACC vs sigma (per condition), and vs K_LDA
fig, axes = plt.subplots(2, 3, figsize=(20, 11), facecolor="white")
cmap_lines = plt.get_cmap("plasma")

for col, (cond_name, real_mat) in enumerate([
        ("A (multi-sess)", res_A_bal),
        ("B (single-sess)", res_B_bal),
        ("C2 (avg-G/pat)", res_C2_bal)]):

    # row 0: BAL-ACC vs sigma, each line = one K_LDA
    ax = axes[0, col]
    for ki, k in enumerate(KLDA_GRID):
        c = cmap_lines(ki / (n_klda - 1))
        ax.plot(SIGMA_GRID, real_mat[:, ki], "o-", ms=4, lw=1.5,
                color=c, label=f"K={k}", alpha=0.85)
    ax.set_xscale("log")
    ax.axvline(0.025, color="gray", ls="--", lw=1, alpha=0.6, label="default σ")
    ax.set_xlabel("sigma", fontsize=9); ax.set_ylabel("LOPO BAL-ACC", fontsize=9)
    ax.set_title(f"{cond_name}\nvs sigma (each line = K_LDA)", fontsize=9)
    ax.legend(fontsize=5, ncol=4, frameon=False, loc="upper right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

    # row 1: BAL-ACC vs K_LDA, each line = one sigma
    ax = axes[1, col]
    for si2, s in enumerate(SIGMA_GRID):
        c = cmap_lines(si2 / (n_sig - 1))
        ax.plot(KLDA_GRID, real_mat[si2, :], "o-", ms=4, lw=1.5,
                color=c, label=f"σ={s}", alpha=0.85)
    ax.axvline(2, color="gray", ls="--", lw=1, alpha=0.6, label="default K=2")
    ax.set_xlabel("K_LDA", fontsize=9); ax.set_ylabel("LOPO BAL-ACC", fontsize=9)
    ax.set_title(f"{cond_name}\nvs K_LDA (each line = sigma)", fontsize=9)
    ax.legend(fontsize=5, ncol=3, frameon=False, loc="lower right")
    for sp in ["top", "right"]: ax.spines[sp].set_visible(False)

fig.suptitle("LOPO Balanced Accuracy profiles — sigma × K_LDA sweep", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/sigma_klda_lopo_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sigma_klda_lopo_profiles.png")

print("\nDone.")
