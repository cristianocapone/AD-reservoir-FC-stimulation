"""
sigma_klda_sweep.py
===================
Sweeps:
  • noise_size (sigma) for W fitting: 10 values log-spaced 0.001 → 2.0
  • K_LDA: number of G-space PCs fed into Fisher LDA: 1 → 20

Conditions evaluated (fair ones + A for reference):
  A) Multi-session W → per-patient  (inflated baseline)
  B) Single-session W → per-patient (fair)
  C2) Per-session W → avg-G per patient (fair)

Saves:
  sigma_klda_heatmap_balACC.png   2-D heatmaps of balanced accuracy
  sigma_klda_heatmap_AUROC.png    2-D heatmaps of AUROC
  sigma_klda_best.txt             best (sigma, K_LDA) per condition
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
K_PC        = 200        # SVD components of X used for W→G projection
TIMES_SKIP  = 10
ff          = 0.1
N_HIDDEN    = 2000
TS_ROOT     = "./timeseries"
OUT_DIR     = "."

SIGMA_GRID  = [0.001, 0.003, 0.007, 0.015, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
KLDA_GRID   = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]

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

print(f"  {N_patients} patients ({(patient_labels==0).sum()} CC, {(patient_labels==1).sum()} AD), "
      f"{N_subj} sessions total")

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

# ── teacher-forced pass (all sessions) ────────────────────────────────────────
print("TF pass (all sessions) ...")
sess_X, sess_Y, sess_tgt = {}, {}, {}
for idx in trange(N_subj, desc="  TF"):
    s      = signals[idx]
    T_s    = s.shape[1]
    tgt    = (s.T @ ev50 @ ev50.T).T
    res.T  = T_s; res.reset()
    X_raw  = []
    for t in range(T_s - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.)
        X_raw.append(res.X.copy())
    Xf          = np.array(X_raw)[TIMES_SKIP:]
    sess_X[idx] = Xf
    sess_Y[idx] = tgt[:, TIMES_SKIP:TIMES_SKIP+len(Xf)].T
    sess_tgt[idx] = tgt

print("  TF done.\n")

# ── helpers ────────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
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
    n   = min(len(c0), len(c1))
    sel = np.concatenate([rng2.choice(c0, n, replace=False),
                          rng2.choice(c1, n, replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

def _bal_acc(lda, G, labels, k):
    pred = lda.predict(G[:, :k])
    sens = float(np.mean(pred[labels==1] == 1))
    spec = float(np.mean(pred[labels==0] == 0))
    return 0.5*(sens+spec)

def _auc(lda, G, labels, k):
    Z = lda.transform(G[:, :k])
    if Z[labels==0].mean() > Z[labels==1].mean(): Z = -Z
    return roc_auc_score(labels, Z)

def _fit_lda(G, labels, k):
    Xl, yl = _balance(G[:, :k], labels, seed=RNG_SEED)
    lda = _LDA().fit(Xl, yl)
    Z   = lda.transform(G[:, :k])
    if Z[labels==0].mean() > Z[labels==1].mean():
        lda.w_ *= -1; lda.thr_ *= -1
    return lda

def project_W(W, Xc):
    """W (N_sites, N_hidden) → flat projected vector in G-space input."""
    W_T = W.T.astype(np.float64)
    Xca = Xc.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(Xca, full_matrices=False)
    kk   = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:kk]
    return (W_T @ Vt_k.T @ Vt_k).flatten()

def build_G(Wproj_list):
    """SVD-based G-space from list of projected W vectors."""
    Wstack = np.array(Wproj_list)
    Wmean  = Wstack.mean(0)
    Wcent  = Wstack - Wmean
    _, _, Vsvd = np.linalg.svd(Wcent, full_matrices=False)
    Meff   = Wstack.shape[0] - 1
    G      = Wcent @ Vsvd[:Meff].T
    return G, Wmean, Vsvd, Meff

# ── pre-build X maps for multi / single session ────────────────────────────────
patX_multi  = {pid: np.vstack([sess_X[i] for i in patient_sids[pid]])
               for pid in unique_pids}
patY_multi  = {pid: np.vstack([sess_Y[i] for i in patient_sids[pid]])
               for pid in unique_pids}
patX_single = {pid: sess_X[patient_sids[pid][0]] for pid in unique_pids}
patY_single = {pid: sess_Y[patient_sids[pid][0]] for pid in unique_pids}

# ── sweep ──────────────────────────────────────────────────────────────────────
n_sig  = len(SIGMA_GRID)
n_klda = len(KLDA_GRID)

# result arrays: [n_sig, n_klda]
res_A_bal  = np.zeros((n_sig, n_klda))
res_B_bal  = np.zeros((n_sig, n_klda))
res_C2_bal = np.zeros((n_sig, n_klda))
res_A_auc  = np.zeros((n_sig, n_klda))
res_B_auc  = np.zeros((n_sig, n_klda))
res_C2_auc = np.zeros((n_sig, n_klda))

print(f"Sweeping {n_sig} sigma values × {n_klda} K_LDA values ...")
rng_w = np.random.default_rng(RNG_SEED + 1)

for si, sigma in enumerate(SIGMA_GRID):
    print(f"\n[sigma={sigma:.4f}] ({si+1}/{n_sig})")

    # ── condition A: multi-session per patient ─────────────────────────────
    Wproj_A = []
    for pid in unique_pids:
        Xc = patX_multi[pid]; Yc = patY_multi[pid]
        noise = rng_w.normal(0, sigma, Xc.shape)
        W = np.linalg.pinv(Xc + noise) @ Yc
        Wproj_A.append(project_W(W, Xc))
    G_A, _, _, _ = build_G(Wproj_A)

    # ── condition B: single session per patient ────────────────────────────
    Wproj_B = []
    for pid in unique_pids:
        Xc = patX_single[pid]; Yc = patY_single[pid]
        noise = rng_w.normal(0, sigma, Xc.shape)
        W = np.linalg.pinv(Xc + noise) @ Yc
        Wproj_B.append(project_W(W, Xc))
    G_B, _, _, _ = build_G(Wproj_B)

    # ── condition C2: per-session W, avg G per patient ─────────────────────
    # Step 1: fit W and project per session
    sess_Wproj = []
    for idx in range(N_subj):
        Xc = sess_X[idx]; Yc = sess_Y[idx]
        noise = rng_w.normal(0, sigma, Xc.shape)
        W = np.linalg.pinv(Xc + noise) @ Yc
        sess_Wproj.append(project_W(W, Xc))

    # Step 2: build G-space from all per-session projections
    Wstack_s    = np.array(sess_Wproj)
    Wmean_s     = Wstack_s.mean(0)
    Wcent_s     = Wstack_s - Wmean_s
    _, _, Vsvd_s = np.linalg.svd(Wcent_s, full_matrices=False)
    Meff_s      = N_subj - 1

    # Step 3: average G across sessions per patient
    G_C2 = np.zeros((N_patients, Meff_s))
    for pi, pid in enumerate(unique_pids):
        idxs = patient_sids[pid]
        g_list = [(sess_Wproj[i] - Wmean_s) @ Vsvd_s[:Meff_s].T for i in idxs]
        G_C2[pi] = np.mean(g_list, axis=0)

    # ── per K_LDA ──────────────────────────────────────────────────────────
    for ki, k in enumerate(KLDA_GRID):
        k_A  = min(k, G_A.shape[1])
        k_B  = min(k, G_B.shape[1])
        k_C2 = min(k, G_C2.shape[1])

        lda_A  = _fit_lda(G_A,  patient_labels, k_A)
        lda_B  = _fit_lda(G_B,  patient_labels, k_B)
        lda_C2 = _fit_lda(G_C2, patient_labels, k_C2)

        res_A_bal[si, ki]  = _bal_acc(lda_A,  G_A,  patient_labels, k_A)
        res_B_bal[si, ki]  = _bal_acc(lda_B,  G_B,  patient_labels, k_B)
        res_C2_bal[si, ki] = _bal_acc(lda_C2, G_C2, patient_labels, k_C2)
        res_A_auc[si, ki]  = _auc(lda_A,  G_A,  patient_labels, k_A)
        res_B_auc[si, ki]  = _auc(lda_B,  G_B,  patient_labels, k_B)
        res_C2_auc[si, ki] = _auc(lda_C2, G_C2, patient_labels, k_C2)

    print(f"  best BAL-ACC → A:{res_A_bal[si].max():.4f} "
          f"B:{res_B_bal[si].max():.4f} "
          f"C2:{res_C2_bal[si].max():.4f}  "
          f"(at K_LDA={KLDA_GRID[res_B_bal[si].argmax()]})")

# ── find global optima ─────────────────────────────────────────────────────────
print("\n" + "="*70)
for name, mat_bal, mat_auc in [
        ("A (multi-sess)",   res_A_bal,  res_A_auc),
        ("B (single-sess)",  res_B_bal,  res_B_auc),
        ("C2 (avg-G/pat)",   res_C2_bal, res_C2_auc)]:
    best_idx = np.unravel_index(mat_bal.argmax(), mat_bal.shape)
    bs, bk   = SIGMA_GRID[best_idx[0]], KLDA_GRID[best_idx[1]]
    print(f"  {name:<20}  best BAL-ACC={mat_bal[best_idx]:.4f}  "
          f"AUROC={mat_auc[best_idx]:.4f}  "
          f"@ sigma={bs}, K_LDA={bk}")
print("="*70)

# ── save best table ────────────────────────────────────────────────────────────
with open(f"{OUT_DIR}/sigma_klda_best.txt", "w") as f:
    f.write("Condition          BAL-ACC  AUROC   sigma    K_LDA\n")
    for name, mat_bal, mat_auc in [
            ("A (multi-sess)",  res_A_bal,  res_A_auc),
            ("B (single-sess)", res_B_bal,  res_B_auc),
            ("C2 (avg-G/pat)",  res_C2_bal, res_C2_auc)]:
        bi = np.unravel_index(mat_bal.argmax(), mat_bal.shape)
        f.write(f"{name:<20} {mat_bal[bi]:.4f}  {mat_auc[bi]:.4f}  "
                f"{SIGMA_GRID[bi[0]]:<8} {KLDA_GRID[bi[1]]}\n")

# ── plots ──────────────────────────────────────────────────────────────────────
sigma_labels = [str(s) for s in SIGMA_GRID]
klda_labels  = [str(k) for k in KLDA_GRID]

def make_heatmap(ax, data, title, vmin=None, vmax=None, cmap="viridis"):
    if vmin is None: vmin = data.min()
    if vmax is None: vmax = data.max()
    im = ax.imshow(data, aspect="auto", origin="lower",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(n_klda)); ax.set_xticklabels(klda_labels, rotation=45, ha="right")
    ax.set_yticks(range(n_sig));  ax.set_yticklabels(sigma_labels)
    ax.set_xlabel("K_LDA (# G-space components)", fontsize=9)
    ax.set_ylabel("sigma (W-fit noise)", fontsize=9)
    ax.set_title(title, fontsize=10)
    # mark global best
    bi = np.unravel_index(data.argmax(), data.shape)
    ax.plot(bi[1], bi[0], "r*", ms=14, label=f"best={data[bi]:.4f}")
    # annotate cells
    for si in range(n_sig):
        for ki in range(n_klda):
            ax.text(ki, si, f"{data[si,ki]:.3f}", ha="center", va="center",
                    fontsize=6, color="white" if data[si,ki] < vmin+0.6*(vmax-vmin) else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.legend(loc="upper right", fontsize=8, frameon=False)

# Balanced accuracy heatmaps
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="white")
vmin_b = min(res_A_bal.min(), res_B_bal.min(), res_C2_bal.min())
vmax_b = max(res_A_bal.max(), res_B_bal.max(), res_C2_bal.max())
make_heatmap(axes[0], res_A_bal,
             "A) Multi-session W → per-patient\n(inflated baseline)", vmin_b, vmax_b)
make_heatmap(axes[1], res_B_bal,
             "B) Single-session W → per-patient\n(fair)", vmin_b, vmax_b)
make_heatmap(axes[2], res_C2_bal,
             "C2) Per-session W → avg-G per patient\n(fair)", vmin_b, vmax_b)
fig.suptitle("Balanced Accuracy — sigma × K_LDA sweep", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/sigma_klda_heatmap_balACC.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sigma_klda_heatmap_balACC.png")

# AUROC heatmaps
fig, axes = plt.subplots(1, 3, figsize=(22, 7), facecolor="white")
vmin_a = min(res_A_auc.min(), res_B_auc.min(), res_C2_auc.min())
vmax_a = max(res_A_auc.max(), res_B_auc.max(), res_C2_auc.max())
make_heatmap(axes[0], res_A_auc,
             "A) Multi-session W → per-patient\n(inflated baseline)", vmin_a, vmax_a)
make_heatmap(axes[1], res_B_auc,
             "B) Single-session W → per-patient\n(fair)", vmin_a, vmax_a)
make_heatmap(axes[2], res_C2_auc,
             "C2) Per-session W → avg-G per patient\n(fair)", vmin_a, vmax_a)
fig.suptitle("AUROC — sigma × K_LDA sweep", fontsize=13, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/sigma_klda_heatmap_AUROC.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sigma_klda_heatmap_AUROC.png")

# Per-condition line plots: best K_LDA per sigma, and best sigma per K_LDA
fig, axes = plt.subplots(2, 3, figsize=(18, 10), facecolor="white")
colors = {"A": "#7B1FA2", "B": "#1565C0", "C2": "#2E7D32"}

for col, (cond_name, real_mat) in enumerate([
        ("A (multi-sess)", res_A_bal),
        ("B (single-sess)", res_B_bal),
        ("C2 (avg-G/pat)", res_C2_bal)]):
    # Row 0: BAL-ACC vs sigma (each line = one K_LDA)
    ax = axes[0, col]
    for ki, k in enumerate(KLDA_GRID):
        ax.plot(SIGMA_GRID, real_mat[:, ki], "o-", ms=4, lw=1.5,
                label=f"K={k}", alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("sigma", fontsize=9)
    ax.set_ylabel("Balanced Accuracy", fontsize=9)
    ax.set_title(f"{cond_name}\nBAL-ACC vs sigma (each line=K_LDA)", fontsize=9)
    ax.legend(fontsize=6, ncol=3, frameon=False, loc="upper right")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

    # Row 1: BAL-ACC vs K_LDA (each line = one sigma)
    ax = axes[1, col]
    for si, s in enumerate(SIGMA_GRID):
        ax.plot(KLDA_GRID, real_mat[si, :], "o-", ms=4, lw=1.5,
                label=f"σ={s}", alpha=0.8)
    ax.set_xlabel("K_LDA", fontsize=9)
    ax.set_ylabel("Balanced Accuracy", fontsize=9)
    ax.set_title(f"{cond_name}\nBAL-ACC vs K_LDA (each line=sigma)", fontsize=9)
    ax.legend(fontsize=6, ncol=2, frameon=False, loc="upper right")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig.suptitle("Sigma × K_LDA sweep — Balanced Accuracy profiles", fontsize=12, fontweight="bold")
plt.tight_layout()
fig.savefig(f"{OUT_DIR}/sigma_klda_profiles.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved sigma_klda_profiles.png")

print("\nDone.")
