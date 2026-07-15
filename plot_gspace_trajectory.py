"""
plot_gspace_trajectory.py
=========================
Combined figure:
  TOP ROW  — G-space (PC1 vs PC2) scatter at N = 10, 20, 30, 40
              All 183 patients in background (faint).
              Selected N/class patients highlighted.
              LDA direction (arrow) + approx. decision line overlaid.
              LOPO BAL-ACC for that rep annotated.
  BOTTOM ROW — LDA learning curve (BAL-ACC and AUROC vs N, 30 reps).
               Vertical markers at the four displayed N values.

Loads:  g_space_cache.npz  (G-space; built by rf_lc.py)
        rf_results.npz      (LDA learning curve)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── settings ───────────────────────────────────────────────────────────────────
K_USE    = 15
RNG_SEED = 42
N_SHOW   = [10, 20, 30, 40]
OUT_FILE = "gspace_trajectory.png"

CC_COL  = "#1565C0"   # blue
AD_COL  = "#C62828"   # red
BG_ALPHA = 0.13
FG_ALPHA = 0.80

# ── helpers ────────────────────────────────────────────────────────────────────
class _LDA:
    def fit(self, X, y):
        c0, c1 = np.unique(y)
        X0, X1 = X[y==c0], X[y==c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
              + 1e-6*np.eye(X0.shape[1]))
        w = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w)+1e-12
        self.w_ = w
        return self
    def transform(self, X): return X @ self.w_

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0i, c1i = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0i), len(c1i))
    sel = np.concatenate([rng2.choice(c0i, n, replace=False),
                          rng2.choice(c1i, n, replace=False)])
    rng2.shuffle(sel)
    return X[sel], y[sel]

def lopo_lda_single(G_sub, y_S, k_use):
    """Returns (preds, scores, per-fold LDA weights) for a single subset."""
    n = len(G_sub); G_k = G_sub[:, :k_use]
    preds = np.full(n, np.nan); scores = np.full(n, np.nan)
    ws = []
    for i in range(n):
        mask = np.arange(n) != i
        G_tr = G_k[mask]; y_tr = y_S[mask]; G_te = G_k[i]
        Xb, yb = _balance(G_tr, y_tr, seed=RNG_SEED)
        try:
            lda = _LDA().fit(Xb, yb)
        except Exception:
            ws.append(None); continue
        z_tr = lda.transform(G_tr)
        if z_tr[y_tr==0].mean() > z_tr[y_tr==1].mean():
            lda.w_ *= -1; z_tr = lda.transform(G_tr)
        thr = 0.5*(z_tr[y_tr==0].mean() + z_tr[y_tr==1].mean())
        z_te = lda.transform(G_te.reshape(1,-1))[0]
        preds[i]  = float(z_te >= thr)
        scores[i] = z_te - thr
        ws.append(lda.w_.copy())
    return preds, scores, ws

# ── load G-space cache ─────────────────────────────────────────────────────────
print("Loading G-space cache ...")
cache = np.load("g_space_cache.npz")
G_pat_full     = cache["G_pat_full"]          # (183, 38)
patient_labels = cache["patient_labels"]
cc_idx         = cache["cc_idx"].astype(int)
ad_idx         = cache["ad_idx"].astype(int)
N_patients     = G_pat_full.shape[0]
print(f"  {N_patients} patients, {G_pat_full.shape[1]} PCs  "
      f"(CC={len(cc_idx)}, AD={len(ad_idx)})")

# ── load LDA learning curve ────────────────────────────────────────────────────
lc = np.load("rf_results.npz")
N_arr   = lc["N_grid"]
ba_mean = lc["LDA_bal_mean"];  ba_std = lc["LDA_bal_std"]
au_mean = lc["LDA_auc_mean"];  au_std = lc["LDA_auc_std"]

# ── pre-compute: for each N_SHOW, pick rep=0 subset and fit aggregate LDA ──────
print("Computing G-space per-N snapshots ...")
snapshots = {}   # N -> dict(G_sub, y_S, w_mean, w_2d, midpoint_2d, bal, auc)

for N in N_SHOW:
    rng_rep  = np.random.default_rng(RNG_SEED + 0*10000 + N)   # rep=0
    sel_cc   = rng_rep.choice(cc_idx, N, replace=False)
    sel_ad   = rng_rep.choice(ad_idx, N, replace=False)
    S        = np.concatenate([sel_cc, sel_ad])
    y_S      = patient_labels[S]
    k_use    = min(K_USE, 2*N - 2)
    G_sub    = G_pat_full[S]

    preds, scores, ws = lopo_lda_single(G_sub, y_S, k_use)
    valid = np.isfinite(preds)
    sens  = np.mean(preds[valid & (y_S==1)] == 1)
    spec  = np.mean(preds[valid & (y_S==0)] == 0)
    bal   = 0.5*(sens+spec)
    try:   auc = roc_auc_score(y_S[valid], scores[valid])
    except: auc = np.nan

    # Aggregate LDA direction: mean of valid fold weights
    ws_valid = [w for w in ws if w is not None]
    if ws_valid:
        W_mat  = np.array(ws_valid)
        # sign-align to first valid weight
        ref = W_mat[0]
        signs = np.sign(W_mat @ ref)
        signs[signs == 0] = 1
        W_mat *= signs[:, None]
        w_mean = W_mat.mean(0); w_mean /= np.linalg.norm(w_mean) + 1e-12
    else:
        w_mean = np.zeros(K_USE)

    # 2-D projection (PC1, PC2 are indices 0 and 1)
    w_2d = w_mean[:2].copy()
    if np.linalg.norm(w_2d) > 1e-10:
        w_2d /= np.linalg.norm(w_2d)

    # Approximate 2-D decision-boundary midpoint
    G2 = G_sub[:, :2]
    mu_cc_2d = G2[y_S==0].mean(0)
    mu_ad_2d = G2[y_S==1].mean(0)
    midpoint_2d = 0.5*(mu_cc_2d + mu_ad_2d)

    snapshots[N] = dict(
        S=S, y_S=y_S,
        w_mean=w_mean, w_2d=w_2d,
        midpoint_2d=midpoint_2d,
        bal=bal, auc=auc, k_use=k_use,
        mu_cc_2d=mu_cc_2d, mu_ad_2d=mu_ad_2d,
    )
    print(f"  N={N:2d}  BAL-ACC={bal:.3f}  AUROC={auc:.3f}  k_use={k_use}")

# ── figure ─────────────────────────────────────────────────────────────────────
print("Plotting ...")
fig = plt.figure(figsize=(18, 10), facecolor="white")
fig.subplots_adjust(hspace=0.42, wspace=0.32)

# ── top row: 4 G-space scatter panels ─────────────────────────────────────────
all_G2 = G_pat_full[:, :2]
# fixed axis limits
x_lim = (all_G2[:,0].min()-0.02, all_G2[:,0].max()+0.02)
y_lim = (all_G2[:,1].min()-0.02, all_G2[:,1].max()+0.02)

for col, N in enumerate(N_SHOW):
    ax = fig.add_subplot(2, 4, col+1)
    sn = snapshots[N]
    S  = sn["S"]; y_S = sn["y_S"]

    # Background: all patients
    ax.scatter(all_G2[cc_idx, 0], all_G2[cc_idx, 1],
               s=14, c=CC_COL, alpha=BG_ALPHA, linewidths=0, zorder=1)
    ax.scatter(all_G2[ad_idx, 0], all_G2[ad_idx, 1],
               s=14, c=AD_COL, alpha=BG_ALPHA, linewidths=0, zorder=1)

    # Foreground: selected patients
    sel_cc_mask = (y_S == 0)
    sel_ad_mask = (y_S == 1)
    G_sel = all_G2[S]
    ax.scatter(G_sel[sel_cc_mask, 0], G_sel[sel_cc_mask, 1],
               s=38, c=CC_COL, alpha=FG_ALPHA, edgecolors="white",
               linewidths=0.4, zorder=3, label=f"CC (N={N})")
    ax.scatter(G_sel[sel_ad_mask, 0], G_sel[sel_ad_mask, 1],
               s=38, c=AD_COL, alpha=FG_ALPHA, edgecolors="white",
               linewidths=0.4, zorder=3, label=f"AD (N={N})")

    # Class means (selected)
    mu_cc = sn["mu_cc_2d"]; mu_ad = sn["mu_ad_2d"]
    ax.scatter(*mu_cc, s=120, c=CC_COL, marker="X", zorder=5,
               edgecolors="white", linewidths=0.8)
    ax.scatter(*mu_ad, s=120, c=AD_COL, marker="X", zorder=5,
               edgecolors="white", linewidths=0.8)

    # LDA direction arrow (from midpoint)
    mid = sn["midpoint_2d"]; w2 = sn["w_2d"]
    arrow_len = 0.12 * (x_lim[1] - x_lim[0])
    ax.annotate("", xy=(mid[0] + w2[0]*arrow_len, mid[1] + w2[1]*arrow_len),
                xytext=(mid[0] - w2[0]*arrow_len, mid[1] - w2[1]*arrow_len),
                arrowprops=dict(arrowstyle="-|>", color="#333",
                                lw=1.8, mutation_scale=12),
                zorder=6)

    # Approximate decision line (perpendicular to w_2d through midpoint)
    perp = np.array([-w2[1], w2[0]])
    line_len = 1.5 * max(x_lim[1]-x_lim[0], y_lim[1]-y_lim[0])
    lx = [mid[0] - perp[0]*line_len, mid[0] + perp[0]*line_len]
    ly = [mid[1] - perp[1]*line_len, mid[1] + perp[1]*line_len]
    ax.plot(lx, ly, "--", color="#555", lw=1.2, alpha=0.7, zorder=4)

    ax.set_xlim(x_lim); ax.set_ylim(y_lim)
    ax.set_xlabel("PC 1", fontsize=9)
    ax.set_ylabel("PC 2", fontsize=9) if col == 0 else None
    ax.set_title(f"N = {N}/class", fontsize=10, fontweight="bold")
    ax.text(0.03, 0.97,
            f"BAL-ACC = {sn['bal']:.3f}\nAUROC = {sn['auc']:.3f}\nk={sn['k_use']}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7))
    # legend only for first panel
    if col == 0:
        ax.legend(handles=[
            plt.Line2D([],[],marker='o',color='w',markerfacecolor=CC_COL,
                       markersize=6, label='CC (all)'),
            plt.Line2D([],[],marker='o',color='w',markerfacecolor=AD_COL,
                       markersize=6, label='AD (all)'),
            plt.Line2D([],[],linestyle='--',color='#555',lw=1.2,label='decision boundary'),
            plt.Line2D([],[],marker='>',color='#333',lw=1.5,label='LDA direction'),
        ], fontsize=7, frameon=False, loc="lower right")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)

# ── bottom row: LDA learning curve ────────────────────────────────────────────
for col, (mean_v, std_v, ylabel, title) in enumerate([
        (ba_mean, ba_std, "LOPO Balanced Accuracy",
         f"BAL-ACC vs N  (K={K_USE})"),
        (au_mean, au_std, "LOPO AUROC",
         f"AUROC vs N  (K={K_USE})")]):

    ax = fig.add_subplot(2, 4, (5+2*col, 6+2*col))

    ax.axhline(0.50, color="gray", ls="--", lw=1.0, alpha=0.6, label="chance")
    ax.fill_between(N_arr, mean_v - std_v, mean_v + std_v,
                    alpha=0.22, color=CC_COL)
    ax.plot(N_arr, mean_v, "-o", ms=6, lw=2.2, color=CC_COL,
            label="LDA (mean ± std, 30 reps)")

    # Vertical markers for the displayed N values
    for N, sn in snapshots.items():
        idx = np.where(N_arr == N)[0]
        if len(idx):
            m = mean_v[idx[0]]
            ax.axvline(N, color="#888", ls=":", lw=1.0, alpha=0.6)
            ax.scatter([N], [m], s=90, color="#FF8F00", edgecolors="k",
                       linewidths=0.7, zorder=5)

    ax.set_xlabel("N patients per class", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(N_arr)
    ax.set_xticklabels(N_arr, rotation=45, fontsize=8)
    ax.set_ylim(0.30, 0.88)
    ax.legend(fontsize=8, frameon=False)
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)

fig.suptitle(
    "G-space classification trajectory — LDA on G-space PCA (702 sessions)\n"
    "Top: PC1–PC2 projection at 4 training sizes  |  "
    "Arrow = LDA direction  |  Dashed = approx. decision boundary\n"
    "Bottom: LOPO learning curve (orange = displayed snapshots)",
    fontsize=10, fontweight="bold")

fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_FILE}")
print("Done.")
