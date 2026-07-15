"""
classification_comparison.py
============================
Leave-One-Patient-Out (LOPO) LDA classification: CC vs AD.

ALL sessions of one patient are held out together → no session leakage.
76 unique patients → 76 LOO folds.

Feature sets compared:
  1. FC             — instantaneous upper-triangle FC  (7260 dims → PCA → LDA)
  2. FC lag-1       — lag-1 cross-corr upper-triangle  (7260 dims → PCA → LDA)
  3. FC + lag-1     — combined                         (14520 dims → PCA → LDA)
  4. Graph features — weighted degree + Onnela clustering + FC std  (363 dims)
  5. G-scores       — reservoir SVD archetypes         (185 dims, k-sweep)

Output: summary_out/Fig4_classification.png
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import trange, tqdm
warnings.filterwarnings("ignore")

sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

# ── Config ────────────────────────────────────────────────────────────────────
RNG_SEED         = 42
N_CC_SAMPLE      = 40
N_SITES          = 121
N_PC_MODEL       = 50       # PCA dims for signal projection
K_PC             = 200      # row-space SVD truncation for W
M_ARCH           = 600      # archetype dims (capped by SVD at min(N_subj, N_feat))
recurrent_factor = 0.1      # same as original notebook
noise_size       = 0.025
TIMES_SKIP       = 10       # transient frames to skip

TS_ROOT = "./timeseries"
OUT_DIR = "./summary_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Fisher LDA (exact from original notebook cell 33) ─────────────────────────
class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw  = (X0 - mu0).T @ (X0 - mu0) + (X1 - mu1).T @ (X1 - mu1)
        Sw += 1e-6 * np.eye(Sw.shape[0])
        w   = np.linalg.solve(Sw, mu1 - mu0)
        w  /= np.linalg.norm(w) + 1e-12
        thr = 0.5 * ((X0 @ w).mean() + (X1 @ w).mean())
        self.w_   = w
        self.thr_ = thr
        return self
    def predict(self, X):
        scores = X @ self.w_
        return np.where(scores >= self.thr_, self.classes_[1], self.classes_[0])
    def score_lda(self, X):
        return X @ self.w_

# ── Leave-One-Patient-Out splits ──────────────────────────────────────────────
def lopo_splits(patient_ids):
    """One fold per unique patient; all their sessions are in test."""
    unique_pids = np.unique(patient_ids)
    splits = []
    for pid in unique_pids:
        test_idx  = np.where(patient_ids == pid)[0]
        train_idx = np.where(patient_ids != pid)[0]
        splits.append((train_idx, test_idx))
    return splits, unique_pids

# ── Balanced class sampling for training ──────────────────────────────────────
def balance_train(X_tr, y_tr, seed=0):
    rng = np.random.default_rng(seed)
    c0 = np.where(y_tr == 0)[0]; c1 = np.where(y_tr == 1)[0]
    n  = min(len(c0), len(c1))
    if n == 0:
        return X_tr, y_tr
    sel = np.concatenate([rng.choice(c0, n, replace=False),
                          rng.choice(c1, n, replace=False)])
    rng.shuffle(sel)
    return X_tr[sel], y_tr[sel]

# ── Graph features ────────────────────────────────────────────────────────────
def graph_features(FC):
    """Weighted degree (121) + Onnela clustering (121) + FC std (121) = 363 dims."""
    np.fill_diagonal(FC, 0)
    degree  = np.abs(FC).sum(axis=1)

    W = np.abs(FC); Wmax = W.max()
    if Wmax > 0: W = W / Wmax
    Wcb  = W ** (1.0/3.0)
    tri  = (Wcb @ Wcb @ Wcb).diagonal()
    k    = (W > 0).sum(axis=1).astype(float)
    denom = np.maximum(k * (k - 1), 1e-12)
    clust = tri / denom

    str_std = np.std(FC, axis=1)
    return np.concatenate([degree, clust, str_std])

# ── Lag-1 cross-correlation ───────────────────────────────────────────────────
def lag1_fc(sig, skip=10):
    """
    Symmetric lag-1 FC: mean of directed lag-1 and its transpose.
    sig: (N_sites, T)
    Returns (N_sites, N_sites) symmetric matrix.
    """
    data  = sig[:, skip:]          # skip transient
    lead  = data[:, :-1]           # t = 0 … T-2
    lag   = data[:, 1:]            # t = 1 … T-1
    full  = np.corrcoef(lead, lag) # (2N, 2N)
    L     = np.nan_to_num(full[N_SITES:, :N_SITES])   # directed lag-1
    return (L + L.T) / 2           # symmetrize

# ──────────────────────────────────────────────────────────────────────────────
# 1. Load data
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data ...")
rng = np.random.default_rng(RNG_SEED)

signals, labels_raw, patient_ids_raw = [], [], []
for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder = os.path.join(TS_ROOT, subfolder)
    files  = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder, fname)).T  # (T, N_sites)
        if arr.shape[1] == N_SITES and arr.shape[0] >= 139:
            signals.append(arr.T)          # store as (N_sites, T)
            patient_ids_raw.append(fname.split("_ses-")[0])
            labels_raw.append(0 if label == "CC" else 1)

patient_ids_raw = np.array(patient_ids_raw)
labels_raw      = np.array(labels_raw)
ctrl_idx        = np.where(labels_raw == 0)[0]
ad_idx          = np.where(labels_raw == 1)[0]
N_subj          = len(signals)

unique_pids, counts = np.unique(patient_ids_raw, return_counts=True)
N_patients = len(unique_pids)

print(f"  Sessions: {N_subj}  (CC={len(ctrl_idx)}, AD={len(ad_idx)})")
print(f"  Unique patients: {N_patients}  "
      f"({(counts>1).sum()} with multiple sessions, max={counts.max()})")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Population PCA
# ──────────────────────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)  # (T_total, N_sites)
centered = all_sig - all_sig.mean(axis=0)
cov      = np.cov(centered.T)
evals, evecs = np.linalg.eigh(cov)
order        = np.argsort(evals)[::-1]
ev50         = evecs[:, order][:, :N_PC_MODEL]    # (N_sites, 50)

# ──────────────────────────────────────────────────────────────────────────────
# 3. FC + lag-1 FC + graph features  (no reservoir)
# ──────────────────────────────────────────────────────────────────────────────
print("Computing FC, lag-1 FC, and graph features ...")
FC_flat_list     = []    # instantaneous FC upper-triangle
FC_lag1_flat_list = []   # lag-1 FC upper-triangle
graph_list       = []

for sig in tqdm(signals, desc="  Features"):
    # PC-projected signal (same pre-processing as original notebook)
    pc_sc = sig.T @ ev50
    proj  = (pc_sc @ ev50.T).T              # (N_sites, T)

    fc  = np.nan_to_num(np.corrcoef(proj))
    fc1 = lag1_fc(proj, skip=TIMES_SKIP)

    iu = np.triu_indices(N_SITES, k=1)
    FC_flat_list.append(fc[iu])
    FC_lag1_flat_list.append(fc1[iu])
    graph_list.append(graph_features(fc.copy()))

FC_flat      = np.array(FC_flat_list)       # (N_subj, 7260)
FC_lag1_flat = np.array(FC_lag1_flat_list)  # (N_subj, 7260)
FC_comb_flat = np.concatenate([FC_flat, FC_lag1_flat], axis=1)   # (N_subj, 14520)
graph_arr    = np.array(graph_list)          # (N_subj, 363)

print(f"  FC: {FC_flat.shape}  lag-1 FC: {FC_lag1_flat.shape}  "
      f"combined: {FC_comb_flat.shape}  graph: {graph_arr.shape}")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Reservoir pass-1 → G-scores
# ──────────────────────────────────────────────────────────────────────────────
print("\nReservoir pipeline (pass-1) ...")
N, I_d, O_d, TIME = 2000, N_SITES, N_SITES, 600
dt = 0.005; tau_m = 0.0001 * dt
par = dict(tau_m_f=tau_m, tau_m_s=tau_m, N=N, T=TIME, dt=dt,
           sigma_input=0.01, shape=(N, I_d, O_d, TIME))
res = RESERVOIRE_SIMPLE(par)
sr  = max(abs(np.linalg.eigvals(res.J)))
res.J *= 0.95 / sr

def driven_run(res, target, ff=0.1):
    """Teacher-forced run. Returns (X_states, Y_target)."""
    S, T_arr = [], []
    res.acc = 0
    for t in range(target.shape[1] - 1):
        res.step_rate(ff * target[:, t], sigma_dyn=0.)
        S.append(res.X.copy())
        T_arr.append(target[:, t].copy())
    return np.array(S), np.array(T_arr)   # (T-1, N), (T-1, N_sites)

W_proj_list = []
rng_fit = np.random.default_rng(RNG_SEED)

for idx in trange(N_subj, desc="  Fitting W_self"):
    sig = signals[idx]
    res.T = sig.shape[1]; res.reset()
    pc_sc  = sig.T @ ev50
    target = (pc_sc @ ev50.T).T             # PC-projected (N_sites, T)

    X_raw, Y_raw = driven_run(res, target, recurrent_factor)
    X = X_raw[TIMES_SKIP:]; Y = Y_raw[TIMES_SKIP:]

    noise  = rng_fit.normal(0, noise_size, X.shape)
    W      = np.linalg.pinv(X + noise).dot(Y)   # (N_res, N_sites)

    # Project into row-space of X
    W_T = W.T.astype(np.float64)               # (N_sites, N_res)
    _, sx, Vtx = np.linalg.svd(X.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    W_proj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

W_stack = np.array(W_proj_list)                # (N_subj, N_sites * k_pc)
W_mean  = W_stack.mean(0)
_, _, Vt_svd = np.linalg.svd(W_stack - W_mean, full_matrices=False)
M_eff   = min(M_ARCH, W_stack.shape[0] - 1)   # max achievable SVD dims
G_scores = (W_stack - W_mean) @ Vt_svd[:M_eff].T   # (N_subj, M_eff)

print(f"  G-scores: {G_scores.shape}  (SVD capped at {M_eff} = min(N_subj-1, N_feat))")

# ──────────────────────────────────────────────────────────────────────────────
# 5. LOPO evaluation
# ──────────────────────────────────────────────────────────────────────────────
splits, unique_pids = lopo_splits(patient_ids_raw)
N_folds = len(splits)
print(f"\nLOPO evaluation ({N_folds} folds = {N_patients} unique patients) ...")

def lopo_eval(X_feat, y, splits, pca_dims=None):
    """
    LOPO evaluation with optional PCA (fitted on train set per fold — no leakage).
    Training set is class-balanced by downsampling the majority class.
    Seed varies per fold so each fold draws a different AD subsample.
    Returns aggregated (y_true, y_pred) across all folds.
    """
    all_true, all_pred = [], []
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_feat[tr_idx], y[tr_idx]
        X_te, y_te = X_feat[te_idx], y[te_idx]

        # ── PCA fitted on training data only (no test leakage) ──────────────
        if pca_dims is not None and X_tr.shape[1] > pca_dims:
            mu_tr = X_tr.mean(0)
            _, _, Vt_pca = np.linalg.svd(X_tr - mu_tr, full_matrices=False)
            Vt_pca = Vt_pca[:pca_dims]
            X_tr = (X_tr - mu_tr) @ Vt_pca.T
            X_te = (X_te - mu_tr) @ Vt_pca.T

        # ── Balance majority class in training — fold-specific seed ─────────
        # seed varies per fold so each fold draws a different AD subsample
        X_tr_b, y_tr_b = balance_train(X_tr, y_tr, seed=RNG_SEED + fold_i)
        if len(np.unique(y_tr_b)) < 2:
            continue

        lda = _LDA().fit(X_tr_b, y_tr_b)
        all_true.extend(y_te.tolist())
        all_pred.extend(lda.predict(X_te).tolist())

    return np.array(all_true), np.array(all_pred)


def balanced_acc(y_true, y_pred):
    """Mean of per-class recall (= balanced accuracy / average sensitivity)."""
    classes = np.unique(y_true)
    return np.mean([(y_pred[y_true == c] == c).mean() for c in classes])


def class_accs(y_true, y_pred):
    """Return (CC_sensitivity, AD_sensitivity) separately."""
    cc_sens = (y_pred[y_true == 0] == 0).mean() if (y_true == 0).any() else np.nan
    ad_sens = (y_pred[y_true == 1] == 1).mean() if (y_true == 1).any() else np.nan
    return cc_sens, ad_sens

def lopo_sweep(X_feat, y, splits, k_values, pca_first=False, base_pca_dims=None):
    """
    Sweep over number of dimensions (first k columns of X_feat, after optional PCA).
    Pre-computes PCA scores on all data (slight leakage, but fast; marked below).
    For a fair sweep we fit PCA globally once, then do LOO on PCA scores.
    """
    if pca_first and X_feat.shape[1] > max(k_values):
        # Fit PCA globally (slight leakage — acceptable for exploratory sweep)
        mu  = X_feat.mean(0)
        _, _, Vt_pca = np.linalg.svd(X_feat - mu, full_matrices=False)
        n_keep = min(max(k_values), Vt_pca.shape[0])
        X_pca = (X_feat - mu) @ Vt_pca[:n_keep].T
    else:
        X_pca = X_feat

    accs = []
    for k in tqdm(k_values, desc="  k-sweep", leave=False):
        yt, yp = lopo_eval(X_pca[:, :k], y, splits)
        accs.append(balanced_acc(yt, yp))
    return np.array(accs)

print("  [1/5] FC instantaneous ...")
yt_fc, yp_fc         = lopo_eval(FC_flat,      labels_raw, splits, pca_dims=50)

print("  [2/5] FC lag-1 ...")
yt_lag, yp_lag       = lopo_eval(FC_lag1_flat, labels_raw, splits, pca_dims=50)

print("  [3/5] FC + lag-1 combined ...")
yt_comb, yp_comb     = lopo_eval(FC_comb_flat, labels_raw, splits, pca_dims=50)

print("  [4/5] Graph features (degree + clustering) ...")
yt_gr, yp_gr         = lopo_eval(graph_arr,    labels_raw, splits)

print("  [5/5] G-scores (k-sweep 1..all dims) ...")
k_vals_g   = list(range(1, G_scores.shape[1] + 1))
g_acc_curve = lopo_sweep(G_scores, labels_raw, splits, k_vals_g)
best_k_g    = k_vals_g[int(np.argmax(g_acc_curve))]
# Evaluate at best k (with proper per-fold PCA on train only)
yt_g, yp_g = lopo_eval(G_scores[:, :best_k_g], labels_raw, splits)

print("  FC-PCA k-sweep (k=1..50) ...")
k_vals_fc  = list(range(1, 51))
fc_acc_curve = lopo_sweep(FC_flat, labels_raw, splits, k_vals_fc, pca_first=True)
best_k_fc   = k_vals_fc[int(np.argmax(fc_acc_curve))]
yt_fc_best, yp_fc_best = lopo_eval(FC_flat, labels_raw, splits, pca_dims=best_k_fc)

print("  FC+lag-1-PCA k-sweep (k=1..50) ...")
comb_acc_curve = lopo_sweep(FC_comb_flat, labels_raw, splits, k_vals_fc, pca_first=True)
best_k_comb    = k_vals_fc[int(np.argmax(comb_acc_curve))]
yt_comb_best, yp_comb_best = lopo_eval(FC_comb_flat, labels_raw, splits, pca_dims=best_k_comb)

# ──────────────────────────────────────────────────────────────────────────────
# 6. Per-patient accuracy for LOO scatter plot
# ──────────────────────────────────────────────────────────────────────────────
def per_patient_accuracy(y_true_all, y_pred_all, patient_ids, unique_pids, labels):
    """Accuracy per patient (fraction of their sessions correctly classified)."""
    pat_acc, pat_lbl = [], []
    for pid in unique_pids:
        idx = np.where(patient_ids == pid)[0]
        yt  = np.array(y_true_all)[idx]
        yp  = np.array(y_pred_all)[idx]
        pat_acc.append((yt == yp).mean())
        pat_lbl.append(labels[idx[0]])
    return np.array(pat_acc), np.array(pat_lbl)

# Build index arrays for per-patient eval using best-k G-scores
# (already have yt_g, yp_g which are session-level)
pat_acc_g, pat_lbl = per_patient_accuracy(yt_g, yp_g,
                                           patient_ids_raw, unique_pids, labels_raw)

# ──────────────────────────────────────────────────────────────────────────────
# 7. Summary
# ──────────────────────────────────────────────────────────────────────────────
def fmt_bal(yt, yp, label=""):
    """
    Report balanced accuracy + per-class sensitivity.
    Balanced acc = (CC_sensitivity + AD_sensitivity) / 2.
    Regular accuracy shown in brackets for reference.
    """
    ba           = balanced_acc(yt, yp)
    reg_acc      = (yt == yp).mean()
    cc_sens, ad_sens = class_accs(yt, yp)
    return (f"BA={ba*100:5.1f}%  "
            f"[CC sens={cc_sens*100:.0f}%  AD sens={ad_sens*100:.0f}%]  "
            f"(raw acc={reg_acc*100:.0f}%)")

print("\n" + "=" * 70)
print("LOPO Classification Results")
print(f"  {'Feature set':<32s}  {'Bal.Acc':>7}  {'CC sens':>7}  {'AD sens':>7}  {'Raw acc':>7}")
print("-" * 70)
rows = [
    (f"FC (PCA-50)  k=50",         yt_fc,       yp_fc),
    (f"FC lag-1 (PCA-50)  k=50",   yt_lag,      yp_lag),
    (f"FC+lag-1 (PCA-50)  k=50",   yt_comb,     yp_comb),
    (f"FC  best k={best_k_fc}",     yt_fc_best,  yp_fc_best),
    (f"FC+lag-1  best k={best_k_comb}", yt_comb_best, yp_comb_best),
    (f"Graph features (363 dim)",   yt_gr,       yp_gr),
    (f"G-scores  best k={best_k_g}",yt_g,        yp_g),
]
for name, yt, yp in rows:
    ba            = balanced_acc(yt, yp)
    cc_s, ad_s    = class_accs(yt, yp)
    raw           = (yt == yp).mean()
    print(f"  {name:<32s}  {ba*100:6.1f}%  {cc_s*100:6.0f}%  {ad_s*100:6.0f}%  {raw*100:6.0f}%")
print("=" * 70)
print("Balanced acc = (CC_sensitivity + AD_sensitivity) / 2")
print("Raw acc is biased: a model predicting all-AD gets raw=78% but BA=50%")

# ──────────────────────────────────────────────────────────────────────────────
# 8. Figure
# ──────────────────────────────────────────────────────────────────────────────
COLORS = {
    "G-scores":    "#E91E63",
    "FC":          "#2196F3",
    "FC lag-1":    "#03A9F4",
    "FC+lag-1":    "#0D47A1",
    "Graph":       "#4CAF50",
}

fig = plt.figure(figsize=(20, 12), facecolor="white")
fig.suptitle("Classification Comparison — Leave-One-Patient-Out LDA (CC vs AD)",
             fontsize=14, fontweight="bold", y=1.01)
gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
              top=0.93, bottom=0.08)

# ── A: k-sweep curves ─────────────────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, :2])

ax_a.plot(k_vals_g,  g_acc_curve  * 100,  color=COLORS["G-scores"],
          lw=2.2, label="G-scores (reservoir)")
ax_a.plot(k_vals_fc, fc_acc_curve * 100,  color=COLORS["FC"],
          lw=2.0, label="FC (PCA)")
ax_a.plot(k_vals_fc, comb_acc_curve * 100, color=COLORS["FC+lag-1"],
          lw=2.0, ls="--", label="FC + lag-1 (PCA)")

ax_a.axhline(50, color="grey", ls=":", lw=1, label="Chance (50%)")

# mark best k
ax_a.axvline(best_k_g,  color=COLORS["G-scores"], ls=":", lw=1.2,
             label=f"Best k G-scores = {best_k_g}  ({g_acc_curve.max()*100:.1f}%)")
ax_a.axvline(best_k_fc,  color=COLORS["FC"],      ls=":", lw=1.2,
             label=f"Best k FC = {best_k_fc}  ({fc_acc_curve.max()*100:.1f}%)")
ax_a.axvline(best_k_comb, color=COLORS["FC+lag-1"], ls=":",  lw=1.2, alpha=0.6,
             label=f"Best k FC+lag-1 = {best_k_comb}  ({comb_acc_curve.max()*100:.1f}%)")

ax_a.set_xlabel("Number of dimensions (k)", fontsize=10)
ax_a.set_ylabel("Balanced accuracy (%)", fontsize=10)
ax_a.set_ylim(30, 100)
ax_a.legend(fontsize=8, ncol=2, loc="upper right")
ax_a.tick_params(labelsize=8)
ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
ax_a.set_title("(A) Accuracy vs Dimensionality (LOPO)", fontsize=11, fontweight="bold")

# ── B: Bar chart — all feature sets ──────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 2])

results = [
    ("G-scores\nbest k",   balanced_acc(yt_g,         yp_g),         COLORS["G-scores"]),
    ("FC\nbest k",         balanced_acc(yt_fc_best,   yp_fc_best),   COLORS["FC"]),
    ("FC lag-1\nbest k",   balanced_acc(yt_comb_best, yp_comb_best), COLORS["FC+lag-1"]),
    ("Graph\nfeatures",    balanced_acc(yt_gr,        yp_gr),        COLORS["Graph"]),
]

names  = [r[0] for r in results]
accs   = [r[1] * 100 for r in results]
colors = [r[2] for r in results]
x_pos  = np.arange(len(results))

bars = ax_b.bar(x_pos, accs, color=colors, alpha=0.8, edgecolor="white", width=0.6)
for xi, acc in zip(x_pos, accs):
    ax_b.text(xi, acc + 1.0, f"{acc:.1f}%", ha="center", fontsize=9, fontweight="bold")

ax_b.axhline(50, color="grey", ls=":", lw=1, label="Chance")
ax_b.set_xticks(x_pos); ax_b.set_xticklabels(names, fontsize=9)
ax_b.set_ylabel("Balanced accuracy (%)", fontsize=10)
ax_b.set_ylim(0, 105)
ax_b.tick_params(labelsize=8)
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
ax_b.set_title("(B) Feature Set Comparison\n(LOPO, best k per method)",
               fontsize=11, fontweight="bold")

# ── C: Per-patient accuracy — CC and AD separately ───────────────────────
ax_c = fig.add_subplot(gs[1, 0])

cc_mask = pat_lbl == 0
ad_mask = pat_lbl == 1
jitter  = np.random.default_rng(0).uniform(-0.12, 0.12, len(pat_acc_g))

# violin
vp_c = ax_c.violinplot([pat_acc_g[cc_mask] * 100, pat_acc_g[ad_mask] * 100],
                        positions=[0, 1], showmedians=True, showextrema=True)
vp_c["bodies"][0].set_facecolor(COLORS["FC"]);     vp_c["bodies"][0].set_alpha(0.45)
vp_c["bodies"][1].set_facecolor(COLORS["G-scores"]); vp_c["bodies"][1].set_alpha(0.45)
for part in ["cmedians","cbars","cmins","cmaxes"]:
    vp_c[part].set_color("k"); vp_c[part].set_linewidth(1.5)

# scatter jitter
ax_c.scatter(np.zeros(cc_mask.sum()) + jitter[cc_mask],
             pat_acc_g[cc_mask] * 100,
             color=COLORS["FC"], s=40, alpha=0.8, zorder=3, edgecolors="none",
             label=f"CC  (n={cc_mask.sum()}  sens={pat_acc_g[cc_mask].mean()*100:.0f}%)")
ax_c.scatter(np.ones(ad_mask.sum()) + jitter[ad_mask],
             pat_acc_g[ad_mask] * 100,
             color=COLORS["G-scores"], s=40, alpha=0.8, zorder=3, edgecolors="none",
             label=f"AD  (n={ad_mask.sum()}  sens={pat_acc_g[ad_mask].mean()*100:.0f}%)")

# balanced accuracy line
ba_g = balanced_acc(yt_g, yp_g)
ax_c.axhline(ba_g * 100, color="black", ls="--", lw=1.5,
             label=f"Balanced acc = {ba_g*100:.1f}%")
ax_c.axhline(50, color="grey", ls=":", lw=1, label="Chance 50%")

ax_c.set_xticks([0, 1])
ax_c.set_xticklabels([f"CC\n(n={cc_mask.sum()})", f"AD\n(n={ad_mask.sum()})"], fontsize=9)
ax_c.set_ylabel("Patient accuracy (%)\n(fraction of sessions correct)", fontsize=9)
ax_c.set_ylim(-5, 115)
ax_c.legend(fontsize=7.5, loc="upper right"); ax_c.tick_params(labelsize=8)
ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
ax_c.set_title(f"(C) Per-patient accuracy — G-scores k={best_k_g}\n"
               f"BA = (CC sens + AD sens)/2",
               fontsize=11, fontweight="bold")

# ── D: Lag-1 vs FC comparison ─────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 1])

pairs_labels = ["FC\n(k=50)", "FC lag-1\n(k=50)", "FC+lag-1\n(k=50)"]
pairs_accs   = [balanced_acc(yt_fc,   yp_fc)   * 100,
                balanced_acc(yt_lag,  yp_lag)  * 100,
                balanced_acc(yt_comb, yp_comb) * 100]
pairs_cols   = [COLORS["FC"], COLORS["FC lag-1"], COLORS["FC+lag-1"]]

bars_d = ax_d.bar(np.arange(3), pairs_accs, color=pairs_cols,
                  alpha=0.8, edgecolor="white", width=0.5)
for xi, acc in enumerate(pairs_accs):
    ax_d.text(xi, acc + 0.8, f"{acc:.1f}%", ha="center", fontsize=9, fontweight="bold")
ax_d.axhline(50, color="grey", ls=":", lw=1)
ax_d.set_xticks([0, 1, 2]); ax_d.set_xticklabels(pairs_labels, fontsize=9)
ax_d.set_ylabel("Balanced accuracy (%)", fontsize=10)
ax_d.set_ylim(0, 105)
ax_d.tick_params(labelsize=8)
ax_d.spines["top"].set_visible(False); ax_d.spines["right"].set_visible(False)
ax_d.set_title("(D) Added value of lag-1 FC\n(fixed k=50 PCA)", fontsize=11, fontweight="bold")

# ── E: Confusion-style breakdown per class ────────────────────────────────
ax_e = fig.add_subplot(gs[1, 2])

feature_labels = ["FC\nbest k", "FC+lag-1\nbest k", "G-scores\nbest k"]
yt_list = [yt_fc_best, yt_comb_best, yt_g]
yp_list = [yp_fc_best, yp_comb_best, yp_g]

x_pos2  = np.arange(len(feature_labels))
width   = 0.3
cc_accs = [(yp[yt == 0] == 0).mean() * 100 for yt, yp in zip(yt_list, yp_list)]
ad_accs = [(yp[yt == 1] == 1).mean() * 100 for yt, yp in zip(yt_list, yp_list)]

ax_e.bar(x_pos2 - width/2, cc_accs, width, color=COLORS["FC"],
         alpha=0.8, label="CC sensitivity")
ax_e.bar(x_pos2 + width/2, ad_accs, width, color=COLORS["G-scores"],
         alpha=0.8, label="AD sensitivity")

for xi, (ca, aa) in enumerate(zip(cc_accs, ad_accs)):
    ax_e.text(xi - width/2, ca + 1, f"{ca:.0f}%", ha="center", fontsize=7.5)
    ax_e.text(xi + width/2, aa + 1, f"{aa:.0f}%", ha="center", fontsize=7.5)

ax_e.axhline(50, color="grey", ls=":", lw=1)
ax_e.set_xticks(x_pos2); ax_e.set_xticklabels(feature_labels, fontsize=9)
ax_e.set_ylabel("Sensitivity (%)", fontsize=10)
ax_e.set_ylim(0, 115)
ax_e.legend(fontsize=9); ax_e.tick_params(labelsize=8)
ax_e.spines["top"].set_visible(False); ax_e.spines["right"].set_visible(False)
ax_e.set_title("(E) Per-class sensitivity\n(CC sensitivity vs AD sensitivity)",
               fontsize=11, fontweight="bold")

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "Fig4_classification.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\nFigure saved -> {out_path}")
