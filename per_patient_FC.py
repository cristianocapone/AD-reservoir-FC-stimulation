"""
per_patient_FC.py
=================
Compute FC by pooling all sessions of a patient (concatenated signals),
giving one better-estimated FC per patient.

Compares to per-session FC baseline (one FC per session, LOPO on sessions).

Feature sets:
  - FC instantaneous (upper-triangle)
  - FC lag-1 (upper-triangle)
  - FC + lag-1 combined

Both per-session (baseline) and per-patient (new) versions.
"""

import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────
RNG_SEED    = 42
N_CC_SAMPLE = 40
N_SITES     = 121
N_PC_MODEL  = 50
TIMES_SKIP  = 10
TS_ROOT     = "./timeseries"
OUT_DIR     = "./summary_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LDA ───────────────────────────────────────────────────────────────────────
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
        self.w_, self.thr_ = w, thr
        return self
    def predict(self, X):
        return np.where(X @ self.w_ >= self.thr_, self.classes_[1], self.classes_[0])

def balance_train(X, y, seed=0):
    rng = np.random.default_rng(seed)
    c0, c1 = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(c0), len(c1))
    if n == 0: return X, y
    sel = np.concatenate([rng.choice(c0, n, replace=False),
                          rng.choice(c1, n, replace=False)])
    rng.shuffle(sel)
    return X[sel], y[sel]

def balanced_acc(yt, yp):
    return np.mean([(yp[yt == c] == c).mean() for c in np.unique(yt)])

def class_accs(yt, yp):
    cc = (yp[yt == 0] == 0).mean() if (yt == 0).any() else np.nan
    ad = (yp[yt == 1] == 1).mean() if (yt == 1).any() else np.nan
    return cc, ad

# ── LOPO on per-session array ─────────────────────────────────────────────────
def lopo_sess_eval(X_feat, y, patient_ids, pca_dims=None):
    unique_pids = np.unique(patient_ids)
    all_true, all_pred = [], []
    for fold_i, pid in enumerate(unique_pids):
        tr_idx = np.where(patient_ids != pid)[0]
        te_idx = np.where(patient_ids == pid)[0]
        X_tr, y_tr = X_feat[tr_idx], y[tr_idx]
        X_te, y_te = X_feat[te_idx], y[te_idx]

        if pca_dims is not None and X_tr.shape[1] > pca_dims:
            mu = X_tr.mean(0)
            _, _, Vt = np.linalg.svd(X_tr - mu, full_matrices=False)
            Vt = Vt[:pca_dims]
            X_tr = (X_tr - mu) @ Vt.T
            X_te = (X_te - mu) @ Vt.T

        X_tr_b, y_tr_b = balance_train(X_tr, y_tr, seed=RNG_SEED + fold_i)
        if len(np.unique(y_tr_b)) < 2: continue
        lda = _LDA().fit(X_tr_b, y_tr_b)
        all_true.extend(y_te.tolist())
        all_pred.extend(lda.predict(X_te).tolist())
    return np.array(all_true), np.array(all_pred)

# ── LOPO on per-patient array (one row per patient) ───────────────────────────
def lopo_patient_eval(X_feat, y, pca_dims=None):
    n = len(y)
    all_true, all_pred = [], []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        X_tr, y_tr = X_feat[tr], y[tr]
        X_te, y_te = X_feat[[i]], y[[i]]

        if pca_dims is not None and X_tr.shape[1] > pca_dims:
            mu = X_tr.mean(0)
            _, _, Vt = np.linalg.svd(X_tr - mu, full_matrices=False)
            Vt = Vt[:pca_dims]
            X_tr = (X_tr - mu) @ Vt.T
            X_te = (X_te - mu) @ Vt.T

        X_tr_b, y_tr_b = balance_train(X_tr, y_tr, seed=RNG_SEED + i)
        if len(np.unique(y_tr_b)) < 2: continue
        lda = _LDA().fit(X_tr_b, y_tr_b)
        all_true.extend(y_te.tolist())
        all_pred.extend(lda.predict(X_te).tolist())
    return np.array(all_true), np.array(all_pred)

def sweep_pca(X_feat, y, eval_fn, k_values, **kwargs):
    accs = []
    for k in tqdm(k_values, desc="  k-sweep", leave=False):
        yt, yp = eval_fn(X_feat, y, pca_dims=k, **kwargs)
        accs.append(balanced_acc(yt, yp))
    return np.array(accs)

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
print(f"  Sessions: {N_subj}  (CC={(labels_raw==0).sum()}, AD={(labels_raw==1).sum()})")

# Patient-level info
unique_pids   = np.unique(patient_ids_raw)
N_patients    = len(unique_pids)
patient_sids  = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}
patient_labels = np.array([labels_raw[patient_sids[pid][0]] for pid in unique_pids])
n_sess = np.array([len(patient_sids[pid]) for pid in unique_pids])

print(f"  Patients: {N_patients}  "
      f"(CC={(patient_labels==0).sum()}, AD={(patient_labels==1).sum()})")
print(f"  Sessions/patient: min={n_sess.min()} max={n_sess.max()} mean={n_sess.mean():.1f}")

# ── Population PCA ────────────────────────────────────────────────────────────
print("Population PCA ...")
all_sig  = np.concatenate([s.T for s in signals], axis=0)
centered = all_sig - all_sig.mean(axis=0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
ev50 = evecs[:, np.argsort(evals)[::-1]][:, :N_PC_MODEL]

# ── Helper: FC and lag-1 FC from a signal ─────────────────────────────────────
def fc_features(sig, skip=TIMES_SKIP):
    """sig: (N_sites, T). Returns (fc_vec, fc_lag1_vec)."""
    data = sig[:, skip:]
    pc   = data.T @ ev50
    proj = (pc @ ev50.T).T          # (N_sites, T_eff)

    fc  = np.nan_to_num(np.corrcoef(proj))
    # lag-1: symmetric
    lead = proj[:, :-1]; lag = proj[:, 1:]
    full = np.corrcoef(lead, lag)
    L    = np.nan_to_num(full[N_SITES:, :N_SITES])
    fc1  = (L + L.T) / 2

    iu = np.triu_indices(N_SITES, k=1)
    return fc[iu], fc1[iu]

# ── Per-session FC (baseline) ─────────────────────────────────────────────────
print("Computing per-session FC ...")
sess_fc_list  = []
sess_fc1_list = []
for sig in tqdm(signals, desc="  Sessions", leave=False):
    f, f1 = fc_features(sig)
    sess_fc_list.append(f)
    sess_fc1_list.append(f1)

FC_sess      = np.array(sess_fc_list)               # (N_subj, 7260)
FC1_sess     = np.array(sess_fc1_list)
FCcomb_sess  = np.concatenate([FC_sess, FC1_sess], axis=1)

# ── Per-patient FC (new): concatenate all sessions ────────────────────────────
print("Computing per-patient FC (concatenated sessions) ...")
pat_fc_list  = []
pat_fc1_list = []
for pid in tqdm(unique_pids, desc="  Patients", leave=False):
    idxs   = patient_sids[pid]
    # Concatenate signals along time axis
    sig_all = np.concatenate([signals[i] for i in idxs], axis=1)  # (N_sites, T_total)
    f, f1   = fc_features(sig_all, skip=TIMES_SKIP)
    pat_fc_list.append(f)
    pat_fc1_list.append(f1)

FC_pat     = np.array(pat_fc_list)                  # (N_patients, 7260)
FC1_pat    = np.array(pat_fc1_list)
FCcomb_pat = np.concatenate([FC_pat, FC1_pat], axis=1)

# ── k-sweep ───────────────────────────────────────────────────────────────────
k_vals = list(range(1, 51))

print("\nLOPO k-sweep ...")

print("  [1/6] Per-session FC ...")
acc_sess_fc = sweep_pca(FC_sess, labels_raw,
                        lambda X, y, pca_dims: lopo_sess_eval(X, y, patient_ids_raw, pca_dims),
                        k_vals)

print("  [2/6] Per-session FC+lag-1 ...")
acc_sess_comb = sweep_pca(FCcomb_sess, labels_raw,
                          lambda X, y, pca_dims: lopo_sess_eval(X, y, patient_ids_raw, pca_dims),
                          k_vals)

print("  [3/6] Per-patient FC ...")
acc_pat_fc = sweep_pca(FC_pat, patient_labels,
                       lambda X, y, pca_dims: lopo_patient_eval(X, y, pca_dims),
                       k_vals)

print("  [4/6] Per-patient FC+lag-1 ...")
acc_pat_comb = sweep_pca(FCcomb_pat, patient_labels,
                         lambda X, y, pca_dims: lopo_patient_eval(X, y, pca_dims),
                         k_vals)

# Best k for each
def best_result(acc_curve, X_feat, y, k_vals, eval_fn, **kwargs):
    best_k = k_vals[int(np.argmax(acc_curve))]
    yt, yp = eval_fn(X_feat, y, pca_dims=best_k, **kwargs)
    ba = balanced_acc(yt, yp)
    cc, ad = class_accs(yt, yp)
    return best_k, ba, cc, ad

bk_sfc,    ba_sfc,    cc_sfc,    ad_sfc    = best_result(
    acc_sess_fc, FC_sess, labels_raw, k_vals,
    lopo_sess_eval, patient_ids=patient_ids_raw)

bk_scomb,  ba_scomb,  cc_scomb,  ad_scomb  = best_result(
    acc_sess_comb, FCcomb_sess, labels_raw, k_vals,
    lopo_sess_eval, patient_ids=patient_ids_raw)

bk_pfc,    ba_pfc,    cc_pfc,    ad_pfc    = best_result(
    acc_pat_fc, FC_pat, patient_labels, k_vals,
    lopo_patient_eval)

bk_pcomb,  ba_pcomb,  cc_pcomb,  ad_pcomb  = best_result(
    acc_pat_comb, FCcomb_pat, patient_labels, k_vals,
    lopo_patient_eval)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"{'Method':<35} {'BA':>6}  {'CC':>5}  {'AD':>5}  {'k':>4}")
print("-" * 65)
rows = [
    ("Per-session FC",              ba_sfc,   cc_sfc,   ad_sfc,   bk_sfc),
    ("Per-session FC+lag-1",        ba_scomb, cc_scomb, ad_scomb, bk_scomb),
    ("Per-patient FC (new)",        ba_pfc,   cc_pfc,   ad_pfc,   bk_pfc),
    ("Per-patient FC+lag-1 (new)",  ba_pcomb, cc_pcomb, ad_pcomb, bk_pcomb),
]
for name, ba, cc, ad, k in rows:
    print(f"  {name:<33} {ba*100:5.1f}%  {cc*100:4.0f}%  {ad*100:4.0f}%  {k:4d}")
print("=" * 65)
print("BA = balanced accuracy, LOPO (patient-held-out)")

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS = {
    "sess_fc":   "#90CAF9",
    "sess_comb": "#1565C0",
    "pat_fc":    "#FFAB91",
    "pat_comb":  "#E64A19",
}

fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

ax.plot(k_vals, acc_sess_fc   * 100, color=COLORS["sess_fc"],   lw=1.8,
        label=f"Per-session FC (best k={bk_sfc})")
ax.plot(k_vals, acc_sess_comb * 100, color=COLORS["sess_comb"], lw=1.8,
        label=f"Per-session FC+lag-1 (best k={bk_scomb})")
ax.plot(k_vals, acc_pat_fc    * 100, color=COLORS["pat_fc"],    lw=2.2, ls="--",
        label=f"Per-patient FC (best k={bk_pfc})")
ax.plot(k_vals, acc_pat_comb  * 100, color=COLORS["pat_comb"],  lw=2.2, ls="--",
        label=f"Per-patient FC+lag-1 (best k={bk_pcomb})")

ax.axhline(50, color="gray", lw=0.8, ls=":", label="Chance")
ax.set_xlabel("PCA dims (k)"); ax.set_ylabel("Balanced accuracy (%)")
ax.set_title("Per-session vs Per-patient FC — LOPO LDA")
ax.legend(fontsize=8, loc="lower right")
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(f"{OUT_DIR}/per_patient_FC_sweep.png", dpi=150, bbox_inches="tight")
print(f"\nPlot saved: {OUT_DIR}/per_patient_FC_sweep.png")
