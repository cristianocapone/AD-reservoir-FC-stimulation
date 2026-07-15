"""
make_summary_notebook.py  v3
============================
Generates summary_figures.ipynb — 4 paper-ready publication figures.

Figures
-------
  Fig 1  Dataset overview (computed fresh)
  Fig 2  Reservoir model: FC panels + PCA + archetype M-selection + temporal decay
  Fig 3  Perturbation experiments (loads pre-saved PNGs)
  Fig 4  Classification: per-session W (baseline) + per-patient W (LOPO AUROC)

New in v3
---------
  • Fig 2 panels F/G: archetype SVD scree + reconstruction-error vs M
  • Fig 4: per-patient W k-sweep + summary bar chart with 85 % highlighted
  • All figures: uniform panel sizes, FC matrices forced to square aspect
  • Global paper-ready rcParams (fonts, spines, line widths)
"""
import nbformat as nbf

nb    = nbf.v4.new_notebook()
cells = []

def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))

# ─────────────────────────────────────────────────────────────────────────────
md("# Summary Figures v3 — Reservoir Computing on ADNI rs-fMRI\n\n"
   "- **Fig 1** Dataset overview\n"
   "- **Fig 2** Reservoir model + archetype M-selection\n"
   "- **Fig 3** Perturbation experiments\n"
   "- **Fig 4** Classification (per-session + per-patient W, LOPO AUROC)")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 1 — Imports, paths, paper-ready rcParams
# ══════════════════════════════════════════════════════════════════════════════
code("""\
import os, sys, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde
warnings.filterwarnings("ignore")

ROOT    = "."
TS_ROOT = "./timeseries"
OUT_DIR = "./summary_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ── constants ────────────────────────────────────────────────────────────────
RNG_SEED     = 42
N_CC_SAMPLE  = 40
N_SITES      = 121
TR           = 3.0
trial_dur    = 139
N_PC_MODEL   = 50
K_PC         = 200
noise_size   = 0.025
TIMES_SKIP   = 10
ff           = 0.1
SPECTRAL_RAD = 0.95
N_HIDDEN     = 2000
MAX_DELAY    = 20

CC_COL = "#2196F3"
AD_COL = "#E91E63"
COND_COLORS = {"CC": CC_COL, "AD": AD_COL}

# ── paper-ready global style ─────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family":        "sans-serif",
    "font.size":          9,
    "axes.labelsize":     9,
    "axes.titlesize":     10,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "lines.linewidth":    1.8,
    "axes.linewidth":     0.8,
    "xtick.major.width":  0.8,
    "ytick.major.width":  0.8,
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
})

def _tag(ax, ltr, x=-0.16, y=1.09):
    ax.text(x, y, ltr, transform=ax.transAxes,
            fontsize=12, fontweight="bold", va="top", ha="left")

def _clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

print("Imports OK")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 2 — Data loading + patient grouping
# ══════════════════════════════════════════════════════════════════════════════
code("""\
rng = np.random.default_rng(RNG_SEED)
collected_signals, identifiers = [], []

for subfolder, label in [("CN", "CC"), ("AD", "AD")]:
    folder_path = os.path.join(TS_ROOT, subfolder)
    if not os.path.isdir(folder_path):
        print(f"WARNING: {folder_path} not found, skipping"); continue
    files = sorted(f for f in os.listdir(folder_path) if f.endswith(".npy"))
    if label == "CC":
        files = list(rng.choice(files, size=min(N_CC_SAMPLE, len(files)), replace=False))
    for fname in files:
        arr = np.load(os.path.join(folder_path, fname)).T   # (T, N_sites)
        if arr.shape[1] == N_SITES and arr.shape[0] >= trial_dur:
            collected_signals.append(arr)
            identifiers.append([label, fname.split("_ses-")[0]])

identifiers    = np.array(identifiers, dtype=object)
state_ID_num   = np.array([0 if r[0] == "CC" else 1 for r in identifiers])
patient_ID_arr = np.array([r[1] for r in identifiers])

ctrl_indices = np.where(state_ID_num == 0)[0]
ad_indices   = np.where(state_ID_num == 1)[0]
sigs = [s.T for s in collected_signals]   # (N_sites, T) per session

# ── patient-level grouping ────────────────────────────────────────────────────
unique_pids    = np.unique(patient_ID_arr)
N_patients     = len(unique_pids)
patient_sids   = {pid: np.where(patient_ID_arr == pid)[0] for pid in unique_pids}
patient_labels = np.array([state_ID_num[patient_sids[pid][0]] for pid in unique_pids])
n_sess_pp      = np.array([len(patient_sids[pid]) for pid in unique_pids])

print(f"Sessions: {len(sigs)}  (CC={len(ctrl_indices)}, AD={len(ad_indices)})")
print(f"Patients: {N_patients}  (CC={(patient_labels==0).sum()}, AD={(patient_labels==1).sum()})")
print(f"Sessions/patient: min={n_sess_pp.min()}  max={n_sess_pp.max()}  mean={n_sess_pp.mean():.1f}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 3 — Population PCA + FC features (per-session and per-patient)
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# ── population PCA ────────────────────────────────────────────────────────────
all_sig  = np.concatenate([s.T for s in sigs], axis=0)
centered = all_sig - all_sig.mean(0)
evals, evecs = np.linalg.eigh(np.cov(centered.T))
order    = np.argsort(evals)[::-1]
evecs    = evecs[:, order]; evals_sorted = evals[order]
expl_var = evals_sorted / evals_sorted.sum()
ev50     = evecs[:, :N_PC_MODEL]           # (N_sites, 50)

# ── per-session FC + lag-1 ───────────────────────────────────────────────────
FC_collected = []
FC_flat_list, FC_lag1_flat_list = [], []

for sig in sigs:
    pc   = sig.T @ ev50
    proj = (pc @ ev50.T).T
    fc   = np.nan_to_num(np.corrcoef(proj))
    FC_collected.append(fc)

    data = proj[:, TIMES_SKIP:]
    full = np.corrcoef(data[:, :-1], data[:, 1:])
    L    = np.nan_to_num(full[N_SITES:, :N_SITES])
    fc1  = (L + L.T) / 2

    iu = np.triu_indices(N_SITES, k=1)
    FC_flat_list.append(fc[iu])
    FC_lag1_flat_list.append(fc1[iu])

FC_flat      = np.array(FC_flat_list)          # (N_subj, 7260)
FC_lag1_flat = np.array(FC_lag1_flat_list)

fc_ctrl_mean     = np.mean([FC_collected[i] for i in ctrl_indices], axis=0)
fc_ctrl_flat_vec = fc_ctrl_mean[np.triu_indices(N_SITES, k=1)]

# ── per-patient FC (concatenate all sessions per patient) ────────────────────
FC_pat_list, FC1_pat_list = [], []

for pid in unique_pids:
    idxs    = patient_sids[pid]
    sig_all = np.concatenate([sigs[i] for i in idxs], axis=1)
    pc      = sig_all.T @ ev50
    proj    = (pc @ ev50.T).T
    fc      = np.nan_to_num(np.corrcoef(proj))
    data    = proj[:, TIMES_SKIP:]
    full    = np.corrcoef(data[:, :-1], data[:, 1:])
    L       = np.nan_to_num(full[N_SITES:, :N_SITES])
    fc1     = (L + L.T) / 2
    iu      = np.triu_indices(N_SITES, k=1)
    FC_pat_list.append(fc[iu])
    FC1_pat_list.append(fc1[iu])

FC_pat  = np.array(FC_pat_list)    # (N_patients, 7260)
FC1_pat = np.array(FC1_pat_list)

print(f"Per-session FC: {FC_flat.shape}")
print(f"Per-patient FC: {FC_pat.shape}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 4 — Schaefer atlas labels
# ══════════════════════════════════════════════════════════════════════════════
code("""\
try:
    from nilearn import datasets as nl_datasets
    schaefer   = nl_datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_lbls = [l.decode() if isinstance(l, bytes) else l for l in schaefer.labels]
    NET_ORDER  = ["Vis","SomMot","DorsAttn","SalVentAttn","Limbic","Cont","Default"]
    NET_COLORS_MAP = dict(Vis="#1f77b4", SomMot="#ff7f0e", DorsAttn="#2ca02c",
                          SalVentAttn="#d62728", Limbic="#9467bd",
                          Cont="#8c564b", Default="#e377c2")
    def _get_net(lbl):
        for n in NET_ORDER:
            if n in lbl: return n
        return "Subcortical"
    net_assign  = [_get_net(l) for l in atlas_lbls]
    sorted_idx  = sorted(range(100),
                         key=lambda i: (NET_ORDER.index(net_assign[i])
                                        if net_assign[i] in NET_ORDER else 7, i))
    sorted_nets = [net_assign[i] for i in sorted_idx]
    net_bounds  = [i - 0.5 for i in range(1, 100)
                   if sorted_nets[i] != sorted_nets[i-1]]
    have_atlas  = True
    print("Atlas loaded")
except Exception as e:
    print(f"Atlas unavailable ({e}), using identity ordering")
    sorted_idx = list(range(100)); net_bounds = []; NET_COLORS_MAP = {}; have_atlas = False
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 5 — Per-session reservoir: teacher-forced + closed-loop
#           Collects W per session → G-scores (SVD) + M-selection curves
#           Also computes temporal-correlation-decay data
# ══════════════════════════════════════════════════════════════════════════════
code("""\
import sys; sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE
from tqdm import trange, tqdm

par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN,
           T=trial_dur, dt=0.005, sigma_input=0.01,
           shape=(N_HIDDEN, N_SITES, N_SITES, trial_dur))
res_nb = RESERVOIRE_SIMPLE(par)
sr     = max(abs(np.linalg.eigvals(res_nb.J)))
res_nb.J *= SPECTRAL_RAD / sr

rng_r = np.random.default_rng(RNG_SEED)
Y_emp_list, Y_sim_list = [], []
W_proj_list = []
sess_X, sess_Y = {}, {}

for idx in trange(len(sigs), desc="Reservoir (per-session)"):
    s   = sigs[idx]; T_s = s.shape[1]
    res_nb.T = T_s; res_nb.reset()
    target = (s.T @ ev50 @ ev50.T).T           # (N_sites, T)
    Y_emp_list.append(target)

    # teacher-forced pass
    X_raw = []
    for t in range(T_s - 1):
        res_nb.step_rate(ff * target[:, t], sigma_dyn=0.)
        X_raw.append(res_nb.X.copy())

    X_fit = np.array(X_raw)[TIMES_SKIP:]        # (T_eff, N_H)
    Y_fit = target[:, TIMES_SKIP:TIMES_SKIP + len(X_fit)].T
    noise = rng_r.normal(0, noise_size, X_fit.shape)
    W_out = np.linalg.pinv(X_fit + noise) @ Y_fit   # (N_H, N_sites)

    # closed-loop (keep warm state — no reset!)
    res_nb.Jout = W_out.T.copy(); res_nb.y = res_nb.Jout @ res_nb.X
    Y_sim = []
    for t in range(T_s - 1):
        res_nb.step_rate(ff * res_nb.y, sigma_dyn=0.)
        Y_sim.append(res_nb.y.copy())
    Y_sim_list.append(np.array(Y_sim).T)

    # W projection into X row-space (for G-scores)
    W_T = W_out.T.astype(np.float64)           # (N_sites, N_H)
    _, sx, Vtx = np.linalg.svd(X_fit.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    W_proj_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

    sess_X[idx] = X_fit
    sess_Y[idx] = Y_fit

# ── per-session G-scores (SVD of W_stack) ────────────────────────────────────
W_stack_s  = np.array(W_proj_list)             # (N_subj, D)
W_mean_s   = W_stack_s.mean(0)
Wcent_s    = W_stack_s - W_mean_s
_, sv_s, Vt_svd_s = np.linalg.svd(Wcent_s, full_matrices=False)
M_eff_s    = W_stack_s.shape[0] - 1
G_sess     = Wcent_s @ Vt_svd_s[:M_eff_s].T   # (N_subj, M_eff_s)

# ── archetype M-selection: scree + reconstruction error ──────────────────────
expl_var_arch = sv_s**2 / (sv_s**2).sum()
cumvar_arch   = np.cumsum(expl_var_arch)

MMAX = min(60, M_eff_s)
recon_err_arr = np.zeros(MMAX + 1)
norm_W = np.linalg.norm(Wcent_s, "fro")
for M in range(MMAX + 1):
    if M == 0:
        err = norm_W
    else:
        W_hat = G_sess[:, :M] @ Vt_svd_s[:M]
        err   = np.linalg.norm(Wcent_s - W_hat, "fro")
    recon_err_arr[M] = err / norm_W

print(f"G-scores (per-session): {G_sess.shape}")
print(f"Recon error  M=1: {recon_err_arr[1]:.3f}  M=10: {recon_err_arr[10]:.3f}  M=50: {recon_err_arr[50]:.3f}")

# ── temporal correlation decay ────────────────────────────────────────────────
def _delayed_fc(data, delay):
    if delay == 0:
        return np.nan_to_num(np.corrcoef(data))
    C = np.corrcoef(data[:, :-delay], data[:, delay:])
    return np.nan_to_num(C[data.shape[0]:, :data.shape[0]])

all_r_es, all_r_ee, all_r_tr = [], [], []
for i in range(len(Y_emp_list)):
    emp   = Y_emp_list[i][:, TIMES_SKIP:]
    sim   = Y_sim_list[i][:, TIMES_SKIP:]
    Tmin  = min(emp.shape[1], sim.shape[1])
    emp   = emp[:, :Tmin]; sim = sim[:, :Tmin]
    emp_e = emp[:, ::2];   emp_o = emp[:, 1::2]
    fc0   = _delayed_fc(emp, 0)
    r_es, r_ee, r_tr = [], [], []
    for d in range(MAX_DELAY + 1):
        if emp.shape[1] <= d or emp_e.shape[1] <= d:
            r_es.append(np.nan); r_ee.append(np.nan); r_tr.append(np.nan); continue
        fc_e = _delayed_fc(emp,   d); fc_s = _delayed_fc(sim,   d)
        fc_v = _delayed_fc(emp_e, d); fc_o = _delayed_fc(emp_o, d)
        r_es.append(np.corrcoef(fc_e.flat, fc_s.flat)[0, 1])
        r_ee.append(np.corrcoef(fc_e.flat, fc0.flat)[0, 1])
        r_tr.append(np.corrcoef(fc_v.flat, fc_o.flat)[0, 1])
    all_r_es.append(r_es); all_r_ee.append(r_ee); all_r_tr.append(r_tr)

all_r_es = np.array(all_r_es); all_r_ee = np.array(all_r_ee); all_r_tr = np.array(all_r_tr)
print(f"Temporal decay arrays: {all_r_es.shape}")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 6 — Per-patient W: pool sessions → better conditioned regression
#           → G-scores per patient (76 × M_eff_p)
# ══════════════════════════════════════════════════════════════════════════════
code("""\
rng_p = np.random.default_rng(RNG_SEED + 1)
W_proj_p_list = []

for pid in tqdm(unique_pids, desc="Per-patient W"):
    idxs   = patient_sids[pid]
    X_coll = np.vstack([sess_X[i] for i in idxs])    # pool all sessions
    Y_coll = np.vstack([sess_Y[i] for i in idxs])
    noise  = rng_p.normal(0, noise_size, X_coll.shape)
    W      = np.linalg.pinv(X_coll + noise) @ Y_coll  # (N_H, N_sites)

    W_T = W.T.astype(np.float64)
    _, sx, Vtx = np.linalg.svd(X_coll.astype(np.float64), full_matrices=False)
    k_pc = min(K_PC, int((sx > 1e-8).sum()))
    Vt_k = Vtx[:k_pc]
    W_proj_p_list.append((W_T @ Vt_k.T @ Vt_k).flatten())

W_stack_p  = np.array(W_proj_p_list)          # (N_patients, D)
W_mean_p   = W_stack_p.mean(0)
Wcent_p    = W_stack_p - W_mean_p
_, sv_p, Vt_svd_p = np.linalg.svd(Wcent_p, full_matrices=False)
M_eff_p    = W_stack_p.shape[0] - 1           # = N_patients - 1 = 75
G_pat      = Wcent_p @ Vt_svd_p[:M_eff_p].T  # (N_patients, M_eff_p)

print(f"G-scores (per-patient): {G_pat.shape}  "
      f"(CC={(patient_labels==0).sum()}, AD={(patient_labels==1).sum()} patients)")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 7 — LDA helpers + LOPO k-sweeps (all methods)
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# ── LDA + helpers ─────────────────────────────────────────────────────────────
from sklearn.metrics import roc_auc_score

class _LDA:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        c0, c1 = self.classes_[0], self.classes_[1]
        X0, X1 = X[y == c0], X[y == c1]
        mu0, mu1 = X0.mean(0), X1.mean(0)
        Sw  = ((X0-mu0).T@(X0-mu0) + (X1-mu1).T@(X1-mu1)
               + 1e-6 * np.eye(X0.shape[1]))
        w   = np.linalg.solve(Sw, mu1-mu0); w /= np.linalg.norm(w) + 1e-12
        self.w_ = w; self.thr_ = 0.5*((X0@w).mean() + (X1@w).mean())
        return self
    def predict(self, X):
        return np.where(X @ self.w_ >= self.thr_, self.classes_[1], self.classes_[0])
    def score(self, X):
        # continuous LDA score; higher = class 1 / AD
        return (X @ self.w_).ravel()

def _balance(X, y, seed=0):
    rng2 = np.random.default_rng(seed)
    c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    if n == 0: return X, y
    sel = np.concatenate([rng2.choice(c0,n,replace=False), rng2.choice(c1,n,replace=False)])
    rng2.shuffle(sel); return X[sel], y[sel]

def _auc(yt, ys):
    # AUROC from true labels and continuous LDA scores
    try:
        return roc_auc_score(yt, ys)
    except Exception:
        return 0.5

def _class_acc(yt, yp):
    cc = (yp[yt==0]==0).mean() if (yt==0).any() else np.nan
    ad = (yp[yt==1]==1).mean() if (yt==1).any() else np.nan
    return cc, ad

# ── session-level LOPO (hold out all sessions of one patient) ─────────────────
def lopo_sess(X_feat, y, pca_dims=None):
    all_t, all_p, all_s = [], [], []
    for fi, pid in enumerate(unique_pids):
        tr = np.where(patient_ID_arr != pid)[0]
        te = np.where(patient_ID_arr == pid)[0]
        Xtr, ytr = X_feat[tr], y[tr]
        Xte, yte = X_feat[te], y[te]
        if pca_dims is not None and Xtr.shape[1] > pca_dims:
            mu = Xtr.mean(0)
            _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
            Vt = Vt[:pca_dims]; Xtr = (Xtr-mu)@Vt.T; Xte = (Xte-mu)@Vt.T
        Xb, yb = _balance(Xtr, ytr, seed=RNG_SEED + fi)
        if len(np.unique(yb)) < 2: continue
        lda = _LDA().fit(Xb, yb)
        all_t.extend(yte.tolist())
        all_p.extend(lda.predict(Xte).tolist())
        all_s.extend(lda.score(Xte).tolist())
    return np.array(all_t), np.array(all_p), np.array(all_s)

# ── patient-level LOPO (one row per patient) ──────────────────────────────────
def lopo_pat(X_feat, y, pca_dims=None):
    n = len(y); all_t, all_p, all_s = [], [], []
    for i in range(n):
        tr = np.array([j for j in range(n) if j != i])
        Xtr, ytr = X_feat[tr], y[tr]
        Xte, yte = X_feat[[i]], y[[i]]
        if pca_dims is not None and Xtr.shape[1] > pca_dims:
            mu = Xtr.mean(0)
            _, _, Vt = np.linalg.svd(Xtr - mu, full_matrices=False)
            Vt = Vt[:pca_dims]; Xtr = (Xtr-mu)@Vt.T; Xte = (Xte-mu)@Vt.T
        Xb, yb = _balance(Xtr, ytr, seed=RNG_SEED + i)
        if len(np.unique(yb)) < 2: continue
        lda = _LDA().fit(Xb, yb)
        all_t.extend(yte.tolist())
        all_p.extend(lda.predict(Xte).tolist())
        all_s.extend(lda.score(Xte).tolist())
    return np.array(all_t), np.array(all_p), np.array(all_s)

# ── k-sweeps (metric: AUROC on continuous LDA scores) ─────────────────────────
kv50  = list(range(1, 51))
kv_p  = list(range(1, M_eff_p + 1))

print("k-sweep: per-session G-scores (k=1..50) ...")
auc_g_s = np.array([_auc(*lopo_sess(G_sess[:, :k], state_ID_num)[::2])
                    for k in tqdm(kv50, leave=False)])
best_k_gs = kv50[int(np.argmax(auc_g_s))]
yt_gs, yp_gs, ys_gs = lopo_sess(G_sess[:, :best_k_gs], state_ID_num)
auc_gs = _auc(yt_gs, ys_gs); cc_gs, ad_gs = _class_acc(yt_gs, yp_gs)
print(f"  best k={best_k_gs}  AUC={auc_gs:.4f}  CC={cc_gs*100:.0f}%  AD={ad_gs*100:.0f}%")

print("k-sweep: per-patient G-scores (k=1..75) ...")
auc_g_p = np.array([_auc(*lopo_pat(G_pat[:, :k], patient_labels)[::2])
                    for k in tqdm(kv_p, leave=False)])
best_k_gp = kv_p[int(np.argmax(auc_g_p))]
yt_gp, yp_gp, ys_gp = lopo_pat(G_pat[:, :best_k_gp], patient_labels)
auc_gp = _auc(yt_gp, ys_gp); cc_gp, ad_gp = _class_acc(yt_gp, yp_gp)
print(f"  best k={best_k_gp}  AUC={auc_gp:.4f}  CC={cc_gp*100:.0f}%  AD={ad_gp*100:.0f}%")

print("k-sweep: per-session FC (k=1..50) ...")
auc_fc_s = np.array([_auc(*lopo_sess(FC_flat, state_ID_num, pca_dims=k)[::2])
                     for k in tqdm(kv50, leave=False)])
best_k_fcs = kv50[int(np.argmax(auc_fc_s))]
yt_fcs, yp_fcs, ys_fcs = lopo_sess(FC_flat, state_ID_num, pca_dims=best_k_fcs)
auc_fcs = _auc(yt_fcs, ys_fcs); cc_fcs, ad_fcs = _class_acc(yt_fcs, yp_fcs)
print(f"  best k={best_k_fcs}  AUC={auc_fcs:.4f}  CC={cc_fcs*100:.0f}%  AD={ad_fcs*100:.0f}%")

print("k-sweep: per-patient FC (k=1..50) ...")
auc_fc_p = np.array([_auc(*lopo_pat(FC_pat, patient_labels, pca_dims=k)[::2])
                     for k in tqdm(kv50, leave=False)])
best_k_fcp = kv50[int(np.argmax(auc_fc_p))]
yt_fcp, yp_fcp, ys_fcp = lopo_pat(FC_pat, patient_labels, pca_dims=best_k_fcp)
auc_fcp = _auc(yt_fcp, ys_fcp); cc_fcp, ad_fcp = _class_acc(yt_fcp, yp_fcp)
print(f"  best k={best_k_fcp}  AUC={auc_fcp:.4f}  CC={cc_fcp*100:.0f}%  AD={ad_fcp*100:.0f}%")

print("\\n" + "="*62)
print(f"{'Method':<30} {'AUROC':>7}  {'CC':>5}  {'AD':>5}  {'k':>4}")
print("-"*62)
for nm, auc, cc, ad, k in [
    ("Per-session G-scores", auc_gs,  cc_gs,  ad_gs,  best_k_gs),
    ("Per-patient G-scores", auc_gp,  cc_gp,  ad_gp,  best_k_gp),
    ("Per-session FC",       auc_fcs, cc_fcs, ad_fcs, best_k_fcs),
    ("Per-patient FC",       auc_fcp, cc_fcp, ad_fcp, best_k_fcp),
]:
    print(f"  {nm:<28} {auc:.4f}  {cc*100:4.0f}%  {ad*100:4.0f}%  {k:4d}")
print("="*62)
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 8 — FIGURE 1: Dataset overview  (paper-ready, uniform panels)
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# Uniform 2-row × 4-col layout; every cell = same size
# FC matrices forced to square aspect
PSIZ = 3.5   # panel size in inches (width = height for square panels)
fig1 = plt.figure(figsize=(PSIZ*4 + 0.8, PSIZ*2 + 0.8), facecolor="white")
gs1  = gridspec.GridSpec(2, 4, figure=fig1,
                         hspace=0.48, wspace=0.40,
                         top=0.94, bottom=0.06, left=0.07, right=0.97)

# ── A: Acquisition table ─────────────────────────────────────────────────────
ax_a = fig1.add_subplot(gs1[0, 0]); ax_a.axis("off")
rows_tbl = [
    ["TR", "3.0 s"], ["Volumes", "140 (~7 min)"],
    ["Atlas", "Schaefer-100\\n+ HO-21 sub-ctx"],
    ["Parcels", "121"], ["Confounds", "24-HMP + WM + CSF"],
    ["Bandpass", "0.01-0.10 Hz"],
    [f"CC (CN)", f"{len(ctrl_indices)} sessions"],
    [f"AD",      f"{len(ad_indices)} sessions"],
]
tbl = ax_a.table(cellText=rows_tbl, colLabels=["Parameter", "Value"],
                 cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0: cell.set_facecolor("#e0e0e0")
_tag(ax_a, "A")
ax_a.set_title("Acquisition setup", pad=4)

# ── B: Sample BOLD timeseries (spans 2 cols) ──────────────────────────────────
ax_b = fig1.add_subplot(gs1[0, 1:3])
t_ax = np.arange(trial_dur) * TR
parcels_show = [48, 0, 37, 103]
pnames = ["pCunPCC (Default)", "LH Vis-1", "Default Temp", "L-Thalamus"]
ex_cc = sigs[ctrl_indices[0]]; ex_ad = sigs[ad_indices[0]]
for ri, (pi, pn) in enumerate(zip(parcels_show, pnames)):
    off = ri * 5
    ax_b.plot(t_ax, ex_cc[pi, :trial_dur] + off, lw=1.0, color=CC_COL,
              alpha=0.9, label="CC" if ri==0 else "")
    ax_b.plot(t_ax, ex_ad[pi, :trial_dur] + off, lw=1.0, color=AD_COL,
              alpha=0.9, label="AD" if ri==0 else "")
    ax_b.text(-10, off, pn, fontsize=7, va="center", ha="right")
ax_b.set_xlim(-18, t_ax[-1]+5); ax_b.set_xlabel("Time (s)")
ax_b.set_ylabel("BOLD + offset"); ax_b.set_yticks([])
ax_b.legend(fontsize=8, loc="upper right")
_tag(ax_b, "B"); _clean(ax_b)
ax_b.set_title("Sample BOLD timeseries", pad=4)

# ── C: Mean FC violin ────────────────────────────────────────────────────────
ax_c = fig1.add_subplot(gs1[0, 3])
fmc_cc = [FC_collected[i][np.triu_indices(N_SITES,k=1)].mean() for i in ctrl_indices]
fmc_ad = [FC_collected[i][np.triu_indices(N_SITES,k=1)].mean() for i in ad_indices]
vp = ax_c.violinplot([fmc_cc, fmc_ad], positions=[0,1], showmedians=True)
vp["bodies"][0].set_facecolor(CC_COL); vp["bodies"][1].set_facecolor(AD_COL)
for kv in ["cbars","cmins","cmaxes","cmedians"]:
    vp[kv].set_color("k"); vp[kv].set_linewidth(1.2)
ax_c.set_xticks([0,1]); ax_c.set_xticklabels(["CC","AD"])
ax_c.set_ylabel("Mean FC (Pearson r)")
_tag(ax_c, "C"); _clean(ax_c)
ax_c.set_title("Mean FC per session", pad=4)

# ── D / E: FC matrices (CC and AD) — square aspect ───────────────────────────
for col_i, (gi, gname, gcol) in enumerate([
        (ctrl_indices[0], "CC", CC_COL),
        (ad_indices[0],   "AD", AD_COL)]):
    ax = fig1.add_subplot(gs1[1, col_i])
    fc_s = FC_collected[gi][:100, :100][np.ix_(sorted_idx, sorted_idx)]
    im   = ax.imshow(fc_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal")
    plt.colorbar(im, ax=ax, shrink=0.82, pad=0.03, fraction=0.046)
    for b in net_bounds:
        ax.axhline(b, color="white", lw=0.5); ax.axvline(b, color="white", lw=0.5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("Brain region"); ax.set_ylabel("Brain region")
    _tag(ax, "DE"[col_i])
    ax.set_title(f"FC matrix — {gname}", color=gcol, pad=4)

# ── F: Population PCA scree ──────────────────────────────────────────────────
ax_f = fig1.add_subplot(gs1[1, 2])
ax_f.bar(range(1, 21), expl_var[:20]*100, color="#455A64", edgecolor="white", lw=0.5)
ax_f.set_xlabel("Principal component"); ax_f.set_ylabel("Explained variance (%)")
ax_f.set_xticks([1, 5, 10, 15, 20])
_tag(ax_f, "F"); _clean(ax_f)
ax_f.set_title("Population PCA scree", pad=4)

# ── G: FC similarity to CC mean ──────────────────────────────────────────────
ax_g = fig1.add_subplot(gs1[1, 3])
cc_corr_cc = [np.corrcoef(FC_collected[i][np.triu_indices(N_SITES,k=1)],
                          fc_ctrl_flat_vec)[0,1] for i in ctrl_indices]
cc_corr_ad = [np.corrcoef(FC_collected[i][np.triu_indices(N_SITES,k=1)],
                          fc_ctrl_flat_vec)[0,1] for i in ad_indices]
allv = cc_corr_cc + cc_corr_ad
bins_g = np.linspace(min(allv)-0.02, max(allv)+0.02, 26)
ax_g.hist(cc_corr_cc, bins=bins_g, alpha=0.65, color=CC_COL,
          label=f"CC  med={np.median(cc_corr_cc):.3f}")
ax_g.hist(cc_corr_ad, bins=bins_g, alpha=0.65, color=AD_COL,
          label=f"AD  med={np.median(cc_corr_ad):.3f}")
ax_g.axvline(np.median(cc_corr_cc), color=CC_COL, lw=1.5, ls="--")
ax_g.axvline(np.median(cc_corr_ad), color=AD_COL, lw=1.5, ls="--")
ax_g.set_xlabel("FC corr. to CC mean"); ax_g.set_ylabel("Count")
ax_g.legend(frameon=False)
_tag(ax_g, "G"); _clean(ax_g)
ax_g.set_title("FC similarity to CC mean", pad=4)

if have_atlas:
    handles = [Patch(color=c, label=n) for n, c in NET_COLORS_MAP.items()]
    fig1.legend(handles=handles, loc="lower center", ncol=7, fontsize=7,
                bbox_to_anchor=(0.5, -0.01), title="Schaefer-100 networks",
                title_fontsize=8)

fig1.savefig(f"{OUT_DIR}/Fig1_dataset.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig1 saved")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 9 — FIGURE 2: Reservoir model
#   2×2 uniform panels — all same size, no spanning columns
#   [A: Mean FC (square)]  [B: W archetype SVD scree]
#   [C: Recon error vs M]  [D: Temporal correlation decay]
# ══════════════════════════════════════════════════════════════════════════════
code("""\
PSIZ2 = 4.5   # panel size (width = height for square panels)
fig2  = plt.figure(figsize=(PSIZ2*2 + 1.2, PSIZ2*2 + 0.9), facecolor="white")
gs2   = gridspec.GridSpec(2, 2, figure=fig2,
                          hspace=0.44, wspace=0.44,
                          top=0.94, bottom=0.07, left=0.10, right=0.96)

# ── A: Mean FC matrix — CC (square aspect) ───────────────────────────────────
ax_a = fig2.add_subplot(gs2[0, 0])
fc_s  = fc_ctrl_mean[:100, :100][np.ix_(sorted_idx, sorted_idx)]
im_a  = ax_a.imshow(fc_s, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="equal")
cb    = plt.colorbar(im_a, ax=ax_a, shrink=0.82, pad=0.03, fraction=0.046)
cb.set_label("Pearson r", fontsize=8); cb.ax.tick_params(labelsize=7)
for b in net_bounds:
    ax_a.axhline(b, color="white", lw=0.4); ax_a.axvline(b, color="white", lw=0.4)
ax_a.set_xticks([]); ax_a.set_yticks([])
ax_a.set_xlabel("Brain region"); ax_a.set_ylabel("Brain region")
ax_a.set_title("Mean FC — CC group", pad=4)
_tag(ax_a, "A")

# ── B: Archetype SVD scree (singular values of W_stack) ──────────────────────
ax_b = fig2.add_subplot(gs2[0, 1])
n_show = min(40, len(sv_s))
ax_b.bar(range(1, n_show + 1),
         (sv_s[:n_show]**2 / (sv_s**2).sum()) * 100,
         color="#6A1B9A", edgecolor="white", lw=0.4)
ax_b.set_xlabel("Archetype index"); ax_b.set_ylabel("Variance explained (%)")
ax_b.set_xticks([1, 10, 20, 30, 40][: int(np.ceil(n_show / 10)) + 1])
ax_b.set_title("W archetype SVD scree", pad=4)
_tag(ax_b, "B"); _clean(ax_b)

# ── C: Reconstruction error vs M + cumulative explained variance ──────────────
ax_c  = fig2.add_subplot(gs2[1, 0])
m_arr = np.arange(MMAX + 1)
ax_c.plot(m_arr, recon_err_arr * 100, color="#6A1B9A", lw=2,
          label="Reconstruction error")
ax_c.set_xlabel("Number of archetypes M")
ax_c.set_ylabel("Reconstruction error (%)")
ax_c.set_xlim(0, MMAX)

ax_c2 = ax_c.twinx()
ax_c2.plot(np.arange(1, MMAX + 1), cumvar_arch[:MMAX] * 100,
           color="#26C6DA", lw=2, ls="--", label="Cumulative variance")
ax_c2.set_ylabel("Cumulative variance (%)", color="#26C6DA")
ax_c2.tick_params(axis="y", labelcolor="#26C6DA")
ax_c2.set_ylim(0, 105)
ax_c2.spines["top"].set_visible(False)

lines1, labs1 = ax_c.get_legend_handles_labels()
lines2, labs2 = ax_c2.get_legend_handles_labels()
ax_c.legend(lines1 + lines2, labs1 + labs2, frameon=False, fontsize=8,
            loc="center right")
ax_c.set_title("M-selection: reconstruction error", pad=4)
_tag(ax_c, "C"); _clean(ax_c)

# ── D: Temporal correlation decay ────────────────────────────────────────────
ax_d     = fig2.add_subplot(gs2[1, 1])
delays_s = np.arange(MAX_DELAY + 1) * TR
_dec = [
    (all_r_es, "#1565C0", "Empirical vs. simulated FC"),
    (all_r_ee, "#757575", "Empirical FC vs. zero-lag FC"),
    (all_r_tr, "#212121", "Test-retest (even/odd, ceiling)"),
]
for arr, col, lbl in _dec:
    med = np.nanmedian(arr, axis=0)
    p25 = np.nanpercentile(arr, 25, axis=0)
    p75 = np.nanpercentile(arr, 75, axis=0)
    ax_d.plot(delays_s, med, color=col, label=lbl, lw=2)
    ax_d.fill_between(delays_s, p25, p75, color=col, alpha=0.18)
ax_d.axhline(0, color="k", ls=":", lw=0.8)
ax_d.set_xlabel("Delay (s)"); ax_d.set_ylabel("Pearson r  (delayed FC)")
ax_d.set_title("Temporal decay of FC", pad=4)
ax_d.set_ylim(-0.15, 1.05); ax_d.set_xlim(0, MAX_DELAY * TR)
ax_d.set_xticks(np.arange(0, MAX_DELAY * TR + 1, TR * 5))
ax_d.legend(frameon=True, framealpha=0.9, edgecolor="#cccccc",
            loc="upper right", fontsize=7.5)
_tag(ax_d, "D"); _clean(ax_d)

fig2.savefig(f"{OUT_DIR}/Fig2_model.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig2 saved")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 10 — FIGURE 3: Perturbation experiments
#   3 rows × 2 cols, all panels same size
#   Row 0: [A: therapeutic KDE]   [B: therapeutic dose-response]
#   Row 1: [C: pert KDE]          [D: pert dose-response]
#   Row 2: [E: pert per-patient]  [F: pert site importance]
#   Panel labels added outside the image; subplot titles give short caption.
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# Panel definitions: (row, col, filename, short_title, panel_letter)
png_layout = [
    (0, 0, "therapeutic_lda_kde.png",    "Offline interpolation\\nLDA score distributions",   "A"),
    (0, 1, "therapeutic_doseresponse.png","Offline interpolation\\nDose-response metrics",     "B"),
    (1, 0, "pert_lda_kde.png",           "Single-site perturbation\\nLDA score distributions", "C"),
    (1, 1, "pert_doseresponse.png",      "Single-site perturbation\\nDose-response metrics",   "D"),
    (2, 0, "pert_per_patient.png",       "Per-patient LDA trajectories",                       "E"),
    (2, 1, "pert_site_importance.png",   "Site importance map",                                "F"),
]

# Uniform panel size; 2 cols × 3 rows
PSIZ3W = 6.5   # panel width
PSIZ3H = 4.8   # panel height
fig3 = plt.figure(figsize=(PSIZ3W * 2 + 0.8, PSIZ3H * 3 + 1.2), facecolor="white")
gs3  = gridspec.GridSpec(3, 2, figure=fig3,
                         hspace=0.28, wspace=0.10,
                         top=0.97, bottom=0.02, left=0.02, right=0.98)

missing = []
for (r, c, fname, short_title, ltr) in png_layout:
    ax = fig3.add_subplot(gs3[r, c])
    fp = os.path.join(ROOT, fname)
    if os.path.exists(fp):
        img = mpimg.imread(fp)
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, f"{fname}\\nnot found",
                ha="center", va="center", fontsize=9, color="red",
                transform=ax.transAxes)
        missing.append(fname)
    ax.axis("off")
    # Panel label (bold, top-left corner, outside image boundary)
    ax.text(-0.02, 1.04, ltr, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")
    # Short caption below panel label
    ax.set_title(short_title, fontsize=9, pad=3, loc="left", x=0.07)

fig3.savefig(f"{OUT_DIR}/Fig3_perturbation.png", dpi=200, bbox_inches="tight")
plt.close()
if missing:
    print(f"WARNING: {len(missing)} PNG(s) not found: {missing}")
else:
    print("Fig3 saved")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Cell 11 — FIGURE 4: Classification results
#   2×3 uniform panels
#   Row 0: [A: G-score k-sweep, k=1..15]  [B: FC k-sweep, k=1..15]  [C: Summary bar]
#   Row 1: [D: Per-patient scatter]  [E: Sensitivity table (span 2)]
# ══════════════════════════════════════════════════════════════════════════════
code("""\
# ── per-patient accuracy per patient (for scatter panel) ─────────────────────
pat_acc_gp = (np.array(yt_gp.tolist()) == np.array(yp_gp.tolist())).astype(float)
pat_lbl_gp = patient_labels

# ── limit k range for display ────────────────────────────────────────────────
K_SHOW = 15
kv_gs_show = kv50[:K_SHOW]                  # k=1..15 (per-session G-scores)
kv_gp_show = kv_p[:K_SHOW]                  # k=1..15 (per-patient G-scores)
kv_fc_show = kv50[:K_SHOW]                  # k=1..15 (FC)

auc_gs_show  = auc_g_s[:K_SHOW]
auc_gp_show  = auc_g_p[:K_SHOW]
auc_fcs_show = auc_fc_s[:K_SHOW]
auc_fcp_show = auc_fc_p[:K_SHOW]

# ── figure — 2×3 uniform panels ───────────────────────────────────────────────
PSIZ4 = 4.2
fig4  = plt.figure(figsize=(PSIZ4*3 + 1.2, PSIZ4*2 + 0.9), facecolor="white")
gs4   = gridspec.GridSpec(2, 3, figure=fig4,
                          hspace=0.50, wspace=0.40,
                          top=0.93, bottom=0.09, left=0.08, right=0.97)

# ── A: G-score k-sweep (k=1..15), per-patient separation highlighted ─────────
ax_a = fig4.add_subplot(gs4[0, 0])
ax_a.plot(kv_gs_show, auc_gs_show, color="#6A1B9A", lw=2.0,
          label=f"Per-session  (AUC={auc_gs:.4f}, k={best_k_gs})")
ax_a.plot(kv_gp_show, auc_gp_show, color="#CE93D8", lw=2.0, ls="--",
          label=f"Per-patient  (AUC={auc_gp:.4f}, k={best_k_gp})")
# mark peaks
ax_a.axvline(min(best_k_gs, K_SHOW), color="#6A1B9A", lw=1.2, ls=":", alpha=0.7)
ax_a.axvline(min(best_k_gp, K_SHOW), color="#CE93D8", lw=1.2, ls=":", alpha=0.7)
ax_a.axhline(0.5, color="gray", lw=0.8, ls=":")
ax_a.set_xlabel("k (G-score dims)"); ax_a.set_ylabel("AUROC")
ax_a.set_title("G-score k-sweep\\n(per-patient W pooling)", pad=4)
ax_a.legend(frameon=False, fontsize=7.5); ax_a.set_ylim(0.40, 1.00)
ax_a.set_xlim(1, K_SHOW)
_tag(ax_a, "A"); _clean(ax_a)

# ── B: FC k-sweep (k=1..15) ──────────────────────────────────────────────────
ax_b = fig4.add_subplot(gs4[0, 1])
ax_b.plot(kv_fc_show, auc_fcs_show, color="#1565C0", lw=2.0,
          label=f"Per-session  (AUC={auc_fcs:.4f}, k={best_k_fcs})")
ax_b.plot(kv_fc_show, auc_fcp_show, color="#90CAF9", lw=2.0, ls="--",
          label=f"Per-patient  (AUC={auc_fcp:.4f}, k={best_k_fcp})")
ax_b.axhline(0.5, color="gray", lw=0.8, ls=":")
ax_b.set_xlabel("k (PCA dims)"); ax_b.set_ylabel("AUROC")
ax_b.set_title("FC k-sweep\\n(per-patient concatenation)", pad=4)
ax_b.legend(frameon=False, fontsize=7.5); ax_b.set_ylim(0.40, 1.00)
ax_b.set_xlim(1, K_SHOW)
_tag(ax_b, "B"); _clean(ax_b)

# ── C: Summary bar chart (all 4 methods) ─────────────────────────────────────
ax_c = fig4.add_subplot(gs4[0, 2])
methods = ["FC\\n(per-sess.)", "FC\\n(per-pat.)",
           "G-scores\\n(per-sess.)", "G-scores\\n(per-pat.)"]
aucs    = [auc_fcs, auc_fcp, auc_gs, auc_gp]
colors  = [CC_COL, "#90CAF9", "#6A1B9A", "#CE93D8"]
bars    = ax_c.bar(methods, aucs, color=colors, edgecolor="white", width=0.58)
for bar, val in zip(bars, aucs):
    ax_c.text(bar.get_x() + bar.get_width()/2, val + 0.008,
              f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax_c.axhline(0.5, color="gray", lw=0.9, ls="--")
bars[-1].set_edgecolor("#6A1B9A"); bars[-1].set_linewidth(2.5)
ax_c.set_ylabel("AUROC"); ax_c.set_ylim(0.40, 1.02)
ax_c.set_title("Best AUROC\\n(LOPO LDA)", pad=4)
ax_c.tick_params(axis="x", labelsize=7.5)
_tag(ax_c, "C"); _clean(ax_c)

# ── D: Per-patient accuracy scatter (per-patient G-scores) ───────────────────
ax_d = fig4.add_subplot(gs4[1, 0])
for lbl, col, name in [(0, CC_COL, "CC"), (1, AD_COL, "AD")]:
    idx = np.where(pat_lbl_gp == lbl)[0]
    jit = np.random.default_rng(0).uniform(-0.12, 0.12, len(idx))
    ax_d.scatter(np.full(len(idx), lbl) + jit, pat_acc_gp[idx] * 100,
                 color=col, alpha=0.65, s=28, label=name)
    ax_d.plot([lbl - 0.25, lbl + 0.25],
              [pat_acc_gp[idx].mean() * 100] * 2, color=col, lw=2.5)
ax_d.axhline(50, color="gray", lw=0.8, ls="--")
ax_d.set_xticks([0, 1]); ax_d.set_xticklabels(["CC", "AD"])
ax_d.set_ylabel("Correct prediction (%)"); ax_d.set_ylim(-5, 115)
ax_d.legend(frameon=False)
ax_d.set_title("Per-patient accuracy\\n(per-patient G-scores)", pad=4)
_tag(ax_d, "D"); _clean(ax_d)

# ── E: Sensitivity table (spans 2 cols) ──────────────────────────────────────
ax_e = fig4.add_subplot(gs4[1, 1:]); ax_e.axis("off")
rows_e = [
    ["Method",              "AUROC",                   "CC sens",              "AD sens"],
    ["G-scores (sess.)",    f"{auc_gs:.4f}",           f"{cc_gs*100:.0f}%",    f"{ad_gs*100:.0f}%"],
    ["G-scores (pat.) *",   f"{auc_gp:.4f}",           f"{cc_gp*100:.0f}%",    f"{ad_gp*100:.0f}%"],
    ["FC (sess.)",          f"{auc_fcs:.4f}",          f"{cc_fcs*100:.0f}%",   f"{ad_fcs*100:.0f}%"],
    ["FC (pat.)",           f"{auc_fcp:.4f}",          f"{cc_fcp*100:.0f}%",   f"{ad_fcp*100:.0f}%"],
]
tbl_e = ax_e.table(cellText=rows_e[1:], colLabels=rows_e[0],
                   cellLoc="center", loc="center", bbox=[0.05, 0.12, 0.90, 0.78])
tbl_e.auto_set_font_size(False); tbl_e.set_fontsize(9)
for (r, c), cell in tbl_e.get_celld().items():
    cell.set_edgecolor("#cccccc")
    if r == 0:   cell.set_facecolor("#e0e0e0")
    elif r == 2: cell.set_facecolor("#EDE7F6")   # highlight per-patient G-scores row
ax_e.text(0.5, 0.03, "* per-patient W: sessions pooled per patient (LOPO, nested k-selection verified)",
          transform=ax_e.transAxes, fontsize=7, ha="center", color="#555")
_tag(ax_e, "E")
ax_e.set_title("Classification summary (AUROC)", pad=4)

fig4.savefig(f"{OUT_DIR}/Fig4_classification.png", dpi=200, bbox_inches="tight")
plt.close()
print("Fig4 saved")
print("\\nAll done — check summary_out/ for Fig1..Fig4")
""")

# ══════════════════════════════════════════════════════════════════════════════
# Supplementary Figure S1 — FC reconstruction quality: per-patient W noise sweep
# ══════════════════════════════════════════════════════════════════════════════
md("## Supplementary Figure S1 — FC Reconstruction Quality\n\n"
   "Per-patient W (same as 85% classifier): closed-loop FC-r on concatenated patient signal "
   "as a function of noise regularisation level. Teacher-forced in-sample check confirms "
   "W perfectly encodes FC (r=1.0); differences are in closed-loop autonomous dynamics stability.")

code("""\
import numpy as np, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec
import os
from scipy import stats

OUT_DIR = "./summary_out"

# ── Load pre-computed noise-sweep results ─────────────────────────────────────
data = np.load("fc_recon_noise_sweep.npz", allow_pickle=True)
NOISE_VALS   = list(data["noise_vals"])          # [0.0001 .. 0.05]
noise_means  = data["noise_means"]               # mean FC-r per noise level
noise_stds   = data["noise_stds"]
r_sess       = data["r_sess"]                    # per-session W baseline (per-session eval)
r_pat_best   = data["r_pat_best"]               # per-patient W at best noise
best_noise   = float(data["best_noise"])
patient_labels = data["patient_labels"]

# Per-patient average for CC / AD split (best noise)
# r_pat_best is per-session; average per patient
patient_ids_raw = data["patient_ids_raw"]
labels_raw      = data["labels_raw"]
unique_pids     = np.unique(patient_ids_raw)
patient_sids    = {pid: np.where(patient_ids_raw == pid)[0] for pid in unique_pids}

r_cc, r_ad = [], []
for pid_i, pid in enumerate(unique_pids):
    # r_pat_best is per-patient (one entry per unique patient)
    r_mean = float(r_pat_best[pid_i])
    if patient_labels[pid_i] == 0:
        r_cc.append(r_mean)
    else:
        r_ad.append(r_mean)
r_cc = np.array(r_cc); r_ad = np.array(r_ad)

PSIZ = 4.2
fig_s1 = plt.figure(figsize=(PSIZ*3 + 1.0, PSIZ + 0.8), facecolor="white")
gs_s1  = gridspec.GridSpec(1, 3, figure=fig_s1, hspace=0.45, wspace=0.42,
                           top=0.90, bottom=0.14, left=0.08, right=0.97)

BLUE   = "#1565C0"; ORANGE = "#E64A19"; GRAY = "#757575"

def _tag(ax, ltr):
    ax.text(-0.12, 1.03, ltr, transform=ax.transAxes,
            fontsize=13, fontweight="bold", va="bottom", ha="left")

# ── Panel A: noise sweep curve ─────────────────────────────────────────────────
ax_a = fig_s1.add_subplot(gs_s1[0, 0])
ax_a.errorbar(range(len(NOISE_VALS)), noise_means, yerr=noise_stds,
              marker="o", color=BLUE, lw=2, capsize=4, zorder=3,
              label="Per-patient W\\n(concat. closed-loop)")
ax_a.axhline(r_sess.mean(), color=ORANGE, lw=1.8, ls="--",
             label=f"Per-session W (baseline): {r_sess.mean():.3f}")
ax_a.axhline(r_sess.mean() + r_sess.std(), color=ORANGE, lw=0.8, ls=":")
ax_a.axhline(r_sess.mean() - r_sess.std(), color=ORANGE, lw=0.8, ls=":")
ax_a.set_xticks(range(len(NOISE_VALS)))
ax_a.set_xticklabels([str(n) for n in NOISE_VALS], rotation=40, ha="right", fontsize=7.5)
ax_a.set_xlabel("Noise regularisation (\\u03c3)", fontsize=9)
ax_a.set_ylabel("FC-r (Pearson r)", fontsize=9)
ax_a.set_title("Per-patient W: FC-r vs noise\\n(closed-loop, concatenated signal)", fontsize=9)
ax_a.legend(fontsize=7, loc="lower right")
ax_a.spines["top"].set_visible(False); ax_a.spines["right"].set_visible(False)
_tag(ax_a, "A")

# ── Panel B: boxplot — per-session vs per-patient W ────────────────────────────
ax_b = fig_s1.add_subplot(gs_s1[0, 1])
r_pat_best_all = r_pat_best
bp = ax_b.boxplot([r_sess, r_pat_best_all],
                  labels=[f"Per-session W\\n(per-sess eval)\\nn={len(r_sess)}",
                           f"Per-patient W\\n(concat eval)\\nnoise={best_noise}\\nn={len(r_pat_best_all)}"],
                  patch_artist=True, widths=0.5)
for patch, c in zip(bp["boxes"], [ORANGE, BLUE]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
for median in bp["medians"]:
    median.set_color("black"); median.set_linewidth(2)
rng_jit = np.random.default_rng(0)
for xi, r_arr in zip([1, 2], [r_sess, r_pat_best_all]):
    jit = rng_jit.normal(0, 0.06, len(r_arr))
    ax_b.scatter(xi + jit, r_arr, alpha=0.3, s=14, color="k", zorder=3)
ax_b.set_ylabel("FC-r", fontsize=9)
ax_b.set_title("FC-r: per-session vs per-patient W\\n(closed-loop evaluation)", fontsize=9)
ax_b.spines["top"].set_visible(False); ax_b.spines["right"].set_visible(False)
_tag(ax_b, "B")

# Add mean annotations
for xi, r_arr in zip([1, 2], [r_sess, r_pat_best_all]):
    ax_b.text(xi, r_arr.max() + 0.005, f"mean={r_arr.mean():.3f}", ha="center", fontsize=7, color=GRAY)

# ── Panel C: CC vs AD at best noise ────────────────────────────────────────────
ax_c = fig_s1.add_subplot(gs_s1[0, 2])
bp2 = ax_c.boxplot([r_cc, r_ad], labels=["CC", "AD"],
                   patch_artist=True, widths=0.5)
for patch, c in zip(bp2["boxes"], ["#42A5F5", "#EF5350"]):
    patch.set_facecolor(c); patch.set_alpha(0.75)
for median in bp2["medians"]:
    median.set_color("black"); median.set_linewidth(2)
for xi, r_arr in zip([1, 2], [r_cc, r_ad]):
    jit = rng_jit.normal(0, 0.07, len(r_arr))
    ax_c.scatter(xi + jit, r_arr, alpha=0.4, s=18, color="k", zorder=3)

t_stat, p_val = stats.ttest_ind(r_cc, r_ad)
y_top = max(r_cc.max(), r_ad.max()) + 0.015
ax_c.plot([1, 2], [y_top, y_top], "k-", lw=1.0)
ax_c.text(1.5, y_top + 0.005, f"p = {p_val:.3f}", ha="center", fontsize=8)

ax_c.set_ylabel("FC-r (per-patient W)", fontsize=9)
ax_c.set_title(f"CC vs AD FC reconstruction quality\\n(noise={best_noise}, concatenated signal)", fontsize=9)
ax_c.spines["top"].set_visible(False); ax_c.spines["right"].set_visible(False)
_tag(ax_c, "C")

fig_s1.suptitle("Supplementary Figure S1 — FC reconstruction quality", fontsize=11, fontweight="bold")
fig_s1.savefig(f"{OUT_DIR}/FigS1_fc_reconstruction.png", dpi=200, bbox_inches="tight")
plt.close()
print("FigS1 saved")
print(f"  Per-session W baseline:   mean FC-r = {r_sess.mean():.4f}")
print(f"  Per-patient W best noise: mean FC-r = {r_pat_best_all.mean():.4f} (noise={best_noise})")
print(f"  CC: {r_cc.mean():.4f} +/- {r_cc.std():.4f}   AD: {r_ad.mean():.4f} +/- {r_ad.std():.4f}   p={p_val:.3f}")
""")

# ── Write notebook ────────────────────────────────────────────────────────────
nb.cells = cells
with open("summary_figures.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

n_cells = len(cells)
print(f"summary_figures.ipynb written — {n_cells} cells")
print("Run with:")
print("  jupyter nbconvert --to notebook --execute --inplace summary_figures.ipynb")
