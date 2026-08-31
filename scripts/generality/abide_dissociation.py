"""
abide_dissociation.py
=====================
GENERALITY TEST: does the graph-vs-activity dissociation reproduce in a second,
unrelated disorder?

Runs the AD paper's pipeline unchanged on ABIDE I (autism spectrum disorder vs
typically developing controls; Harvard-Oxford 111 parcels, C-PAC filt_noglobal),
and asks the three questions that carry the paper's claim:

  Q1  Do the sites with the largest read-out change (||dW||, "most affected")
      differ from the sites where a focal resonant drive best moves the disease
      discriminant ("effective targets")?
  Q2  Is ||dW|| disease-specific, or does a control-vs-control null reproduce it?
  Q3  Are the most-affected sites the weakly-connected ones (the actuator /
      controllability account)?

Nothing here overwrites the ADNI analysis: every output goes to
abide_generality/ under new filenames.
"""
# --- path bootstrap: import shared modules from scripts/common ---
import sys as _sys, pathlib as _pathlib
for _p in _pathlib.Path(__file__).resolve().parents:
    if (_p / "scripts" / "common").is_dir():
        _sys.path.insert(0, str(_p / "scripts" / "common")); break
# --- end bootstrap ---

import os, csv
import numpy as np
from scipy.linalg import eig as sla_eig
from scipy.stats import spearmanr, mannwhitneyu
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
from res import RESERVOIRE_SIMPLE

RNG_SEED = 42
N_SITES  = 111                 # Harvard-Oxford parcels in the ABIDE C-PAC release
N_PC_MODEL, TIMES_SKIP = 50, 10
ff, N_HIDDEN, SIGMA, SR = 0.1, 2000, 0.05, 0.95
K_LDA, MAX_LAG = 25, 2
N_PER_GROUP = int(os.environ.get("N_PER_GROUP", 60))   # balanced subsample
A_REF       = float(os.environ.get("A_REF", 4.0))
OUT = "abide_generality"
os.makedirs(OUT, exist_ok=True)

# ── load ABIDE ────────────────────────────────────────────────────────────────
d  = np.load("ABIDE/abide_timeseries.npz", allow_pickle=True)
ts, ph = d["timeseries"], d["phenotypic"]
dx   = np.array([int(r[7]) for r in ph])       # 1 = autism, 2 = control
site = np.array([str(r[5]) for r in ph])
sid  = np.array([str(r[6]) for r in ph])
y_all = (dx == 1).astype(int)                  # 1 = ASD (the "patient" class)

# use only subjects with an adequate run length (ADNI used >=139 volumes);
# truncating everything to the global minimum (78) would cripple the read-out fit
T_MIN = int(os.environ.get("T_MIN", 150))
T_all = np.array([np.asarray(t).shape[0] for t in ts])
ok = T_all >= T_MIN
T_use = T_MIN
rng = np.random.default_rng(RNG_SEED)
idx_c = np.where((y_all == 0) & ok)[0]; idx_a = np.where((y_all == 1) & ok)[0]
sel = np.concatenate([rng.choice(idx_c, min(N_PER_GROUP, len(idx_c)), replace=False),
                      rng.choice(idx_a, min(N_PER_GROUP, len(idx_a)), replace=False)])
rng.shuffle(sel)

signals = [np.asarray(ts[i])[:T_use].T for i in sel]      # (111, T)
plabel  = y_all[sel]; psite = site[sel]; pid = sid[sel]
n_pat   = len(sel)
print(f"ABIDE: {n_pat} subjects ({(plabel==0).sum()} control, {(plabel==1).sum()} ASD), "
      f"{len(np.unique(psite))} sites, T={T_use}")

# ── population PCA + reservoir (same construction as the ADNI pipeline) ───────
all_sig = np.concatenate([s.T for s in signals], 0)
ev, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
ev50 = evec[:, np.argsort(ev)[::-1]][:, :N_PC_MODEL]

np.random.seed(RNG_SEED)
par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN, T=T_use, dt=0.005,
           sigma_input=0.01, shape=(N_HIDDEN, N_SITES, N_SITES, T_use))
res = RESERVOIRE_SIMPLE(par); res.J *= SR / max(abs(np.linalg.eigvals(res.J)))

def teacher_force(s):
    T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T - 1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:], tgt

patX, patT = [], []
for s in tqdm(signals, desc="  TF"):
    X, tgt = teacher_force(s); patX.append(X); patT.append(tgt)
rw = np.random.default_rng(RNG_SEED + 1)
patW = []
for X, tgt in zip(patX, patT):
    Y = tgt[:, TIMES_SKIP:TIMES_SKIP + X.shape[0]].T
    patW.append(np.linalg.pinv(X + rw.normal(0, SIGMA, X.shape)) @ Y)
patW = np.array(patW)

ctrl = np.where(plabel == 0)[0]; asd = np.where(plabel == 1)[0]
Wc = patW[ctrl].mean(0)

# ── resonant frequency of the reservoir ──────────────────────────────────────
wv, vl, vr = sla_eig(res.J, left=True, right=True)
pos = np.where(wv.imag > 1e-8)[0]; order = pos[np.argsort(np.abs(wv[pos]))[::-1]]
f1 = float(abs(np.angle(wv[order[0]])) / (2 * np.pi))
print(f"  resonant f1 = {f1:.4f} cycles/step")

# ── FC-lag discriminant ──────────────────────────────────────────────────────
class LDA:
    def fit(s, X, y):
        c0, c1 = np.unique(y); X0, X1 = X[y == c0], X[y == c1]
        m0, m1 = X0.mean(0), X1.mean(0)
        Sw = (X0-m0).T@(X0-m0) + (X1-m1).T@(X1-m1) + 1e-6*np.eye(X.shape[1])
        w = np.linalg.solve(Sw, m1 - m0); s.w = w/(np.linalg.norm(w)+1e-12); return s
    def tr(s, X): return X @ s.w

def lagc(S, l):
    if l == 0: return np.corrcoef(S.T)
    T = S.shape[0]; A = S[:T-l].astype(float); B = S[l:].astype(float)
    A -= A.mean(0); B -= B.mean(0); A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - l)

def feat(W, X):
    S = (W.T.astype(float) @ X.T.astype(float)).T; fs = []
    for l in range(MAX_LAG + 1):
        fc = np.nan_to_num(lagc(S, l))
        fs.append(fc[np.triu_indices(N_SITES, 1)] if l == 0 else fc.flatten())
    return np.concatenate(fs)

fb = np.array([feat(patW[i], patX[i]) for i in tqdm(range(n_pat), desc="  feat", leave=False)])
fm = fb.mean(0); fcc = fb - fm
evf, evecf = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(evf)[::-1]
evf = np.maximum(evf[o], 0); evecf = evecf[:, o]; Gf = evecf * np.sqrt(evf)
lda = LDA().fit(Gf[:, :K_LDA], plabel); Z = lda.tr(Gf[:, :K_LDA])
if Z[plabel == 0].mean() > Z[plabel == 1].mean(): lda.w *= -1; Z = -Z
thr = 0.5 * (Z[plabel == 0].mean() + Z[plabel == 1].mean())
from sklearn.metrics import roc_auc_score
print(f"  in-sample FC-lag discriminant AUROC = {roc_auc_score(plabel, Z):.3f} "
      f"(orientation ASD > control)")

def fscore(W, X):
    f = feat(W, X) - fm
    g = (f @ fcc.T @ evecf) / (np.sqrt(evf) + 1e-12)
    return float(lda.tr(g[:K_LDA].reshape(1, -1))[0])

def osc(i, sites, freqs, amp):
    s = signals[i]; T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T - 1):
        inp = ff * tgt[:, t].copy()
        for st, fr in zip(sites, freqs): inp[st] += amp * np.sin(2*np.pi*fr*t)
        res.step_rate(inp, sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]

# ── Q1: "most affected" (dW) sites vs "effective target" (LDA-resonant) sites ─
DW = np.array([np.linalg.norm(Wc - patW[i], axis=0) for i in asd])   # (n_asd,111)
path_top5 = set()
for r in DW: path_top5.update(np.argsort(r)[::-1][:5].tolist())

print(f"\nSite scan: drive each of {N_SITES} sites at f1, A={A_REF} ...")
fbase = np.array([fscore(patW[i], patX[i]) for i in asd])
red = np.zeros((N_SITES, len(asd)))
for k in tqdm(range(N_SITES), desc="  scan"):
    for j, i in enumerate(asd):
        red[k, j] = fbase[j] - fscore(patW[i], osc(i, [k], [f1], A_REF))
ther_top1 = {j: int(np.argmax(red[:, j])) for j in range(len(asd))}
ther_set = set(ther_top1.values())

shared = path_top5 & ther_set
print("\n=== Q1  DISSOCIATION ===")
print(f"  most-affected (dW top-5 union) : {len(path_top5)} sites")
print(f"  effective targets (top-1/patient): {len(ther_set)} sites")
print(f"  shared: {len(shared)}  "
      f"({100*len(shared)/max(len(path_top5),1):.0f}% / {100*len(shared)/max(len(ther_set),1):.0f}% of each set)")

# ── Q2: is dW disease-specific? control-vs-control null ──────────────────────
DW_ctrl = np.array([np.linalg.norm(patW[[c for c in ctrl if c != i]].mean(0) - patW[i], axis=0)
                    for i in ctrl])
r_map, p_map = spearmanr(DW.mean(0), DW_ctrl.mean(0))
pv = np.array([mannwhitneyu(DW[:, k], DW_ctrl[:, k])[1] for k in range(N_SITES)])
q = pv * N_SITES / (np.argsort(np.argsort(pv)) + 1)
print("\n=== Q2  IS dW DISEASE-SPECIFIC? ===")
print(f"  ASD dW map vs control-null map: rho = {r_map:+.3f} (p={p_map:.2g})")
print(f"  amplitude ratio ASD/control   : {DW.mean()/DW_ctrl.mean():.3f}")
print(f"  sites surviving BH-FDR q<0.05 : {(q<0.05).sum()}/{N_SITES}")

# ── Q3: are most-affected sites the weakly connected ones? ───────────────────
FCm = np.mean([np.nan_to_num(np.corrcoef(s)) for s in signals], 0)
strength = np.abs(FCm * (1 - np.eye(N_SITES))).sum(1)
flat = strength < 1e-8            # parcels with no signal in this release
print(f"  ({int(flat.sum())} parcels have no usable signal and are excluded from Q3)")
ok_p = ~flat
r_s, p_s = spearmanr(DW.mean(0)[ok_p], strength[ok_p])
r_t, p_t = spearmanr(red.mean(1)[ok_p], strength[ok_p])
print("\n=== Q3  CONNECTIVITY ACCOUNT ===")
print(f"  dW magnitude   vs FC node strength: rho = {r_s:+.3f} (p={p_s:.2g})")
print(f"  drive efficacy vs FC node strength: rho = {r_t:+.3f} (p={p_t:.2g})")
ps = sorted(path_top5); tsx = sorted(ther_set)
print(f"  mean strength  most-affected sites: {strength[ps].mean():.2f}")
print(f"  mean strength  effective targets  : {strength[tsx].mean():.2f}")
print(f"  mean strength  all sites          : {strength.mean():.2f}")

np.savez(os.path.join(OUT, "abide_dissociation_results.npz"),
         plabel=plabel, psite=psite, pid=pid, Z=Z, thr=thr, f1=f1,
         DW=DW, DW_ctrl=DW_ctrl, red=red, strength=strength,
         path_top5=np.array(sorted(path_top5)), ther_set=np.array(sorted(ther_set)),
         shared=np.array(sorted(shared)), pv=pv)
print(f"\nSaved {OUT}/abide_dissociation_results.npz")
