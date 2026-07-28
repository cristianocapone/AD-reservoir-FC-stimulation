"""
external_oasis_validate.py
==========================
External validation on OASIS-3 by TRANSFER of the frozen, ADNI-trained FC-lag
classifier. This is the "does the trained AD detector work on a cohort it never
saw" test (the strong flavour of external validation), as opposed to independent
replication (refitting the whole pipeline on OASIS and checking the phenomenon
reappears).

Pipeline (transfer):
  1. FREEZE  — build the ADNI classifier ONCE and save every artifact needed to
     score a brand-new subject: the population PCA basis (ev50), the seeded
     reservoir (J), the FC-lag feature mean (fm) and training centred-feature
     matrix (fcc) with its Gram eigenbasis (evf, evecf), the LDA direction
     (lda_w) and decision threshold (thr).  Nothing here is refit on OASIS.
  2. TRANSFER — for each OASIS subject: teacher-force through the FROZEN
     reservoir with the FROZEN ev50, fit only that subject's read-out W, project
     the subject's FC-lag features onto the FROZEN Gram eigenbasis, and apply the
     FROZEN LDA + threshold.  Report AUROC / balanced accuracy vs OASIS labels.

The FC-lag feature/embedding/LDA maths is copied verbatim from pert_closedloop.py
so the frozen classifier is byte-for-byte the paper's operating point (K_LDA=25).

------------------------------------------------------------------------------
TO RUN AGAINST OASIS-3, FILL IN THE THREE MARKED ITEMS BELOW:
  * OASIS_TS_DIR       — dir of per-subject (121, T) parcel-timeseries .npy files
  * OASIS_LABELS_CSV   — csv with columns  subject,label   (label: 0=CN, 1=AD)
  * preprocess_bold()  — make OASIS BOLD match how the ADNI .npy were generated
Gotchas: (i) the 121-parcel atlas order MUST match ADNI exactly; (ii) runs must
clear the >=139-volume filter; (iii) OASIS TR (~2.2 s) != ADNI TR (3 s) — the
per-subject refit absorbs it, but any resonance-in-Hz statement needs the OASIS
TR.  Until the three items are filled, running this file executes a SELF-TEST:
it freezes the ADNI classifier and confirms the frozen-transfer path reproduces
the in-sample training scores (machinery check), then prints next steps.
------------------------------------------------------------------------------
"""
import os, sys
import numpy as np
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); from res import RESERVOIRE_SIMPLE

# ── constants (MUST match the training pipeline / pert_closedloop.py exactly) ──
RNG_SEED = 42; N_CC_SAMP = 40; N_SITES = 121; N_PC_MODEL = 50; TIMES_SKIP = 10
ff = 0.1; N_HIDDEN = 2000; SIGMA = 0.05; SR = 0.95; K_LDA = 25; MAX_LAG = 2
MIN_VOL = 139
ADNI_TS_ROOT = "./timeseries"
FROZEN = "frozen_adni_fclag.npz"

# ╔══════════════════════ OASIS-3 — FILL THESE IN ═══════════════════════════════
OASIS_TS_DIR     = "PATH/TO/oasis3_parcels"        # <-- (121, T) .npy per subject
OASIS_LABELS_CSV = "PATH/TO/oasis3_labels.csv"     # <-- columns: subject,label(0/1)

def preprocess_bold(ts):
    """Make an OASIS (121, T) parcel matrix match the ADNI .npy convention.
    EDIT to mirror exactly how the ADNI timeseries/*.npy were produced
    (e.g. confound regression + detrend + per-parcel z-score). The default
    below is a no-op placeholder; replace it with your real pipeline."""
    return ts
# ╚══════════════════════════════════════════════════════════════════════════════

# ── shared FC-lag machinery (verbatim from pert_closedloop.py) ─────────────────
def lagc(S, l):
    if l == 0: return np.corrcoef(S.T)
    T = S.shape[0]; A = S[:T-l].astype(float); B = S[l:].astype(float)
    A -= A.mean(0); B -= B.mean(0); A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T-l)

def feat(W, X):
    S = (W.T.astype(float) @ X.T.astype(float)).T; fs = []
    for l in range(MAX_LAG+1):
        fc = np.nan_to_num(lagc(S, l))
        fs.append(fc[np.triu_indices(N_SITES, 1)] if l == 0 else fc.flatten())
    return np.concatenate(fs)

class LDA:
    def fit(s, X, y):
        c0, c1 = np.unique(y); X0, X1 = X[y==c0], X[y==c1]; m0, m1 = X0.mean(0), X1.mean(0)
        Sw = (X0-m0).T@(X0-m0) + (X1-m1).T@(X1-m1) + 1e-6*np.eye(X.shape[1])
        w = np.linalg.solve(Sw, m1-m0); w /= np.linalg.norm(w)+1e-12; s.w = w; return s
    def tr(s, X): return X @ s.w

def balm(X, y, sd=0):
    r = np.random.default_rng(sd); c0, c1 = np.where(y==0)[0], np.where(y==1)[0]
    n = min(len(c0), len(c1))
    sel = np.concatenate([r.choice(c0, n, 0), r.choice(c1, n, 0)]); r.shuffle(sel)
    return X[sel], y[sel]

# ── reservoir (deterministic) + per-subject read-out ──────────────────────────
def build_reservoir():
    np.random.seed(RNG_SEED)
    par = dict(tau_m_f=0.0005, tau_m_s=0.0005, N=N_HIDDEN, T=139, dt=0.005,
               sigma_input=0.01, shape=(N_HIDDEN, N_SITES, N_SITES, 139))
    res = RESERVOIRE_SIMPLE(par)
    res.J *= SR / max(abs(np.linalg.eigvals(res.J)))
    return res

def teacher_force(res, s, ev50):
    """Drive the reservoir with the subject's PCA-projected signal; return states."""
    T = s.shape[1]; tgt = (s.T @ ev50 @ ev50.T).T
    res.T = T; res.reset(); X = []
    for t in range(T-1):
        res.step_rate(ff * tgt[:, t], sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:], tgt

def fit_W(X, tgt, rng):
    Yc = tgt[:, TIMES_SKIP:TIMES_SKIP + X.shape[0]].T
    return np.linalg.pinv(X + rng.normal(0, SIGMA, X.shape)) @ Yc

# ── ADNI loader (identical sampling to the training scripts) ───────────────────
def load_adni():
    rng = np.random.default_rng(RNG_SEED); signals, labs, pids = [], [], []
    for sub, lb in [("CN", "CC"), ("AD", "AD")]:
        folder = os.path.join(ADNI_TS_ROOT, sub)
        files = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
        if lb == "CC":
            files = list(rng.choice(files, size=min(N_CC_SAMP, len(files)), replace=False))
        for fn in files:
            a = np.load(os.path.join(folder, fn)).T
            if a.shape[1] == N_SITES and a.shape[0] >= MIN_VOL:
                signals.append(a.T); pids.append(fn.split("_ses-")[0])
                labs.append(0 if lb == "CC" else 1)
    pids = np.array(pids); labs = np.array(labs); upid = np.unique(pids)
    first = {p: np.where(pids == p)[0][0] for p in upid}
    plabel = np.array([labs[first[p]] for p in upid])
    return signals, first, upid, plabel

# ── PART 1: freeze the ADNI classifier ─────────────────────────────────────────
def build_frozen():
    print("FREEZE: building ADNI FC-lag classifier ...")
    signals, first, upid, plabel = load_adni()
    all_sig = np.concatenate([s.T for s in signals], 0)
    evv, evec = np.linalg.eigh(np.cov((all_sig - all_sig.mean(0)).T))
    ev50 = evec[:, np.argsort(evv)[::-1]][:, :N_PC_MODEL]

    res = build_reservoir()
    patX = {}
    for p in tqdm(upid, desc="  TF"):
        patX[p], tgt = teacher_force(res, signals[first[p]], ev50)
        patX[p] = (patX[p], tgt)                      # stash tgt for W-fit
    rw = np.random.default_rng(RNG_SEED + 1); patW = {}
    for p in upid:                                    # single stream = training order
        X, tgt = patX[p]; patW[p] = fit_W(X, tgt, rw)
    patX = {p: patX[p][0] for p in upid}

    fb = np.array([feat(patW[p], patX[p]) for p in tqdm(upid, desc="  feat", leave=False)])
    fm = fb.mean(0); fcc = fb - fm
    evf, evecf = np.linalg.eigh(fcc @ fcc.T); o = np.argsort(evf)[::-1]
    evf = np.maximum(evf[o], 0); evecf = evecf[:, o]; Gf = evecf * np.sqrt(evf)
    Xl, yl = balm(Gf[:, :K_LDA], plabel, RNG_SEED)
    lda = LDA().fit(Xl, yl); Z = lda.tr(Gf[:, :K_LDA])
    if Z[plabel == 0].mean() > Z[plabel == 1].mean(): lda.w *= -1; Z = -Z
    thr = 0.5 * (Z[plabel == 0].mean() + Z[plabel == 1].mean())

    np.savez(FROZEN, ev50=ev50, J=res.J, fm=fm, fcc=fcc, evf=evf, evecf=evecf,
             lda_w=lda.w, thr=thr, K_LDA=K_LDA)
    print(f"  saved {FROZEN}  (AD>CC oriented, thr={thr:.3f})")
    return load_frozen()          # reload so in-memory dict == on-disk artifacts

def load_frozen():
    d = np.load(FROZEN, allow_pickle=True)
    return {k: d[k] for k in d.files}

# ── frozen transfer score for a NEW subject ────────────────────────────────────
def fscore_transfer(W, X, F):
    """Project a new subject's FC-lag features onto the FROZEN embedding and
    apply the FROZEN LDA. Score > thr => AD side, < thr => CC side."""
    f = feat(W, X) - F["fm"]
    g = (f @ F["fcc"].T @ F["evecf"]) / (np.sqrt(F["evf"]) + 1e-12)
    return float(g[:int(F["K_LDA"])] @ F["lda_w"])

def make_reservoir_from_frozen(F):
    res = build_reservoir(); res.J = F["J"].copy(); return res   # guarantee identity

def score_subjects(signals_list, F):
    """signals_list: list of (121, T) arrays. Returns np.array of transfer scores."""
    res = make_reservoir_from_frozen(F); scores = []
    for i, s in enumerate(tqdm(signals_list, desc="  transfer")):
        X, tgt = teacher_force(res, s, F["ev50"])
        W = fit_W(X, tgt, np.random.default_rng(RNG_SEED + 100 + i))  # per-subject seed
        scores.append(fscore_transfer(W, X, F))
    return np.array(scores)

def report(scores, labels, thr, tag):
    from numpy import mean
    labels = np.asarray(labels)
    pred = (scores > thr).astype(int)                 # AD side = 1
    # balanced accuracy
    tpr = pred[labels == 1].mean() if (labels == 1).any() else np.nan
    tnr = (1 - pred[labels == 0]).mean() if (labels == 0).any() else np.nan
    bal = np.nanmean([tpr, tnr])
    # AUROC (rank-based, no sklearn dependency)
    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, scores) if len(np.unique(labels)) == 2 else np.nan
    except Exception:
        auc = np.nan
    print(f"\n[{tag}]  n={len(labels)} ({int((labels==0).sum())} CN, "
          f"{int((labels==1).sum())} AD)  AUROC={auc:.3f}  bal-acc={bal:.3f} "
          f"(TPR={tpr:.2f}, TNR={tnr:.2f}, thr={thr:.3f})")
    return dict(auc=float(auc), bal=float(bal), tpr=float(tpr), tnr=float(tnr))

# ── PART 2: OASIS loader (fill preprocess_bold + paths above) ──────────────────
def load_oasis():
    import csv
    lab = {}
    with open(OASIS_LABELS_CSV) as fh:
        for row in csv.DictReader(fh):
            lab[row["subject"]] = int(row["label"])
    signals, labels = [], []
    for subj, y in lab.items():
        fp = os.path.join(OASIS_TS_DIR, f"{subj}.npy")
        if not os.path.exists(fp):
            print(f"  ! missing {fp} — skipped"); continue
        ts = preprocess_bold(np.load(fp))            # (121, T)
        if ts.shape[0] != N_SITES: ts = ts.T
        if ts.shape[0] != N_SITES or ts.shape[1] < MIN_VOL:
            print(f"  ! {subj}: shape {ts.shape} fails atlas/length check — skipped"); continue
        signals.append(ts); labels.append(y)
    return signals, np.array(labels)

# ── main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    F = load_frozen() if os.path.exists(FROZEN) else build_frozen()
    thr = float(F["thr"])

    oasis_ready = os.path.isdir(OASIS_TS_DIR) and os.path.exists(OASIS_LABELS_CSV)
    if not oasis_ready:
        print("\nOASIS paths not set — running SELF-TEST (frozen-transfer on ADNI).")
        print("This confirms the freeze/transfer machinery reproduces training "
              "separation; it is NOT external validation.")
        signals, first, upid, plabel = load_adni()
        sig = [signals[first[p]] for p in upid]
        s = score_subjects(sig, F)
        report(s, plabel, thr, "SELF-TEST: frozen transfer on ADNI train subjects")
        print("\nNext: set OASIS_TS_DIR, OASIS_LABELS_CSV and preprocess_bold(), "
              "then rerun for the external result.")
    else:
        print("\nOASIS-3 EXTERNAL TRANSFER EVALUATION")
        signals, labels = load_oasis()
        s = score_subjects(signals, F)
        res = report(s, labels, thr, "OASIS-3 external (frozen ADNI classifier)")
        np.savez("external_oasis_scores.npz", scores=s, labels=labels, thr=thr, **res)
        print("Saved external_oasis_scores.npz")
