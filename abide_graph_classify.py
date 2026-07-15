"""
ABIDE I — graph-theoretic FC feature classification.

Features extracted from the thresholded FC matrix (Pearson correlation of
Harvard-Oxford 111-parcel time series):

  Proportional thresholding at d = 5%, 10%, 15%, 20% density
  For each density, binary & weighted variants:
    - degree / strength (mean, std, max)
    - clustering coefficient (mean, std)
    - global efficiency
    - local efficiency (mean)
    - betweenness centrality (mean) — binary only
    - modularity (greedy community detection) — binary only
    - largest-component size (sanity check)

  Plus unthresholded weighted graph:
    - mean absolute correlation (mean FC strength)
    - std of FC matrix
    - variance explained by top-5 eigenvalues

All per-subject features are concatenated into one vector and classified
with the same repeated 5-fold CV (LDA + RF) used in abide_reservoir_classify.py.
"""

import os, time
import numpy as np
from nilearn.datasets import fetch_abide_pcp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import networkx as nx
from tqdm import tqdm

# ── config ───────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "ABIDE")
T_CUTOFF   = 116          # same as reservoir script
DENSITIES  = [0.05, 0.10, 0.15, 0.20]  # proportional thresholds
N_FOLDS    = 5
N_REPEATS  = 10
SEED       = 42
N_JOBS_NX  = 1            # networkx is single-threaded

rng = np.random.default_rng(SEED)

# ── load data ─────────────────────────────────────────────────────────────────
print("Loading ABIDE time series (from nilearn cache)...")
abide   = fetch_abide_pcp(
    data_dir=DATA_DIR, pipeline="cpac",
    band_pass_filtering=True, global_signal_regression=False,
    derivatives=["rois_ho"], quality_checked=True, verbose=0,
)
pheno   = abide.phenotypic
ts_list = abide.rois_ho

dx  = np.array(pheno["DX_GROUP"]).astype(int)
y_  = (dx == 1).astype(int)              # 1=ASD, 0=Control

keep    = [ts.shape[0] >= T_CUTOFF for ts in ts_list]
ts_kept = [ts_list[i][:T_CUTOFF, :] for i in range(len(ts_list)) if keep[i]]
y       = y_[np.array(keep)]

N, P = len(ts_kept), ts_kept[0].shape[1]
n_asd, n_ctrl = y.sum(), (y == 0).sum()
print(f"  N={N}  ASD={n_asd}  Ctrl={n_ctrl}  parcels={P}")

# ── helper: graph features from one FC matrix ─────────────────────────────────
def _threshold_proportional(W, density):
    """Keep top (density * N*(N-1)/2) absolute correlation edges; return binary adj."""
    W = W.copy()
    np.fill_diagonal(W, 0)
    n = W.shape[0]
    vals = np.abs(W[np.triu_indices(n, k=1)])
    thresh = np.quantile(vals, 1.0 - density)
    A = (np.abs(W) >= thresh).astype(float)
    np.fill_diagonal(A, 0)
    return A


def graph_features_one_subject(ts):
    """
    ts: (T, P) time series array
    Returns 1-D feature vector.
    """
    feats = []

    # ── raw FC (Pearson) ──────────────────────────────────────────────────────
    FC = np.corrcoef(ts.T)   # (P, P)
    np.fill_diagonal(FC, 0)
    FC = np.nan_to_num(FC, nan=0.0, posinf=0.0, neginf=0.0)  # flat parcels → 0

    # unthresholded statistics
    fc_upper = FC[np.triu_indices(P, k=1)]
    feats.extend([
        fc_upper.mean(),           # mean connectivity
        fc_upper.std(),            # spread
        (fc_upper > 0).mean(),     # fraction positive
    ])

    # eigenvalue spectrum
    evals = np.linalg.eigvalsh(FC)[::-1]   # descending
    evals_pos = np.maximum(evals, 0)
    total_var = evals_pos.sum() + 1e-12
    feats.extend([
        evals_pos[0] / total_var,   # fraction var in PC1
        evals_pos[:5].sum() / total_var,
    ])

    # ── graph features at each density ───────────────────────────────────────
    for d in DENSITIES:
        A = _threshold_proportional(FC, d)
        W_d = np.abs(FC) * A          # weighted (positive only)

        # ---- strength (weighted degree)
        S = W_d.sum(axis=1)
        feats.extend([S.mean(), S.std(), S.max()])

        # ---- binary degree
        deg = A.sum(axis=1)
        feats.extend([deg.mean(), deg.std(), deg.max()])

        # ---- build networkx graph (binary)
        G = nx.from_numpy_array(A)

        # connected? record largest component fraction
        lcc = max(nx.connected_components(G), key=len)
        feats.append(len(lcc) / P)

        # ---- clustering coefficient (binary)
        cc = nx.clustering(G)
        cc_vals = np.array(list(cc.values()))
        feats.extend([cc_vals.mean(), cc_vals.std()])

        # ---- weighted clustering coefficient
        Gw = nx.from_numpy_array(W_d)
        ccw = nx.clustering(Gw, weight="weight")
        ccw_vals = np.array(list(ccw.values()))
        feats.extend([ccw_vals.mean(), ccw_vals.std()])

        # ---- global efficiency
        ge = nx.global_efficiency(G)
        feats.append(ge)

        # ---- local efficiency (mean over nodes)
        le = nx.local_efficiency(G)
        feats.append(le)

        # ---- betweenness centrality (binary, mean)
        bc = nx.betweenness_centrality(G, normalized=True)
        bc_vals = np.array(list(bc.values()))
        feats.extend([bc_vals.mean(), bc_vals.std(), bc_vals.max()])

        # ---- transitivity (global clustering)
        feats.append(nx.transitivity(G))

        # ---- modularity via greedy modularity communities
        try:
            communities = nx.algorithms.community.greedy_modularity_communities(G)
            Q = nx.algorithms.community.quality.modularity(G, communities)
            n_mod = len(communities)
        except Exception:
            Q, n_mod = 0.0, 1
        feats.extend([Q, float(n_mod)])

    return np.array(feats, dtype=np.float32)


# ── compute features for all subjects ────────────────────────────────────────
print(f"\nComputing graph features ({len(DENSITIES)} densities each subject)...")
t0 = time.time()

all_feats = []
for ts in tqdm(ts_kept, desc="  subjects"):
    all_feats.append(graph_features_one_subject(ts))

X_graph = np.vstack(all_feats)   # (N, n_feats)
print(f"  Done in {time.time()-t0:.1f}s  feature dim: {X_graph.shape[1]}")

# ── classification: repeated stratified k-fold ────────────────────────────────
def evaluate(X, y, label):
    all_auc, all_bac = [], []
    for rep in range(N_REPEATS):
        cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=SEED + rep)
        for tr, te in cv.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr])
            X_te = scaler.transform(X[te])

            n_comp = min(50, X_tr.shape[1], X_tr.shape[0] - 1)
            pca = PCA(n_components=n_comp, random_state=SEED)
            X_tr = pca.fit_transform(X_tr)
            X_te = pca.transform(X_te)

            for name, clf in [
                ("LDA", LinearDiscriminantAnalysis()),
                ("RF",  RandomForestClassifier(n_estimators=200,
                                               random_state=SEED + rep)),
            ]:
                clf.fit(X_tr, y[tr])
                prob = clf.predict_proba(X_te)[:, 1]
                pred = clf.predict(X_te)
                all_auc.append((name, roc_auc_score(y[te], prob)))
                all_bac.append((name, balanced_accuracy_score(y[te], pred)))

    for name in ["LDA", "RF"]:
        aucs = [v for n, v in all_auc if n == name]
        bacs = [v for n, v in all_bac if n == name]
        print(f"  {label:12s} {name:4s}   "
              f"AUROC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
              f"BAcc={np.mean(bacs):.3f}+/-{np.std(bacs):.3f}")


print(f"\nClassification (repeated {N_REPEATS}x{N_FOLDS}-fold CV)...")

# Full graph feature vector
evaluate(X_graph, y, "Graph-full")

# Subsets for ablation
n5 = 5  # number of features per eigenvalue block — rough split
print("\n  --- ablation by feature group ---")

# unthresholded block: first 5 features
evaluate(X_graph[:, :5], y, "FC-raw-stats")

# per-density blocks
feat_per_density = (3 + 3 + 1 + 2 + 2 + 1 + 1 + 3 + 1 + 2)  # = 19
offset = 5
for i, d in enumerate(DENSITIES):
    block = X_graph[:, offset + i * feat_per_density:
                       offset + (i + 1) * feat_per_density]
    evaluate(block, y, f"d={int(d*100):02d}%")

print("\nDone.")
