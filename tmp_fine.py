import numpy as np, os
from collections import defaultdict
from sklearn.metrics import roc_auc_score

RNG_SEED = 42; N_SITES = 121; MAX_LAG = 5; TS_ROOT = "./timeseries"

def lagged_corrcoef(S, lag):
    if lag == 0:
        return np.corrcoef(S.T)
    T = S.shape[0]
    A = S[:T-lag,:].astype(np.float64); B = S[lag:,:].astype(np.float64)
    A -= A.mean(0); B -= B.mean(0)
    A /= A.std(0)+1e-12; B /= B.std(0)+1e-12
    return (A.T @ B) / (T - lag)

def session_features(S, max_lag):
    feats = []
    for lag in range(max_lag+1):
        fc = np.nan_to_num(lagged_corrcoef(S, lag))
        if lag == 0:
            feats.append(fc[np.triu_indices(N_SITES, k=1)])
        else:
            feats.append(fc.flatten())
    return np.concatenate(feats)

pid_feats = defaultdict(list); pid_labels = {}
for grp, lbl in [("CN",0),("AD",1)]:
    for f in sorted(os.listdir(os.path.join(TS_ROOT,grp))):
        if not f.endswith(".npy"): continue
        arr = np.load(os.path.join(TS_ROOT,grp,f)).T
        if arr.shape[1]!=N_SITES or arr.shape[0]<10: continue
        pid = f.split("_ses-")[0]
        pid_feats[pid].append(session_features(arr, MAX_LAG))
        pid_labels[pid] = lbl

pids = sorted(pid_feats.keys())
y = np.array([pid_labels[p] for p in pids])
X = np.array([np.mean(pid_feats[p],0) for p in pids], dtype=np.float64)

def make_G(X, max_lag_use):
    lag0_d = N_SITES*(N_SITES-1)//2
    lagk_d = N_SITES**2
    cols = lag0_d + max_lag_use*lagk_d if max_lag_use > 0 else lag0_d
    Xc = X[:,:cols]; Xc = Xc - Xc.mean(0)
    C = Xc @ Xc.T
    ev, evec = np.linalg.eigh(C)
    o = np.argsort(ev)[::-1]; ev=np.maximum(ev[o],0); evec=evec[:,o]
    return evec * np.sqrt(ev)

class LDA:
    def fit(self,X,y):
        c0,c1=np.unique(y); X0,X1=X[y==c0],X[y==c1]
        mu0,mu1=X0.mean(0),X1.mean(0)
        Sw=(X0-mu0).T@(X0-mu0)+(X1-mu1).T@(X1-mu1)+1e-6*np.eye(X.shape[1])
        w=np.linalg.solve(Sw,mu1-mu0); w/=np.linalg.norm(w)+1e-12
        self.w_=w; return self
    def transform(self,X): return X@self.w_

def balance(X,y,seed=0):
    rng=np.random.default_rng(seed)
    c0i=np.where(y==0)[0]; c1i=np.where(y==1)[0]; n=min(len(c0i),len(c1i))
    sel=np.concatenate([rng.choice(c0i,n,replace=False),rng.choice(c1i,n,replace=False)])
    rng.shuffle(sel); return X[sel],y[sel]

def lopo(G, y, k):
    n=len(y); Gk=G[:,:k]; preds=np.full(n,np.nan); scores=np.full(n,np.nan)
    for i in range(n):
        mask=np.arange(n)!=i
        G_tr=Gk[mask]; y_tr=y[mask]; G_te=Gk[i]
        Xb,yb=balance(G_tr,y_tr,seed=RNG_SEED)
        try: lda=LDA().fit(Xb,yb)
        except: continue
        z_tr=lda.transform(G_tr)
        if z_tr[y_tr==0].mean()>z_tr[y_tr==1].mean():
            lda.w_*=-1; z_tr=lda.transform(G_tr)
        thr=0.5*(z_tr[y_tr==0].mean()+z_tr[y_tr==1].mean())
        z_te=lda.transform(G_te.reshape(1,-1))[0]
        preds[i]=float(z_te>=thr); scores[i]=z_te-thr
    v=np.isfinite(preds)
    if v.sum()<4: return np.nan,np.nan
    sens=np.mean(preds[v&(y==1)]==1); spec=np.mean(preds[v&(y==0)]==0)
    try: auc=roc_auc_score(y[v],scores[v])
    except: auc=np.nan
    return 0.5*(sens+spec), auc

# Fine K sweep for lags 0-2
print("Fine K sweep, lags 0-2:")
G02 = make_G(X, 2)
Ks_fine = list(range(15, 36))
best_ba = 0; best_k = 0
for k in Ks_fine:
    ba, au = lopo(G02, y, k)
    marker = " <--" if ba > best_ba else ""
    if ba > best_ba: best_ba=ba; best_k=k
    print(f"  K={k:3d}  BAL={ba:.4f}  AUC={au:.4f}{marker}")

print(f"\nBest: K={best_k}  BAL={best_ba:.4f}")

# Also check lags 0-2 vs 0-5 at best K
print("\nLag comparison at K="+str(best_k)+":")
for ml in [0,1,2,3,4,5]:
    G = make_G(X, ml)
    ba, au = lopo(G, y, best_k)
    print(f"  lags 0-{ml}  BAL={ba:.4f}  AUC={au:.4f}")
