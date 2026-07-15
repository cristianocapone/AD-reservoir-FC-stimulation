"""
pert_eigsites.py
================
Test whether choosing the top-5 sites by coupling to the reservoir's dominant
eigenmode (rather than by the disease-correction norm ||dW||) re-sharpens the
k=5 frequency tuning at the eigenmode frequency f_eig.

For the leading complex eigenmode of J we use the LEFT eigenvector w (w^H J = lam w^H),
which gives the modal excitation coefficient for an input direction b as w^H b.
The eigenmode coupling of input site k is |w^H Jin[:,k]|; the top-5 such sites
are driven (globally, same for all patients) and the FC-lag frequency tuning is
compared with the per-patient ||dW||-selected top-5.

Saves: paper_figures/figure_eigsites.{png,pdf}, pert_eigsites_data.npz
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eig as sla_eig
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; K_LDA=25; MAX_LAG=2
TS_ROOT="./timeseries"; OUT="paper_figures"; AMP=4.0; KTOP=5

def load_labels(path="./timeseries/parcel_labels.txt"):
    lab={}
    for line in open(path):
        p=line.split(None,1)
        if len(p)==2 and p[0].strip().isdigit(): lab[int(p[0])-1]=p[1].strip()
    return lab
def short(n):
    if n is None: return "?"
    n=n.replace("Left ","L ").replace("Right ","R ")
    if n.startswith("7Networks_"):
        q=n.replace("7Networks_","").split("_"); return q[0]+" "+"".join(q[1:])
    return n
labels=load_labels()

print("Loading data ...")
rng=np.random.default_rng(RNG_SEED)
signals,labs,pids=[],[],[]
for sub,lb in [("CN","CC"),("AD","AD")]:
    folder=os.path.join(TS_ROOT,sub)
    files=sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if lb=="CC": files=list(rng.choice(files,size=min(N_CC_SAMP,len(files)),replace=False))
    for fn in files:
        a=np.load(os.path.join(folder,fn)).T
        if a.shape[1]==N_SITES and a.shape[0]>=139:
            signals.append(a.T); pids.append(fn.split("_ses-")[0])
            labs.append(0 if lb=="CC" else 1)
pids=np.array(pids); labs=np.array(labs)
upid=np.unique(pids); psid={p:np.where(pids==p)[0] for p in upid}
plabel=np.array([labs[psid[p][0]] for p in upid])
cc=[upid[i] for i in np.where(plabel==0)[0]]; ad=[upid[i] for i in np.where(plabel==1)[0]]
n_ad=len(ad); print(f"  {len(upid)} patients ({len(cc)} CC, {n_ad} AD)")

all_sig=np.concatenate([s.T for s in signals],0)
ev,evec=np.linalg.eigh(np.cov((all_sig-all_sig.mean(0)).T))
ev50=evec[:,np.argsort(ev)[::-1]][:,:N_PC_MODEL]

print("Reservoir (seeded) ...")
np.random.seed(RNG_SEED)
par=dict(tau_m_f=0.0005,tau_m_s=0.0005,N=N_HIDDEN,T=139,dt=0.005,
         sigma_input=0.01,shape=(N_HIDDEN,N_SITES,N_SITES,139))
res=RESERVOIRE_SIMPLE(par); res.J*=SR/max(abs(np.linalg.eigvals(res.J)))

first={p:psid[p][0] for p in upid}; patX={}
for p in tqdm(upid,desc="  TF"):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1): res.step_rate(ff*tgt[:,t],sigma_dyn=0.); X.append(res.X.copy())
    patX[p]=np.array(X)[TIMES_SKIP:]
rw=np.random.default_rng(RNG_SEED+1); patW={}
for p in upid:
    s=signals[first[p]]; tgt=(s.T@ev50@ev50.T).T
    Xc=patX[p]; Yc=tgt[:,TIMES_SKIP:TIMES_SKIP+Xc.shape[0]].T
    patW[p]=np.linalg.pinv(Xc+rw.normal(0,SIGMA,Xc.shape))@Yc
Wcc=np.mean([patW[p] for p in cc],0)

# ── dominant eigenmode: left eigenvector w (w^H J = lam w^H) ──────────────────
print("Eigendecomposition of J ...")
w_eig, vl, vr = sla_eig(res.J, left=True, right=True)
cmask=np.abs(w_eig.imag)>1e-8
top=np.where(cmask)[0][np.argsort(np.abs(w_eig[cmask]))[::-1][0]]
lam=w_eig[top]; f_eig=float(np.abs(np.angle(lam))/(2*np.pi))
wL=vl[:,top]                                   # left eigenvector (2000,)
coupling=np.abs(wL.conj() @ res.Jin)           # (121,) modal excitation per site
eig_sites=np.argsort(coupling)[::-1][:KTOP]
print(f"  f_eig={f_eig:.4f}")
print("  eigenmode top-5 sites:", [f"{s}:{short(labels.get(s))}" for s in eig_sites])
# dW top-5 (per patient) most common, for reference
import collections
cnt=collections.Counter()
for p in ad: cnt.update(np.argsort(np.linalg.norm(Wcc-patW[p],axis=0))[::-1][:KTOP].tolist())
dW_top=[s for s,_ in cnt.most_common(KTOP)]
print("  ||dW|| top-5 (most frequent):", [f"{s}:{short(labels.get(s))}" for s in dW_top])

# FFT peak
Te=patX[ad[0]].shape[0]; psd=np.zeros(Te//2+1)
for p in ad:
    Xr=patX[p].astype(float); psd+=(np.abs(np.fft.rfft(Xr-Xr.mean(0),axis=0))**2).mean(1)
f_fft=float(np.fft.rfftfreq(Te)[1+int(np.argmax(psd[1:]/n_ad))])

FREQS=np.unique(np.round(np.concatenate([np.linspace(0.03,0.45,11),[f_eig,f_fft]]),4))

# ── FC-lag LDA ────────────────────────────────────────────────────────────────
def lagc(S,l):
    if l==0: return np.corrcoef(S.T)
    T=S.shape[0]; A=S[:T-l].astype(float); B=S[l:].astype(float)
    A-=A.mean(0); B-=B.mean(0); A/=A.std(0)+1e-12; B/=B.std(0)+1e-12
    return (A.T@B)/(T-l)
def feat(W,X):
    S=(W.T.astype(float)@X.T.astype(float)).T; fs=[]
    for l in range(MAX_LAG+1):
        fc=np.nan_to_num(lagc(S,l))
        fs.append(fc[np.triu_indices(N_SITES,1)] if l==0 else fc.flatten())
    return np.concatenate(fs)
class LDA:
    def fit(s,X,y):
        c0,c1=np.unique(y); X0,X1=X[y==c0],X[y==c1]; m0,m1=X0.mean(0),X1.mean(0)
        Sw=(X0-m0).T@(X0-m0)+(X1-m1).T@(X1-m1)+1e-6*np.eye(X.shape[1])
        w=np.linalg.solve(Sw,m1-m0); w/=np.linalg.norm(w)+1e-12; s.w=w; s.t=0.5*(m0@w+m1@w); return s
    def tr(s,X): return X@s.w
def bal(X,y,sd=0):
    r=np.random.default_rng(sd); c0,c1=np.where(y==0)[0],np.where(y==1)[0]; n=min(len(c0),len(c1))
    sel=np.concatenate([r.choice(c0,n,0),r.choice(c1,n,0)]); r.shuffle(sel); return X[sel],y[sel]
print("FC-lag LDA ...")
fb=np.array([feat(patW[p],patX[p]) for p in tqdm(upid,leave=False)])
fm=fb.mean(0); fcc=fb-fm; evf,evecf=np.linalg.eigh(fcc@fcc.T); o=np.argsort(evf)[::-1]
evf=np.maximum(evf[o],0); evecf=evecf[:,o]; G=evecf*np.sqrt(evf)
Xl,yl=bal(G[:,:K_LDA],plabel,RNG_SEED); lda=LDA().fit(Xl,yl); Z=lda.tr(G[:,:K_LDA])
if Z[plabel==0].mean()>Z[plabel==1].mean(): lda.w*=-1; lda.t*=-1; Z=-Z
thr=0.5*(Z[plabel==0].mean()+Z[plabel==1].mean())
def flscore(W,X):
    f=feat(W,X)-fm; g=(f@fcc.T@evecf)/(np.sqrt(evf)+1e-12); return float(lda.tr(g[:K_LDA].reshape(1,-1))[0])
def osc(pid,sites,fr,amp):
    s=signals[first[pid]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy(); inp[sites]+=amp*np.sin(2*np.pi*fr*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]

print(f"Frequency sweep (A={AMP}) for eigenmode-top5 vs ||dW||-top5 ...")
FL_eig=np.zeros((len(FREQS),n_ad)); FL_dw=np.zeros((len(FREQS),n_ad))
for fi,fr in enumerate(FREQS):
    for pi,p in enumerate(ad):
        dw_sites=np.argsort(np.linalg.norm(Wcc-patW[p],axis=0))[::-1][:KTOP]
        FL_eig[fi,pi]=flscore(patW[p],osc(p,eig_sites,fr,AMP))
        FL_dw[fi,pi]=flscore(patW[p],osc(p,dw_sites,fr,AMP))
    print(f"  f={fr:.4f}  eig:FL={FL_eig[fi].mean():+.3f}  dW:FL={FL_dw[fi].mean():+.3f}",flush=True)

fie=int(np.argmin(np.abs(FREQS-f_eig)))
print(f"\nmin-FL freq:  eigenmode-sel -> {FREQS[np.argmin(FL_eig.mean(1))]:.4f}"
      f"   ||dW||-sel -> {FREQS[np.argmin(FL_dw.mean(1))]:.4f}   (f_eig={f_eig:.4f})")

np.savez("pert_eigsites_data.npz",freqs=FREQS,FL_eig=FL_eig,FL_dw=FL_dw,
         thr=thr,f_eig=f_eig,f_fft=f_fft,eig_sites=eig_sites,coupling=coupling)

def sem(a): return a.std(1)/np.sqrt(a.shape[1])
fig,ax=plt.subplots(figsize=(7,5),facecolor="white")
ax.fill_between(FREQS,FL_dw.mean(1)-sem(FL_dw),FL_dw.mean(1)+sem(FL_dw),color="#2E7D32",alpha=0.15)
ax.plot(FREQS,FL_dw.mean(1),"-o",ms=4,color="#2E7D32",lw=2,label="top-5 by $\\|\\Delta W\\|$ (disease)")
ax.fill_between(FREQS,FL_eig.mean(1)-sem(FL_eig),FL_eig.mean(1)+sem(FL_eig),color="#1565C0",alpha=0.15)
ax.plot(FREQS,FL_eig.mean(1),"-s",ms=4,color="#1565C0",lw=2,label="top-5 by eigenmode coupling")
ax.axhline(thr,color="gray",ls="-.",lw=1,label="boundary")
ax.axvline(f_eig,color="#C62828",ls="--",lw=1.5,label=f"$f_\\mathrm{{eig}}$={f_eig:.3f}")
ax.set_xlabel("stimulation frequency (cycles/step)"); ax.set_ylabel("FC-lag LDA score (mean $\\pm$ SEM)")
ax.set_title(f"Frequency tuning by site-selection criterion (k=5, A={AMP:.0f})")
ax.legend(frameon=False,fontsize=8)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT,f"figure_eigsites.{ext}"),dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figure_eigsites.{ext}")
plt.close(fig); print("Done.")
