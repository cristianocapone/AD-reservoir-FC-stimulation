"""
pert_twofreq.py — test two ideas at a FIXED site (eigenmode site 71):
  (1) does driving at the most CC/AD-discriminative mode frequency (f_disc)
      underperform the resonant dominant frequency (f1)?
  (2) does an energy-matched TWO-frequency drive  f1 (+amplitude/drivability)
      + f_disc (+discriminant direction)  beat single-frequency drive?
f_disc = mode with the largest AD power DEFICIT (min CC-vs-AD AUC of modal power),
because an oscillatory drive can only ADD power, so only a deficit is correctable.
Energy-matched: single-freq uses amplitude A; two-freq uses A/sqrt(2) per component.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eig as sla_eig
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; K_LDA=25; MAX_LAG=2
TS_ROOT="./timeseries"; OUT="paper_figures"; SITE=71
AMPS=np.array([0.0,1.0,2.0,4.0,6.0,8.0,10.0])

print("Loading data + reservoir ...")
rng=np.random.default_rng(RNG_SEED); signals,labs,pids=[],[],[]
for sub,lb in [("CN","CC"),("AD","AD")]:
    folder=os.path.join(TS_ROOT,sub); files=sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if lb=="CC": files=list(rng.choice(files,size=min(N_CC_SAMP,len(files)),replace=False))
    for fn in files:
        a=np.load(os.path.join(folder,fn)).T
        if a.shape[1]==N_SITES and a.shape[0]>=139:
            signals.append(a.T); pids.append(fn.split("_ses-")[0]); labs.append(0 if lb=="CC" else 1)
pids=np.array(pids); labs=np.array(labs); upid=np.unique(pids)
psid={p:np.where(pids==p)[0] for p in upid}
plabel=np.array([labs[psid[p][0]] for p in upid]); first={p:psid[p][0] for p in upid}
cc=[upid[i] for i in np.where(plabel==0)[0]]; ad=[upid[i] for i in np.where(plabel==1)[0]]; n_ad=len(ad)
all_sig=np.concatenate([s.T for s in signals],0)
evv,evec=np.linalg.eigh(np.cov((all_sig-all_sig.mean(0)).T)); ev50=evec[:,np.argsort(evv)[::-1]][:,:N_PC_MODEL]
np.random.seed(RNG_SEED)
par=dict(tau_m_f=0.0005,tau_m_s=0.0005,N=N_HIDDEN,T=139,dt=0.005,sigma_input=0.01,shape=(N_HIDDEN,N_SITES,N_SITES,139))
res=RESERVOIRE_SIMPLE(par); res.J*=SR/max(abs(np.linalg.eigvals(res.J)))
patX={}
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

# ── FC-lag classifier ────────────────────────────────────────────────────────
class LDA:
    def fit(s,X,y):
        c0,c1=np.unique(y); X0,X1=X[y==c0],X[y==c1]; m0,m1=X0.mean(0),X1.mean(0)
        Sw=(X0-m0).T@(X0-m0)+(X1-m1).T@(X1-m1)+1e-6*np.eye(X.shape[1])
        w=np.linalg.solve(Sw,m1-m0); w/=np.linalg.norm(w)+1e-12; s.w=w; return s
    def tr(s,X): return X@s.w
def bal(X,y,sd=0):
    r=np.random.default_rng(sd); c0,c1=np.where(y==0)[0],np.where(y==1)[0]; n=min(len(c0),len(c1))
    sel=np.concatenate([r.choice(c0,n,0),r.choice(c1,n,0)]); r.shuffle(sel); return X[sel],y[sel]
def lagc(S,l):
    if l==0: return np.corrcoef(S.T)
    T=S.shape[0]; A=S[:T-l].astype(float); B=S[l:].astype(float)
    A-=A.mean(0); B-=B.mean(0); A/=A.std(0)+1e-12; B/=B.std(0)+1e-12
    return (A.T@B)/(T-l)
def feat(W,X):
    S=(W.T.astype(float)@X.T.astype(float)).T; fs=[]
    for l in range(MAX_LAG+1):
        fc=np.nan_to_num(lagc(S,l)); fs.append(fc[np.triu_indices(N_SITES,1)] if l==0 else fc.flatten())
    return np.concatenate(fs)
fb=np.array([feat(patW[p],patX[p]) for p in tqdm(upid,leave=False,desc="  feat")])
fm=fb.mean(0); fcc=fb-fm; evf,evecf=np.linalg.eigh(fcc@fcc.T); o=np.argsort(evf)[::-1]
evf=np.maximum(evf[o],0); evecf=evecf[:,o]; Gf=evecf*np.sqrt(evf)
Xl,yl=bal(Gf[:,:K_LDA],plabel,RNG_SEED); lda_f=LDA().fit(Xl,yl); Zf=lda_f.tr(Gf[:,:K_LDA])
if Zf[plabel==0].mean()>Zf[plabel==1].mean(): lda_f.w*=-1; Zf=-Zf
thr_f=0.5*(Zf[plabel==0].mean()+Zf[plabel==1].mean()); cc_f=Zf[plabel==0]
def fscore(W,X):
    f=feat(W,X)-fm; g=(f@fcc.T@evecf)/(np.sqrt(evf)+1e-12); return float(lda_f.tr(g[:K_LDA].reshape(1,-1))[0])

# ── frequencies: dominant f1 and discriminative-deficit f_disc ───────────────
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; pos=pos[np.argsort(np.abs(wv[pos]))[::-1]]; modes=pos[:40]
fk=np.abs(np.angle(wv[modes]))/(2*np.pi); VL=vl[:,modes]
P=np.array([np.log(np.mean(np.abs(patX[p].astype(np.float64)@VL.conj())**2,0)+1e-12) for p in upid])
auc=np.array([roc_auc_score(plabel,P[:,k]) for k in range(len(modes))])
f1=float(fk[0])                       # dominant (least-damped) mode
kd=int(np.argmin(auc))                 # biggest AD deficit -> correctable by adding power
f_disc=float(fk[kd])
print(f"  f1(dominant)={f1:.3f} AUC={auc[0]:.3f} ; f_disc(min-AUC)={f_disc:.3f} AUC={auc[kd]:.3f}")

def osc(p,comps):                      # comps = list of (freq, amp)
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy()
        for fr,am in comps: inp[SITE]+=am*np.sin(2*np.pi*fr*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]

fbase=np.array([fscore(patW[p],patX[p]) for p in ad])
nd=len(AMPS); r2=1/np.sqrt(2)
COND={"f1":[], "fdisc":[], "f1+fdisc":[]}
S={k:np.zeros((nd,n_ad)) for k in COND}
print(f"\nFixed site {SITE}; energy-matched sweep ...")
for di,A in enumerate(AMPS):
    comp={"f1":[(f1,A)], "fdisc":[(f_disc,A)], "f1+fdisc":[(f1,A*r2),(f_disc,A*r2)]}
    for k in COND:
        if A==0: S[k][di]=fbase; continue
        for pi,p in enumerate(ad): S[k][di,pi]=fscore(patW[p],osc(p,comp[k]))
    rc={k:(S[k][di]<thr_f).mean()*100 for k in COND}
    print(f"  A={A:4.1f}  f1:{S['f1'][di].mean():+.2f}({rc['f1']:3.0f}%)  "
          f"fdisc:{S['fdisc'][di].mean():+.2f}({rc['fdisc']:3.0f}%)  "
          f"f1+fdisc:{S['f1+fdisc'][di].mean():+.2f}({rc['f1+fdisc']:3.0f}%)",flush=True)

recl={k:(S[k]<thr_f).mean(1)*100 for k in COND}
np.savez("pert_twofreq_data.npz",amps=AMPS,thr_f=thr_f,f1=f1,f_disc=f_disc,site=SITE,
         S_f1=S["f1"],S_fdisc=S["fdisc"],S_both=S["f1+fdisc"],cc_f=cc_f)

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
COL={"f1":"#C62828","fdisc":"#6A1B9A","f1+fdisc":"#00838F"}
LBL={"f1":f"$f_1$ resonant ({f1:.3f})","fdisc":f"$f_{{disc}}$ ({f_disc:.3f})",
     "f1+fdisc":f"$f_1+f_{{disc}}$ (energy-matched)"}
fig=plt.figure(figsize=(11,4.2),facecolor="white")
gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.28,left=0.08,right=0.98,top=0.86,bottom=0.16)
def tag(ax,t): ax.text(-0.14,1.04,t,transform=ax.transAxes,fontsize=13,fontweight="bold")
ax=fig.add_subplot(gs[0,0])
ax.axhline(thr_f,color="gray",ls="-.",lw=1,label="boundary")
ax.axhline(cc_f.mean(),color="#1565C0",ls="--",lw=1.2,label="CC mean")
for k in COND:
    m=S[k].mean(1); e=S[k].std(1)/np.sqrt(n_ad)
    ax.fill_between(AMPS,m-e,m+e,color=COL[k],alpha=0.13); ax.plot(AMPS,m,"-o",ms=4,color=COL[k],lw=2,label=LBL[k])
ax.set_xlabel("stimulation amplitude $A$"); ax.set_ylabel("FC-lag LDA score")
ax.set_title(f"Single vs two-frequency drive (site {SITE})"); ax.legend(frameon=False,fontsize=7.5); tag(ax,"A")
ax=fig.add_subplot(gs[0,1])
for k in COND: ax.plot(AMPS,recl[k],"-o",ms=4,color=COL[k],lw=2,label=LBL[k])
ax.set_xlabel("stimulation amplitude $A$"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification vs amplitude"); ax.set_ylim(-2,105); ax.legend(frameon=False,fontsize=7.5); tag(ax,"B")
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figure_twofreq.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figure_twofreq.{ext}")
plt.close(fig); print("Done.")
