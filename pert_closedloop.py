"""
pert_closedloop.py — closed-loop (biomarker-titrated) stimulation that adapts
AMPLITUDE and SITE per patient to cross the FC-lag boundary at minimal distance
from the unstimulated simulation. Drive at the resonance f1.
Conditions:
  (1) open-loop      : personalised site (argmax red_full), fixed amplitude A_FIX
  (2) CL amplitude   : personalised site, minimal amplitude that crosses the boundary
  (3) CL amp + site  : among top-K candidate sites, the (site,amplitude) crossing at
                       the smallest distance from the unstimulated FC
Distance = 1 - corr(stimulated free-run FC, unstimulated free-run FC).
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.linalg import eig as sla_eig
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"."); from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; K_LDA=25; MAX_LAG=2; DRIVE_STEPS=5
TS_ROOT="./timeseries"; OUT="paper_figures"
AMPS=np.array([0.25,0.5,0.75,1.0,1.5,2.0,3.0,4.0,6.0])   # titration ladder
A_FIX=6.0; KCAND=15
iu=np.triu_indices(N_SITES,1)

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

class LDA:
    def fit(s,X,y):
        c0,c1=np.unique(y); X0,X1=X[y==c0],X[y==c1]; m0,m1=X0.mean(0),X1.mean(0)
        Sw=(X0-m0).T@(X0-m0)+(X1-m1).T@(X1-m1)+1e-6*np.eye(X.shape[1])
        w=np.linalg.solve(Sw,m1-m0); w/=np.linalg.norm(w)+1e-12; s.w=w; return s
    def tr(s,X): return X@s.w
def balm(X,y,sd=0):
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
Xl,yl=balm(Gf[:,:K_LDA],plabel,RNG_SEED); lda_f=LDA().fit(Xl,yl); Zf=lda_f.tr(Gf[:,:K_LDA])
if Zf[plabel==0].mean()>Zf[plabel==1].mean(): lda_f.w*=-1; Zf=-Zf
thr_f=0.5*(Zf[plabel==0].mean()+Zf[plabel==1].mean())
def fscore(W,X):
    f=feat(W,X)-fm; g=(f@fcc.T@evecf)/(np.sqrt(evf)+1e-12); return float(lda_f.tr(g[:K_LDA].reshape(1,-1))[0])
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; f1=float(abs(np.angle(wv[pos[np.argsort(np.abs(wv[pos]))[::-1][0]]]))/(2*np.pi))
def osc(p,site,amp):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy(); inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]
def freerun_FC(Wout,p,site=None,amp=0.0):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); res.Jout=Wout.T.copy(); Y=[]
    for t in range(T-1):
        fbk=tgt[:,t] if t<=DRIVE_STEPS else res.y
        inp=ff*np.asarray(fbk,dtype=float).copy()
        if site is not None and amp!=0.: inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); Y.append(np.asarray(res.y,dtype=float).copy())
    return np.nan_to_num(np.corrcoef(np.array(Y).T[:,TIMES_SKIP-1:]))
fc_ref={p:freerun_FC(patW[p],p) for p in ad}
def dist(p,site,amp): return 1.0-float(np.corrcoef(freerun_FC(patW[p],p,site,amp)[iu],fc_ref[p][iu])[0,1])

red_full=np.load("pert_compare3_data.npz",allow_pickle=True)["red_full"]
pers_site={p:int(np.argmax(red_full[:,pi])) for pi,p in enumerate(ad)}
cand={p:list(np.argsort(red_full[:,pi])[::-1][:KCAND]) for pi,p in enumerate(ad)}

INF=np.inf
res_ol=dict(rec=np.zeros(n_ad),dist=np.full(n_ad,np.nan),amp=np.full(n_ad,A_FIX))
res_ca=dict(rec=np.zeros(n_ad),dist=np.full(n_ad,np.nan),amp=np.full(n_ad,np.nan))
res_cs=dict(rec=np.zeros(n_ad),dist=np.full(n_ad,np.nan),amp=np.full(n_ad,np.nan),site=np.full(n_ad,-1))
print(f"\nClosed-loop (f1={f1:.3f}, A_fix={A_FIX}, {KCAND} candidate sites) ...")
def min_cross_amp(p,site):           # smallest amp on ladder with score<thr; np.inf if none
    for A in AMPS:
        if fscore(patW[p],osc(p,site,A))<thr_f: return A
    return INF
for pi,p in enumerate(tqdm(ad,desc="  patients")):
    s0=pers_site[p]
    # (1) open-loop: pers site, fixed amplitude
    res_ol["rec"][pi]=fscore(patW[p],osc(p,s0,A_FIX))<thr_f
    res_ol["dist"][pi]=dist(p,s0,A_FIX)
    # (2) CL amplitude: pers site, min crossing amplitude
    a0=min_cross_amp(p,s0)
    if np.isfinite(a0):
        res_ca["rec"][pi]=1; res_ca["amp"][pi]=a0; res_ca["dist"][pi]=dist(p,s0,a0)
    # (3) CL amp+site: candidate site crossing at smallest distance
    best=(INF,None,None)
    for k in cand[p]:
        ak=min_cross_amp(p,k)
        if np.isfinite(ak):
            dk=dist(p,k,ak)
            if dk<best[0]: best=(dk,k,ak)
    if best[1] is not None:
        res_cs["rec"][pi]=1; res_cs["dist"][pi]=best[0]; res_cs["site"][pi]=best[1]; res_cs["amp"][pi]=best[2]

def summ(r,name):
    m=r["rec"]>0; print(f"  {name:14s} reclass {m.mean()*100:5.1f}%  mean dist {np.nanmean(r['dist'][m]):.3f}"
                         f"  mean amp {np.nanmean(r['amp'][m]):.2f}")
print(); summ(res_ol,"open-loop"); summ(res_ca,"CL amplitude"); summ(res_cs,"CL amp+site")
np.savez("pert_closedloop_data.npz",amps=AMPS,A_FIX=A_FIX,f1=f1,thr_f=thr_f,
         ol_rec=res_ol["rec"],ol_dist=res_ol["dist"],
         ca_rec=res_ca["rec"],ca_dist=res_ca["dist"],ca_amp=res_ca["amp"],
         cs_rec=res_cs["rec"],cs_dist=res_cs["dist"],cs_amp=res_cs["amp"],cs_site=res_cs["site"])

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
COL=["#455A64","#1565C0","#2E7D32"]; NAMES=["open-loop\n(fixed $A$)","CL amplitude","CL amp+site"]
R=[res_ol,res_ca,res_cs]
fig=plt.figure(figsize=(13.5,4.3),facecolor="white")
gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.32,left=0.06,right=0.985,top=0.87,bottom=0.18)
def tag(ax,s): ax.text(-0.16,1.04,s,transform=ax.transAxes,fontsize=13,fontweight="bold")

ax=fig.add_subplot(gs[0,0])
ax.bar(range(3),[r["rec"].mean()*100 for r in R],color=COL,alpha=0.85)
ax.set_xticks(range(3)); ax.set_xticklabels(NAMES); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_ylim(0,105); ax.set_title("Reclassification rate"); tag(ax,"A")

ax=fig.add_subplot(gs[0,1])
data=[r["dist"][r["rec"]>0] for r in R]; data=[d[~np.isnan(d)] for d in data]
vp=ax.violinplot(data,positions=range(3),showmedians=True,widths=0.7)
for b,c in zip(vp["bodies"],COL): b.set_facecolor(c); b.set_alpha(0.5)
for kk in ["cmedians","cbars","cmins","cmaxes"]: vp[kk].set_color("k")
for xi,d in enumerate(data): ax.scatter(xi+np.random.uniform(-0.07,0.07,len(d)),d,s=8,c=COL[xi],alpha=0.6)
ax.set_xticks(range(3)); ax.set_xticklabels(NAMES); ax.set_ylabel("distance from original FC ($1-$corr)")
ax.set_title("Distance among reclassified patients"); tag(ax,"B")

ax=fig.add_subplot(gs[0,2])
m=(res_ol["rec"]>0)&(res_cs["rec"]>0)
ax.scatter(res_ol["dist"][m],res_cs["dist"][m],s=22,color="#2E7D32",alpha=0.7,edgecolors="none")
lim=[0,max(np.nanmax(res_ol["dist"]),np.nanmax(res_cs["dist"]))*1.05]
ax.plot(lim,lim,"k--",lw=1,alpha=0.6); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("open-loop distance"); ax.set_ylabel("CL amp+site distance")
ax.set_title("Per-patient: closed-loop reduces distance"); tag(ax,"C")

fig.suptitle("Closed-loop (biomarker-titrated) stimulation adapting amplitude and site: "
             "reclassify at lower distance from the unstimulated simulation",
             fontsize=11,fontweight="bold",y=0.99)
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figureS10_closedloop.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS10_closedloop.{ext}")
plt.close(fig); print("Done.")
