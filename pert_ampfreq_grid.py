"""
pert_ampfreq_grid.py — per-patient 2D (amplitude x frequency) stimulation scan.
At each patient's personalised LDA-resonant site, drive A*sin(2*pi*f*t) and record:
  - reclassification (FC-lag classifier score < boundary; TF reconstruction = paper's space)
  - distance to the UNSTIMULATED simulation: 1 - corr(free-run FC_stim, free-run FC_unstim)
Outputs one example patient (2 heatmaps) + population averages (reclass rate, mean distance),
and the optimum that maximises reclassification while minimising distance.
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
AMPS=np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0])
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

# FC-lag classifier (paper's)
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
thr_f=0.5*(Zf[plabel==0].mean()+Zf[plabel==1].mean())
def fscore(W,X):
    f=feat(W,X)-fm; g=(f@fcc.T@evecf)/(np.sqrt(evf)+1e-12); return float(lda_f.tr(g[:K_LDA].reshape(1,-1))[0])
def osc(p,site,amp,fr):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy(); inp[site]+=amp*np.sin(2*np.pi*fr*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]
def freerun_FC(Wout,p,site=None,amp=0.0,fr=0.0):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); res.Jout=Wout.T.copy(); Y=[]
    for t in range(T-1):
        fbk=tgt[:,t] if t<=DRIVE_STEPS else res.y
        inp=ff*np.asarray(fbk,dtype=float).copy()
        if site is not None and amp!=0.: inp[site]+=amp*np.sin(2*np.pi*fr*t)
        res.step_rate(inp,sigma_dyn=0.); Y.append(np.asarray(res.y,dtype=float).copy())
    return np.nan_to_num(np.corrcoef(np.array(Y).T[:,TIMES_SKIP-1:]))
def fcmatch(fc,ref): return float(np.corrcoef(fc[iu],ref[iu])[0,1])

# eigenmode f1 and personalised site
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; f1=float(abs(np.angle(wv[pos[np.argsort(np.abs(wv[pos]))[::-1][0]]]))/(2*np.pi))
red_full=np.load("pert_compare3_data.npz",allow_pickle=True)["red_full"]
pers_site={p:int(np.argmax(red_full[:,pi])) for pi,p in enumerate(ad)}
fc_ref={p:freerun_FC(patW[p],p) for p in ad}
fbase={p:fscore(patW[p],patX[p]) for p in ad}

FREQS=np.unique(np.round(np.concatenate([[f1],np.linspace(0.02,0.45,9)]),4))
na,nf=len(AMPS),len(FREQS)
print(f"\nGrid {na} amp x {nf} freq x {n_ad} AD at personalised sites; f1={f1:.3f}")
SCORE=np.zeros((na,nf,n_ad)); DIST=np.zeros((na,nf,n_ad))
for ai,A in enumerate(tqdm(AMPS,desc="  amp")):
    for fi,fr in enumerate(FREQS):
        for pi,p in enumerate(ad):
            if A==0:
                SCORE[ai,fi,pi]=fbase[p]; DIST[ai,fi,pi]=0.0; continue
            SCORE[ai,fi,pi]=fscore(patW[p],osc(p,pers_site[p],A,fr))
            DIST[ai,fi,pi]=1.0-fcmatch(freerun_FC(patW[p],p,pers_site[p],A,fr),fc_ref[p])
RECL=(SCORE<thr_f)
recl_rate=RECL.mean(2)*100; dist_mean=DIST.mean(2)

# example patient: representative = median baseline score among AD
bvals=np.array([fbase[p] for p in ad]); ex=int(np.argsort(bvals)[len(bvals)//2])
exname=ad[ex]
# optimum on average: maximise reclassification while minimising distance
obj=recl_rate/100.0 - dist_mean
io_=np.unravel_index(np.argmax(obj),obj.shape); aopt,fopt=AMPS[io_[0]],FREQS[io_[1]]
print(f"\nExample patient idx {ex} (baseline score {bvals[ex]:+.2f})")
print(f"Optimum (max reclass - distance): A={aopt:.2f}, f={fopt:.3f}  "
      f"-> reclass {recl_rate[io_]:.0f}%, distance {dist_mean[io_]:.3f}  (f1={f1:.3f})")
# min-distance to reach >=80% reclassification
hi=recl_rate>=80
if hi.any():
    di=np.where(hi,dist_mean,np.inf); j=np.unravel_index(np.argmin(di),di.shape)
    print(f"Cheapest route to >=80% reclass: A={AMPS[j[0]]:.2f}, f={FREQS[j[1]]:.3f}, distance {dist_mean[j]:.3f}")
np.savez("pert_ampfreq_data.npz",amps=AMPS,freqs=FREQS,f1=f1,thr_f=thr_f,
         SCORE=SCORE,DIST=DIST,recl_rate=recl_rate,dist_mean=dist_mean,ex=ex,aopt=aopt,fopt=fopt)

# ── figure: example (2 heatmaps) + averages (2 heatmaps) ─────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":7.5,"ytick.labelsize":8,"figure.dpi":300,"savefig.dpi":300})
fig=plt.figure(figsize=(11.5,8.6),facecolor="white")
gs=gridspec.GridSpec(2,2,figure=fig,wspace=0.30,hspace=0.40,left=0.09,right=0.97,top=0.91,bottom=0.09)
def tag(ax,s): ax.text(-0.16,1.06,s,transform=ax.transAxes,fontsize=13,fontweight="bold")
def hmap(ax,M,cmap,title,cbar,vmin=None,vmax=None,center=None):
    if center is not None:
        import matplotlib.colors as mc
        norm=mc.TwoSlopeNorm(vcenter=center,vmin=M.min(),vmax=M.max())
        im=ax.pcolormesh(FREQS,AMPS,M,cmap=cmap,norm=norm,shading="nearest")
    else:
        im=ax.pcolormesh(FREQS,AMPS,M,cmap=cmap,vmin=vmin,vmax=vmax,shading="nearest")
    ax.axvline(f1,color="k",ls=":",lw=1.2)
    ax.set_xlabel("frequency (cycles/step)"); ax.set_ylabel("amplitude $A$"); ax.set_title(title)
    fig.colorbar(im,ax=ax,label=cbar,fraction=0.046,pad=0.03)
    return im

ax=fig.add_subplot(gs[0,0])
hmap(ax,SCORE[:,:,ex],"RdBu","Example patient: FC-lag score","score (low=CC)",center=thr_f)
cs=ax.contour(FREQS,AMPS,SCORE[:,:,ex],levels=[thr_f],colors="k",linewidths=1.6)
ax.clabel(cs,fmt="boundary",fontsize=6); tag(ax,"A")

ax=fig.add_subplot(gs[0,1])
hmap(ax,DIST[:,:,ex],"viridis","Example patient: distance to unstim. FC","$1-$corr",vmin=0)
tag(ax,"B")

ax=fig.add_subplot(gs[1,0])
hmap(ax,recl_rate,"magma","Average: reclassification rate","% reclassified",vmin=0,vmax=100)
ax.contour(FREQS,AMPS,recl_rate,levels=[50,80],colors="cyan",linewidths=1.2,linestyles=["--","-"])
ax.plot(fopt,aopt,"*",ms=16,color="lime",mec="k",mew=0.7); tag(ax,"C")

ax=fig.add_subplot(gs[1,1])
hmap(ax,dist_mean,"viridis","Average: distance to unstim. FC","$1-$corr",vmin=0)
ax.plot(fopt,aopt,"*",ms=16,color="red",mec="k",mew=0.7); tag(ax,"D")

fig.suptitle("Per-patient amplitude$\\times$frequency stimulation landscape: "
             "maximise reclassification, minimise distance to the unstimulated simulation",
             fontsize=11,fontweight="bold",y=0.975)
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figureS9_ampfreq.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS9_ampfreq.{ext}")
plt.close(fig); print("Done.")
