"""
pert_archetype_grid.py
======================
Supplementary: training and test reconstruction error of the read-out archetype
model, on a 2-D grid of (K_PC, M)  [cf. Fig1DEF_MC_data.ipynb cells 23-25].

Per patient the fitted read-out W_i (N_sites x N_hidden) is projected onto the
top-K_PC right singular vectors of its reservoir states X (W_proj), then the
projected read-outs are compressed across patients to M SVD "archetypes".  The
error is measured in signal space, NMSE = <(Y - X W_g^T)^2> / <Y^2>, where W_g is
the archetype reconstruction.  For each (K_PC, M) the archetype basis is built on
a training split and evaluated on a held-out test split (repeated, averaged).

Fixed regularisation noise sigma = 0.025.
Saves: paper_figures/figureS3_archetype.{png,pdf}, pert_archetype_grid_data.npz
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.025; SR=0.95; TS_ROOT="./timeseries"; OUT="paper_figures"
KPC_GRID=np.array([10,25,50,75,100,150,200])
M_GRID  =np.array([1,2,3,5,8,12,20,30,40,50])
N_REP=4; TEST_FRAC=0.30

print("Loading data ...")
rng=np.random.default_rng(RNG_SEED)
signals,pids=[],[]
for sub in ["CN","AD"]:
    folder=os.path.join(TS_ROOT,sub)
    files=sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    if sub=="CN": files=list(rng.choice(files,size=min(N_CC_SAMP,len(files)),replace=False))
    for fn in files:
        a=np.load(os.path.join(folder,fn)).T
        if a.shape[1]==N_SITES and a.shape[0]>=139:
            signals.append(a.T); pids.append(fn.split("_ses-")[0])
pids=np.array(pids); upid=np.unique(pids)
psid={p:np.where(pids==p)[0] for p in upid}; Npat=len(upid)
print(f"  {Npat} patients")

all_sig=np.concatenate([s.T for s in signals],0)
ev,evec=np.linalg.eigh(np.cov((all_sig-all_sig.mean(0)).T))
ev50=evec[:,np.argsort(ev)[::-1]][:,:N_PC_MODEL]

print("Reservoir + W fit (sigma=0.025) ...")
np.random.seed(RNG_SEED)
par=dict(tau_m_f=0.0005,tau_m_s=0.0005,N=N_HIDDEN,T=139,dt=0.005,
         sigma_input=0.01,shape=(N_HIDDEN,N_SITES,N_SITES,139))
res=RESERVOIRE_SIMPLE(par); res.J*=SR/max(abs(np.linalg.eigvals(res.J)))
first={p:psid[p][0] for p in upid}
rw=np.random.default_rng(RNG_SEED+1)
Xc={}; Yc={}; Wi={}; Vt={}
KPCmax=int(KPC_GRID.max())
for p in tqdm(upid,desc="  fit"):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1): res.step_rate(ff*tgt[:,t],sigma_dyn=0.); X.append(res.X.copy())
    Xf=np.array(X)[TIMES_SKIP:].astype(np.float64)
    Yf=tgt[:,TIMES_SKIP:TIMES_SKIP+Xf.shape[0]].T.astype(np.float64)
    W=np.linalg.pinv(Xf+rw.normal(0,SIGMA,Xf.shape))@Yf      # (N_hidden, N_sites)
    Xc[p]=Xf; Yc[p]=Yf; Wi[p]=W.T                            # W_i = (N_sites, N_hidden)
    _,sx,V=np.linalg.svd(Xf,full_matrices=False)
    Vt[p]=V[:min(KPCmax,int((sx>1e-8).sum()))]
Yvar={p:float((Yc[p]**2).mean()) for p in upid}

def wproj_flat(p,K):
    Vk=Vt[p][:K]; return (Wi[p]@Vk.T@Vk).ravel()            # (N_sites*N_hidden,)

# ── sweep K_PC x M with repeated train/test split ─────────────────────────────
nK=len(KPC_GRID); nM=len(M_GRID)
tr=np.zeros((nK,nM)); te=np.zeros((nK,nM))
rsp=np.random.default_rng(0)
print(f"\nSweep {nK} K_PC x {nM} M x {N_REP} splits ...")
for ki,K in enumerate(KPC_GRID):
    Wp={p:wproj_flat(p,int(K)) for p in upid}               # flattened projected W
    tr_acc=np.zeros(nM); te_acc=np.zeros(nM)
    for rep in range(N_REP):
        idx=rsp.permutation(Npat); ntest=int(TEST_FRAC*Npat)
        test=[upid[i] for i in idx[:ntest]]; train=[upid[i] for i in idx[ntest:]]
        Wtr=np.stack([Wp[p] for p in train])
        Wmean=Wtr.mean(0,keepdims=True); Wc=Wtr-Wmean
        _,_,Vs=np.linalg.svd(Wc,full_matrices=False)         # archetypes Vs (rows)
        Mmax=min(M_GRID.max(),Vs.shape[0])
        arch=Vs[:Mmax]                                       # (Mmax, D)
        for split,grp,acc in [("tr",train,tr_acc),("te",test,te_acc)]:
            nm=np.zeros(nM)
            for p in grp:
                g=(Wp[p]-Wmean.ravel())@arch.T               # (Mmax,)
                Yhat=Xc[p]@Wmean.reshape(N_SITES,N_HIDDEN).T # (T,N_sites)
                mi=0
                for j in range(Mmax):
                    Yhat=Yhat+g[j]*(Xc[p]@arch[j].reshape(N_SITES,N_HIDDEN).T)
                    if (j+1)==M_GRID[mi]:
                        nm[mi]+=((Yc[p]-Yhat)**2).mean()/Yvar[p]; mi+=1
                        if mi>=nM: break
                # M values beyond Mmax: clamp to last
                for jj in range(mi,nM): nm[jj]+=nm[mi-1] if mi>0 else 1.0
            acc+= nm/len(grp)
    tr[ki]=tr_acc/N_REP; te[ki]=te_acc/N_REP
    print(f"  K_PC={K:3d}  train NMSE[min..max]=[{tr[ki].min():.3f},{tr[ki].max():.3f}]"
          f"  test=[{te[ki].min():.3f},{te[ki].max():.3f}]",flush=True)

np.savez("pert_archetype_grid_data.npz",kpc=KPC_GRID,M=M_GRID,train=tr,test=te,sigma=SIGMA)

# ── figure ────────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,4.6),facecolor="white")
gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.34,left=0.06,right=0.98,top=0.86,bottom=0.16)
def tag(ax,t): ax.text(-0.16,1.04,t,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom")
vmax=max(tr.max(),te.max()); vmin=min(tr.min(),te.min())

def heat(ax,Z,title):
    im=ax.imshow(Z,aspect="auto",origin="lower",cmap="viridis_r",vmin=vmin,vmax=vmax)
    ax.set_xticks(range(nM)); ax.set_xticklabels(M_GRID)
    ax.set_yticks(range(nK)); ax.set_yticklabels(KPC_GRID)
    ax.set_xlabel("number of archetypes  $M$"); ax.set_ylabel("$K_\\mathrm{PC}$")
    ax.set_title(title)
    for yi in range(nK):
        for xi in range(nM):
            ax.text(xi,yi,f"{Z[yi,xi]:.2f}",ha="center",va="center",fontsize=5.5,
                    color="white" if Z[yi,xi]>(vmin+0.55*(vmax-vmin)) else "black")
    plt.colorbar(im,ax=ax,shrink=0.85,label="NMSE")

ax=fig.add_subplot(gs[0,0]); heat(ax,tr,"Training error (NMSE)"); tag(ax,"A")
ax=fig.add_subplot(gs[0,1]); heat(ax,te,"Test error (NMSE)"); tag(ax,"B")

ax=fig.add_subplot(gs[0,2])
cmap=plt.cm.plasma(np.linspace(0.1,0.9,nK))
for ki,K in enumerate(KPC_GRID):
    ax.plot(M_GRID,te[ki],"-o",ms=3,color=cmap[ki],lw=1.6,label=f"$K_\\mathrm{{PC}}$={K}")
ax.set_xlabel("number of archetypes  $M$"); ax.set_ylabel("test NMSE")
ax.set_title("Test error vs $M$"); ax.legend(frameon=False,fontsize=6.5,ncol=2)
tag(ax,"C")

fig.suptitle("Read-out archetype model: training and test reconstruction error vs "
             f"($K_\\mathrm{{PC}}$, $M$)   ($\\sigma$={SIGMA}, {N_REP}$\\times$ "
             f"{int((1-TEST_FRAC)*100)}/{int(TEST_FRAC*100)} split)",
             fontsize=11,fontweight="bold",y=0.99)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT,f"figureS3_archetype.{ext}"),dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS3_archetype.{ext}")
plt.close(fig); print("Done.")
