"""
pert_single_compare.py — single-site: theoretical read-out correction vs physical
resonant oscillatory drive, at the SAME top-1 (Delta-W most-affected) site per patient.
  - theoretical:  W_int = (1-a) W_p + a W_CC  on the top-1 column only, alpha in [0,50]
  - resonant osc: A*sin(2*pi*f1*t) at the top-1 site, A in [0,10]
Records reclassification (FC-lag classifier) and distance from the unstimulated
simulation (1 - corr to the unstimulated free-run FC). Main-text figure.
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
ALPHAS=np.array([0,1,2,4,7,12,18,25,32,40,45,50.])   # theoretical Delta-W on the single column
AMPS=np.array([0,0.5,1,1.5,2,3,4,6,9,12,16,20.])     # resonant oscillatory amplitude (extended)
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
Wcc=np.mean([patW[p] for p in cc],0)
top1={p:int(np.argmax(np.linalg.norm(Wcc-patW[p],axis=0))) for p in ad}

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
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; f1=float(abs(np.angle(wv[pos[np.argsort(np.abs(wv[pos]))[::-1][0]]]))/(2*np.pi))
def freerun_FC(Wout,p,site=None,amp=0.0):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); res.Jout=Wout.T.copy(); Y=[]
    for t in range(T-1):
        fbk=tgt[:,t] if t<=DRIVE_STEPS else res.y
        inp=ff*np.asarray(fbk,dtype=float).copy()
        if site is not None and amp!=0.: inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); Y.append(np.asarray(res.y,dtype=float).copy())
    return np.nan_to_num(np.corrcoef(np.array(Y).T[:,TIMES_SKIP-1:]))
def osc(p,site,amp):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy(); inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]
fc_ref={p:freerun_FC(patW[p],p) for p in ad}
def dist(fc,p): return 1.0-float(np.corrcoef(fc[iu],fc_ref[p][iu])[0,1])

nd=len(ALPHAS)
# personalised LDA-resonant site (cached scan) = the discriminant-aligned "good" target
red_full=np.load("pert_compare3_data.npz",allow_pickle=True)["red_full"]
pers_site={p:int(np.argmax(red_full[:,pi])) for pi,p in enumerate(ad)}
S_dw=np.zeros((nd,n_ad)); D_dw=np.zeros((nd,n_ad))
S_os=np.zeros((nd,n_ad)); D_os=np.zeros((nd,n_ad))
S_lp=np.zeros((nd,n_ad)); D_lp=np.zeros((nd,n_ad))
print(f"\nSingle-site comparison, f1={f1:.3f} ...")
for di in range(nd):
    a=ALPHAS[di]; A=AMPS[di]
    for pi,p in enumerate(ad):
        W=patW[p]; s=top1[p]; sl=pers_site[p]
        Wi=W.copy(); Wi[:,s]=(1-a)*W[:,s]+a*Wcc[:,s]
        S_dw[di,pi]=fscore(Wi,patX[p]);   D_dw[di,pi]=dist(freerun_FC(Wi,p),p)
        S_os[di,pi]=fscore(W,osc(p,s,A)); D_os[di,pi]=dist(freerun_FC(W,p,s,A),p)
        S_lp[di,pi]=fscore(W,osc(p,sl,A));D_lp[di,pi]=dist(freerun_FC(W,p,sl,A),p)
    print(f"  d{di:2d} dW(a={a:4.0f}) {(S_dw[di]<thr_f).mean()*100:3.0f}% | "
          f"osc-top1(A={A:4.1f}) {(S_os[di]<thr_f).mean()*100:3.0f}% | "
          f"osc-LDA {(S_lp[di]<thr_f).mean()*100:3.0f}%",flush=True)

recl_dw=(S_dw<thr_f).mean(1)*100; recl_os=(S_os<thr_f).mean(1)*100; recl_lp=(S_lp<thr_f).mean(1)*100
ddw=D_dw.mean(1); dos=D_os.mean(1); dlp=D_lp.mean(1)
np.savez("pert_single_compare_data.npz",alphas=ALPHAS,amps=AMPS,f1=f1,thr_f=thr_f,
         S_dw=S_dw,D_dw=D_dw,S_os=S_os,D_os=D_os,S_lp=S_lp,D_lp=D_lp,
         recl_dw=recl_dw,recl_os=recl_os,recl_lp=recl_lp,ddw=ddw,dos=dos,dlp=dlp)

# closed-loop results (cached) for the bottom row
cl=np.load("pert_closedloop_data.npz",allow_pickle=True)
CLR=[(cl["ol_dist"],cl["ol_rec"],"open-loop\n(LDA site, fixed $A$)","#455A64"),
     (cl["ca_dist"],cl["ca_rec"],"CL amplitude\n(LDA site)","#1565C0"),
     (cl["cs_dist"],cl["cs_rec"],"CL amp+site","#2E7D32")]
# closed-loop operating points (mean dose, mean distance, 100% reclassified) for A-C overlay
AMAX=float(np.load("pert_single_compare_data.npz")["amps"].max()) if False else 20.0
CLPTS=[(float(cl["A_FIX"])/20.0,            float(np.nanmean(cl["ol_dist"])),"#455A64","X","open-loop (fixed $A$)"),
       (float(np.nanmean(cl["ca_amp"]))/20.0,float(np.nanmean(cl["ca_dist"])),"#1565C0","P","closed-loop (amp)"),
       (float(np.nanmean(cl["cs_amp"]))/20.0,float(np.nanmean(cl["cs_dist"])),"#2E7D32","*","closed-loop (amp+site)")]

# ── merged figure (2x3): single-site dose-response (top) + closed-loop (bottom) ─
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
CDW="#1A237E"; COS="#E65100"; CLA="#2E7D32"
dwn=ALPHAS/ALPHAS.max(); osn=AMPS/AMPS.max()
fig=plt.figure(figsize=(13.5,8.6),facecolor="white")
gs=gridspec.GridSpec(2,3,figure=fig,wspace=0.30,hspace=0.42,left=0.07,right=0.985,top=0.91,bottom=0.07)
def tag(ax,s): ax.text(-0.17,1.05,s,transform=ax.transAxes,fontsize=13,fontweight="bold")
L_DW="theoretical $\\Delta W$ (top-1, $\\alpha\\!\\to\\!50$)"
L_OS="resonant osc ($\\Delta W$ top-1 site)"
L_LP="resonant osc (LDA-resonant site)"

# A: reclassification vs dose
ax=fig.add_subplot(gs[0,0])
ax.plot(dwn,recl_dw,"-o",ms=4,color=CDW,lw=2,label=L_DW)
ax.plot(osn,recl_os,"-s",ms=4,color=COS,lw=2,label=L_OS)
ax.plot(osn,recl_lp,"-^",ms=4,color=CLA,lw=2,label=L_LP)
for x,_,c,mk,_ in CLPTS: ax.scatter(x,100,marker=mk,s=130,c=c,edgecolors="k",lw=0.6,zorder=6)
ax.set_xlabel("relative dose"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification vs dose"); ax.set_ylim(-2,107); ax.legend(frameon=False); tag(ax,"A")

# B: distance vs dose
ax=fig.add_subplot(gs[0,1])
ax.plot(dwn,ddw,"-o",ms=4,color=CDW,lw=2,label=L_DW)
ax.plot(osn,dos,"-s",ms=4,color=COS,lw=2,label=L_OS)
ax.plot(osn,dlp,"-^",ms=4,color=CLA,lw=2,label=L_LP)
for x,dd,c,mk,_ in CLPTS: ax.scatter(x,dd,marker=mk,s=130,c=c,edgecolors="k",lw=0.6,zorder=6)
ax.set_xlabel("relative dose"); ax.set_ylabel("distance from original FC ($1-$corr)")
ax.set_title("Distance from unstimulated FC vs dose"); ax.legend(frameon=False); tag(ax,"B")

# C: efficacy vs perturbation cost
ax=fig.add_subplot(gs[0,2])
ax.plot(ddw,recl_dw,"-o",ms=4,color=CDW,lw=2,label=L_DW)
ax.plot(dos,recl_os,"-s",ms=4,color=COS,lw=2,label=L_OS)
ax.plot(dlp,recl_lp,"-^",ms=4,color=CLA,lw=2,label=L_LP)
for dd,_,c,mk,lab in [(p[1],p[0],p[2],p[3],p[4]) for p in CLPTS]:
    ax.scatter(dd,100,marker=mk,s=150,c=c,edgecolors="k",lw=0.6,zorder=6,label=lab)
ax.set_xlabel("distance from original FC ($1-$corr)"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Efficacy vs perturbation cost"); ax.set_ylim(-2,107)
ax.legend(frameon=False,fontsize=6.6,loc="lower left"); tag(ax,"C")

# D: closed-loop reclassification rate
ax=fig.add_subplot(gs[1,0])
ax.bar(range(3),[r.mean()*100 for _,r,_,_ in CLR],color=[c for *_,c in CLR],alpha=0.85)
ax.set_xticks(range(3)); ax.set_xticklabels([n for _,_,n,_ in CLR])
ax.set_ylabel("AD reclassified as CC (%)"); ax.set_ylim(0,108)
ax.set_title("Closed-loop: efficacy matched (all $100\\%$)")
for i,(_,r,_,_) in enumerate(CLR): ax.text(i,r.mean()*100+1,f"{r.mean()*100:.0f}%",ha="center",fontsize=8)
tag(ax,"D")

# E: closed-loop distance among reclassified patients
ax=fig.add_subplot(gs[1,1])
data=[d[(r>0)&~np.isnan(d)] for d,r,_,_ in CLR]
vp=ax.violinplot(data,positions=range(3),showmedians=True,widths=0.75)
for b,(*_,c) in zip(vp["bodies"],CLR): b.set_facecolor(c); b.set_alpha(0.5)
for kk in ["cmedians","cbars","cmins","cmaxes"]: vp[kk].set_color("k")
for xi,d in enumerate(data):
    ax.scatter(xi+np.random.uniform(-0.07,0.07,len(d)),d,s=8,c=CLR[xi][3],alpha=0.6,edgecolors="none")
    ax.text(xi,np.mean(d),f"{np.mean(d):.2f}",ha="center",va="center",fontsize=8,fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.8))
ax.set_xticks(range(3)); ax.set_xticklabels([n for _,_,n,_ in CLR])
ax.set_ylabel("distance from original FC ($1-$corr)"); ax.set_title("Closed-loop: perturbation cost"); tag(ax,"E")

# F: per-patient open-loop vs closed-loop distance
ax=fig.add_subplot(gs[1,2])
old=cl["ol_dist"]; csd=cl["cs_dist"]; m=(~np.isnan(old))&(~np.isnan(csd))
ax.scatter(old[m],csd[m],s=24,color="#2E7D32",alpha=0.75,edgecolors="none")
lim=[0,max(np.nanmax(old),np.nanmax(csd))*1.05]; ax.plot(lim,lim,"k--",lw=1,alpha=0.6)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.text(0.62,0.16,"closed-loop\nbetter",transform=ax.transAxes,fontsize=8,color="#2E7D32",ha="center")
ax.set_xlabel("open-loop distance (fixed $A$)"); ax.set_ylabel("closed-loop distance (amp+site)")
ax.set_title("Per-patient improvement"); tag(ax,"F")

fig.suptitle("Single-site stimulation and closed-loop control: a single read-out column is inert, a resonant "
             "drive cures only at the discriminant-aligned site, and feedback minimises the perturbation",
             fontsize=10.8,fontweight="bold",y=0.975)
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figure6_singlecompare.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figure6_singlecompare.{ext}")
plt.close(fig); print("Done.")
