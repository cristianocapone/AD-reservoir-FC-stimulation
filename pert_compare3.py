"""
pert_compare3.py
================
Focal-stimulation comparison (Fig S2) + where-to-drive-the-resonance analysis
(Fig S4), same seeded reservoir, both classifiers.

Strategies (dose-response, FC-lag & G-space):
  (1) single-site         : W-interpolation W_int=(1-a)W_AD+a*W_CC, top-1 ||dW|| site
  (2) resonant oscillatory: A sin(2 pi f1 t) at the top-1 ||dW|| site
  (3) 2-site eigenmode     : A sin(2 pi f1 t)@s1 + A sin(2 pi f2 t)@s2, s_i = site of
                             maximal coupling to eigenmode i
  (4) LDA-resonant         : A sin(2 pi f1 t) at the site whose resonant drive moves
                             the FC-lag discriminant most toward CC (= projection of
                             the stimulation effect onto the LDA)  -> "drive toward health"

Fig S4: per-site FC-lag score reduction under resonant drive (the LDA projection)
        ranked + on a glass brain, for anatomical/literature interpretation.

Saves: paper_figures/figureS2_compare.{png,pdf}, paper_figures/figureS4_ldares.{png,pdf},
       paper_figures/figureS4_brain.{png,pdf}, pert_compare3_data.npz
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              errors="replace", line_buffering=True)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from scipy.linalg import eig as sla_eig
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; K_PC=200; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; K_LDA=25; MAX_LAG=2
TS_ROOT="./timeseries"; OUT="paper_figures"; A_REF=4.0
ALPHAS=np.linspace(0,5,11); AMPS=np.linspace(0,10,11)

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
            signals.append(a.T); pids.append(fn.split("_ses-")[0]); labs.append(0 if lb=="CC" else 1)
pids=np.array(pids); labs=np.array(labs); upid=np.unique(pids)
psid={p:np.where(pids==p)[0] for p in upid}
plabel=np.array([labs[psid[p][0]] for p in upid]); Npat=len(upid)
cc=[upid[i] for i in np.where(plabel==0)[0]]; ad=[upid[i] for i in np.where(plabel==1)[0]]
n_ad=len(ad); print(f"  {Npat} patients ({len(cc)} CC, {n_ad} AD)")

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
top1={p:int(np.argmax(np.linalg.norm(Wcc-patW[p],axis=0))) for p in ad}

print("Eigenmodes ...")
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; order=pos[np.argsort(np.abs(wv[pos]))[::-1]]
i1,i2=order[0],order[1]
f1=float(abs(np.angle(wv[i1]))/(2*np.pi)); f2=float(abs(np.angle(wv[i2]))/(2*np.pi))
site1=int(np.argmax(np.abs(vl[:,i1].conj()@res.Jin)))
site2=int(np.argmax(np.abs(vl[:,i2].conj()@res.Jin)))
print(f"  eig1 f1={f1:.4f} site {site1}({short(labels.get(site1))}); "
      f"eig2 f2={f2:.4f} site {site2}({short(labels.get(site2))})")

# ── classifiers ──────────────────────────────────────────────────────────────
class LDA:
    def fit(s,X,y):
        c0,c1=np.unique(y); X0,X1=X[y==c0],X[y==c1]; m0,m1=X0.mean(0),X1.mean(0)
        Sw=(X0-m0).T@(X0-m0)+(X1-m1).T@(X1-m1)+1e-6*np.eye(X.shape[1])
        w=np.linalg.solve(Sw,m1-m0); w/=np.linalg.norm(w)+1e-12; s.w=w; return s
    def tr(s,X): return X@s.w
def bal(X,y,sd=0):
    r=np.random.default_rng(sd); c0,c1=np.where(y==0)[0],np.where(y==1)[0]; n=min(len(c0),len(c1))
    sel=np.concatenate([r.choice(c0,n,0),r.choice(c1,n,0)]); r.shuffle(sel); return X[sel],y[sel]
patVt={}
for p in upid:
    _,sx,V=np.linalg.svd(patX[p].astype(np.float64),full_matrices=False)
    patVt[p]=V[:min(K_PC,int((sx>1e-8).sum()))]
def projW(W,p):
    Vk=patVt[p]; return (W.T.astype(np.float64)@Vk.T@Vk).flatten()
Wproj=np.array([projW(patW[p],p) for p in upid]); Wmean=Wproj.mean(0)
Wcent=Wproj-Wmean; _,_,Vsvd=np.linalg.svd(Wcent,full_matrices=False); Meff=Npat-1
G_B=Wcent@Vsvd[:Meff].T
Xl,yl=bal(G_B[:,:K_LDA],plabel,RNG_SEED); lda_g=LDA().fit(Xl,yl); Zg=lda_g.tr(G_B[:,:K_LDA])
if Zg[plabel==0].mean()>Zg[plabel==1].mean(): lda_g.w*=-1; Zg=-Zg
thr_g=0.5*(Zg[plabel==0].mean()+Zg[plabel==1].mean()); cc_g=Zg[plabel==0]
def gscore(W,p):
    wp=projW(W,p); g=((wp-Wmean)@Vsvd[:Meff].T)[:K_LDA]; return float(lda_g.tr(g.reshape(1,-1))[0])
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
fb=np.array([feat(patW[p],patX[p]) for p in tqdm(upid,leave=False)])
fm=fb.mean(0); fcc=fb-fm; evf,evecf=np.linalg.eigh(fcc@fcc.T); o=np.argsort(evf)[::-1]
evf=np.maximum(evf[o],0); evecf=evecf[:,o]; Gf=evecf*np.sqrt(evf)
Xl,yl=bal(Gf[:,:K_LDA],plabel,RNG_SEED); lda_f=LDA().fit(Xl,yl); Zf=lda_f.tr(Gf[:,:K_LDA])
if Zf[plabel==0].mean()>Zf[plabel==1].mean(): lda_f.w*=-1; Zf=-Zf
thr_f=0.5*(Zf[plabel==0].mean()+Zf[plabel==1].mean()); cc_f=Zf[plabel==0]
def fscore(W,X):
    f=feat(W,X)-fm; g=(f@fcc.T@evecf)/(np.sqrt(evf)+1e-12); return float(lda_f.tr(g[:K_LDA].reshape(1,-1))[0])
def osc_multi(p,sites,freqs,amp):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy()
        for st,fr in zip(sites,freqs): inp[st]+=amp*np.sin(2*np.pi*fr*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]

# ── site scan: per-patient resonant-drive LDA projection (score reduction at f1)
print(f"\nLDA-resonance site scan: drive each of {N_SITES} sites at f1, A={A_REF} ...")
fbase=np.array([fscore(patW[p],patX[p]) for p in ad])
red_full=np.zeros((N_SITES,n_ad))           # per-site, per-patient reduction (toward CC)
for k in tqdm(range(N_SITES),desc="  scan"):
    for pi,p in enumerate(ad):
        red_full[k,pi]=fbase[pi]-fscore(patW[p],osc_multi(p,[k],[f1],A_REF))
site_red=red_full.mean(1)                    # population-average reduction per site
ld_order=np.argsort(site_red)[::-1]
ldares=int(ld_order[0])                       # AVERAGE best site (one global target)
pers_site={p:int(np.argmax(red_full[:,pi])) for pi,p in enumerate(ad)}  # PERSONALIZED
pers_counts=np.bincount([pers_site[p] for p in ad],minlength=N_SITES)
print(f"  average best site: {ldares} ({short(labels.get(ldares))})  red={site_red[ldares]:+.3f}")
print("  top average sites:")
for k in ld_order[:8]:
    print(f"    site {k:3d}  {short(labels.get(k)):22s}  red={site_red[k]:+.3f}")
print("  most frequent personalized sites:")
for k in np.argsort(pers_counts)[::-1][:8]:
    if pers_counts[k]>0:
        print(f"    site {k:3d}  {short(labels.get(k)):22s}  chosen by {pers_counts[k]}/{n_ad}")

# ── dose-response: 4 strategies x 2 classifiers ───────────────────────────────
nd=len(ALPHAS)
def Z(): return np.zeros((nd,n_ad))
G_single,G_osc,G_eig2,G_ld,G_ldp=Z(),Z(),Z(),Z(),Z()
F_single,F_osc,F_eig2,F_ld,F_ldp=Z(),Z(),Z(),Z(),Z()
print("\nDose-response sweep ...")
for di in range(nd):
    a=ALPHAS[di]; amp=AMPS[di]
    for pi,p in enumerate(ad):
        W=patW[p]; s=top1[p]
        Wi=W.copy(); Wi[:,s]=(1-a)*W[:,s]+a*Wcc[:,s]
        G_single[di,pi]=gscore(Wi,p); F_single[di,pi]=fscore(Wi,patX[p])
        F_osc[di,pi]=fscore(W,osc_multi(p,[s],[f1],amp));        G_osc[di,pi]=gscore(W,p)
        F_eig2[di,pi]=fscore(W,osc_multi(p,[site1,site2],[f1,f2],amp)); G_eig2[di,pi]=gscore(W,p)
        F_ld[di,pi]=fscore(W,osc_multi(p,[ldares],[f1],amp));    G_ld[di,pi]=gscore(W,p)
        F_ldp[di,pi]=fscore(W,osc_multi(p,[pers_site[p]],[f1],amp)); G_ldp[di,pi]=gscore(W,p)
    print(f"  dose {di}/10  F: single={F_single[di].mean():+.2f} osc={F_osc[di].mean():+.2f}"
          f" eig2={F_eig2[di].mean():+.2f} ld-avg={F_ld[di].mean():+.2f} ld-pers={F_ldp[di].mean():+.2f}",
          flush=True)

parcel_coords=np.load("pert_sites_data.npz",allow_pickle=True)["parcel_coords"]
np.savez("pert_compare3_data.npz",alphas=ALPHAS,amps=AMPS,
         G_single=G_single,G_osc=G_osc,G_eig2=G_eig2,G_ld=G_ld,G_ldp=G_ldp,
         F_single=F_single,F_osc=F_osc,F_eig2=F_eig2,F_ld=F_ld,F_ldp=F_ldp,
         thr_g=thr_g,thr_f=thr_f,cc_g=cc_g,cc_f=cc_f,f1=f1,f2=f2,
         site1=site1,site2=site2,ldares=ldares,site_red=site_red,
         red_full=red_full,pers_counts=pers_counts)

# ── glass brain of LDA-resonance map ──────────────────────────────────────────
print("Rendering brain ...")
from nilearn import plotting
sel=site_red>max(0,site_red.max()*0.15)
disp=plotting.plot_markers(site_red[sel],parcel_coords[sel],
    node_size=30+120*site_red[sel]/site_red.max(),node_cmap="autumn_r",
    node_vmin=0,node_vmax=float(site_red.max()),display_mode="lzry",alpha=0.9,
    colorbar=True,title="FC-lag score reduction under resonant drive (toward CC)")
disp.savefig(os.path.join(OUT,"figureS4_brain.png"),dpi=300)
disp.savefig(os.path.join(OUT,"figureS4_brain.pdf")); disp.close()

# ── Figure S2 (4 strategies, both classifiers) ────────────────────────────────
dose=np.linspace(0,1,nd)
COL={"single":"#6A1B9A","osc":"#E65100","eig2":"#00838F","ld":"#2E7D32","ldp":"#C2185B"}
LBL={"single":"single-site ($\\Delta W$, top-1)","osc":"resonant osc (top-1 $\\Delta W$, $f_1$)",
     "eig2":"2-site eigenmode ($f_1$@s1,$f_2$@s2)",
     "ld":"LDA-resonant, average (global site)","ldp":"LDA-resonant, personalised (per-patient)"}
def cure(FL,thr): return (FL<thr).mean(1)*100
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(11,8.4),facecolor="white")
gs=gridspec.GridSpec(2,2,figure=fig,wspace=0.28,hspace=0.42,left=0.08,right=0.97,top=0.90,bottom=0.08)
def tag(ax,t): ax.text(-0.13,1.05,t,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom")
STRAT=[("single",G_single,F_single),("osc",G_osc,F_osc),("eig2",G_eig2,F_eig2),
       ("ld",G_ld,F_ld),("ldp",G_ldp,F_ldp)]
def score_panel(ax,which,thr,ccv,ylab,title):
    ax.axhspan(ccv.mean()-ccv.std(),ccv.mean()+ccv.std(),alpha=0.1,color="#1565C0")
    ax.axhline(ccv.mean(),color="#1565C0",ls="--",lw=1.2,label="CC mean ±1σ")
    ax.axhline(thr,color="gray",ls="-.",lw=1,label="boundary")
    for key,Gd,Fd in STRAT:
        FL=Gd if which=="G" else Fd; m=FL.mean(1); e=FL.std(1)/np.sqrt(n_ad)
        ax.fill_between(dose,m-e,m+e,color=COL[key],alpha=0.13)
        ax.plot(dose,m,"-o",ms=3.5,color=COL[key],lw=2,label=LBL[key])
    ax.set_xlabel("relative stimulation dose"); ax.set_ylabel(ylab); ax.set_title(title)
def cure_panel(ax,which,thr,base,title):
    for key,Gd,Fd in STRAT:
        FL=Gd if which=="G" else Fd
        ax.plot(dose,cure(FL,thr),"-o",ms=3.5,color=COL[key],lw=2,label=LBL[key])
    ax.axhline(base,color="#C62828",ls=":",lw=1.2,label="baseline")
    ax.set_xlabel("relative stimulation dose"); ax.set_ylabel("AD reclassified as CC (%)")
    ax.set_title(title); ax.set_ylim(-2,105)
ax=fig.add_subplot(gs[0,0]); score_panel(ax,"G",thr_g,cc_g,"G-space LDA score","G-space (read-out geometry)"); ax.legend(frameon=False,fontsize=6.5); tag(ax,"A")
ax=fig.add_subplot(gs[0,1]); cure_panel(ax,"G",thr_g,cure(G_single[:1],thr_g)[0],"G-space — reclassified"); ax.legend(frameon=False,fontsize=6.5); tag(ax,"B")
ax=fig.add_subplot(gs[1,0]); score_panel(ax,"F",thr_f,cc_f,"FC-lag LDA score","FC-lag (reconstructed FC)"); ax.legend(frameon=False,fontsize=6.5); tag(ax,"C")
ax=fig.add_subplot(gs[1,1]); cure_panel(ax,"F",thr_f,cure(F_single[:1],thr_f)[0],"FC-lag — reclassified"); ax.legend(frameon=False,fontsize=6.5); tag(ax,"D")
fig.suptitle("Focal stimulation incl. LDA-resonant site selection (drive resonance toward health)",
             fontsize=11,fontweight="bold",y=0.975)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT,f"figureS2_compare.{ext}"),dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS2_compare.{ext}")
plt.close(fig)

# ── Figure S4 (avg site ranking + brain + personalised-site frequency) ────────
fig=plt.figure(figsize=(16,4.4),facecolor="white")
gs=gridspec.GridSpec(1,3,figure=fig,width_ratios=[1,1.3,1],wspace=0.30,left=0.06,right=0.99,top=0.84,bottom=0.32)
ax=fig.add_subplot(gs[0,0])
nb=12; top=ld_order[:nb]
ax.bar(range(nb),site_red[top],color="#2E7D32",alpha=0.85)
ax.set_xticks(range(nb)); ax.set_xticklabels([short(labels.get(k)) for k in top],rotation=55,ha="right",fontsize=6.5)
ax.set_ylabel("mean FC-lag reduction\n(toward CC)"); ax.set_title("Most effective sites (population average)")
ax.text(-0.16,1.05,"A",transform=ax.transAxes,fontsize=13,fontweight="bold")
ax=fig.add_subplot(gs[0,1]); ax.imshow(imread(os.path.join(OUT,"figureS4_brain.png"))); ax.axis("off")
ax.set_title("Anatomical map (average, driven at $f_1$)",pad=2)
ax.text(-0.02,1.03,"B",transform=ax.transAxes,fontsize=13,fontweight="bold")
ax=fig.add_subplot(gs[0,2])
pc_order=np.argsort(pers_counts)[::-1]; pc_sel=pc_order[pers_counts[pc_order]>0][:12]
ax.bar(range(len(pc_sel)),pers_counts[pc_sel],color="#C2185B",alpha=0.85)
ax.set_xticks(range(len(pc_sel))); ax.set_xticklabels([short(labels.get(k)) for k in pc_sel],rotation=55,ha="right",fontsize=6.5)
ax.set_ylabel(f"# patients (of {n_ad})"); ax.set_title("Personalised best site — selection frequency")
ax.text(-0.16,1.05,"C",transform=ax.transAxes,fontsize=13,fontweight="bold")
fig.suptitle("Where to drive the resonance toward health: per-site FC-lag improvement; "
             "population-average (A,B) vs personalised targets (C)",
             fontsize=10.5,fontweight="bold",y=0.99)
for ext in ("png","pdf"):
    fig.savefig(os.path.join(OUT,f"figureS4_ldares.{ext}"),dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS4_ldares.{ext}")
plt.close(fig); print("Done.")
