"""
persdist_compare.py — Fig S5: compare the per-patient top-1 stimulation site chosen
by the three selection methods, on the SAME seeded reservoir as pert_compare3.py:
  (1) Delta-W (most relevant in W; the single-site DW stimulation)  -> top1 ||Wcc-Wp||
  (2) eigenmode (first site in leading eigenmode)                   -> global site1
  (3) LDA-resonant (projection of resonant drive onto discriminant) -> pers_counts (npz)
No 121-site scan / no dose sweep -> fast. Saves paper_figures/figureS5_persdist.{png,pdf}.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from scipy.linalg import eig as sla_eig
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; TS_ROOT="./timeseries"; OUT="paper_figures"

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
i1=order[0]; site1=int(np.argmax(np.abs(vl[:,i1].conj()@res.Jin)))

# ── distributions over patients ──────────────────────────────────────────────
dw_counts=np.bincount([top1[p] for p in ad],minlength=N_SITES)
eig_counts=np.zeros(N_SITES,int); eig_counts[site1]=n_ad
pers_counts=np.load("pert_compare3_data.npz",allow_pickle=True)["pers_counts"].astype(int)
coords=np.load("pert_sites_data.npz",allow_pickle=True)["parcel_coords"]

METH=[("$\\Delta W$ (most relevant in W)",dw_counts,"#6A1B9A","Greens"),
      ("eigenmode (leading-mode site)",eig_counts,"#00838F","Blues"),
      ("LDA-resonant (per patient)",pers_counts,"#C2185B","autumn_r")]
for name,c,_,_ in METH:
    nd=int((c>0).sum())
    print(f"\n{name}: {nd} distinct site(s) over {int(c.sum())} patients")
    for k in np.argsort(c)[::-1]:
        if c[k]>0: print(f"  site {k:3d} {short(labels.get(k)):24s} {c[k]}")

# ── brains ───────────────────────────────────────────────────────────────────
from nilearn import plotting
brain_png=[]
for i,(name,c,_,cmap) in enumerate(METH):
    sel=np.where(c>0)[0]
    disp=plotting.plot_markers(c[sel].astype(float),coords[sel],
         node_size=45+45*c[sel]/max(1,c[sel].max())*2.5, node_cmap=cmap,
         node_vmin=0,node_vmax=float(c[sel].max()),display_mode="lzr",
         alpha=0.9,colorbar=False)
    fn=f"{OUT}/_s5_brain_{i}.png"; disp.savefig(fn,dpi=220); disp.close(); brain_png.append(fn)

# ── composite figure ─────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":8.5,
    "axes.titlesize":9.5,"xtick.labelsize":7,"ytick.labelsize":7.5,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(13.5,6.6),facecolor="white")
gs=gridspec.GridSpec(2,3,figure=fig,height_ratios=[1.15,1.0],hspace=0.45,wspace=0.22,
                     left=0.05,right=0.99,top=0.88,bottom=0.20)
TAGS=["A","B","C"]
for i,(name,c,col,cmap) in enumerate(METH):
    sel=np.argsort(c)[::-1]; sel=sel[c[sel]>0]; nd=len(sel)
    # brain
    axb=fig.add_subplot(gs[0,i]); axb.imshow(imread(brain_png[i])); axb.axis("off")
    axb.set_title(f"{name}\n({nd} distinct site{'s' if nd!=1 else ''}, N={int(c.sum())})",
                  fontsize=9.5,pad=2)
    axb.text(-0.02,1.04,TAGS[i],transform=axb.transAxes,fontsize=14,fontweight="bold")
    # histogram (cap at 18 labelled sites)
    axh=fig.add_subplot(gs[1,i]); sh=sel[:18]
    axh.bar(range(len(sh)),c[sh],color=col,alpha=0.85)
    axh.set_xticks(range(len(sh)))
    axh.set_xticklabels([short(labels.get(k)) for k in sh],rotation=65,ha="right",fontsize=5.6)
    axh.set_ylabel(f"# patients (of {n_ad})"); axh.set_ylim(0,max(3,c.max()+0.5))
    if nd>18: axh.text(0.98,0.92,f"+{nd-18} more",transform=axh.transAxes,ha="right",fontsize=6,style="italic")
fig.suptitle("Per-patient top-1 stimulation site: comparison of the three selection criteria",
             fontsize=11.5,fontweight="bold",y=0.965)
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figureS5_persdist.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS5_persdist.{ext}")
plt.close(fig)
print("Done.")
