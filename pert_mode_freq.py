"""
pert_mode_freq.py — test the "mode, not site" hypothesis.
(A) Site-robustness at the dominant-mode frequency f1: from cached red_full
    (each of 121 sites driven at f1, A=4), the per-site reclassification rate.
    If 'any site works', this distribution should be tight & high.
(B) Which eigenmode best separates CC from AD?  Project each patient's reservoir
    state onto every (left) eigenmode, take modal power, and score CC-vs-AD AUC
    per mode -> is the most discriminative mode the dominant one (f1)?
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
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; TS_ROOT="./timeseries"; OUT="paper_figures"

# ── (A) site-robustness from cached scan ─────────────────────────────────────
d=np.load("pert_compare3_data.npz",allow_pickle=True)
red_full=d["red_full"]; fbase=d["F_single"][0]; thr=float(d["thr_f"]); f1=float(d["f1"])
recl_site=np.array([np.mean((fbase-red_full[k])<thr) for k in range(N_SITES)])*100
site_eig=71
print(f"(A) Per-site reclassification at f1={f1:.3f}, A=4 (all 121 sites):")
print(f"    mean={recl_site.mean():.0f}%  sd={recl_site.std():.0f}%  min={recl_site.min():.0f}%  "
      f"max={recl_site.max():.0f}%  eigenmode-site(71)={recl_site[site_eig]:.0f}%")
print(f"    sites reclassifying >=90%: {(recl_site>=90).sum()}/121;  <=40%: {(recl_site<=40).sum()}/121")

# ── data + seeded reservoir (for modal analysis) ─────────────────────────────
print("\nLoading data + reservoir ...")
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
plabel=np.array([labs[psid[p][0]] for p in upid])
first={p:psid[p][0] for p in upid}
all_sig=np.concatenate([s.T for s in signals],0)
evv,evec=np.linalg.eigh(np.cov((all_sig-all_sig.mean(0)).T))
ev50=evec[:,np.argsort(evv)[::-1]][:,:N_PC_MODEL]
np.random.seed(RNG_SEED)
par=dict(tau_m_f=0.0005,tau_m_s=0.0005,N=N_HIDDEN,T=139,dt=0.005,sigma_input=0.01,
         shape=(N_HIDDEN,N_SITES,N_SITES,139))
res=RESERVOIRE_SIMPLE(par); res.J*=SR/max(abs(np.linalg.eigvals(res.J)))
patX={}
for p in tqdm(upid,desc="  TF"):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1): res.step_rate(ff*tgt[:,t],sigma_dyn=0.); X.append(res.X.copy())
    patX[p]=np.array(X)[TIMES_SKIP:]

# ── (B) modal power discriminability ─────────────────────────────────────────
print("\n(B) Eigenmode discriminability (CC vs AD modal power) ...")
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; pos=pos[np.argsort(np.abs(wv[pos]))[::-1]]   # complex modes, by |lambda|
M=min(40,len(pos)); modes=pos[:M]
fk=np.abs(np.angle(wv[modes]))/(2*np.pi); lam=np.abs(wv[modes])
VL=vl[:,modes]                                   # (N_hidden, M) complex left eigenvectors
P=np.zeros((len(upid),M))
for i,p in enumerate(upid):
    z=patX[p].astype(np.float64)@VL.conj()       # (timesteps, M) modal coordinates
    P[i]=np.log(np.mean(np.abs(z)**2,0)+1e-12)
auc=np.array([roc_auc_score(plabel,P[:,k]) for k in range(M)])
disc=np.abs(auc-0.5)
o=np.argsort(disc)[::-1]
ki_f1=int(np.argmin(np.abs(fk-f1)))              # the dominant mode (closest freq to f1)
print(f"    dominant mode f1={f1:.3f}: AUC={auc[ki_f1]:.3f} (|AUC-.5|={disc[ki_f1]:.3f}), rank "
      f"{list(o).index(ki_f1)+1}/{M}")
print("    most discriminative modes (freq, AUC):")
for k in o[:6]:
    tag=" <- f1" if k==ki_f1 else ""
    print(f"      f={fk[k]:.3f}  |lambda|={lam[k]:.3f}  AUC={auc[k]:.3f}{tag}")

np.savez("pert_mode_freq_data.npz",recl_site=recl_site,f1=f1,site_eig=site_eig,
         fk=fk,lam=lam,auc=auc,ki_f1=ki_f1)

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(11,4.2),facecolor="white")
gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.28,left=0.08,right=0.98,top=0.88,bottom=0.16)
def tag(ax,t): ax.text(-0.14,1.04,t,transform=ax.transAxes,fontsize=13,fontweight="bold")

ax=fig.add_subplot(gs[0,0])
ax.hist(recl_site,bins=np.arange(0,101,7),color="#5C6BC0",alpha=0.85,edgecolor="white")
ax.axvline(recl_site.mean(),color="k",ls="--",lw=1.3,label=f"mean {recl_site.mean():.0f}%")
ax.axvline(recl_site[site_eig],color="#00838F",lw=2,label=f"eigenmode site 71 ({recl_site[site_eig]:.0f}%)")
ax.axvline(100,color="#C2185B",lw=2,label="LDA per-patient (100%)")
ax.set_xlabel("AD reclassified as CC (%)"); ax.set_ylabel("# of 121 driven sites")
ax.set_title(f"Site matters at fixed $f_1$ ($A=4$):\nper-site reclassification spans the whole range")
ax.legend(frameon=False,fontsize=7); tag(ax,"A")

ax=fig.add_subplot(gs[0,1])
ax.axhline(0.5,color="gray",ls="-.",lw=1)
ax.scatter(fk,auc,s=24,color="#455A64",alpha=0.8)
ax.scatter([fk[ki_f1]],[auc[ki_f1]],s=70,color="#C62828",zorder=5,label=f"dominant mode $f_1$={f1:.3f}")
kbest=o[0]; ax.scatter([fk[kbest]],[auc[kbest]],s=70,facecolors="none",edgecolors="#2E7D32",
                        lw=2,zorder=5,label=f"most discriminative (f={fk[kbest]:.3f})")
ax.set_xlabel("mode frequency (cycles/step)"); ax.set_ylabel("CC-vs-AD AUC of modal power")
ax.set_title("Which eigenmode separates CC from AD?"); ax.legend(frameon=False,fontsize=7.5); tag(ax,"B")

for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figure_modefreq.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figure_modefreq.{ext}")
plt.close(fig); print("Done.")
