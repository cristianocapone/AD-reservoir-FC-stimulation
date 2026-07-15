"""
pert_corr_dose.py — fidelity vs efficacy.
For two interventions on the seeded reservoir,
  (1) full-site EXACT Delta-W :  W_int = (1-a) W_p + a W_CC  over ALL sites
  (2) personalised LDA-resonant : A sin(2 pi f1 t) at each patient's own resonant site
measure, as a function of dose, the correlation between the *stimulated simulation*
(reconstructed FC) and the patient's *original recording* (real FC), together with the
reclassification rate. The key output compares the two at the SAME reclassification level.
Reuses red_full from pert_compare3_data.npz (no 121-site scan re-run).
Saves paper_figures/figureS6_fidelity.{png,pdf}, pert_corr_dose_data.npz.
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
sys.path.insert(0, ".")
from res import RESERVOIRE_SIMPLE

RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; K_PC=200; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; K_LDA=25; MAX_LAG=2; DRIVE_STEPS=5
TS_ROOT="./timeseries"; OUT="paper_figures"
ALPHAS=np.linspace(0,2,11)                              # full Delta-W to alpha=2 (Fig 4 range; reverts at ~1.5-2)
AMPS=np.array([0,0.1,0.25,0.5,0.75,1,1.5,2,3,5,10.])    # finer low end to resolve the resonant onset
iu=np.triu_indices(N_SITES,1)

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


print("Eigenmodes ...")
wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; order=pos[np.argsort(np.abs(wv[pos]))[::-1]]
f1=float(abs(np.angle(wv[order[0]]))/(2*np.pi))
site_eig=int(np.argmax(np.abs(vl[:,order[0]].conj()@res.Jin)))   # eigenmode-coupling site (no LDA)
print(f"  eigenmode-coupling site = {site_eig}")

# ── FC-lag classifier (same as pert_compare3) ────────────────────────────────
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
def recon(W,X): return (W.T.astype(float)@X.T.astype(float)).T          # (timesteps, N_sites)
def feat(W,X):
    S=recon(W,X); fs=[]
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
def osc(p,site,amp):                       # teacher-forced reconstruction (for the classifier)
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy(); inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]
def freerun_FC(Wout,p,site=None,amp=0.0):   # closed-loop FREE-RUN (Fig 2 protocol) -> simulated FC
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); res.Jout=Wout.T.copy(); Ysim=[]
    for t in range(T-1):
        fb=tgt[:,t] if t<=DRIVE_STEPS else res.y
        inp=ff*np.asarray(fb,dtype=float).copy()
        if site is not None and amp!=0.: inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); Ysim.append(np.asarray(res.y,dtype=float).copy())
    return np.nan_to_num(np.corrcoef(np.array(Ysim).T[:,TIMES_SKIP-1:]))
def fcmatch(fc,ref): return float(np.corrcoef(fc[iu],ref[iu])[0,1])   # corr of FC off-diagonals
def fcmatch_ex(fc,ref,s):            # same, but excluding the driven site's row & column
    m=np.ones(N_SITES,bool); m[s]=False
    a=fc[np.ix_(m,m)]; b=ref[np.ix_(m,m)]; ju=np.triu_indices(N_SITES-1,1)
    return float(np.corrcoef(a[ju],b[ju])[0,1])

# personalised resonant site from saved red_full (no scan re-run)
red_full=np.load("pert_compare3_data.npz",allow_pickle=True)["red_full"]
pers_site={p:int(np.argmax(red_full[:,pi])) for pi,p in enumerate(ad)}

# ── dose sweep ───────────────────────────────────────────────────────────────
nd=len(ALPHAS)
C_dw,C_lp,C_eg=np.zeros((nd,n_ad)),np.zeros((nd,n_ad)),np.zeros((nd,n_ad))  # corr to UNSTIM-MODEL FC (=1 at 0)
C_lp_ex,C_eg_ex=np.zeros((nd,n_ad)),np.zeros((nd,n_ad))                      # same, driven site excluded
R_dw,R_lp,R_eg=np.zeros((nd,n_ad)),np.zeros((nd,n_ad)),np.zeros((nd,n_ad))  # FC-lag score (reclass if <thr)
fc_ref={p:freerun_FC(patW[p],p) for p in ad}        # unstimulated free-run model FC, per patient
# sanity: unstimulated free-run reproduces the data FC at the Fig-2 level
emp_match=np.mean([fcmatch(fc_ref[p],np.nan_to_num(np.corrcoef(signals[first[p]]))) for p in ad])
print(f"\n  (unstimulated free-run vs data FC: r={emp_match:.3f}, cf. Fig 2 ~0.70)")
print("Dose sweep (full Delta-W vs personalised LDA-resonant vs eigenmode-site resonant) ...")
print("  perturbation = corr(stimulated free-run FC, UNSTIMULATED model FC); reclassification = FC-lag classifier")
for di in range(nd):
    a=ALPHAS[di]; amp=AMPS[di]
    for pi,p in enumerate(ad):
        W=patW[p]; X=patX[p]
        Wi=(1-a)*W+a*Wcc                       # full-site exact Delta-W (alpha in [0,2])
        C_dw[di,pi]=fcmatch(freerun_FC(Wi,p),fc_ref[p]); R_dw[di,pi]=fscore(Wi,X)
        fc_lp=freerun_FC(W,p,pers_site[p],amp)                           # personalised resonant drive
        C_lp[di,pi]=fcmatch(fc_lp,fc_ref[p]); C_lp_ex[di,pi]=fcmatch_ex(fc_lp,fc_ref[p],pers_site[p])
        R_lp[di,pi]=fscore(W,osc(p,pers_site[p],amp))                    # reclassification (classifier space)
        fc_eg=freerun_FC(W,p,site_eig,amp)                              # eigenmode-site resonant (no LDA)
        C_eg[di,pi]=fcmatch(fc_eg,fc_ref[p]); C_eg_ex[di,pi]=fcmatch_ex(fc_eg,fc_ref[p],site_eig)
        R_eg[di,pi]=fscore(W,osc(p,site_eig,amp))
    print(f"  d{di:2d}  dW(a={a:.2f}) corr={C_dw[di].mean():.3f} recl={(R_dw[di]<thr_f).mean()*100:4.0f}%"
          f" | LDp(A={amp:.2f}) corr={C_lp[di].mean():.3f} recl={(R_lp[di]<thr_f).mean()*100:4.0f}%"
          f" | eig corr={C_eg[di].mean():.3f} recl={(R_eg[di]<thr_f).mean()*100:4.0f}%",flush=True)

recl_dw=(R_dw<thr_f).mean(1)*100; recl_lp=(R_lp<thr_f).mean(1)*100; recl_eg=(R_eg<thr_f).mean(1)*100
cdw=C_dw.mean(1); clp=C_lp.mean(1); ceg=C_eg.mean(1)
clp_ex=C_lp_ex.mean(1); ceg_ex=C_eg_ex.mean(1)
cdw_e=C_dw.std(1)/np.sqrt(n_ad); clp_e=C_lp.std(1)/np.sqrt(n_ad); ceg_e=C_eg.std(1)/np.sqrt(n_ad)
np.savez("pert_corr_dose_data.npz",alphas=ALPHAS,amps=AMPS,thr_f=thr_f,site_eig=site_eig,
         C_dw=C_dw,C_lp=C_lp,C_eg=C_eg,C_lp_ex=C_lp_ex,C_eg_ex=C_eg_ex,
         R_dw=R_dw,R_lp=R_lp,R_eg=R_eg,recl_dw=recl_dw,recl_lp=recl_lp,recl_eg=recl_eg,f1=f1)

print("\nResonant-drive FC corr to unstimulated model: ALL sites vs DRIVEN SITE EXCLUDED")
print("  A      LDA(all) LDA(excl)   eig(all) eig(excl)")
for di in range(nd):
    print(f"  {AMPS[di]:5.2f}   {clp[di]:6.3f}  {clp_ex[di]:6.3f}    {ceg[di]:6.3f}  {ceg_ex[di]:6.3f}")

print(f"\nBaseline corr to unstimulated model (no stim): {cdw[0]:.3f} (=1 by construction)")
print(f"Reclassification ceiling: dW={recl_dw.max():.0f}%  LDA-pers={recl_lp.max():.0f}%  eigenmode={recl_eg.max():.0f}%")
def at_recl(levels,recl,corr):
    o=np.argsort(recl); return np.interp(levels,recl[o],corr[o])
levels=np.array([25,50,75,90.])
print("\nCorr to unstimulated model FC at matched reclassification:")
print("  recl%   fullDeltaW   LDA-pers   eigenmode")
for L,a_,b_,c_ in zip(levels,at_recl(levels,recl_dw,cdw),at_recl(levels,recl_lp,clp),at_recl(levels,recl_eg,ceg)):
    print(f"  {L:5.0f}    {a_:8.3f}   {b_:8.3f}   {c_:8.3f}")

# ── figure ───────────────────────────────────────────────────────────────────
plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
CDW="#1A237E"; CLP="#C2185B"; CEG="#00838F"
dose_dw=ALPHAS/ALPHAS.max(); dose_lp=AMPS/AMPS.max()   # both normalised to [0,1] (full Delta-W alpha<=2)
fig=plt.figure(figsize=(13.5,4.3),facecolor="white")
gs=gridspec.GridSpec(1,3,figure=fig,wspace=0.30,left=0.06,right=0.985,top=0.87,bottom=0.16)
def tag(ax,t): ax.text(-0.16,1.04,t,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom")

ax=fig.add_subplot(gs[0,0])
ax.fill_between(dose_dw,cdw-cdw_e,cdw+cdw_e,color=CDW,alpha=0.13)
ax.fill_between(dose_lp,clp-clp_e,clp+clp_e,color=CLP,alpha=0.13)
ax.fill_between(dose_lp,ceg-ceg_e,ceg+ceg_e,color=CEG,alpha=0.13)
ax.plot(dose_dw,cdw,"-o",ms=4,color=CDW,lw=2,label="full exact $\\Delta W$ ($\\alpha\\in[0,2]$)")
ax.plot(dose_lp,clp,"-o",ms=4,color=CLP,lw=2,label="personalised LDA-resonant")
ax.plot(dose_lp,ceg,"-^",ms=4,color=CEG,lw=2,label="eigenmode-site resonant (no LDA)")
ax.set_xlabel("relative stimulation dose"); ax.set_ylabel("corr(stimulated FC, unstimulated-model FC)")
ax.set_title("Perturbation of the simulation vs dose"); ax.set_ylim(0.0,1.02); ax.legend(frameon=False,loc="lower left"); tag(ax,"A")

ax=fig.add_subplot(gs[0,1])
ax.plot(dose_dw,recl_dw,"-o",ms=4,color=CDW,lw=2,label="full exact $\\Delta W$ ($\\alpha\\in[0,2]$)")
ax.plot(dose_lp,recl_lp,"-o",ms=4,color=CLP,lw=2,label="personalised LDA-resonant")
ax.plot(dose_lp,recl_eg,"-^",ms=4,color=CEG,lw=2,label="eigenmode-site resonant (no LDA)")
ax.set_xlabel("relative stimulation dose"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification vs dose"); ax.set_ylim(-2,105); ax.legend(frameon=False); tag(ax,"B")

ax=fig.add_subplot(gs[0,2])
o=np.argsort(recl_dw); ax.plot(recl_dw[o],cdw[o],"-o",ms=4,color=CDW,lw=2,label="full exact $\\Delta W$")
o=np.argsort(recl_lp); ax.plot(recl_lp[o],clp[o],"-o",ms=4,color=CLP,lw=2,label="personalised LDA-resonant")
o=np.argsort(recl_eg); ax.plot(recl_eg[o],ceg[o],"-^",ms=4,color=CEG,lw=2,label="eigenmode-site resonant (no LDA)")
ax.set_xlabel("AD reclassified as CC (%)"); ax.set_ylabel("corr(stimulated FC, unstimulated-model FC)")
ax.set_title("Perturbation at matched reclassification"); ax.set_ylim(0.0,1.02); ax.legend(frameon=False,loc="lower left"); tag(ax,"C")

fig.suptitle("How much each intervention perturbs the simulation (vs the unstimulated model) for a given "
             "therapeutic effect: full exact $\\Delta W$ vs personalised LDA-resonant vs eigenmode-site resonant",
             fontsize=10.5,fontweight="bold",y=0.99)
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figureS6_fidelity.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figureS6_fidelity.{ext}")
plt.close(fig); print("Done.")
