"""
pert_lopo_target.py
===================
Leakage-free (nested / leave-one-AD-patient-out) version of the LDA-resonant
therapy-target result, to answer the circularity concern: the therapy target is
currently selected AND scored with a discriminant fit on ALL patients (including
the test patient). Here the FC-lag embedding, the LDA direction, the decision
threshold, AND the per-patient site selection are all rebuilt from the TRAINING
patients only, then applied to the held-out AD patient.

Two selection modes are reported, each transductive (paper's current scheme) vs
leakage-free LOPO:
  - population-average : one global site = argmax over sites of the mean per-site
    resonance reduction across the *training* AD patients (no per-test-patient
    choice). Conservative.
  - personalised       : per held-out patient, the site maximising THAT patient's
    own reduction under the *train-only* discriminant. Upper bound on
    personalisation; still leakage-free in the classifier.
Baseline (unstimulated) reclassification is reported for reference.

Efficiency: the FC-lag embedding projection is a Gram/kernel form, so every fold
is exact linear algebra on the base-patient Gram (76x76) and a 76-vector per
stimulated feature; the reservoir is simulated once per (AD patient, site) at the
selection amplitude, plus once per selected (patient, site, amplitude).

Saves: pert_lopo_target_data.npz, paper_figures/figure_lopo_target.{png,pdf}
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import eig as sla_eig
from tqdm import tqdm
import warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "."); from res import RESERVOIRE_SIMPLE

# ── params (identical to pert_compare3.py so sites/f1/W match) ─────────────────
RNG_SEED=42; N_CC_SAMP=40; N_SITES=121; N_PC_MODEL=50; K_PC=200; TIMES_SKIP=10
ff=0.1; N_HIDDEN=2000; SIGMA=0.05; SR=0.95; K_LDA=25; MAX_LAG=2; DRIVE_STEPS=5
TS_ROOT="./timeseries"; OUT="paper_figures"; A_SEL=4.0
AMPS_EVAL=[4.0, 7.0, 10.0]           # report reclassification at these drive amplitudes

# ── data ──────────────────────────────────────────────────────────────────────
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
cc=[upid[i] for i in np.where(plabel==0)[0]]; ad=[upid[i] for i in np.where(plabel==1)[0]]
n_ad=len(ad); Npat=len(upid)
ad_idx=np.array([list(upid).index(p) for p in ad])   # positions of AD patients in upid/fb
print(f"  {Npat} patients ({len(cc)} CC, {n_ad} AD)")

all_sig=np.concatenate([s.T for s in signals],0)
ev,evec=np.linalg.eigh(np.cov((all_sig-all_sig.mean(0)).T))
ev50=evec[:,np.argsort(ev)[::-1]][:,:N_PC_MODEL]

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
rw=np.random.default_rng(RNG_SEED+1); patW={}
for p in upid:
    s=signals[first[p]]; tgt=(s.T@ev50@ev50.T).T
    Xc=patX[p]; Yc=tgt[:,TIMES_SKIP:TIMES_SKIP+Xc.shape[0]].T
    patW[p]=np.linalg.pinv(Xc+rw.normal(0,SIGMA,Xc.shape))@Yc

wv,vl,vr=sla_eig(res.J,left=True,right=True)
pos=np.where(wv.imag>1e-8)[0]; i1=pos[np.argsort(np.abs(wv[pos]))[::-1][0]]
f1=float(abs(np.angle(wv[i1]))/(2*np.pi))
print(f"  resonant frequency f1={f1:.4f}")

# ── FC-lag features ────────────────────────────────────────────────────────────
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
def osc(p,site,amp):
    s=signals[first[p]]; T=s.shape[1]; tgt=(s.T@ev50@ev50.T).T
    res.T=T; res.reset(); X=[]
    for t in range(T-1):
        inp=ff*tgt[:,t].copy(); inp[site]+=amp*np.sin(2*np.pi*f1*t)
        res.step_rate(inp,sigma_dyn=0.); X.append(res.X.copy())
    return np.array(X)[TIMES_SKIP:]

print("Base features (all patients) ...")
fb=np.array([feat(patW[p],patX[p]) for p in tqdm(upid,leave=False)], dtype=np.float64)  # (76, D)
Kbb=fb@fb.T                                    # base Gram (76x76)

def d_of(f):  return fb@f                       # 76-vector of dots with base features

# ── site scan at A_SEL: stimulated-feature dot-vectors d_ref[p][k] (cached) ────
CACHE="pert_lopo_sims.npz"
if os.path.exists(CACHE):
    print(f"Loading cached site scan from {CACHE} ...")
    _c=np.load(CACHE); d_ref=_c["d_ref"]
    assert d_ref.shape==(n_ad,N_SITES,Npat), "cache shape mismatch; delete pert_lopo_sims.npz"
else:
    print(f"\nSite scan (drive each of {N_SITES} sites at f1, A={A_SEL}) ...")
    d_ref=np.zeros((n_ad,N_SITES,Npat))            # (40,121,76)
    for pi,p in enumerate(tqdm(ad,desc="  scan")):
        for k in range(N_SITES):
            d_ref[pi,k]=d_of(feat(patW[p],osc(p,k,A_SEL)))
    np.savez(CACHE, d_ref=d_ref)
    print(f"Saved {CACHE}")

# ══════════════════════════════════════════════════════════════════════════════
#  Kernel FC-lag embedding + balanced-LDA scoring, from base Gram + d-vectors
# ══════════════════════════════════════════════════════════════════════════════
def bal_idx(y,sd=0):
    r=np.random.default_rng(sd); c0,c1=np.where(y==0)[0],np.where(y==1)[0]; n=min(len(c0),len(c1))
    sel=np.concatenate([r.choice(c0,n,0),r.choice(c1,n,0)]); r.shuffle(sel); return sel

class KernelLDA:
    """FC-lag embedding (kernel PCA of centred base features) + balanced Fisher
    LDA, all fit on a chosen TRAIN subset of base patients. Scores any feature
    given its dot-vector d = fb @ f."""
    def __init__(self, train_pos, y_train):
        self.T=np.asarray(train_pos); nt=len(self.T)
        K=Kbb[np.ix_(self.T,self.T)]                     # train raw Gram
        one=np.ones((nt,nt))/nt
        Kc=K - one@K - K@one + one@K@one                 # centred train Gram
        w_,U=np.linalg.eigh(Kc); o=np.argsort(w_)[::-1]
        self.lam=np.maximum(w_[o],1e-12); self.U=U[:,o]
        self.krow=K.mean(0); self.kall=K.mean()          # centring terms
        Gtr=(self.U*np.sqrt(self.lam))[:,:K_LDA]         # train coords
        # balanced Fisher LDA on train coords
        si=bal_idx(y_train,RNG_SEED); Xl=Gtr[si]; yl=y_train[si]
        c0,c1=np.unique(yl); X0,X1=Xl[yl==c0],Xl[yl==c1]; m0,m1=X0.mean(0),X1.mean(0)
        Sw=(X0-m0).T@(X0-m0)+(X1-m1).T@(X1-m1)+1e-6*np.eye(Xl.shape[1])
        w=np.linalg.solve(Sw,m1-m0); w/=np.linalg.norm(w)+1e-12
        Z=Gtr@w
        if Z[y_train==0].mean()>Z[y_train==1].mean(): w=-w; Z=-Z
        self.w=w; self.thr=0.5*(Z[y_train==0].mean()+Z[y_train==1].mean())
    def coord(self, d):
        # out-of-sample kernel-PCA projection of feature with dot-vector d
        dT=d[self.T]
        c=dT - dT.mean() - self.krow + self.kall          # (f-fm_T)·fcc_T
        g=(c@self.U)/np.sqrt(self.lam)
        return g[:K_LDA]
    def score(self, d):  return float(self.coord(d)@self.w)

# base patients' own dot-vectors = rows of Kbb
d_base=[Kbb[i] for i in range(Npat)]

def eval_scheme(train_pos, y_train, held_pos_list):
    """Fit on train_pos; return the fitted model + convenience closures."""
    return KernelLDA(train_pos, y_train)

# ── amplitude-dependent stim dots for a given (patient, site) ─────────────────
_stim_cache={}
def d_stim(p_i, site, amp):
    key=(p_i,site,round(amp,4))
    if key not in _stim_cache:
        p=ad[p_i]
        _stim_cache[key]=d_of(feat(patW[p],osc(p,site,amp)))
    return _stim_cache[key]
# seed cache with the A_SEL scan
for pi in range(n_ad):
    for k in range(N_SITES):
        _stim_cache[(pi,k,round(A_SEL,4))]=d_ref[pi,k]

# ══════════════════════════════════════════════════════════════════════════════
#  TRANSDUCTIVE (paper scheme): fit on ALL patients
# ══════════════════════════════════════════════════════════════════════════════
print("\nTransductive scheme (fit on all patients) ...")
all_pos=np.arange(Npat)
M_all=KernelLDA(all_pos, plabel)
base_score_all=np.array([M_all.score(d_base[ad_idx[pi]]) for pi in range(n_ad)])
red_all=np.array([[base_score_all[pi]-M_all.score(d_ref[pi,k]) for k in range(N_SITES)]
                  for pi in range(n_ad)])          # (40,121)
site_red_all=red_all.mean(0)
global_all=int(np.argmax(site_red_all))
pers_all=np.array([int(np.argmax(red_all[pi])) for pi in range(n_ad)])
print(f"  global site (transductive): {global_all}  | {len(set(pers_all))} distinct personalised sites")

# ══════════════════════════════════════════════════════════════════════════════
#  LEAKAGE-FREE LOPO: leave out each AD patient, refit everything on the rest
# ══════════════════════════════════════════════════════════════════════════════
print("Leakage-free LOPO (leave-one-AD-patient-out) ...")
lopo_global_site=np.zeros(n_ad,dtype=int)
lopo_pers_site  =np.zeros(n_ad,dtype=int)
lopo_thr        =np.zeros(n_ad)
lopo_base_score =np.zeros(n_ad)
for pi,p in enumerate(tqdm(ad,desc="  fold")):
    held=ad_idx[pi]
    train_pos=np.array([j for j in range(Npat) if j!=held])
    M=KernelLDA(train_pos, plabel[train_pos])
    lopo_thr[pi]=M.thr
    lopo_base_score[pi]=M.score(d_base[held])
    # per-site reduction for the held-out patient under the TRAIN-only discriminant
    red_held=np.array([lopo_base_score[pi]-M.score(d_ref[pi,k]) for k in range(N_SITES)])
    lopo_pers_site[pi]=int(np.argmax(red_held))
    # global site = argmax over sites of mean reduction across TRAIN AD patients
    train_ad=[j for j in range(n_ad) if j!=pi]
    base_tr=np.array([M.score(d_base[ad_idx[j]]) for j in train_ad])
    red_tr=np.array([[base_tr[a]-M.score(d_ref[train_ad[a],k]) for k in range(N_SITES)]
                     for a in range(len(train_ad))])
    lopo_global_site[pi]=int(np.argmax(red_tr.mean(0)))

# transductive model/threshold is the same for all patients
def tm(pi): return M_all, M_all.thr
print("Caching LOPO fold models ...")
lopo_models=[]
for pi,p in enumerate(ad):
    held=ad_idx[pi]; train_pos=np.array([j for j in range(Npat) if j!=held])
    lopo_models.append(KernelLDA(train_pos, plabel[train_pos]))
def lm(pi): return lopo_models[pi], lopo_models[pi].thr

# ── per-patient CC-side booleans, baseline and stimulated ─────────────────────
def cc_side(get_site, get_model_thr, amp):
    """Boolean array: is each AD patient scored on the CC side after stim at amp?"""
    out=np.zeros(n_ad,bool)
    for pi in range(n_ad):
        M,thr=get_model_thr(pi)
        out[pi]=M.score(d_stim(pi,get_site(pi),amp))<thr
    return out

base_cc_trans=np.array([M_all.score(d_base[ad_idx[pi]])<M_all.thr for pi in range(n_ad)])
base_cc_lopo =np.array([lopo_models[pi].score(d_base[ad_idx[pi]])<lopo_models[pi].thr for pi in range(n_ad)])
base_recl_trans=base_cc_trans.mean(); base_recl_lopo=base_cc_lopo.mean()

def net_cure(stim_cc, base_cc):
    """Among patients on the AD side at baseline, fraction moved to the CC side."""
    need=~base_cc; n=need.sum()
    return (stim_cc & need).sum()/n if n else float("nan"), int(n)

print("\nComputing reclassification + net cure vs amplitude ...")
results={}
SEL={"trans_global":(lambda pi: global_all, tm, base_cc_trans),
     "trans_pers":  (lambda pi: pers_all[pi], tm, base_cc_trans),
     "lopo_global": (lambda pi: lopo_global_site[pi], lm, base_cc_lopo),
     "lopo_pers":   (lambda pi: lopo_pers_site[pi], lm, base_cc_lopo)}
for amp in AMPS_EVAL:
    r={}
    for key,(gs,gm,bcc) in SEL.items():
        scc=cc_side(gs,gm,amp)
        raw=scc.mean(); nc,nneed=net_cure(scc,bcc)
        r[key+"_raw"]=raw; r[key+"_net"]=nc; r[key+"_nneed"]=nneed
    results[amp]=r
    print(f"  A={amp:4.1f} | raw reclassified %  "
          f"T-glob {r['trans_global_raw']*100:5.1f}  T-pers {r['trans_pers_raw']*100:5.1f} || "
          f"L-glob {r['lopo_global_raw']*100:5.1f}  L-pers {r['lopo_pers_raw']*100:5.1f}")
    print(f"        | net cure (of AD-side) %  "
          f"T-glob {r['trans_global_net']*100:5.1f}  T-pers {r['trans_pers_net']*100:5.1f} || "
          f"L-glob {r['lopo_global_net']*100:5.1f}  L-pers {r['lopo_pers_net']*100:5.1f}")

print(f"\nbaseline (unstim) misclassified as CC: transductive {base_recl_trans*100:.1f}%  LOPO {base_recl_lopo*100:.1f}%")
print(f"  -> under honest CV {int((~base_cc_lopo).sum())}/{n_ad} AD patients are on the AD side at baseline (need stimulation)")
print(f"global-site agreement transductive-vs-LOPO: "
      f"{np.mean(lopo_global_site==global_all)*100:.0f}% of folds pick site {global_all}")
print(f"personalised-site agreement: {np.mean(lopo_pers_site==pers_all)*100:.0f}% of patients unchanged under LOPO")

np.savez("pert_lopo_target_data.npz",
         amps_eval=np.array(AMPS_EVAL), f1=f1, n_ad=n_ad,
         global_all=global_all, pers_all=pers_all,
         lopo_global_site=lopo_global_site, lopo_pers_site=lopo_pers_site,
         base_recl_trans=base_recl_trans, base_recl_lopo=base_recl_lopo,
         base_cc_trans=base_cc_trans, base_cc_lopo=base_cc_lopo,
         **{f"A{amp}_{k}":v for amp,r in results.items() for k,v in r.items()})
print("Saved pert_lopo_target_data.npz")

# ── figure ─────────────────────────────────────────────────────────────────────
#  (A) baseline vs stimulated reclassification at a reference amplitude, per scheme
#      -> shows honest CV lifts the baseline misclassification and how far stim adds
#  (B) net cure among AD-side-at-baseline patients vs amplitude, transductive vs LOPO
plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.labelsize":10.5,
    "axes.titlesize":10.5,"legend.fontsize":8.5,"figure.dpi":300,"savefig.dpi":300,
    "axes.spines.top":False,"axes.spines.right":False})
A=np.array(AMPS_EVAL); A_ref=7.0; ridx=AMPS_EVAL.index(A_ref)
CT="#455A64"; CL="#C62828"
fig,axes=plt.subplots(1,2,figsize=(12,4.7))

# panel A: grouped bars baseline vs stimulated at A_ref
ax=axes[0]
groups=[("Transductive\nglobal","trans_global",base_recl_trans,CT),
        ("LOPO\nglobal","lopo_global",base_recl_lopo,CL),
        ("Transductive\npersonalised","trans_pers",base_recl_trans,CT),
        ("LOPO\npersonalised","lopo_pers",base_recl_lopo,CL)]
x=np.arange(len(groups)); w=0.38
for i,(lab,key,base,col) in enumerate(groups):
    ax.bar(x[i]-w/2, base*100, w, color=col, alpha=0.35, edgecolor=col,
           label="baseline (unstim)" if i==0 else None)
    ax.bar(x[i]+w/2, results[A_ref][key+"_raw"]*100, w, color=col, alpha=0.9,
           label="after stimulation" if i==0 else None)
    ax.annotate(f"+{(results[A_ref][key+'_raw']-base)*100:.0f}", (x[i]+w/2, results[A_ref][key+'_raw']*100+1.5),
                ha="center", fontsize=8, color=col, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_ylabel("AD scored as CC (%)"); ax.set_ylim(0,108)
ax.set_title(f"Baseline vs stimulated reclassification ($A={A_ref:g}$)")
ax.legend(frameon=False, loc="upper left")
ax.text(-0.09,1.06,"A",transform=ax.transAxes,fontsize=13,fontweight="bold")

# panel B: net cure among AD-side patients vs amplitude
ax=axes[1]
for mode,ls,mk in [("global","-","o"),("pers","--","s")]:
    ax.plot(A,[results[a][f"trans_{mode}_net"]*100 for a in A],ls,color=CT,marker=mk,lw=2,ms=6,
            label=f"transductive, {'global' if mode=='global' else 'personalised'}")
    ax.plot(A,[results[a][f"lopo_{mode}_net"]*100 for a in A],ls,color=CL,marker=mk,lw=2,ms=6,
            label=f"LOPO, {'global' if mode=='global' else 'personalised'}")
ax.set_xlabel("resonant drive amplitude $A$")
ax.set_ylabel("net cure among AD-side patients (%)")
ax.set_ylim(-3,105); ax.set_title("Net therapeutic effect (leakage-free vs transductive)")
ax.legend(frameon=False, loc="lower right", ncol=1)
ax.text(-0.09,1.06,"B",transform=ax.transAxes,fontsize=13,fontweight="bold")

fig.suptitle("The therapeutic reclassification effect survives a leakage-free (leave-one-AD-patient-out) "
             "discriminant\n(honest cross-validation raises the baseline misclassification; the stimulation "
             "effect is the increment over that baseline)",
             fontsize=10.6,fontweight="bold",y=1.02)
fig.tight_layout()
for ext in ("png","pdf"):
    fig.savefig(f"{OUT}/figure_lopo_target.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved {OUT}/figure_lopo_target.{ext}")
plt.close(fig); print("Done.")
