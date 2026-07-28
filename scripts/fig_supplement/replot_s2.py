"""replot_s2.py — Fig S2 FC-lag only (G-space panels removed). From cached npz."""
import sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

d=np.load("../pert_compare3_data.npz", allow_pickle=True)
thr=float(d["thr_f"]); cc=d["cc_f"]; n_ad=d["F_single"].shape[1]
dose=np.linspace(0,1,d["F_single"].shape[0])
STR=[("single",d["F_single"],"#6A1B9A","single-site ($\\Delta W$, top-1)"),
     ("osc",d["F_osc"],"#E65100","resonant osc (top-1 $\\Delta W$, $f_1$)"),
     ("eig2",d["F_eig2"],"#00838F","2-site eigenmode ($f_1$@s1,$f_2$@s2)"),
     ("ld",d["F_ld"],"#2E7D32","LDA-resonant, average (global site)"),
     ("ldp",d["F_ldp"],"#C2185B","LDA-resonant, personalised (per-patient)")]
def cure(FL): return (FL<thr).mean(1)*100

plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(11,4.4),facecolor="white")
gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.27,left=0.08,right=0.97,top=0.86,bottom=0.15)
def tag(ax,t): ax.text(-0.13,1.04,t,transform=ax.transAxes,fontsize=13,fontweight="bold",va="bottom")

ax=fig.add_subplot(gs[0,0])
ax.axhspan(cc.mean()-cc.std(),cc.mean()+cc.std(),alpha=0.1,color="#1565C0")
ax.axhline(cc.mean(),color="#1565C0",ls="--",lw=1.2,label="CC mean ±1σ")
ax.axhline(thr,color="gray",ls="-.",lw=1,label="boundary")
for k,FL,c,l in STR:
    m=FL.mean(1); e=FL.std(1)/np.sqrt(n_ad)
    ax.fill_between(dose,m-e,m+e,color=c,alpha=0.13); ax.plot(dose,m,"-o",ms=3.5,color=c,lw=2,label=l)
ax.set_xlabel("relative stimulation dose"); ax.set_ylabel("FC-lag LDA score")
ax.set_title("FC-lag score vs dose"); ax.legend(frameon=False,fontsize=6.8); tag(ax,"A")

ax=fig.add_subplot(gs[0,1])
for k,FL,c,l in STR: ax.plot(dose,cure(FL),"-o",ms=3.5,color=c,lw=2,label=l)
ax.axhline(cure(STR[0][1][:1])[0],color="#C62828",ls=":",lw=1.2,label="baseline")
ax.set_xlabel("relative stimulation dose"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Patients reclassified vs dose"); ax.set_ylim(-2,105)
ax.legend(frameon=False,fontsize=6.8); tag(ax,"B")

fig.suptitle("Focal oscillatory stimulation (FC-lag): site-selection criterion and personalisation",
             fontsize=11,fontweight="bold",y=0.98)
for ext in ("png","pdf"):
    fig.savefig(f"figureS2_compare.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved figureS2_compare.{ext}")
plt.close(fig)
