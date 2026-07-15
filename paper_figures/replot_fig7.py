"""replot_fig7.py — condensed MAIN-TEXT closed-loop vs open-loop figure
from pert_closedloop_data.npz. Two panels: (A) distance among reclassified patients
for the three control schemes (all 100% reclassified); (B) per-patient open-loop vs
closed-loop distance."""
import sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

d=np.load("../pert_closedloop_data.npz",allow_pickle=True)
ol=d["ol_dist"]; ca=d["ca_dist"]; cs=d["cs_dist"]
olr=d["ol_rec"]; car=d["ca_rec"]; csr=d["cs_rec"]; A_FIX=float(d["A_FIX"])
R=[(ol,olr,"open-loop\n(fixed $A$)","#455A64"),
   (ca,car,"CL amplitude","#1565C0"),
   (cs,csr,"CL amp+site","#2E7D32")]

plt.rcParams.update({"font.family":"sans-serif","font.size":9.5,"axes.labelsize":10,
    "axes.titlesize":10.5,"xtick.labelsize":8.5,"ytick.labelsize":8.5,"legend.fontsize":8.5,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(9.2,4.2),facecolor="white")
gs=gridspec.GridSpec(1,2,figure=fig,wspace=0.30,left=0.09,right=0.985,top=0.82,bottom=0.17)
def tag(ax,s): ax.text(-0.17,1.05,s,transform=ax.transAxes,fontsize=13,fontweight="bold")

ax=fig.add_subplot(gs[0,0])
data=[r[~np.isnan(r)] for r,_,_,_ in R]
vp=ax.violinplot(data,positions=range(3),showmedians=True,widths=0.75)
for b,(_,_,_,c) in zip(vp["bodies"],R): b.set_facecolor(c); b.set_alpha(0.5)
for kk in ["cmedians","cbars","cmins","cmaxes"]: vp[kk].set_color("k"); vp[kk].set_linewidth(1.0)
rng=np.random.default_rng(0)
for xi,(r,rec,_,c) in enumerate(R):
    rr=r[~np.isnan(r)]; ax.scatter(xi+rng.uniform(-0.08,0.08,len(rr)),rr,s=10,c=c,alpha=0.6,edgecolors="none")
    ax.text(xi,1.02,f"{np.mean(rec)*100:.0f}%\ncured",ha="center",va="bottom",fontsize=8,color="#333",
            transform=ax.get_xaxis_transform())
    ax.text(xi,np.nanmean(r),f"{np.nanmean(r):.2f}",ha="center",va="center",fontsize=8,
            fontweight="bold",bbox=dict(boxstyle="round,pad=0.15",fc="white",ec="none",alpha=0.8))
ax.set_xticks(range(3)); ax.set_xticklabels([x[2] for x in R])
ax.set_ylabel("distance from original FC  ($1-$corr)")
ax.set_ylim(0.15,1.05); tag(ax,"A")

ax=fig.add_subplot(gs[0,1])
m=(~np.isnan(ol))&(~np.isnan(cs))
ax.scatter(ol[m],cs[m],s=26,color="#2E7D32",alpha=0.75,edgecolors="none")
lim=[0,max(np.nanmax(ol),np.nanmax(cs))*1.05]
ax.plot(lim,lim,"k--",lw=1,alpha=0.6); ax.fill_between(lim,lim,[lim[1],lim[1]],color="#2E7D32",alpha=0.06)
ax.text(0.62,0.18,"closed-loop\nbetter",transform=ax.transAxes,fontsize=8.5,color="#2E7D32",ha="center")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("open-loop distance (fixed $A$)"); ax.set_ylabel("closed-loop distance (amp+site)")
ax.set_title("Per-patient improvement"); tag(ax,"B")

fig.suptitle("Closed-loop, biomarker-titrated stimulation reverts AD at lower cost than open-loop",
             fontsize=11.5,fontweight="bold",y=0.97)
for ext in ("png","pdf"):
    fig.savefig(f"figure7_closedloop.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved figure7_closedloop.{ext}")
plt.close(fig)
