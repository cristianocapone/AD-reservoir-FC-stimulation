"""replot_S7.py — Fig S7: site vs mode/frequency selection.
(A) site-robustness at f1; (B) eigenmode CC/AD discriminability;
(C,D) single- vs two-frequency drive at a fixed site. From cached npz (no re-run)."""
import sys, io, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

m=np.load("../pert_mode_freq_data.npz",allow_pickle=True)
t=np.load("../pert_twofreq_data.npz",allow_pickle=True)
c=np.load("../pert_compare3_data.npz",allow_pickle=True)
coords=np.load("../pert_sites_data.npz",allow_pickle=True)["parcel_coords"]
recl_site=m["recl_site"]; f1=float(m["f1"]); site_eig=int(m["site_eig"])
fk=m["fk"]; auc=m["auc"]; ki_f1=int(m["ki_f1"]); kbest=int(np.argmin(auc))
AMPS=t["amps"]; thr=float(t["thr_f"]); f_disc=float(t["f_disc"]); cc_f=t["cc_f"]
S={"f1":t["S_f1"],"fdisc":t["S_fdisc"],"both":t["S_both"]}; n_ad=S["f1"].shape[1]
recl={k:(S[k]<thr).mean(1)*100 for k in S}
# signed per-site effect at f1 (red_full>0 = toward CC): mean across patients
red=c["red_full"]; site_red=red.mean(1); fbase=c["F_single"][0]; thr_c=float(c["thr_f"])
n_neg=int((site_red<0).sum()); frac_worse=float(np.mean(red<0)*100)
kworst=int(np.argmin(site_red))
print(f"net-harmful sites: {n_neg}/121 (worst site {kworst}, mean {site_red[kworst]:+.3f}); "
      f"{frac_worse:.0f}% of (site,patient) drives worsen")
sf=S["f1"]
print(f"variance-collapse check (eigenmode site, f1): baseline score {sf[0].mean():+.2f}+-{sf[0].std():.2f}"
      f"  ->  A=10 {sf[-1].mean():+.2f}+-{sf[-1].std():.2f}  (boundary {thr:+.2f}, CC mean {cc_f.mean():+.2f})")

plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9.5,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":7.5,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
COL={"f1":"#C62828","fdisc":"#6A1B9A","both":"#00838F"}
LBL={"f1":f"$f_1$ resonant ({f1:.3f})","fdisc":f"$f_{{disc}}$ ({f_disc:.3f})",
     "both":"$f_1+f_{disc}$ (energy-matched)"}
# signed-effect glass brain (panel E)
from nilearn import plotting
disp=plotting.plot_markers(site_red, coords, node_size=45, node_cmap="coolwarm",
     node_vmin=-1.0, node_vmax=1.0, display_mode="lzry", alpha=0.9, colorbar=True,
     title="Per-site mean effect at f1 (red: toward CC, blue: toward AD)")
disp.savefig("_s7_brain.png", dpi=220); disp.close()

fig=plt.figure(figsize=(11,11.6),facecolor="white")
gs=gridspec.GridSpec(3,2,figure=fig,height_ratios=[1,1,0.85],wspace=0.27,hspace=0.40,
                     left=0.08,right=0.98,top=0.93,bottom=0.05)
def tag(ax,s): ax.text(-0.14,1.05,s,transform=ax.transAxes,fontsize=13,fontweight="bold")

# A — site robustness
ax=fig.add_subplot(gs[0,0])
ax.hist(recl_site,bins=np.arange(0,101,7),color="#5C6BC0",alpha=0.85,edgecolor="white")
ax.axvline(recl_site.mean(),color="k",ls="--",lw=1.3,label=f"mean {recl_site.mean():.0f}%")
ax.axvline(recl_site[site_eig],color="#00838F",lw=2,label=f"eigenmode site 71 ({recl_site[site_eig]:.0f}%)")
ax.axvline(100,color="#C2185B",lw=2,label="LDA per-patient (100%)")
ax.set_xlabel("AD reclassified as CC (%)"); ax.set_ylabel("# of 121 driven sites")
ax.set_title(f"Site matters at fixed $f_1$ ($A=4$)"); ax.legend(frameon=False); tag(ax,"A")

# B — mode discriminability
ax=fig.add_subplot(gs[0,1])
ax.axhline(0.5,color="gray",ls="-.",lw=1)
ax.scatter(fk,auc,s=24,color="#455A64",alpha=0.8)
ax.scatter([fk[ki_f1]],[auc[ki_f1]],s=70,color="#C62828",zorder=5,label=f"dominant $f_1$={f1:.3f}")
ax.scatter([fk[kbest]],[auc[kbest]],s=80,facecolors="none",edgecolors="#6A1B9A",lw=2,zorder=5,
           label=f"$f_{{disc}}$ (min AUC, f={fk[kbest]:.3f})")
ax.set_xlabel("mode frequency (cycles/step)"); ax.set_ylabel("CC-vs-AD AUC of modal power")
ax.set_title("Discriminative mode $\\neq$ dominant mode"); ax.legend(frameon=False); tag(ax,"B")

# C — single vs two-frequency score
ax=fig.add_subplot(gs[1,0])
ax.axhline(thr,color="gray",ls="-.",lw=1,label="boundary")
ax.axhline(cc_f.mean(),color="#1565C0",ls="--",lw=1.2,label="CC mean")
for k in S:
    mu=S[k].mean(1); e=S[k].std(1)/np.sqrt(n_ad)
    ax.fill_between(AMPS,mu-e,mu+e,color=COL[k],alpha=0.13); ax.plot(AMPS,mu,"-o",ms=4,color=COL[k],lw=2,label=LBL[k])
ax.set_xlabel("stimulation amplitude $A$"); ax.set_ylabel("FC-lag LDA score")
ax.set_title(f"Single- vs two-frequency drive (site {site_eig})"); ax.legend(frameon=False); tag(ax,"C")

# D — reclassification
ax=fig.add_subplot(gs[1,1])
for k in S: ax.plot(AMPS,recl[k],"-o",ms=4,color=COL[k],lw=2,label=LBL[k])
ax.set_xlabel("stimulation amplitude $A$"); ax.set_ylabel("AD reclassified as CC (%)")
ax.set_title("Reclassification vs amplitude"); ax.set_ylim(-2,105); ax.legend(frameon=False); tag(ax,"D")

# E — signed glass brain
from matplotlib.image import imread
ax=fig.add_subplot(gs[2,:]); ax.imshow(imread("_s7_brain.png")); ax.axis("off")
ax.set_title(f"Almost every site is therapeutic on average at $f_1$ "
             f"({121-n_neg}/121 toward CC; only site {kworst} net-harmful), "
             f"but {frac_worse:.0f}% of individual drives worsen the patient",pad=4,fontsize=9.5)
ax.text(-0.02,1.04,"E",transform=ax.transAxes,fontsize=13,fontweight="bold")

fig.suptitle("Site vs mode/frequency selection: efficacy needs a drivable (resonant) mode AND the right site",
             fontsize=11,fontweight="bold",y=0.985)
for ext in ("png","pdf"):
    fig.savefig(f"figureS7_modefreq.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved figureS7_modefreq.{ext}")
plt.close(fig)
