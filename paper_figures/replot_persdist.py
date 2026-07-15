"""
replot_persdist.py — distribution over patients of the personalised LDA-resonant
single-site target (from pert_compare3_data.npz). No experiment re-run.
"""
import os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread

d = np.load("../pert_compare3_data.npz", allow_pickle=True)
pc = d["pers_counts"].astype(int)
coords = np.load("../pert_sites_data.npz", allow_pickle=True)["parcel_coords"]

def load_labels(path="../timeseries/parcel_labels.txt"):
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

n_ad=int(pc.sum()); ndist=int((pc>0).sum())
print(f"{n_ad} patients, {ndist} distinct personalised sites (max {pc.max()} patients/site)")
order=np.argsort(pc)[::-1]; sel=order[pc[order]>0]
for k in sel: print(f"  site {k:3d} {short(labels.get(k)):22s} {pc[k]}")

# brain
from nilearn import plotting
disp=plotting.plot_markers(pc[sel].astype(float), coords[sel],
     node_size=40+40*pc[sel], node_cmap="autumn_r", node_vmin=0,
     node_vmax=float(pc.max()), display_mode="lzry", alpha=0.9, colorbar=True,
     title=f"Personalised LDA-resonant target ({ndist} distinct sites, N={n_ad})")
disp.savefig("figureS5_brain.png", dpi=300); disp.savefig("figureS5_brain.pdf"); disp.close()

plt.rcParams.update({"font.family":"sans-serif","font.size":9,"axes.labelsize":9,
    "axes.titlesize":10,"xtick.labelsize":8,"ytick.labelsize":8,"legend.fontsize":8,
    "figure.dpi":300,"savefig.dpi":300,"axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(14,4.4),facecolor="white")
gs=gridspec.GridSpec(1,2,figure=fig,width_ratios=[1,1.3],wspace=0.22,
                     left=0.07,right=0.99,top=0.84,bottom=0.34)
ax=fig.add_subplot(gs[0,0])
ax.bar(range(len(sel)),pc[sel],color="#C2185B",alpha=0.85)
ax.set_xticks(range(len(sel))); ax.set_xticklabels([short(labels.get(k)) for k in sel],
              rotation=60,ha="right",fontsize=6)
ax.set_ylabel(f"# patients (of {n_ad})")
ax.set_title(f"Personalised single-site target ({ndist} distinct sites)")
ax.text(-0.12,1.05,"A",transform=ax.transAxes,fontsize=13,fontweight="bold")
ax=fig.add_subplot(gs[0,1]); ax.imshow(imread("figureS5_brain.png")); ax.axis("off")
ax.set_title("Anatomical distribution across patients",pad=2)
ax.text(-0.02,1.03,"B",transform=ax.transAxes,fontsize=13,fontweight="bold")
fig.suptitle("Distribution over patients of the personalised LDA-resonant "
             "stimulation site (one site per patient)",fontsize=10.5,fontweight="bold",y=0.99)
for ext in ("png","pdf"):
    fig.savefig(f"figureS5_persdist.{ext}",dpi=300,bbox_inches="tight",facecolor="white")
    print(f"Saved figureS5_persdist.{ext}")
plt.close(fig)
