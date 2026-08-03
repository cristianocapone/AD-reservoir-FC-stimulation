"""Generate evocative conceptual figures (vector PDF) for the talk.
Reuses the HTML deck's brain-network motif: peripheral cortical hubs + central
weakly-connected subcortical nodes. Palette matches the slides."""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

TEAL = "#0e8c7f"; BLUE = "#1565c0"; RED = "#c62828"
INK = "#16202f"; FAINT = "#9aa7b5"; GRID = "#cfd8e3"
OUT = os.path.join(os.path.dirname(__file__), "figs")
os.makedirs(OUT, exist_ok=True)

# node coords (y flipped to y-up), hubs then weak
H = [(70,80),(140,44),(255,42),(345,80),(362,168),(300,238),(150,242),(62,176)]
W = [(196,120),(172,150),(224,158),(205,186)]
NODES = [(x, 300-y) for (x,y) in H] + [(x, 300-y) for (x,y) in W]
NHUB = len(H)
EDGES = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0),
         (0,2),(1,3),(7,4),(0,5),(6,3),(1,7),(2,5),
         (8,9),(10,11),(9,1),(8,2)]
deg = [0]*len(NODES)
for a,b in EDGES: deg[a]+=1; deg[b]+=1

def base_ax(w=5.2,h=3.7):
    fig,ax = plt.subplots(figsize=(w,h)); ax.set_xlim(30,395); ax.set_ylim(20,285)
    ax.set_aspect("equal"); ax.axis("off"); return fig,ax

def draw_net(ax, mode="plain"):
    for a,b in EDGES:
        col,lw,al = FAINT,1.0,0.55
        if mode=="graph": col,lw,al = TEAL,2.0,0.9
        ax.plot([NODES[a][0],NODES[b][0]],[NODES[a][1],NODES[b][1]],
                color=col,lw=lw,alpha=al,zorder=1,solid_capstyle="round")
    for i,(x,y) in enumerate(NODES):
        weak = i>=NHUB
        r,col = (7,FAINT)
        if mode=="dissociation": col,r = (RED,7) if weak else (TEAL,9)
        elif mode=="hubs":       r = 4+deg[i]*1.4; col = RED if weak else TEAL
        elif mode=="activity":   col,r = (TEAL,10) if i==3 else (FAINT,7)
        elif mode=="graph":      col = FAINT
        elif mode=="motivation": col,r = (RED,6.5) if weak else ("#e08a8a",7)
        # soft glow
        ax.scatter([x],[y],s=(r*4)**1.1,color=col,alpha=0.14,zorder=2,edgecolors="none")
        ax.scatter([x],[y],s=(r*3.4),color=col,zorder=3,edgecolors="white",linewidths=1.1)
        if mode=="activity" and i==3:
            for rr,aa in [(16,0.5),(26,0.28),(36,0.12)]:
                ax.add_patch(Circle((x,y),rr,fill=False,ec=TEAL,alpha=aa,lw=1.6,zorder=2))

# 1 MOTIVATION: distributed alteration + one focal target
fig,ax = base_ax()
draw_net(ax,"motivation")
tx,ty = NODES[3]
ax.add_patch(Circle((tx,ty),15,fill=False,ec=TEAL,lw=2.2,zorder=5))
ax.plot([tx-22,tx-15],[ty,ty],color=TEAL,lw=2.2); ax.plot([tx+15,tx+22],[ty,ty],color=TEAL,lw=2.2)
ax.plot([tx,tx],[ty-22,ty-15],color=TEAL,lw=2.2); ax.plot([tx,tx],[ty+15,ty+22],color=TEAL,lw=2.2)
ax.text(tx, ty+34, "stimulate?", color=TEAL, ha="center", va="bottom", fontsize=11, fontweight="bold")
fig.savefig(f"{OUT}/concept_motivation.pdf", bbox_inches="tight", transparent=True); plt.close(fig)

# 2 DIGITAL TWIN: fMRI trace -> model -> reproduced trace
fig,ax = plt.subplots(figsize=(6.6,2.9)); ax.set_xlim(0,400); ax.set_ylim(0,150); ax.axis("off")
t = np.linspace(0,120,300)
def trace(x0,col):
    y = 75 + 20*np.sin(t/9) + 12*np.sin(t/23+1)
    ax.plot(x0+t*0.9, y, color=col, lw=2.2, solid_capstyle="round")
trace(5,BLUE)
ax.text(60,20,"resting-state fMRI",color=FAINT,ha="center",fontsize=10)
ax.add_patch(FancyBboxPatch((150,55),70,50,boxstyle="round,pad=3,rounding_size=10",
            fill=False,ec=TEAL,lw=1.8)); ax.text(185,80,"model",color=TEAL,ha="center",va="center",fontsize=12,fontweight="bold")
for x0 in (128,240):
    ax.add_patch(FancyArrowPatch((x0,80),(x0+18,80),arrowstyle="-|>",mutation_scale=14,color=FAINT,lw=1.6))
trace(258,TEAL)
ax.text(315,20,"reproduced dynamics",color=FAINT,ha="center",fontsize=10)
fig.savefig(f"{OUT}/concept_twin.pdf", bbox_inches="tight", transparent=True); plt.close(fig)

# 3 TWO INTERVENTIONS: graph vs activity
fig,axes = plt.subplots(1,2,figsize=(7.4,3.4))
for ax in axes: ax.set_xlim(30,395); ax.set_ylim(20,285); ax.set_aspect("equal"); ax.axis("off")
draw_net(axes[0],"graph"); axes[0].set_title("perturb the graph", color=INK, fontsize=13, fontweight="bold", pad=10)
draw_net(axes[1],"activity"); axes[1].set_title("perturb the activity", color=INK, fontsize=13, fontweight="bold", pad=10)
# little sine at driven node in right panel
xx,yy = NODES[3]; s = np.linspace(0,20,40)
axes[1].plot(xx-45+s, yy+34+6*np.sin(s), color=TEAL, lw=1.8)
fig.savefig(f"{OUT}/concept_interventions.pdf", bbox_inches="tight", transparent=True); plt.close(fig)

# 4 WHY / HUBS: node size ~ degree, hubs teal vs weak red
fig,ax = base_ax(5.4,3.8)
draw_net(ax,"hubs")
ax.scatter([],[],s=90,color=TEAL,label="well-connected hub"); ax.scatter([],[],s=40,color=RED,label="weak node")
ax.legend(loc="lower center", bbox_to_anchor=(0.5,-0.14), ncol=2, frameon=False, fontsize=9,
          handletextpad=0.3, columnspacing=1.2, labelcolor=INK)
fig.savefig(f"{OUT}/concept_hubs.pdf", bbox_inches="tight", transparent=True); plt.close(fig)

print("wrote:", os.listdir(OUT))
