"""
replot_figure5_topsites.py
===========================
Extend Figure 5 into an explicit comparison: the sites where the AD read-out
deviates most (pathology, Delta-W) vs the sites where single-site resonant
stimulation is most effective (therapy, LDA-resonant / "stim sites"), showing
that the two site sets are largely disjoint. No resimulation: reuses cached
per-site counts and MNI coordinates.

Data:
  Delta-W (pathology) top-5 / top-1 per-patient selection counts and parcel
  coordinates: ../pert_sites_data.npz (top5_site_counts, top1_site_counts,
  parcel_coords), from pert_sites_stimulation.py (same seeded reservoir/W-fit
  as this figure's original version).

  LDA-resonant (therapy) per-patient top-1 site (argmax over 121 candidate
  sites of the resonant-drive FC-lag reduction toward CC), and the full
  121 x 40 reduction matrix: ../pert_compare3_data.npz (pers_counts, red_full),
  from pert_compare3.py.

Panels:
  A  Glass brain, Delta-W top-5 selection frequency (pathology)
  B  Selection-frequency bars, Delta-W (top-5 orange, top-1 purple)
  C  Overlap (2-set Venn): Delta-W top-5-union sites vs LDA-resonant top-1
     sites -- the great majority of sites in each set are exclusive to that
     criterion
  D  Glass brain, LDA-resonant top-1 selection frequency (therapy)
  E  Selection-frequency bars, LDA-resonant top-1

Saves: figure5_topsites.{png,pdf}
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.image import imread
from matplotlib_venn import venn2
import warnings; warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9,
    "axes.labelsize": 9, "axes.titlesize": 10,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ── parcel labels ─────────────────────────────────────────────────────────────
def load_labels(path="../timeseries/parcel_labels.txt"):
    lab = {}
    with open(path) as f:
        for line in f:
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].strip().isdigit():
                lab[int(parts[0]) - 1] = parts[1].strip()
    return lab

def short_label(name):
    if name is None:
        return "?"
    name = name.replace("Left ", "L ").replace("Right ", "R ")
    if name.startswith("7Networks_"):
        p = name.replace("7Networks_", "").split("_")
        return (p[0] + " " + "".join(p[1:])) if len(p) > 1 else p[0]
    return name

labels = load_labels()

# ── cached data (no resimulation) ──────────────────────────────────────────────
sites = np.load("../pert_sites_data.npz", allow_pickle=True)
counts5 = sites["top5_site_counts"]; counts1 = sites["top1_site_counts"]
parcel_coords = sites["parcel_coords"]
N_SITES = len(counts5)

cmp3 = np.load("../pert_compare3_data.npz", allow_pickle=True)
pers_counts = cmp3["pers_counts"].astype(int)   # per-patient top-1 LDA-resonant site
n_ad = int(pers_counts.sum())

dw5_set = set(np.where(counts5 > 0)[0])          # Delta-W top-5-union sites
stim_set = set(np.where(pers_counts > 0)[0])      # LDA-resonant top-1 sites
overlap_sites = sorted(dw5_set & stim_set)
n_dw_only = len(dw5_set - stim_set)
n_stim_only = len(stim_set - dw5_set)
n_overlap = len(overlap_sites)
print(f"Delta-W top-5-union sites: {len(dw5_set)}  |  LDA-resonant top-1 sites: {len(stim_set)}")
print(f"Overlap: {n_overlap} sites {[short_label(labels.get(s)) for s in overlap_sites]}")
print(f"Delta-W-only: {n_dw_only}  |  shared: {n_overlap}  |  stim-only: {n_stim_only}")

order5 = np.argsort(counts5)[::-1]; sel5 = order5[counts5[order5] > 0]
order_p = np.argsort(pers_counts)[::-1]; sel_p = order_p[pers_counts[order_p] > 0]

# ══════════════════════════════════════════════════════════════════════════════
# BRAIN RENDERS
# ══════════════════════════════════════════════════════════════════════════════
print("\nRendering glass brains ...")
from nilearn import plotting

sel = counts5 > 0
disp = plotting.plot_markers(
    node_values=counts5[sel].astype(float), node_coords=parcel_coords[sel],
    node_size=25 + 10 * counts5[sel], node_cmap="YlOrRd",
    node_vmin=0, node_vmax=float(counts5.max()),
    display_mode="lzry", alpha=0.85, colorbar=True,
    title="top-5 selection frequency")
disp.savefig("figure5_brain.png", dpi=300)
disp.savefig("figure5_brain.pdf")
disp.close()

sel = pers_counts > 0
disp = plotting.plot_markers(
    node_values=pers_counts[sel].astype(float), node_coords=parcel_coords[sel],
    node_size=25 + 10 * pers_counts[sel], node_cmap="RdPu",
    node_vmin=0, node_vmax=float(pers_counts.max()),
    display_mode="lzry", alpha=0.85, colorbar=True,
    title="top-1 selection frequency")
disp.savefig("figure5_stimbrain.png", dpi=300)
disp.savefig("figure5_stimbrain.pdf")
disp.close()
print("Saved figure5_brain.{png,pdf} and figure5_stimbrain.{png,pdf}")

# ══════════════════════════════════════════════════════════════════════════════
# COMPOSITE FIGURE 5
# ══════════════════════════════════════════════════════════════════════════════
print("Rendering composite figure ...")
fig = plt.figure(figsize=(15.5, 8.2), facecolor="white")
gs = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.05, 1.0, 0.85],
                        height_ratios=[1.0, 1.0], hspace=0.48, wspace=0.32,
                        left=0.055, right=0.98, top=0.88, bottom=0.11)

def _tag(ax, t, x=-0.08, y=1.05):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=13,
             fontweight="bold", va="bottom", ha="left")

# ── A: Delta-W glass brain (pathology) ─────────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
ax_a.imshow(imread("figure5_brain.png")); ax_a.axis("off")
ax_a.set_title("Pathology: $\\Delta W$ top-5 sites", pad=2, color="#B71C1C")
_tag(ax_a, "A", x=-0.04, y=1.02)

# ── B: Delta-W selection-frequency bars ────────────────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
nb = min(15, len(sel5))
xs = np.arange(nb)
ax_b.bar(xs - 0.2, counts5[sel5[:nb]], 0.4, color="#E65100", alpha=0.85, label="in top-5")
ax_b.bar(xs + 0.2, counts1[sel5[:nb]], 0.4, color="#6A1B9A", alpha=0.85, label="top-1")
ax_b.set_xticks(xs)
ax_b.set_xticklabels([short_label(labels.get(s)) for s in sel5[:nb]],
                      rotation=55, ha="right", fontsize=6.5)
ax_b.set_ylabel(f"# AD patients (of {n_ad})")
ax_b.set_title("Pathology: site-selection frequency")
ax_b.legend(frameon=False, fontsize=7.5)
_tag(ax_b, "B", x=-0.14)

# ── C: overlap (Venn) ──────────────────────────────────────────────────────────
ax_c = fig.add_subplot(gs[:, 2])
ax_c.set_position([ax_c.get_position().x0, ax_c.get_position().y0 + 0.16,
                    ax_c.get_position().width, ax_c.get_position().height * 0.62])
v = venn2(subsets=(n_dw_only, n_stim_only, n_overlap),
          set_labels=(f"$\\Delta W$ pathology\nsites (top-5 union, N={len(dw5_set)})",
                       f"LDA-resonant\ntherapy sites (top-1, N={len(stim_set)})"),
          set_colors=("#E65100", "#C2185B"), alpha=0.55, ax=ax_c)
for txt in v.subset_labels:
    if txt is not None:
        txt.set_fontsize(13); txt.set_fontweight("bold")
for txt in v.set_labels:
    if txt is not None:
        txt.set_fontsize(8.5)
ax_c.set_title("Pathology vs therapy sites\nlargely disjoint", pad=8)
_tag(ax_c, "C", x=0.0, y=1.16)

overlap_names = [short_label(labels.get(s)) for s in overlap_sites]
half = (len(overlap_names) + 1) // 2
col1 = "\n".join(overlap_names[:half]); col2 = "\n".join(overlap_names[half:])
ax_c.text(0.5, -0.28, f"{n_overlap} shared sites:", transform=ax_c.transAxes,
           ha="center", va="top", fontsize=8, fontweight="bold")
ax_c.text(0.30, -0.36, col1, transform=ax_c.transAxes, ha="center", va="top", fontsize=7)
ax_c.text(0.70, -0.36, col2, transform=ax_c.transAxes, ha="center", va="top", fontsize=7)

# ── D: LDA-resonant glass brain (therapy) ──────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
ax_d.imshow(imread("figure5_stimbrain.png")); ax_d.axis("off")
ax_d.set_title("Therapy: LDA-resonant top-1 sites", pad=2, color="#880E4F")
_tag(ax_d, "D", x=-0.04, y=1.02)

# ── E: LDA-resonant selection-frequency bars ────────────────────────────────────
ax_e = fig.add_subplot(gs[1, 1])
ne = min(15, len(sel_p))
xs = np.arange(ne)
ax_e.bar(xs, pers_counts[sel_p[:ne]], 0.55, color="#C2185B", alpha=0.85,
          label="top-1 (personalised)")
ax_e.set_xticks(xs)
ax_e.set_xticklabels([short_label(labels.get(s)) for s in sel_p[:ne]],
                      rotation=55, ha="right", fontsize=6.5)
ax_e.set_ylabel(f"# AD patients (of {n_ad})")
ax_e.set_title("Therapy: site-selection frequency")
ax_e.legend(frameon=False, fontsize=7.5)
_tag(ax_e, "E", x=-0.14)

fig.suptitle(
    "Pathology sites vs therapeutic-stimulation sites: where the read-out "
    "deviates most is not where single-site resonant stimulation is most "
    f"effective\n(N={n_ad} AD; $\\Delta W$ = per-patient top-5 read-out "
    "correction magnitude; LDA-resonant = per-patient top-1 site, chosen by "
    "its effect on the disease discriminant under resonant drive)",
    fontsize=11.3, fontweight="bold", y=0.985)

for ext in ("png", "pdf"):
    fig.savefig(f"figure5_topsites.{ext}", dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure5_topsites.{ext}")
plt.close(fig)
print("\nDone.")
