import json, re, os

os.makedirs('figures_MC_twoclasses', exist_ok=True)

# ── Load notebooks ────────────────────────────────────────────────────────────
with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb_part1 = json.load(f)

with open(r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML__.ipynb',
          encoding='utf-8-sig') as f:
    nb_orig = json.load(f)

# ── Helper: fix plt.pause + reduce epochs in a cell source ───────────────────
def fix_training_cell(src):
    # Reduce 10000 → 1000 epochs
    src = src.replace('trange(10000)', 'trange(1000)')
    src = src.replace('trange(30000)', 'trange(1000)')
    # Replace plt.pause with savefig + close so nbconvert captures plots
    src = src.replace(
        'plt.pause(0.01)',
        'plt.savefig(f"figures_MC_twoclasses/arch_train_{epoch:05d}.png", '
        'dpi=72, bbox_inches="tight"); plt.close("all")'
    )
    return src

# ── Data loading source (same as before) ─────────────────────────────────────
new_src_loading = (
    "import os\n"
    "import numpy as np\n"
    "from scipy.signal import butter, filtfilt\n"
    "\n"
    "def lowpass_filter_rows(X, fs, cutoff, order=4):\n"
    "    nyq = 0.5 * fs\n"
    "    normal_cutoff = cutoff / nyq\n"
    "    b, a = butter(order, normal_cutoff, btype='low')\n"
    "    return np.array([filtfilt(b, a, row) for row in X])\n"
    "\n"
    "fs = 1000\n"
    "cutoff = 150\n"
    "\n"
    r'timeseries_base = r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries"' + "\n"
    "\n"
    "T_STANDARD = 140\n"
    "N_PARCELS  = 121\n"
    "\n"
    "collected_signals = []\n"
    "identifiers_list = []\n"
    "\n"
    'for group, state_id in [("AD", "AD"), ("MCI", "MCI"), ("CN", "CC")]:\n'
    "    folder = os.path.join(timeseries_base, group)\n"
    "    if not os.path.isdir(folder):\n"
    "        continue\n"
    '    fnames = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])\n'
    "    for fname in fnames:\n"
    "        arr = np.load(os.path.join(folder, fname))\n"
    "        if arr.shape[1] != T_STANDARD:\n"
    '            print(f"Skipped (T={arr.shape[1]}) {fname}")\n'
    "            continue\n"
    "        if arr.shape[0] != N_PARCELS:\n"
    '            print(f"Skipped (N={arr.shape[0]}) {fname}")\n'
    "            continue\n"
    "        arr = arr.T\n"
    '        sub_id = fname.split("_")[0]\n'
    '        identifiers_list.append([state_id, sub_id, "Resting_State_fMRI"])\n'
    "        collected_signals.append(arr)\n"
    '        print(f"Loaded {fname}, shape={arr.shape}")\n'
    "\n"
    "identifiers = np.array(identifiers_list, dtype=object)\n"
)

# ── Patch cells ───────────────────────────────────────────────────────────────
cells = nb_orig['cells']
patched = []

for cell in cells:
    if cell['cell_type'] != 'code':
        continue
    cid = cell.get('id', '')
    src = ''.join(cell['source'])
    new_src = src

    if cid == '56c85796':
        new_src = new_src_loading
    elif cid == '3080171f':
        new_src = "patient_ID = [identifiers[k][1] for k in range(len(identifiers))]\n"
    elif cid == 'ff3bcf9b':
        pass  # keep == 121 (correct parcel count)
    elif cid == '8a830229':
        pass  # keep N_sites = 121 (correct parcel count)
    elif cid == 'c851c817':
        pass  # keep [121:,:121] (correct parcel count)
    elif cid in ('29dbf9b4', '601c995a'):   # slow training loops
        new_src = fix_training_cell(src)

    if new_src != src:
        cell['source'] = new_src
        patched.append(cid)

    cell['outputs'] = []
    cell['execution_count'] = None

print(f'Patched: {patched}')

# ── Replace slow function definitions with optimised versions ─────────────────
opt_res_class  = nb_part1['cells'][16]   # optimised RESERVOIRE_SIMPLE
opt_pinv       = nb_part1['cells'][17]   # optimised train_test_pinv
opt_train_test = nb_part1['cells'][18]   # optimised train_test

for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    if 'def train_test_pinv' in src:
        cells[i] = {**opt_pinv, 'outputs': [], 'execution_count': None}
        print(f'Replaced train_test_pinv at cell {i}')
    elif 'def train_test(' in src and 'def train_test_pinv' not in src:
        cells[i] = {**opt_train_test, 'outputs': [], 'execution_count': None}
        print(f'Replaced train_test at cell {i}')

# ── Insert optimised RESERVOIRE_SIMPLE after cell 0 ───────────────────────────
if not any('Pre-compute per-unit decay' in ''.join(c['source']) for c in cells):
    cells.insert(1, {**opt_res_class, 'outputs': [], 'execution_count': None})
    print('Inserted optimised RESERVOIRE_SIMPLE at position 1')

# ── Append stimulation-protocol cells from partI_MC ──────────────────────────
# These are cells 27-29 from partI_MC: FC analysis + delayed FC (stimulation response)
stim_cell_ids = {'c851c817', 'b58b4a5c'}   # already in TWOCLASSES main loop
# Add cells 27, 28, 29 from partI_MC (FC and delayed FC analysis)
existing_ids = {c.get('id') for c in cells}

print('\nAppending stimulation/FC protocol cells from Fig1DEF_partI_MC:')
for i in [27, 28, 29]:
    src_cell = nb_part1['cells'][i]
    cid = src_cell.get('id', f'partI-{i}')
    if cid not in existing_ids:
        new_cell = {**src_cell, 'outputs': [], 'execution_count': None}
        cells.append(new_cell)
        print(f'  + cell {i} id={cid}  ({str(src_cell["source"])[:60]}...)')
    else:
        print(f'  ~ cell {i} id={cid} already present, skipping')

# ── Save ─────────────────────────────────────────────────────────────────────
out = 'metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML___MC.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb_orig, f, indent=1, ensure_ascii=False)
print(f'\nSaved {out}  ({len(cells)} cells)')
