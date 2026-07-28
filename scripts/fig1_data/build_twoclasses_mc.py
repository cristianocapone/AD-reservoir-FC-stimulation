import json, shutil, os

# ── 1. Load source notebooks ─────────────────────────────────────────────────
with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb_part1 = json.load(f)

src_path = r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML__.ipynb'
with open(src_path, encoding='utf-8-sig') as f:
    nb_orig = json.load(f)

# ── 2. Extract optimised cells from partI_MC ─────────────────────────────────
# cell 16 = optimised RESERVOIRE_SIMPLE
# cell 17 = optimised train_test_pinv
# cell 18 = optimised train_test
opt_res_class  = nb_part1['cells'][16]
opt_pinv       = nb_part1['cells'][17]
opt_train_test = nb_part1['cells'][18]

# ── 3. New data-loading source (same as partI_MC cell 1) ─────────────────────
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
    "# Load motion-corrected timeseries from fmriprep/nilearn extraction.\n"
    "# Files stored as (N_parcels x T); transpose to (T x N_parcels).\n"
    "# Extended recordings (T!=140) and incomplete parcellations (N!=114) are discarded.\n"
    r'timeseries_base = r"C:\Users\user\Desktop\2026.AD_MotionCorrection\timeseries"' + "\n"
    "\n"
    "T_STANDARD = 140\n"
    "N_PARCELS  = 114\n"
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
    "        arr = np.load(os.path.join(folder, fname))   # (N_parcels, T)\n"
    "        if arr.shape[1] != T_STANDARD:\n"
    '            print(f"Skipped (T={arr.shape[1]}) {fname}")\n'
    "            continue\n"
    "        if arr.shape[0] != N_PARCELS:\n"
    '            print(f"Skipped (N={arr.shape[0]}) {fname}")\n'
    "            continue\n"
    "        arr = arr.T                                   # -> (T, N_parcels)\n"
    '        sub_id = fname.split("_")[0]\n'
    '        identifiers_list.append([state_id, sub_id, "Resting_State_fMRI"])\n'
    "        collected_signals.append(arr)\n"
    '        print(f"Loaded {fname}, shape={arr.shape}")\n'
    "\n"
    "identifiers = np.array(identifiers_list, dtype=object)\n"
)

new_src_patient_id = "patient_ID = [identifiers[k][1] for k in range(len(identifiers))]\n"

# ── 4. Patch cells by id ──────────────────────────────────────────────────────
cells = nb_orig['cells']
patched = set()

for cell in cells:
    cid = cell.get('id', '')
    src = ''.join(cell['source'])
    new_src = None

    if cid == '56c85796':
        new_src = new_src_loading
    elif cid == '3080171f':
        new_src = new_src_patient_id
    elif cid == 'ff3bcf9b':
        new_src = src.replace('np.shape(collected_signals[i])[1] == 121',
                              'np.shape(collected_signals[i])[1] == 114')
    elif cid == '8a830229':
        new_src = src.replace(
            'N_sites = 121 # Number of sites in the Schaefer atlas',
            'N_sites = 114 # Number of sites in the Schaefer atlas (100 cortical + 14 subcortical)')
    elif cid == 'c851c817':
        new_src = src.replace('fc_matrix_delayed[121:,:121]', 'fc_matrix_delayed[114:,:114]')
        new_src = new_src.replace('fc_matrix_delayed_sim[121:,:121]', 'fc_matrix_delayed_sim[114:,:114]')

    if new_src is not None and new_src != src:
        cell['source'] = new_src
        cell['outputs'] = []
        cell['execution_count'] = None
        patched.add(cid)

print(f'Patched cells: {patched}')

# ── 5. Replace slow train_test_pinv and train_test with optimised versions ────
# Find cell indices for cells 14 and 15 (train_test_pinv and train_test)
for i, cell in enumerate(cells):
    src = ''.join(cell['source'])
    if 'def train_test_pinv' in src and cell.get('id') != opt_pinv.get('id'):
        # Replace with optimised version
        cells[i] = dict(opt_pinv)
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None
        print(f'Replaced train_test_pinv at cell {i}')
    elif 'def train_test(' in src and cell.get('id') != opt_train_test.get('id'):
        cells[i] = dict(opt_train_test)
        cells[i]['outputs'] = []
        cells[i]['execution_count'] = None
        print(f'Replaced train_test at cell {i}')

# ── 6. Insert optimised RESERVOIRE_SIMPLE right after cell 0 ─────────────────
# Check if it's already there
has_opt = any('Pre-compute per-unit decay' in ''.join(c['source']) for c in cells)
if not has_opt:
    insert_cell = dict(opt_res_class)
    insert_cell['outputs'] = []
    insert_cell['execution_count'] = None
    cells.insert(1, insert_cell)
    print('Inserted optimised RESERVOIRE_SIMPLE at position 1')

# ── 7. Clear all existing outputs ────────────────────────────────────────────
for cell in cells:
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None

# ── 8. Save ───────────────────────────────────────────────────────────────────
out_path = 'metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML___MC.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(nb_orig, f, indent=1, ensure_ascii=False)

print(f'Saved {out_path}')
print(f'Total cells: {len(cells)}')
