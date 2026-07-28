import json

with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)

new_src_56 = (
    "import os\n"
    "import numpy as np\n"
    "from scipy.signal import butter, filtfilt\n"
    "\n"
    "def lowpass_filter_rows(X, fs, cutoff, order=4):\n"
    "    # X: N x T matrix\n"
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

new_src_3080 = "patient_ID = [identifiers[k][1] for k in range(len(identifiers))]\n"

for cell in nb['cells']:
    if cell.get('id') == '56c85796':
        cell['source'] = new_src_56
        cell['outputs'] = []
        cell['execution_count'] = None
    if cell.get('id') == '3080171f':
        cell['source'] = new_src_3080
        cell['outputs'] = []
        cell['execution_count'] = None

with open('Fig1DEF_partI_MC.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Saved OK, no BOM')
