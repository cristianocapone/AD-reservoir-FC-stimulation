import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML___MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
print(f'Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    for line in src.split('\n'):
        if any(x in line for x in ['N_PARCELS', 'N_sites', '== 121', '== 114', 'plt.pause', 'trange']):
            print(f'  cell {i}: {line.strip()}')
