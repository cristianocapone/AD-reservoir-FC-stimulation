import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if 'N_PARCELS' in src or 'N_sites' in src:
        cid = cell.get('id', '?')
        print(f'cell {i} id={cid}')
        print(src[:500])
        print('---')
