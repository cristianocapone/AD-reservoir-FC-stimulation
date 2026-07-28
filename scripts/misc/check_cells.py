import json

with open('metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML___MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if 'identifiers' in src or 'processed_data' in src or 'timeseries' in src:
        print(f'\n=== cell {i}  id={cell.get("id","?")} ===')
        print(src)
