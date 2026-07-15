import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML__.ipynb',
          encoding='utf-8-sig') as f:
    nb = json.load(f)

for i in [19, 20, 21]:
    cell = nb['cells'][i]
    if cell['cell_type'] != 'code':
        continue
    print(f'\n=== cell {i} id={cell.get("id","?")} ===')
    print(''.join(cell['source']).encode('ascii','replace').decode())
