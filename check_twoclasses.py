import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open(r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML__.ipynb',
          encoding='utf-8-sig') as f:
    nb = json.load(f)
for cell in nb['cells']:
    cid = cell.get('id', '')
    if cid in ('ff3bcf9b', '8a830229', 'c851c817'):
        src = ''.join(cell['source'])
        print(f'=== cell id={cid} ===')
        print(src[:300])
        print()
