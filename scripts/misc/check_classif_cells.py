import json, sys

sys.stdout.reconfigure(encoding='utf-8')

with open(r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML__.ipynb',
          encoding='utf-8-sig') as f:
    nb = json.load(f)

for i in range(19, 26):
    if i >= len(nb['cells']):
        break
    cell = nb['cells'][i]
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    # Show which variables are referenced
    refs = [v for v in ['X_coll','Y_coll','all_weights','FC_collected',
                        'FC_plus_collected_sel','fitted_weights_list',
                        'state_ID_numeric','patient_ID_resting']
            if v in src]
    print(f'cell {i}  uses: {refs}')
    print(src[:300].encode('ascii','replace').decode())
    print()
