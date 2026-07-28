import json, sys

sys.stdout.reconfigure(encoding='utf-8')

for path, label in [
    (r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_ALLCLASSES_DEF2_ML__.ipynb', 'ALLCLASSES'),
    (r'C:\Users\user\Desktop\2026.AD\metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML__.ipynb', 'TWOCLASSES'),
]:
    with open(path, encoding='utf-8-sig') as f:
        nb = json.load(f)
    print(f'\n{"="*60}')
    print(f'{label}  — {len(nb["cells"])} cells')
    print('='*60)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        preview = src[:120].replace('\n', ' ').encode('ascii','replace').decode()
        print(f'  cell {i:2d}  {preview}')
