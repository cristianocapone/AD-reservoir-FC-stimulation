import json
with open('Fig1DEF_partI.ipynb', encoding='utf-8-sig') as f:
    nb = json.load(f)
print(f'Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    preview = src[:120].replace('\n', ' ') if ctype == 'code' else '[markdown]'
    print(f'cell {i:2d}  | {preview}')
