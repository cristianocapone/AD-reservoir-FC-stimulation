import json, sys

fname = sys.argv[1]
with open(fname, encoding='utf-8-sig') as f:
    nb = json.load(f)
print(f'File: {fname}  Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    ctype = cell['cell_type']
    if ctype != 'code':
        continue
    preview = src[:150].replace('\n', ' ')
    print(f'cell {i:2d}  | {preview}')
