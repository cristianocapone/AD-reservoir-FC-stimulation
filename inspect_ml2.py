import json, sys

fname = sys.argv[1]
with open(fname, encoding='utf-8-sig') as f:
    nb = json.load(f)

# Show cells 0-10 in full, and scan all for 121
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])
    if cell['cell_type'] != 'code':
        continue
    if i <= 10 or '121' in src:
        print(f'\n=== cell {i}  id={cell.get("id","?")} ===')
        print(src)
