import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

# Show full source of cells 1 and 20
for idx in [1, 20]:
    cell = nb['cells'][idx]
    cid = cell.get('id', '?')
    src = ''.join(cell['source'])
    print(f'=== cell {idx} id={cid} ===')
    print(src)
    print()
