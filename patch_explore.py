import json, sys
sys.stdout.reconfigure(encoding='utf-8')
with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    src = ''.join(cell['source'])
    if 'if_explore_parameters = True' in src:
        cell['source'] = src.replace('if_explore_parameters = True',
                                     'if_explore_parameters = False')
        cell['outputs'] = []
        cell['execution_count'] = None
        patched += 1

# Also clear all outputs so notebook re-executes cleanly
for cell in nb['cells']:
    cell['outputs'] = []
    cell['execution_count'] = None

with open('Fig1DEF_partI_MC.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print(f'Patched {patched} cell(s). Outputs cleared.')
