"""Update Fig1DEF_partI_MC.ipynb: 114 → 121 parcels throughout."""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

patched = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    new_src = src

    # Loading cell: N_PARCELS and comment
    new_src = new_src.replace(
        'N_PARCELS  = 114',
        'N_PARCELS  = 121')
    new_src = new_src.replace(
        'incomplete parcellations (N!=114) are discarded.',
        'incomplete parcellations (N!=121) are discarded.')

    # Reservoir cell: N_sites
    new_src = new_src.replace(
        'N_sites = 114 # Number of sites in the Schaefer atlas (100 cortical + 14 subcortical)',
        'N_sites = 121 # Number of sites (100 cortical + 21 subcortical)')
    new_src = new_src.replace(
        'N_sites = 114 # Number of',
        'N_sites = 121 # Number of')

    # Shape filter cell (ff3bcf9b style): == 114 → == 121
    new_src = new_src.replace('== 114', '== 121')

    # FC delayed slicing: [114:,:114] → [121:,:121]
    new_src = new_src.replace('[114:,:114]', '[121:,:121]')
    new_src = new_src.replace('[114:, :114]', '[121:, :121]')

    if new_src != src:
        cell['source'] = new_src
        patched.append(i)
        print(f'  Patched cell {i} id={cell.get("id","?")}')

with open('Fig1DEF_partI_MC.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f'\nSaved. Patched cells: {patched}')
