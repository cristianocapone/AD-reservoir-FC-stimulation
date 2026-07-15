import json
with open('Fig1DEF_partI_MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)
cell = nb['cells'][26]
print('Cell 26 id:', cell.get('id'))
print('Source:')
print(''.join(cell['source']))
