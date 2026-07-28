import json

with open('metaTwin_AD_WEIGHTS_clean_TWOCLASSES_DEF2_ML___MC.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    ec = cell.get('execution_count')
    outs = cell.get('outputs', [])
    n_out = len(outs)
    if ec is not None or n_out > 0:
        last_txt = ''
        for out in outs[-3:]:
            t = ''.join(out.get('text', out.get('traceback', [])))
            if t.strip():
                last_txt = t.strip()[-150:]
        print(f'cell {i:2d}  exec={ec}  outputs={n_out}  | {last_txt}')
