import numpy as np, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ts = Path('timeseries')
shape_counts = {}
for f in ts.rglob('*.npy'):
    a = np.load(f)
    shape_counts[a.shape] = shape_counts.get(a.shape, 0) + 1
for s, c in sorted(shape_counts.items()):
    print(f'  shape={s}  count={c}')
good = shape_counts.get((121, 140), 0)
print(f'\nFiles with (121,140): {good}  (notebooks will use these)')
