#!/usr/bin/env python3
from pathlib import Path
import re

root = Path('MCI_bids')
pattern = re.compile(r'(_task-[a-zA-Z0-9]+_run-[0-9]+)_task-[a-zA-Z0-9]+_bold')
fixes = []

for path in root.rglob('*'):
    if not path.is_file():
        continue
    name = path.name
    new_name = name
    new_name = pattern.sub(r'\1_bold', new_name)
    new_name = new_name.replace('_task-rest_bolda', '_task-rest_bold')
    if new_name != name:
        fixes.append((path, path.with_name(new_name)))

print(f'Found {len(fixes)} file(s) to rename')
for old, new in fixes:
    if new.exists():
        raise FileExistsError(f'Target file already exists: {new}')
    print(f'Renaming {old.relative_to(root)} -> {new.relative_to(root)}')
    old.rename(new)

print('Rename complete.')
