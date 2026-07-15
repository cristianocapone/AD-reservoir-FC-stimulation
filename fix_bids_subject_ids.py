#!/usr/bin/env python3
from pathlib import Path
import re

root = Path('MCI_bids')
# Only rename subject labels containing underscores after 'sub-'
subjects = [p for p in root.iterdir() if p.is_dir() and p.name.startswith('sub-')]

mapping = {}
for subj in subjects:
    label = subj.name[4:]
    if '_' in label:
        new_label = 'sub-' + label.replace('_', '')
        mapping[subj.name] = new_label

if not mapping:
    print('No subject labels to rename.')
    raise SystemExit(0)

print('Subject renaming mapping:')
for old, new in mapping.items():
    print(f'  {old} -> {new}')

for old_label, new_label in mapping.items():
    old_dir = root / old_label
    new_dir = root / new_label
    if new_dir.exists():
        raise FileExistsError(f'Target directory already exists: {new_dir}')

    # Rename files and directories within current subject directory
    for path in sorted(old_dir.rglob('*'), key=lambda p: len(p.parts), reverse=True):
        rel = path.relative_to(old_dir)
        new_name = rel.name.replace(old_label, new_label)
        if new_name != rel.name:
            target = path.with_name(new_name)
            print(f'Renaming {path} -> {target}')
            path.rename(target)

    # Rename the subject top-level directory
    print(f'Renaming top-level directory {old_dir} -> {new_dir}')
    old_dir.rename(new_dir)

# Update participants.tsv
participants = root / 'participants.tsv'
if participants.exists():
    text = participants.read_text()
    for old_label, new_label in mapping.items():
        text = text.replace(old_label, new_label)
    participants.write_text(text)
    print('Updated participants.tsv')
else:
    print('No participants.tsv found to update.')
