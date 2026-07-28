#!/usr/bin/env python3
"""Remove the duplicate _task-rest_ entity from functional filenames in AD_bids.

dcm2niix was called with a filename already containing _task-rest_, and
SERIES_MAP had 'task-rest_bold' as the suffix, so every file ended up with
the pattern duplicated:
  ..._task-rest_run-01_task-rest_bold.nii.gz   (wrong)
  ..._task-rest_run-01_bold.nii.gz             (correct)

Run without arguments to perform the rename.
Pass --dry-run to preview without touching files.
"""

import argparse
import re
import sys
from pathlib import Path

BIDS_ROOT = Path(__file__).parent / "AD_bids"

# Matches the second _task-<label>_ that appears just before bold/bolda
PATTERN = re.compile(r"(_task-[A-Za-z0-9]+_run-\d+)_task-[A-Za-z0-9]+_(bold)")


def collect_renames(root: Path):
    renames = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        new_name = PATTERN.sub(r"\1_\2", path.name)
        if new_name != path.name:
            new_path = path.with_name(new_name)
            renames.append((path, new_path))
    return renames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned renames without touching files.")
    args = parser.parse_args()

    renames = collect_renames(BIDS_ROOT)
    if not renames:
        print("No files to rename — already fixed or wrong directory.")
        return

    print(f"{'DRY RUN: ' if args.dry_run else ''}Found {len(renames)} file(s) to rename\n")
    for old, new in renames:
        rel = old.relative_to(BIDS_ROOT)
        print(f"  {rel}")
        print(f"  -> {new.name}\n")
        if not args.dry_run:
            if new.exists():
                print(f"ERROR: target already exists: {new}", file=sys.stderr)
                sys.exit(1)
            old.rename(new)

    if not args.dry_run:
        print(f"Renamed {len(renames)} file(s).")
    else:
        print("Dry run complete — no files were changed.")


if __name__ == "__main__":
    main()
