#!/usr/bin/env python3
"""
Safely rename subdirectories matching a pattern to a compact sequential
naming scheme: run_000, run_001, ...

Usage examples:
  python prepare/rename_runs.py /path/to/parent --dry-run
  python prepare/rename_runs.py /path/to/parent --start 0 --pad 3

Features:
- Finds directories matching a regex with a single capturing group for the
  numeric part (default: r"run_(\d+)").
- Sorts by numeric value when present, otherwise by name.
- Performs safe two-phase renaming using temporary names to avoid collisions.
- Supports `--dry-run` to print planned operations without applying them.
"""
from pathlib import Path
import argparse
import re
import os
import sys
import uuid


def parse_args():
    p = argparse.ArgumentParser(description="Rename run_* subdirectories into a compact sequential order")
    p.add_argument("dir", help="Parent directory containing run_* subfolders")
    p.add_argument("--pattern", default=r"run_(\d+)",
                   help="Regex with one capturing group for the numeric part (default: 'run_(\\d+)')")
    p.add_argument("--prefix", default="run_", help="Prefix for the output names (default: run_)")
    p.add_argument("--start", type=int, default=0, help="Start index for renaming (default: 0)")
    p.add_argument("--pad", type=int, default=3, help="Zero-padding width (default: 3 => 000)")
    p.add_argument("--dry-run", action="store_true", help="Print planned renames without applying them")
    return p.parse_args()


def collect_dirs(parent: Path, pattern: str):
    prog = re.compile(pattern)
    entries = []
    for p in parent.iterdir():
        if not p.is_dir():
            continue
        m = prog.search(p.name)
        if not m:
            continue
        # prefer numeric sort if capturing group yields digits
        num = None
        try:
            num = int(m.group(1)) if m.groups() else None
        except Exception:
            num = None
        entries.append((p, num))
    return entries


def plan_and_apply(parent: Path, entries, prefix: str, start: int, pad: int, dry_run: bool):
    # sort: numeric first (None -> large), then name
    def sort_key(item):
        p, num = item
        return (num if num is not None else 10**12, p.name)

    entries_sorted = sorted(entries, key=sort_key)
    n = len(entries_sorted)
    if n == 0:
        print("No matching directories found. Nothing to do.")
        return 0

    # build final names
    finals = [f"{prefix}{i:0{pad}d}" for i in range(start, start + n)]

    # check for immediate conflicts (final name exists but not in our list)
    final_paths = [parent / fn for fn in finals]
    existing_conflicts = [p for p in final_paths if p.exists() and p not in [e[0] for e in entries_sorted]]
    if existing_conflicts:
        print("Error: The following target names already exist and are not part of the current matched set:")
        for p in existing_conflicts:
            print("  ", p)
        print("Resolve or move them before running the script.")
        return 2

    # build mapping old -> final
    mapping = {}
    for (entry, _), final_name in zip(entries_sorted, finals):
        mapping[entry] = parent / final_name

    # print plan
    print("Planned renames:")
    for src, dst in mapping.items():
        print(f"  {src.name}  ->  {dst.name}")

    if dry_run:
        print("Dry-run mode: no changes applied.")
        return 0

    # perform two-phase renaming to avoid collisions: rename to temp names, then to finals
    temp_map = {}
    try:
        for src in mapping:
            tmp_name = f".tmp_rename_{uuid.uuid4().hex[:8]}_{src.name}"
            tmp_path = parent / tmp_name
            if tmp_path.exists():
                raise FileExistsError(f"Temporary path unexpectedly exists: {tmp_path}")
            print(f"Renaming: {src.name} -> {tmp_path.name}")
            os.rename(src, tmp_path)
            temp_map[tmp_path] = mapping[src]

        # now rename temps to final names
        for tmp, final in temp_map.items():
            if final.exists():
                raise FileExistsError(f"Target final path exists: {final}")
            print(f"Renaming: {tmp.name} -> {final.name}")
            os.rename(tmp, final)

    except Exception as e:
        print("Error during renaming:", e)
        print("Attempting to roll back any temporary renames...")
        # try to roll back: move any tmp back to their original names where possible
        for tmp, final in list(temp_map.items()):
            try:
                orig_name = tmp.name.split('_', 3)[-1]
                orig_path = parent / orig_name
                if tmp.exists() and not orig_path.exists():
                    os.rename(tmp, orig_path)
                    print(f"Rolled back: {tmp.name} -> {orig_name}")
            except Exception:
                pass
        return 3

    print("Renaming complete.")
    return 0


def main():
    args = parse_args()
    parent = Path(args.dir)
    if not parent.exists() or not parent.is_dir():
        print(f"Directory not found: {parent}")
        sys.exit(1)
    entries = collect_dirs(parent, args.pattern)
    rc = plan_and_apply(parent, entries, args.prefix, args.start, args.pad, args.dry_run)
    sys.exit(rc)


if __name__ == "__main__":
    main()
