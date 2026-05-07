#!/usr/bin/env python3
"""Convert numeric byte fields in a summary CSV to human-readable units.

Usage:
  python scripts/convert_summary_human.py --input logs/diff_frame/summary.csv --output logs/diff_frame/summary_human.csv
"""
import argparse
import csv
import os


def human_bytes(n):
    if n is None:
        return ''
    try:
        n = float(n)
    except Exception:
        return str(n)
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024.0 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    if units[idx] == 'B':
        return f"{int(n)}{units[idx]}"
    return f"{n:.1f}{units[idx]}"


def convert_row(row):
    # convert known numeric byte columns if present
    for k in ('system_mem_total_before', 'system_mem_total_after'):
        if k in row and row[k] not in (None, ''):
            try:
                row[k + '_human'] = human_bytes(int(row[k]))
            except Exception:
                row[k + '_human'] = row[k]
        else:
            row[k + '_human'] = ''
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='logs/diff_frame/summary.csv')
    p.add_argument('--output', '-o', default='logs/diff_frame/summary_human.csv')
    p.add_argument('--print', action='store_true', dest='print_table', help='Also print a simple table to stdout')
    args = p.parse_args()

    inp = args.input
    out = args.output
    if not os.path.exists(inp):
        print(f'Input not found: {inp}')
        raise SystemExit(1)

    with open(inp, 'r', encoding='utf-8', newline='') as inf:
        reader = csv.DictReader(inf)
        rows = [convert_row(r) for r in reader]
        # build fieldnames: original + added human cols (if not already present)
        fieldnames = list(reader.fieldnames or [])
        for extra in ('system_mem_total_before_human', 'system_mem_total_after_human'):
            if extra not in fieldnames:
                fieldnames.append(extra)

    os.makedirs(os.path.dirname(out), exist_ok=True) if os.path.dirname(out) else None
    with open(out, 'w', encoding='utf-8', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # ensure keys exist
            for fn in fieldnames:
                if fn not in r:
                    r[fn] = ''
            # map our generated keys
            r['system_mem_total_before_human'] = r.get('system_mem_total_before_human') or r.get('system_mem_total_before_human')
            r['system_mem_total_after_human'] = r.get('system_mem_total_after_human') or r.get('system_mem_total_after_human')
            writer.writerow(r)

    print(f'Wrote converted CSV: {out}')

    if args.print_table:
        # simple aligned print
        cols = ['data_name', 'classifier_duration_hms', 'classifier_max_cpu', 'classifier_max_mem',
                'aggregate_duration_hms', 'aggregate_max_cpu', 'aggregate_max_mem',
                'system_mem_total_before_human', 'system_mem_total_after_human']
        widths = {c: max(len(c), 8) for c in cols}
        for r in rows:
            for c in cols:
                widths[c] = max(widths[c], len(str(r.get(c, ''))))
        fmt = '  '.join('{:'+str(widths[c])+'}' for c in cols)
        print(fmt.format(*cols))
        for r in rows:
            print(fmt.format(*(str(r.get(c, '')) for c in cols)))


if __name__ == '__main__':
    main()
