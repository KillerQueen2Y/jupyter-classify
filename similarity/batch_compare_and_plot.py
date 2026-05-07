#!/usr/bin/env python3
"""
batch_compare_and_plot.py

对参考目录（例如 classifier/output/classify_256）与多个目标目录（128/64/32/16/8）批量比较，
保存每个比较的 JSON，并生成一张折线图（global_cosine 与 mean_cosine_layers）。

用法示例：
python similarity/batch_compare_and_plot.py \
  --ref classifier/output/classify_256 \
  --targets classifier/output/classify_128 classifier/output/classify_64 classifier/output/classify_32 classifier/output/classify_16 classifier/output/classify_8 \
  --out_json similarity/batch_compare_summary.json \
  --out_png similarity/batch_compare_plot.png
"""
import argparse
import json
import os
import sys

# ensure repo root in sys.path so we can import compare function
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from similarity.compare_classification_by_runs import compare_dirs
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def numeric_label_from_path(p):
    # try to extract final numeric token like classify_128 -> 128
    base = os.path.basename(p.rstrip('/'))
    parts = base.split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    return base


def main(ref, targets, out_json, out_png):
    results = {}
    xs = []
    labels = []
    global_cos = []

    for t in targets:
        print('Comparing', ref, 'vs', t)
        res = compare_dirs(ref, t)
        key = os.path.basename(t.rstrip('/'))
        # save per-comparison json
        cmp_out = os.path.join(os.path.dirname(out_json), f'{key}_vs_ref.json')
        os.makedirs(os.path.dirname(cmp_out), exist_ok=True)
        with open(cmp_out, 'w', encoding='utf-8') as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        results[key] = res

        lab = numeric_label_from_path(t)
        labels.append(key)
        try:
            xs.append(float(lab))
        except Exception:
            xs.append(len(xs))
        global_cos.append(res.get('overall', {}).get('global_cosine', res.get('summary', {}).get('global_cosine')))

    # write batch summary
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump({'ref': ref, 'targets': targets, 'results': results}, f, indent=2, ensure_ascii=False)

    # sort by numeric x if possible
    try:
        order = sorted(range(len(xs)), key=lambda i: xs[i], reverse=False)
    except Exception:
        order = list(range(len(xs)))

    xs_ord = [xs[i] for i in order]
    labs_ord = [labels[i] for i in order]
    global_ord = [global_cos[i] for i in order]

    # Convert similarities to percentage (60s baseline = 100%) for clearer plotting
    global_pct = [v * 100.0 for v in global_ord]

    plt.figure(figsize=(8,4))
    plt.plot(labs_ord, global_pct, marker='o', label='global_cosine (%)')
    plt.axhline(100.0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    # Show only the 99.98%~100% band and keep 100% as the top boundary.
    plt.ylim(99.98, 100.0)

    for x, y in zip(labs_ord, global_pct):
        label_offset = (0, 8)
        vertical_align = 'bottom'
        if y >= 99.85:
            label_offset = (0, -10)
            vertical_align = 'top'
        plt.annotate(
            f'{y:.4f}%',
            xy=(x, y),
            xytext=label_offset,
            textcoords='offset points',
            ha='center',
            va=vertical_align,
            fontsize=9,
            clip_on=True,
        )
    plt.xlabel('target (numeric from dir name)')
    plt.ylabel('cosine similarity (%)')
    ref_name = os.path.basename(os.path.normpath(ref))
    plt.title(f'Classification similarity vs {ref_name}', fontsize=13, pad=16)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.xticks(rotation=0)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.93])
    plt.savefig(out_png, dpi=200)
    print('Wrote plot to', out_png)
    print('Wrote batch summary to', out_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', required=True)
    parser.add_argument('--targets', nargs='+', required=True)
    parser.add_argument('--out_json', required=True)
    parser.add_argument('--out_png', required=True)
    args = parser.parse_args()
    main(args.ref, args.targets, args.out_json, args.out_png)
