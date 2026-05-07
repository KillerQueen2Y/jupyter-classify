#!/usr/bin/env python3
"""
compare_classification_by_runs.py

按 run 聚合 `classification_results.csv`（每个 run 包含 30 行 x 12 列的标签），
对每个 (layer, head) 统计标签出现频次（跨多个 run），然后计算两个目录之间每个 (layer,head) 的
分布相似度（余弦）。输出 JSON 包含每层每个 head 的相似度、每层平均相似度，以及整体平均/中位数。

用法：
python similarity/compare_classification_by_runs.py \
  --dir_a classifier/output/classify_256 \
  --dir_b classifier/output/classify_128 \
  --out_json similarity/classify_256_vs_128_by_runs.json

脚本假定每个目录下包含若干 run_NNN 子目录，并且每个 run 目录包含 `classification_results.csv`。
"""

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict
from glob import glob


def read_runs(dir_root):
    # find all run_*/classification_results.csv
    # accept either 'classification_results.csv' or 'labels.csv' in run_* dirs
    patterns = [os.path.join(dir_root, 'run_*', 'classification_results.csv'),
                os.path.join(dir_root, 'run_*', 'labels.csv')]
    files = []
    for p in patterns:
        files.extend(glob(p))
    # remove duplicates and sort
    files = sorted(set(files))
    runs = []
    for fp in files:
        mat = []
        with open(fp, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for r in reader:
                if not r:
                    continue
                row = []
                for x in r:
                    xs = x.strip()
                    if xs == '':
                        continue
                    try:
                        row.append(int(xs))
                    except Exception:
                        try:
                            row.append(int(float(xs)))
                        except Exception:
                            row.append(xs)
                if row:
                    mat.append(row)
        if mat:
            runs.append({'path': fp, 'mat': mat})
    return runs


def union_label_set(runs_a, runs_b):
    s = set()
    for r in runs_a:
        for row in r['mat']:
            s.update(row)
    for r in runs_b:
        for row in r['mat']:
            s.update(row)
    return sorted(s)


def cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(y * y for y in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def aggregate_label_counts_per_position(runs):
    """Return dict: (layer_idx, head_idx) -> Counter of labels across runs."""
    counts = defaultdict(Counter)
    for run in runs:
        mat = run['mat']
        n_layers = len(mat)
        for li in range(n_layers):
            row = mat[li]
            n_heads = len(row)
            for hi in range(n_heads):
                counts[(li, hi)][row[hi]] += 1
    return counts


def normalize_counter_to_vector(counter, labels):
    total = sum(counter.values())
    if total == 0:
        return [0.0] * len(labels)
    return [counter.get(l, 0) / total for l in labels]


def compare_dirs(dir_a, dir_b):
    runs_a = read_runs(dir_a)
    runs_b = read_runs(dir_b)
    if not runs_a:
        raise RuntimeError(f'No runs found under {dir_a}')
    if not runs_b:
        raise RuntimeError(f'No runs found under {dir_b}')

    labels = union_label_set(runs_a, runs_b)

    counts_a = aggregate_label_counts_per_position(runs_a)
    counts_b = aggregate_label_counts_per_position(runs_b)

    # determine max layer/head indices present in either
    keys = set(counts_a.keys()) | set(counts_b.keys())
    if not keys:
        raise RuntimeError('No (layer,head) positions found')

    max_layer = max(k[0] for k in keys)
    max_head = max(k[1] for k in keys)

    # compute global label counts across all positions (for summary)
    from collections import Counter
    total_counter_a = Counter()
    total_counter_b = Counter()
    for c in counts_a.values():
        total_counter_a.update(c)
    for c in counts_b.values():
        total_counter_b.update(c)
    total_a = sum(total_counter_a.values())
    total_b = sum(total_counter_b.values())
    # global frequency vectors
    vec_tot_a = [total_counter_a.get(l, 0) / total_a if total_a else 0.0 for l in labels]
    vec_tot_b = [total_counter_b.get(l, 0) / total_b if total_b else 0.0 for l in labels]
    # global cosine
    def _cos(u, v):
        dot = sum(x * y for x, y in zip(u, v))
        nu = math.sqrt(sum(x * x for x in u))
        nv = math.sqrt(sum(y * y for y in v))
        return dot / (nu * nv) if nu > 0 and nv > 0 else 0.0
    global_cosine = _cos(vec_tot_a, vec_tot_b)

    per_layer = []
    all_sims = []
    for li in range(max_layer + 1):
        per_head_sims = []
        for hi in range(max_head + 1):
            ca = counts_a.get((li, hi), Counter())
            cb = counts_b.get((li, hi), Counter())
            va = normalize_counter_to_vector(ca, labels)
            vb = normalize_counter_to_vector(cb, labels)
            s = cosine(va, vb)
            per_head_sims.append({
                'head_index': hi,
                'cosine': s,
                'vec_a': va,
                'vec_b': vb,
                'count_a_total': sum(ca.values()),
                'count_b_total': sum(cb.values()),
            })
            all_sims.append(s)
        # layer mean
        layer_mean = sum(h['cosine'] for h in per_head_sims) / len(per_head_sims) if per_head_sims else 0.0
        per_layer.append({'layer_index': li, 'mean_cosine': layer_mean, 'per_head': per_head_sims})

    # overall stats
    mean_all = sum(all_sims) / len(all_sims) if all_sims else 0.0
    sims_sorted = sorted(all_sims)
    mid = len(sims_sorted) // 2
    if len(sims_sorted) % 2 == 1:
        median_all = sims_sorted[mid]
    else:
        median_all = (sims_sorted[mid - 1] + sims_sorted[mid]) / 2.0 if sims_sorted else 0.0

    overall = {
        'global_cosine': global_cosine,
        'mean_cosine_layers': mean_all,
        'median_cosine_layers': median_all,
        'n_positions_compared': len(keys),
    }

    summary = {
        'n_runs_a': len(runs_a),
        'n_runs_b': len(runs_b),
        'n_layers': max_layer + 1,
        'n_heads_per_layer': max_head + 1,
        'n_positions_compared': len(keys),
        'total_counts_a': int(total_a),
        'total_counts_b': int(total_b),
        'labels': labels,
        'global_label_freq_a': vec_tot_a,
        'global_label_freq_b': vec_tot_b,
        'global_cosine': global_cosine,
    }

    return {
        'overall': overall,
        'dir_a': dir_a,
        'dir_b': dir_b,
        'summary': summary,
        'per_layer': per_layer,
        'mean_cosine_all': mean_all,
        'median_cosine_all': median_all,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_a', required=True, help='First classifier output root (contains run_*/classification_results.csv)')
    parser.add_argument('--dir_b', required=True, help='Second classifier output root')
    parser.add_argument('--out_json', required=True, help='Output JSON path')
    args = parser.parse_args()

    res = compare_dirs(args.dir_a, args.dir_b)
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as fo:
        json.dump(res, fo, indent=2, ensure_ascii=False)
    print('Wrote', args.out_json)


if __name__ == '__main__':
    main()
