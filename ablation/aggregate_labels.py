"""
Aggregate labels.csv files under a directory (e.g. output\ablation\lab3\th7),
perform majority vote per matrix position and write a combined CSV.

Usage:
    uv run python -m ablation.aggregate_labels --csv_dir output\ablation\lab3\th7 --output ./output/label/

Default output file: <output>/aggregated_labels.csv

Tie-breaking: choose smallest numeric label among those tied for max count.
Labels: -1 Wave, 1 Anchor, 2 Veil, 0 Unknown
"""
from pathlib import Path
import argparse
import numpy as np
from collections import Counter


def collect_label_arrays(csv_dir: Path):
    files = sorted(csv_dir.rglob("labels.csv"))
    arrays = []
    for f in files:
        try:
            a = np.loadtxt(f, delimiter=",", dtype=int)
            if a.ndim == 1:
                a = np.atleast_2d(a)
            arrays.append((f, a))
        except Exception as e:
            print(f"Skipping {f}: failed to read ({e})")
    return arrays


def pad_arrays(arrays):
    # arrays: list of (Path, np.array)
    if not arrays:
        return []
    max_rows = max(a.shape[0] for _, a in arrays)
    max_cols = max(a.shape[1] for _, a in arrays)
    padded = []
    for f, a in arrays:
        pr = np.zeros((max_rows, max_cols), dtype=int)
        pr[:] = 0  # default unknown
        pr[: a.shape[0], : a.shape[1]] = a
        padded.append((f, pr))
    return padded


def majority_vote(padded_arrays):
    if not padded_arrays:
        return None
    _, sample = padded_arrays[0]
    R, C = sample.shape
    out = np.zeros((R, C), dtype=int)
    # for each position, collect votes
    for i in range(R):
        for j in range(C):
            vals = [arr[i, j] for _, arr in padded_arrays]
            c = Counter(vals)
            if not c:
                out[i, j] = 0
            else:
                # get most common; if tie, pick smallest label
                most = c.most_common()
                max_count = most[0][1]
                candidates = [val for val, cnt in most if cnt == max_count]
                out[i, j] = int(min(candidates))
    return out


def save_output(mat: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aggregated_labels.csv"
    np.savetxt(out_path, mat, fmt='%d', delimiter=',')
    print(f"Wrote aggregated labels to: {out_path}")
    return out_path


def summarize_counts(mat: np.ndarray):
    flat = mat.flatten()
    c = Counter(flat)
    print("Final label counts:")
    for k, v in sorted(c.items()):
        print(f"  {k}: {v}")


def main(csv_dir: str, output: str):
    csv_path = Path(csv_dir)
    if not csv_path.exists():
        print(f"CSV directory does not exist: {csv_dir}")
        return
    arrays = collect_label_arrays(csv_path)
    if not arrays:
        print(f"No labels.csv files found under {csv_dir}")
        return
    padded = pad_arrays(arrays)
    mat = majority_vote(padded)
    if mat is None:
        print("No data to aggregate")
        return
    out_path = save_output(mat, Path(output))
    summarize_counts(mat)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Aggregate labels.csv files by majority vote')
    p.add_argument('--csv_dir', required=True, help='Top-level directory containing run_* subdirs with labels.csv')
    p.add_argument('--output', default='./output/label/', help='Output directory for aggregated CSV')
    args = p.parse_args()
    main(args.csv_dir, args.output)
