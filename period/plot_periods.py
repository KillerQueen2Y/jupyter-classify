"""
Plot max (global) period per head across all layers in a run using rFFT logic
from `classifier.core.classify_attention`.

Outputs:
 - CSV with columns: layer, head, global_period, global_amp
 - PNG and SVG figure with a line + scatter of global_periods ordered by layer/head

Usage:
    uv run python -m period.plot_periods --run_dir <path/to/run_000> --output ./output/periods/ --ignore-last-frames 3
"""
from pathlib import Path
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import math
import traceback

from classifier.core import classify_attention
import torch

# default FFT ranges used previously
FFT_RANGES = [
    {"label": "0-68", "start": 0, "end": 69},
    {"label": "0-71", "start": 0, "end": None},
]


def collect_periods(run_dir: Path, fft_ranges=FFT_RANGES, ignore_last_frames: int = 3, wave_only: bool = False, label_csv: Path = None):
    rows = []  # (layer, head, global_period, global_amp)
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    layer_files = sorted(run_dir.glob("layer*.pt"))
    if not layer_files:
        raise FileNotFoundError(f"No layer*.pt files found under {run_dir}")

    # 如果 wave_only 且 label_csv 提供，则先读取 label
    wave_mask = None
    if wave_only and label_csv is not None:
        # 读取 label csv，假定格式 layer,head,label,...
        import csv as _csv
        wave_set = set()
        with open(label_csv, 'r', encoding='utf-8') as f:
            reader = _csv.reader(f)
            header = next(reader)
            # 支持不同表头
            try:
                idx_layer = header.index('layer')
                idx_head = header.index('head')
                idx_label = header.index('label') if 'label' in header else 2
            except Exception:
                idx_layer, idx_head, idx_label = 0, 1, 2
            for row in reader:
                try:
                    l = int(row[idx_layer])
                    h = int(row[idx_head])
                    label = int(row[idx_label])
                    if label == 0:  # 0 视为 Wave Head
                        wave_set.add((l, h))
                except Exception:
                    continue
        # wave_set 里只保留 (layer, head) 属于 Wave 的

    for lf in layer_files:
        layer_name = lf.stem
        def get_num_heads(pt_path: Path):
            try:
                payload = torch.load(pt_path, map_location="cpu", weights_only=False)
                per_head = payload.get("last_frame_attention_per_head")
                if per_head is None:
                    raise KeyError("last_frame_attention_per_head not found in payload")
                return int(per_head.shape[0])
            except Exception:
                raise

        try:
            n_heads = get_num_heads(lf)
        except Exception:
            try:
                layer_idx = int(layer_name.replace('layer', ''))
            except Exception:
                continue
            print(f"Failed to read {lf}, skipping")
            continue

        layer_idx = int(layer_name.replace("layer", ""))
        for h in range(n_heads):
            # 如果 wave_only 且 (layer, head) 不在 wave_set，则跳过
            if wave_only and label_csv is not None:
                if (layer_idx, h) not in wave_set:
                    continue
            try:
                res = classify_attention(run_dir, layer=layer_idx, head=h, fft_ranges=fft_ranges, ignore_last_frames=ignore_last_frames)
                gp = None
                ga = None
                try:
                    gp = float(res.get('raw_results')[0].get('global_period'))
                    ga = float(res.get('raw_results')[0].get('global_amp'))
                except Exception:
                    gp = math.nan
                    ga = math.nan
                rows.append((layer_idx, h, gp, ga))
            except Exception:
                # record NaN if classify failed for this head
                traceback.print_exc()
                rows.append((layer_idx, h, math.nan, math.nan))
    return rows


def save_csv(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "periods_per_head.csv"
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "head", "global_period", "global_amp"])
        for r in rows:
            writer.writerow(r)
    return out_path


def plot_periods(rows, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No data to plot")

    # sort by layer then head
    rows_sorted = sorted(rows, key=lambda x: (x[0], x[1]))
    layers = [r[0] for r in rows_sorted]
    heads = [r[1] for r in rows_sorted]
    periods = np.array([r[2] for r in rows_sorted], dtype=float)
    amps = np.array([r[3] for r in rows_sorted], dtype=float)

    # x positions: index across all heads
    x = np.arange(len(periods))

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.set(style='whitegrid')

    # line plot connecting valid points (ignore NaN)
    valid_mask = ~np.isnan(periods)
    ax.plot(x[valid_mask], periods[valid_mask], '-', color='tab:blue', linewidth=1.0, alpha=0.8)

    # scatter: color by layer
    uniq_layers = sorted(set(layers))
    cmap = plt.get_cmap('tab20')
    layer_to_color = {lv: cmap(i % 20) for i, lv in enumerate(uniq_layers)}
    colors = [layer_to_color[l] for l in layers]

    sc = ax.scatter(x, periods, c=colors, s=20, edgecolors='k', linewidths=0.2, alpha=0.9)

    # create legend for layers (sampling few if too many)
    handles = []
    for lv in uniq_layers:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Layer {lv}', markerfacecolor=layer_to_color[lv], markersize=6))
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc='upper left')

    ax.set_xlabel('Head index (layer-major order)')
    ax.set_ylabel('Global period (frames)')
    ax.set_title('Global Period per Head — line + scatter')

    out_png = out_dir / 'periods_line_scatter.png'
    out_svg = out_dir / 'periods_line_scatter.svg'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)
    return out_png, out_svg


def plot_histogram(rows, out_dir: Path, bins: int = 50, max_period: float = None, style: str = 'bar', drop_zero_bins: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No data to plot")

    periods = np.array([r[2] for r in rows], dtype=float)
    periods = periods[np.isfinite(periods)]
    if periods.size == 0:
        raise ValueError("No finite period values found to histogram")

    if max_period is not None:
        periods = periods[periods <= float(max_period)]

    counts, edges = np.histogram(periods, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # optionally drop zero-count bins from both plot and CSV (apply before plotting)
    if drop_zero_bins:
        mask = counts > 0
        edges_filtered_left = edges[:-1][mask]
        edges_filtered_right = edges[1:][mask]
        centers = centers[mask]
        counts = counts[mask]

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set(style='whitegrid')
    # width for bars: use mean bin width from original edges
    width = float(np.diff(edges).mean())
    if style == 'bar':
        ax.bar(centers, counts, width=width * 0.95, color='tab:blue', edgecolor='k', alpha=0.8)
    else:
        # line style: plot centers vs counts with markers
        ax.plot(centers, counts, '-o', color='tab:blue', markersize=4, linewidth=1.5, markeredgecolor='k', alpha=0.9)
    ax.set_xlabel('Global period (frames)')
    ax.set_ylabel('Count')
    ax.set_title('Histogram of global periods per head')

    out_png = out_dir / 'periods_histogram.png'
    out_svg = out_dir / 'periods_histogram.svg'
    fig.savefig(out_png, dpi=200, bbox_inches='tight')
    fig.savefig(out_svg, bbox_inches='tight')
    plt.close(fig)
    
    # write CSV of bin counts (after optional filtering)
    csv_path = out_dir / 'periods_histogram.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['bin_left', 'bin_right', 'center', 'count'])
        if drop_zero_bins:
            for l, r, c, cnt in zip(edges_filtered_left, edges_filtered_right, centers, counts):
                writer.writerow([l, r, c, int(cnt)])
        else:
            for l, r, c, cnt in zip(edges[:-1], edges[1:], centers, counts):
                writer.writerow([l, r, c, int(cnt)])

    return out_png, out_svg, csv_path


def parse_args():
    p = argparse.ArgumentParser(description='Plot global-period per head for a run')
    p.add_argument('--run_dir', required=True, help='Path to run directory containing layer*.pt')
    p.add_argument('--output', default='./output/periods/', help='Output directory for CSV and figures')
    p.add_argument('--ignore-last-frames', type=int, default=3, help='Frames at the end to ignore')
    p.add_argument('--hist', action='store_true', help='Produce histogram (period -> count) instead of line+scatter')
    p.add_argument('--bins', type=int, default=50, help='Number of bins for histogram')
    p.add_argument('--hist-style', choices=['bar', 'line'], default='bar', help='Histogram plot style: bar or line')
    p.add_argument('--drop-zero-bins', action='store_true', help='Do not show bins with zero counts (filters them from plot and CSV)')
    p.add_argument('--max-period', type=float, default=None, help='Optional max period to include in histogram')
    p.add_argument('--wave', type=str, default=None, help='Path to labels.csv, only count Wave Head (label==0)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output)

    wave_only = args.wave is not None
    label_csv = Path(args.wave) if args.wave else None
    rows = collect_periods(run_dir, fft_ranges=FFT_RANGES, ignore_last_frames=args.ignore_last_frames, wave_only=wave_only, label_csv=label_csv)
    csv_path = save_csv(rows, out_dir)
    if args.hist:
        png, svg, hist_csv = plot_histogram(rows, out_dir, bins=args.bins, max_period=args.max_period, style=args.hist_style, drop_zero_bins=args.drop_zero_bins)
        print(f'Wrote CSV: {csv_path}')
        print(f'Wrote histogram CSV: {hist_csv}')
        print(f'Wrote histogram figures: {png}, {svg}')
    else:
        png, svg = plot_periods(rows, out_dir)
        print(f'Wrote CSV: {csv_path}')
        print(f'Wrote figures: {png}, {svg}')
