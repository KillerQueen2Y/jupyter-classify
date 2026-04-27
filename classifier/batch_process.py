"""
Batch processing script that walks `lastframe_8` runs and applies classification
from `core.classify_attention` for each layer and head. Results are saved as JSON
under `classifier/output/<run>/<layer>_head<k>.json`.

Run as:
    uv run python classifier/batch_process.py
"""
from pathlib import Path
import json
import traceback
import argparse
from classifier.core import classify_attention, load_attention
from classifier.labeling import label_head_from_result
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


DEFAULT_DATA_ROOT = Path("../lastframe_8")
DEFAULT_OUTPUT_ROOT = Path("output")

# Default FFT ranges used by the notebook; adapt if needed
FFT_RANGES = [
    {"label": "0-68", "start": 0, "end": 69},
    {"label": "0-71", "start": 0, "end": None},
]


def heads_in_layer(attn_file: Path):
    payload = torch.load(attn_file, map_location="cpu", weights_only=False)
    per_head = payload["last_frame_attention_per_head"]
    return int(per_head.shape[0])


def process_run(run_dir: Path, output_root: Path, cache_path: Path = None, period_threshold: float = 6.0, ignore_last_frames: int = 3):
    print(f"Processing run: {run_dir}")
    run_out = output_root / run_dir.name
    run_out.mkdir(parents=True, exist_ok=True)
    labels_by_layer = {}
    for layer_file in sorted(run_dir.glob("layer*.pt")):
        layer_name = layer_file.stem  # e.g., layer0
        try:
            n_heads = heads_in_layer(layer_file)
        except Exception as e:
            print(f"  Skipping {layer_file.name}: failed to read heads: {e}")
            continue

        layer_index = int(layer_name.replace("layer", ""))
        layer_out = run_out / layer_name
        layer_out.mkdir(parents=True, exist_ok=True)
        head_labels = []
        head_attns = []
        head_results = []
        for head in range(n_heads):
            try:
                # determine attention source: prefer cache_path/run_name, then cache_path itself, else input run_dir
                if cache_path:
                    candidate = cache_path / run_dir.name
                    if candidate.exists():
                        attn_source = candidate
                    elif cache_path.exists() and any(cache_path.glob("layer*.pt")):
                        attn_source = cache_path
                    else:
                        attn_source = run_dir
                else:
                    attn_source = run_dir

                result = classify_attention(
                    attn_source,
                    layer=layer_index,
                    head=head,
                    fft_ranges=FFT_RANGES,
                    ignore_last_frames=ignore_last_frames,
                )
                # add label and accumulate per-layer results (we'll write one JSON per layer)
                try:
                    label = label_head_from_result(result, period_threshold=period_threshold)
                except Exception:
                    label = "Unknown"
                result["label"] = label
                head_results.append(result)

                # collect for layer plot and labels
                head_labels.append(label)
                try:
                    attn_vec, frame_idx, _ = load_attention(attn_source, layer_index, head)
                except Exception:
                    try:
                        attn_vec, frame_idx, _ = load_attention(run_dir, layer_index, head)
                    except Exception:
                        attn_vec = None
                        frame_idx = None
                head_attns.append((attn_vec, frame_idx, result))
            except Exception:
                print(f"  Error processing {layer_file.name} head {head}:")
                traceback.print_exc()

        # create FL 图 for the layer (per-head bar+line grid, following reference style)
        try:
            n_heads_layer = len(head_attns)
            cols = min(5, max(1, n_heads_layer))
            rows = int(np.ceil(n_heads_layer / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(16, 3 * rows))
            axes = np.atleast_1d(axes).flatten()

            BAR_COLOR = sns.color_palette("colorblind")[0]

            # determine tick positions based on number of frames (use first non-None attn)
            sample_attn = None
            for attn_vec, frame_idx, _ in head_attns:
                if attn_vec is not None:
                    sample_attn = attn_vec
                    break
            num_frames = len(sample_attn) if sample_attn is not None else 0
            if num_frames <= 30:
                tick_step = 5
            elif num_frames <= 60:
                tick_step = 10
            else:
                tick_step = 20
            tick_pos = list(range(0, max(1, num_frames), tick_step)) + ([num_frames - 1] if num_frames>0 else [])
            tick_pos = sorted(set(tick_pos))

            for h in range(n_heads_layer):
                ax = axes[h]
                attn_vec, frame_idx, res = head_attns[h]

                if attn_vec is not None and frame_idx is not None:
                    key_indices = np.arange(len(attn_vec))
                    ax.bar(key_indices, attn_vec, alpha=0.85, width=0.8, color=BAR_COLOR)
                    markersize = max(1, 4 - num_frames // 20) if num_frames>0 else 3
                    ax.plot(key_indices, attn_vec, "o-", color="black", linewidth=1, markersize=markersize)
                    try:
                        ax.set_ylim(np.nanmin(attn_vec) - 0.01, np.nanmax(attn_vec) + 0.01)
                    except Exception:
                        pass
                    ax.set_xticks(tick_pos)
                ht_str = head_labels[h] if h < len(head_labels) else 'unknown'
                # annotate periods if available
                period_str = ''
                try:
                    metrics = {}
                    if res.get('raw_results'):
                        rr = res.get('raw_results')[0]
                        metrics['p'] = rr.get('global_period')
                        metrics['W'] = rr.get('global_amp')
                    parts = []
                    if metrics.get('p') is not None:
                        parts.append(f"p={metrics['p']:.1f}")
                    if metrics.get('W') is not None:
                        parts.append(f"W={metrics['W']:.3f}")
                    if parts:
                        period_str = ' ' + ' '.join(parts)
                except Exception:
                    period_str = ''

                ax.set_title(f"H{h} ({ht_str}){period_str}", fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=7)

            # hide unused axes
            for k in range(n_heads_layer, len(axes)):
                axes[k].axis('off')

            fig.suptitle(f"Layer {layer_index}: Per-Head Attention Distribution", fontsize=12, fontweight='bold')
            plt.tight_layout()
            # save SVG (preferred) and PNG
            svg_path = layer_out / f"{layer_name}_FL.svg"
            png_path = layer_out / f"{layer_name}_FL.png"
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            fig.savefig(png_path, format='png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        except Exception:
            print(f"  Failed to create FL image for {layer_name}")
            traceback.print_exc()

        def _label_to_num(lbl):
            if lbl == 'Wave Head':
                return -1
            if lbl == 'Anchor Head':
                return 1
            if lbl == 'Veil Head':
                return 2
            return 0

        labels_by_layer[layer_index] = [_label_to_num(x) for x in head_labels]

        # write one JSON per layer containing all heads' results
        try:
            layer_json = {
                "layer": layer_index,
                "n_heads": n_heads,
                "heads": head_results,
            }
            layer_json_path = layer_out / f"{layer_name}.json"
            with open(layer_json_path, "w", encoding="utf-8") as f:
                json.dump(layer_json, f, indent=2)
            print(f"  Wrote {layer_json_path.relative_to(Path.cwd())}")
        except Exception:
            print(f"  Failed to write layer JSON for {layer_name}")
            traceback.print_exc()
    return labels_by_layer


def parse_args():
    p = argparse.ArgumentParser(description="Batch classify attention files in a directory of runs")
    p.add_argument("--cache", required=False, default=str(DEFAULT_DATA_ROOT),
                   help="Path to top-level lastframe directory to process (used as data root).")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT),
                   help="Where to write per-run output directories. Default: classifier/output")
    p.add_argument("--period-threshold", type=float, default=6.0,
                   help="Period threshold (frames) below which a head is labeled 'Wave Head'. Default: 6.0")
    p.add_argument("--ignore-last-frames", type=int, default=3,
                   help="Number of frames at the end to ignore when computing periods (default: 3)")
    return p.parse_args()


def main(data_root: Path, output_root: Path, cache_root: Path = None, period_threshold: float = 6.0, ignore_last_frames: int = 3):
    data_root = data_root.resolve()
    output_root = output_root.resolve()

    if not data_root.exists():
        print(f"Data root {data_root} does not exist. Adjust the path.")
        return

    cache_path = cache_root.resolve() if cache_root else None

    for run_dir in sorted(data_root.iterdir()):
        if not run_dir.is_dir():
            continue
        labels_by_layer = process_run(run_dir, output_root, cache_path=cache_path, period_threshold=period_threshold, ignore_last_frames=ignore_last_frames)
        # assemble CSV and compose one large FL image (vertical concatenation of per-layer FLs)
        try:
            if labels_by_layer:
                max_layer = max(labels_by_layer.keys())
                n_layers = max_layer + 1
                max_heads = max((len(v) for v in labels_by_layer.values()), default=0)
                n_cols = max(12, max_heads)
                mat = np.zeros((n_layers, n_cols), dtype=int)
                for li, lbls in labels_by_layer.items():
                    for hi, val in enumerate(lbls):
                        mat[li, hi] = int(val)

                run_out = output_root / run_dir.name
                run_out.mkdir(parents=True, exist_ok=True)
                csv_path = run_out / "labels.csv"
                np.savetxt(csv_path, mat, fmt='%d', delimiter=',')

                # compose big FL image by reading each layer's PNG and stacking vertically
                layer_images = []
                widths = []
                heights = []
                for li in range(n_layers):
                    layer_name = f"layer{li}"
                    img_path = run_out / layer_name / f"{layer_name}_FL.png"
                    if img_path.exists():
                        try:
                            img = Image.open(img_path).convert('RGBA')
                            layer_images.append(img)
                            widths.append(img.width)
                            heights.append(img.height)
                        except Exception:
                            print(f"  Failed to open {img_path}")
                            continue

                if layer_images:
                    max_w = max(widths)
                    total_h = sum(heights)
                    composite = Image.new('RGBA', (max_w, total_h), (255, 255, 255, 255))
                    y = 0
                    for img in layer_images:
                        # pad to max width if needed
                        if img.width < max_w:
                            padded = Image.new('RGBA', (max_w, img.height), (255, 255, 255, 255))
                            padded.paste(img, (0, 0))
                            composite.paste(padded, (0, y))
                        else:
                            composite.paste(img, (0, y))
                        y += img.height

                    out_img_path = run_out / "all_layers_FL.png"
                    composite.convert('RGB').save(out_img_path, format='PNG')
                    print(f"  Wrote combined FL image: {out_img_path.relative_to(Path.cwd())}")
                else:
                    print("  No layer FL images found to compose a combined FL image.")
        except Exception:
            print(f"Failed to write labels CSV or compose FL image for {run_dir}")
            traceback.print_exc()


if __name__ == "__main__":
    args = parse_args()
    data_root = Path(args.cache)
    output_root = Path(args.output_root)
    # use --cache as the top-level data root; no separate cache path
    main(data_root, output_root, cache_root=None, period_threshold=args.period_threshold, ignore_last_frames=args.ignore_last_frames)
