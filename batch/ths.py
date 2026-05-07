#!/usr/bin/env python3
"""
Parallel runner for multiple `ths` pipelines.

For each th in the provided list (e.g. 70,80,90) this script will run:
  1) uv run python -m classifier.batch_process --cache <cache> --output-root <output_root>/th{th} --period-threshold <period> --sign-threshold <sign> --ignore-last-frames <n> --direct
  2) uv run python -m ablation.aggregate_labels --csv_dir <output_root>/th{th} --output <agg_output> --name <name>

By default it runs the three ths 70,80,90 in parallel (one pipeline per worker).

Example:
  python batch/ths.py --cache .\\cache\\diff_prompts\\lastframe_256 --output-root .\\output\\period6\\diff_ths --agg-output .\\output\\csv\\final2.0\\ths --period 6.5 --ths 70 80 90 --workers 3

Note: This script shells out to the `uv` runner exactly as in your examples, so it should behave like running the commands in your shell.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import argparse
import os
from pathlib import Path
import shlex
import sys


def run_command(cmd, cwd=None):
    # Accept either a sequence (recommended) or a shell string.
    if isinstance(cmd, (list, tuple)):
        print("Running:", " ".join(cmd))
        try:
            completed = subprocess.run(cmd, cwd=cwd)
            return completed.returncode
        except Exception as e:
            print(f"Command failed to start: {e}")
            return 2
    else:
        print(f"Running (shell): {cmd}")
        try:
            completed = subprocess.run(cmd, shell=True, cwd=cwd)
            return completed.returncode
        except Exception as e:
            print(f"Command failed to start: {e}")
            return 2


def pipeline(th, cache, output_root, period, sign, ignore_last_frames, agg_output, name_template, cwd=None,
             compare=False, origin=None, compare_output_dir=None):
    """Run classifier then aggregation for a single threshold value (th in percent, e.g. 70).
    Returns (th, rc, message)
    """
    sign_val = float(sign)
    # if sign was given as percent (like 70) convert to fraction; but caller will pass fraction.
    # We accept both: if sign > 1 assume percent.
    if sign_val > 1:
        sign_val = sign_val / 100.0

    th_dir = Path(output_root) / f"th{th}"
    th_dir_str = str(th_dir)

    # build classifier command as a list to avoid quoting issues on Windows
    classifier_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "classifier.batch_process",
        "--cache",
        str(cache),
        "--output-root",
        str(th_dir),
        "--period-threshold",
        str(period),
        "--sign-threshold",
        str(sign_val),
        "--ignore-last-frames",
        str(ignore_last_frames),
        "--direct",
    ]

    rc = run_command(classifier_cmd, cwd=cwd)
    if rc != 0:
        return (th, rc, f"classifier failed with rc={rc}")

    # build aggregation name
    # name_template may contain placeholders: {th}, {period}, {sign}
    name = name_template.format(th=th, period=period, sign=int(sign_val*100))

    agg_cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "ablation.aggregate_labels",
        "--csv_dir",
        th_dir_str,
        "--output",
        str(agg_output),
        "--name",
        name,
    ]

    rc2 = run_command(agg_cmd, cwd=cwd)
    if rc2 != 0:
        return (th, rc2, f"aggregate_labels failed with rc={rc2}")
    # Optionally run grid_graph compare
    if compare:
        try:
            target_csv = str(Path(agg_output) / name)
            compare_out_dir = Path(compare_output_dir)
            compare_out_dir.mkdir(parents=True, exist_ok=True)
            # format period as safe string (replace '.' with 'p') so names like thp6p5_th80.png
            period_safe = str(period).replace('.', 'p')
            out_img = compare_out_dir / f"thp{period_safe}_th{th}.png"
            compare_cmd = [
                "uv",
                "run",
                r"grid_graph\compare.py",
                "--origin",
                str(origin),
                "--target",
                target_csv,
                "--output",
                str(out_img),
            ]
            rc3 = run_command(compare_cmd, cwd=cwd)
            if rc3 != 0:
                return (th, rc3, f"compare failed with rc={rc3}")
        except Exception as e:
            return (th, 4, f"compare step exception: {e}")

    return (th, 0, "ok")


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple ths pipelines in parallel")
    p.add_argument('--cache', required=True, help='Cache path to pass to classifier (e.g. .\\cache\\diff_prompts\\lastframe_256)')
    p.add_argument('--output-root', required=True, help='Base output root for classifier (each th will create a subdir th{th})')
    p.add_argument('--agg-output', required=True, help='Directory where aggregated CSVs will be written')
    p.add_argument('--period', type=float, default=6.5, help='Period threshold to pass to classifier (e.g. 6.5)')
    p.add_argument('--sign', type=float, default=0.7, help='Default sign threshold (fraction or percent>1)')
    p.add_argument('--ignore-last-frames', type=int, default=3, help='ignore-last-frames passed to classifier')
    p.add_argument('--ths', nargs='+', type=int, default=[70,80,90], help='List of th percentages to run (default: 70 80 90)')
    p.add_argument('--workers', type=int, default=3, help='Number of parallel pipelines to run')
    p.add_argument('--name-template', default='256prompt_{period}thp_{th}ths.csv',
                   help='Filename template for aggregated CSVs. Placeholders: {th}, {period}, {sign}')
    p.add_argument('--compare', action='store_true', help='If set, run grid_graph compare after aggregation')
    p.add_argument('--origin', default='output/csv/final/diff_ths/256prompt_35thp_80ths.csv',
                   help='Default origin CSV used by grid_graph compare')
    p.add_argument('--compare-output-dir', default='output/grid_graph/ths',
                   help='Directory where compare PNGs will be written')
    return p.parse_args()


def main():
    args = parse_args()
    cwd = Path.cwd()

    # Ensure output dirs exist
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.agg_output).mkdir(parents=True, exist_ok=True)

    jobs = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for th in args.ths:
            # compute sign value from th if desired: user may want to vary sign by th.
            # But from the examples, sign corresponds directly to th/100 for 70->0.7 etc.
            sign_for_th = th / 100.0
            fut = ex.submit(pipeline, th, args.cache, args.output_root, args.period, sign_for_th,
                             args.ignore_last_frames, args.agg_output, args.name_template, cwd,
                             compare=args.compare, origin=args.origin, compare_output_dir=args.compare_output_dir)
            futures[fut] = th

        for fut in as_completed(futures):
            th = futures[fut]
            try:
                th, rc, msg = fut.result()
                if rc == 0:
                    print(f"Pipeline th={th} completed successfully")
                else:
                    print(f"Pipeline th={th} FAILED: {msg}")
            except Exception as e:
                print(f"Pipeline th={th} raised exception: {e}")

    print("All pipelines finished.")


if __name__ == '__main__':
    main()
