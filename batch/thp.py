#!/usr/bin/env python3
"""
Parallel runner that varies the `--period-threshold` (thp) parameter for the
classifier pipeline, analogous to `batch/ths.py` which varied `ths`.

For each thp value this script will run:
  1) uv run python -m classifier.batch_process --cache <cache> --output-root <output_root>/thp{thp_safe} --period-threshold <thp> --sign-threshold <sign> --ignore-last-frames <n> --direct
  2) uv run python -m ablation.aggregate_labels --csv_dir <output_root>/thp{thp_safe} --output <agg_output> --name <name>

Supports optional `--compare` step to run `grid_graph\compare.py` after aggregation.

Example:
  python batch/thp.py --cache .\\cache\\diff_prompts\\lastframe_256 --output-root .\\output\\periods\\diff_thp --agg-output .\\output\\csv\\final2.0\\thp --thps 6.5 7.0 8.0 --sign 0.8 --workers 3 --compare
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import argparse
from pathlib import Path
import sys


def run_command(cmd, cwd=None):
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


def sanitize_thp(thp):
    # Convert float like 6.5 -> 6p5 (safe for filenames)
    s = str(thp)
    return s.replace('.', 'p')


def pipeline(thp, ths, cache, output_root, sign, ignore_last_frames, agg_output, name_template, cwd=None,
             compare=False, origin=None, compare_output_dir=None):
    thp_val = float(thp)
    thp_safe = sanitize_thp(thp_val)
    ths_val = int(ths)
    sign_val = float(ths_val) / 100.0
    out_dir = Path(output_root) / f"thp{thp_safe}_th{ths_val}"
    out_dir_str = str(out_dir)

    classifier_cmd = [
        "uv", "run", "python", "-m", "classifier.batch_process",
        "--cache", str(cache),
        "--output-root", out_dir_str,
        "--period-threshold", str(thp_val),
        "--sign-threshold", str(sign_val),
        "--ignore-last-frames", str(ignore_last_frames),
        "--direct",
    ]

    rc = run_command(classifier_cmd, cwd=cwd)
    if rc != 0:
        return (thp, rc, f"classifier failed with rc={rc}")

    name = name_template.format(thp=thp_val, thp_safe=thp_safe, ths=ths_val, sign=int(sign_val*100))

    agg_cmd = [
        "uv", "run", "python", "-m", "ablation.aggregate_labels",
        "--csv_dir", out_dir_str,
        "--output", str(agg_output),
        "--name", name,
    ]

    rc2 = run_command(agg_cmd, cwd=cwd)
    if rc2 != 0:
        return (thp, rc2, f"aggregate_labels failed with rc={rc2}")

    if compare:
        try:
            target_csv = str(Path(agg_output) / name)
            compare_out_dir = Path(compare_output_dir)
            compare_out_dir.mkdir(parents=True, exist_ok=True)
            out_img = compare_out_dir / f"thp{thp_safe}_th{ths_val}.png"
            compare_cmd = [
                "uv", "run", r"grid_graph\compare.py",
                "--origin", str(origin),
                "--target", target_csv,
                "--output", str(out_img),
            ]
            rc3 = run_command(compare_cmd, cwd=cwd)
            if rc3 != 0:
                return (thp, rc3, f"compare failed with rc={rc3}")
        except Exception as e:
            return (thp, 4, f"compare step exception: {e}")

    return (thp, 0, "ok")


def parse_args():
    p = argparse.ArgumentParser(description="Run multiple thp pipelines in parallel")
    p.add_argument('--cache', required=True)
    p.add_argument('--output-root', required=True)
    p.add_argument('--agg-output', required=True)
    p.add_argument('--period', nargs='+', type=float, required=True, help='List of period-thresholds to run (e.g. 6.5 7.0)')
    p.add_argument('--ths', nargs='+', type=int, required=True, help='List of ths percentages to run (e.g. 70 80 90)')
    p.add_argument('--sign', type=float, default=None, help='(deprecated) Sign threshold; use --ths to provide percentage(s)')
    p.add_argument('--ignore-last-frames', type=int, default=3)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--name-template', default='256prompt_{thp}thp_{ths}ths.csv',
                   help='Filename template for aggregated CSVs. Placeholders: {thp}, {thp_safe}, {ths}, {sign}')
    p.add_argument('--compare', action='store_true')
    p.add_argument('--origin', default='output/csv/final/diff_ths/256prompt_35thp_80ths.csv')
    p.add_argument('--compare-output-dir', default='output/grid_graph/thp')
    return p.parse_args()


def main():
    args = parse_args()
    cwd = Path.cwd()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.agg_output).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for thp in args.period:
            for ths in args.ths:
                fut = ex.submit(pipeline, thp, ths, args.cache, args.output_root, args.sign,
                                 args.ignore_last_frames, args.agg_output, args.name_template, cwd,
                                 compare=args.compare, origin=args.origin, compare_output_dir=args.compare_output_dir)
                futures[fut] = (thp, ths)
        for fut in as_completed(futures):
            thp, ths = futures[fut]
            try:
                res_thp, res_ths, rc, msg = None, None, None, None
                result = fut.result()
                # pipeline returns tuple (thp, 0/rc, msg) for backward compat; we adapt
                if isinstance(result, tuple) and len(result) == 3:
                    res_thp, rc, msg = result
                    res_ths = ths
                elif isinstance(result, tuple) and len(result) == 4:
                    res_thp, res_ths, rc, msg = result
                else:
                    res_thp, rc, msg = thp, 1, f"unexpected result: {result}"

                if rc == 0:
                    print(f"Pipeline thp={res_thp} ths={res_ths} completed successfully")
                else:
                    print(f"Pipeline thp={res_thp} ths={res_ths} FAILED: {msg}")
            except Exception as e:
                print(f"Pipeline thp={thp} ths={ths} raised exception: {e}")

    print("All thp pipelines finished.")


if __name__ == '__main__':
    main()
