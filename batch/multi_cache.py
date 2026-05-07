#!/usr/bin/env python3
"""
Run classifier + aggregate_labels over multiple cache subdirectories (e.g.
cache/diff_frame/lastframe_5s, lastframe_10s, ...).

Behavior:
- For each matched data dir, run:
    uv run python -m classifier.batch_process --cache <data_dir> --output-root <output_root>/<data_dir_name> --period-threshold <period> --sign-threshold <sign> --ignore-last-frames <n> --direct
  then
    uv run python -m ablation.aggregate_labels --csv_dir <output_root>/<data_dir_name> --output <agg_output> --name <name>

Defaults: period=10, ths=80 (sign=0.8), no compare step.

Example:
  python batch/multi_cache.py --cache-root .\\cache\\diff_frame --output-root .\\output\\period10\\diff_frame --agg-output .\\output\\csv\\final2.0\\multi --workers 4
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import argparse
from pathlib import Path
import sys
import fnmatch
import time
import json
import torch
import os
import threading

# psutil optional
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False


def human_bytes(n):
    """Return human-readable string for bytes (e.g., '12.3MB')."""
    if n is None:
        return None
    try:
        n = float(n)
    except Exception:
        return None
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = 0
    while n >= 1024.0 and idx < len(units) - 1:
        n /= 1024.0
        idx += 1
    # show one decimal for KB+ else integer for bytes
    if units[idx] == 'B':
        return f"{int(n)}{units[idx]}"
    return f"{n:.1f}{units[idx]}"


def run_command(cmd, cwd=None, dry_run=False, capture_output=True):
    """Run command, sample memory/cpu and return structured result.

    This implementation uses psutil when available for accurate sampling.
    """
    if dry_run:
        text = " ".join(cmd) if isinstance(cmd, (list, tuple)) else str(cmd)
        print("DRY-RUN:", text)
        return {
            "cmd": cmd,
            "start_iso": None,
            "end_iso": None,
            "duration_s": 0.0,
            "duration_hms": "0:00:00",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "max_rss_bytes": 0,
            "avg_rss_bytes": None,
            "max_cpu_percent": None,
            "avg_cpu_percent": None,
        }

    start_ts = time.time()
    start_iso = __import__("datetime").datetime.fromtimestamp(start_ts, __import__("datetime").timezone.utc).isoformat().replace('+00:00','Z')

    # Launch process
    popen_args = {}
    if capture_output:
        # merge stderr into stdout so we can stream a single stream
        popen_args.update({"stdout": subprocess.PIPE, "stderr": subprocess.STDOUT, "text": True})

    try:
        # print the command so user sees progress when long-running
        try:
            if isinstance(cmd, (list, tuple)):
                print("Running:", " ".join(cmd), flush=True)
            else:
                print(f"Running (shell): {cmd}", flush=True)
        except Exception:
            pass

        if isinstance(cmd, (list, tuple)):
            proc = subprocess.Popen(cmd, cwd=cwd, **popen_args)
        else:
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, **popen_args)
    except Exception as e:
        end_ts = time.time()
        end_iso = __import__("datetime").datetime.fromtimestamp(end_ts, __import__("datetime").timezone.utc).isoformat().replace('+00:00','Z')
        return {
            "cmd": cmd,
            "start_iso": start_iso,
            "end_iso": end_iso,
            "duration_s": end_ts - start_ts,
            "duration_hms": "0:00:00",
            "returncode": 2,
            "stdout": "",
            "stderr": str(e),
            "max_rss_bytes": 0,
            "avg_rss_bytes": None,
            "max_cpu_percent": None,
            "avg_cpu_percent": None,
        }

    mem_samples = []
    cpu_samples = []
    ps_proc = None
    if HAS_PSUTIL:
        try:
            ps_proc = psutil.Process(proc.pid)
            try:
                ps_proc.cpu_percent(None)
            except Exception:
                pass
        except Exception:
            ps_proc = None

    # if capturing output, start a thread to stream and accumulate stdout
    out_lines = []
    def _reader_thread(pipe, collector):
        try:
            for line in iter(pipe.readline, ''):
                if not line:
                    break
                collector.append(line)
                try:
                    # print live to console
                    print(line, end='', flush=True)
                except Exception:
                    pass
        except Exception:
            pass

    reader = None
    if capture_output and proc.stdout is not None:
        reader = threading.Thread(target=_reader_thread, args=(proc.stdout, out_lines), daemon=True)
        reader.start()

    # poll loop
    while True:
        ret = proc.poll()
        try:
            if ps_proc is not None:
                mi = ps_proc.memory_info()
                mem_samples.append(int(getattr(mi, 'rss', 0)))
                try:
                    cpu_samples.append(ps_proc.cpu_percent(None))
                except Exception:
                    pass
            else:
                # minimal fallback sampling (may be coarse)
                try:
                    if os.name == 'nt':
                        out = subprocess.check_output(['tasklist', '/fi', f'PID eq {proc.pid}', '/fo', 'CSV', '/nh'], text=True)
                        parts = out.strip().split(',')
                        if parts and len(parts) >= 5:
                            mem_str = parts[-1].strip().strip('"').replace(' K', '').replace(',', '')
                            mem_samples.append(int(mem_str) * 1024)
                    else:
                        out = subprocess.check_output(['ps', '-o', 'rss=', '-p', str(proc.pid)], text=True)
                        rss_k = int(out.strip())
                        mem_samples.append(rss_k * 1024)
                except Exception:
                    pass
        except Exception:
            pass

        if ret is not None:
            break
        time.sleep(0.1)

    # collect outputs: if we used a reader thread it already filled out_lines
    out = ''
    err = ''
    try:
        if capture_output:
            # wait a short while for reader to finish
            if reader is not None:
                reader.join(timeout=2)
            out = ''.join(out_lines)
            err = ''
        else:
            try:
                out, err = proc.communicate(timeout=1)
            except Exception:
                out, err = ('', '')
    except Exception:
        out, err = ('', '')

    end_ts = time.time()
    end_iso = __import__("datetime").datetime.fromtimestamp(end_ts, __import__("datetime").timezone.utc).isoformat().replace('+00:00','Z')

    stdout = out if out is not None else ''
    stderr = err if err is not None else ''
    MAX_OUT = 20000
    if len(stdout) > MAX_OUT:
        stdout = stdout[:MAX_OUT] + "\n...TRUNCATED..."
    if len(stderr) > MAX_OUT:
        stderr = stderr[:MAX_OUT] + "\n...TRUNCATED..."

    duration = end_ts - start_ts
    secs = int(round(duration))
    h = secs // 3600
    m_ = (secs % 3600) // 60
    s = secs % 60
    duration_hms = f"{h}:{m_:02d}:{s:02d}"

    max_rss = max(mem_samples) if mem_samples else 0
    avg_rss = int(sum(mem_samples) / len(mem_samples)) if mem_samples else None
    max_cpu = max(cpu_samples) if cpu_samples else None
    avg_cpu = float(sum(cpu_samples) / len(cpu_samples)) if cpu_samples else None

    res = {
        "cmd": cmd,
        "start_iso": start_iso,
        "end_iso": end_iso,
        "duration_s": duration,
        "duration_hms": duration_hms,
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "max_rss_bytes": max_rss,
        "avg_rss_bytes": avg_rss,
        "max_cpu_percent": max_cpu,
        "avg_cpu_percent": avg_cpu,
        "max_rss_human": human_bytes(max_rss) if max_rss is not None else None,
        "avg_rss_human": human_bytes(avg_rss) if avg_rss is not None else None,
    }
    return res


def sanitize_period(p):
    return str(p).replace('.', 'p')


def process_data_dir(data_dir: Path, output_root: Path, agg_output: Path, period: float, ths: int,
                     ignore_last_frames: int, name_template: str, dry_run: bool, cwd: Path,
                     log_root: Path = Path("logs")):
    data_name = data_dir.name
    out_dir = output_root / data_name
    out_dir_str = str(out_dir)

    sign_val = float(ths) / 100.0

    classifier_cmd = [
        "uv", "run", "python", "-m", "classifier.batch_process",
        "--cache", str(data_dir),
        "--output-root", out_dir_str,
        "--period-threshold", str(period),
        "--sign-threshold", str(sign_val),
        "--ignore-last-frames", str(ignore_last_frames),
        "--direct",
    ]

    # try to infer number of frames from a layer*.pt file (for per-frame cost)
    frames = None
    try:
        for pf in data_dir.glob('layer*.pt'):
            try:
                payload = torch.load(pf, map_location='cpu', weights_only=False)
                per_head = payload.get('last_frame_attention_per_head')
                if per_head is not None:
                    # per_head shape expected (n_heads, n_frames)
                    shape = getattr(per_head, 'shape', None)
                    if shape and len(shape) >= 2:
                        frames = int(shape[1])
                        break
            except Exception:
                continue
    except Exception:
        frames = None

    print(f"[{data_name}] START classifier", flush=True)
    # system snapshot before classifier
    sys_before = None
    if HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            sys_before = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_total": int(getattr(vm, 'total', 0)),
                "mem_available": int(getattr(vm, 'available', 0)),
            }
        except Exception:
            sys_before = None

    clf_res = run_command(classifier_cmd, cwd=cwd, dry_run=dry_run)
    # system snapshot after classifier
    sys_after = None
    if HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            sys_after = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_total": int(getattr(vm, 'total', 0)),
                "mem_available": int(getattr(vm, 'available', 0)),
            }
        except Exception:
            sys_after = None
    rc = clf_res.get("returncode", 1) if isinstance(clf_res, dict) else (clf_res or 1)
    # annotate system snapshots into classifier result (for logging)
    try:
        if isinstance(clf_res, dict):
            clf_res['system_before'] = sys_before
            clf_res['system_after'] = sys_after
            if sys_before and sys_before.get('mem_total'):
                clf_res['system_before']['mem_total_human'] = human_bytes(sys_before.get('mem_total'))
            if sys_after and sys_after.get('mem_total'):
                clf_res['system_after']['mem_total_human'] = human_bytes(sys_after.get('mem_total'))
    except Exception:
        pass

    print(f"[{data_name}] classifier rc={rc} duration={clf_res.get('duration_hms')} max_cpu={clf_res.get('max_cpu_percent')} max_mem={clf_res.get('max_rss_human')}", flush=True)
    if rc != 0:
        # write classifier log even on failure (if possible)
        try:
            if not dry_run:
                data_log_dir = Path(log_root) / data_name
                data_log_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                with open(data_log_dir / f"classifier_{ts}.json", "w", encoding="utf-8") as f:
                    json.dump(clf_res, f, indent=2)
        except Exception:
            pass
        return (data_name, rc, "classifier failed")

    # annotate classifier result with per-frame cost (if we inferred frames)
    try:
        if isinstance(clf_res, dict) and frames:
            dur = clf_res.get('duration_s') or 0.0
            clf_res['frames'] = frames
            clf_res['secs_per_frame'] = dur / frames if frames else None
    except Exception:
        pass

    # record classifier timing/log
    try:
        if not dry_run:
            data_log_dir = Path(log_root) / data_name
            data_log_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            with open(data_log_dir / f"classifier_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(clf_res, f, indent=2)
    except Exception:
        pass

    name = name_template.format(period=period, period_safe=sanitize_period(period), ths=ths, data=data_name)

    # write aggregated CSVs into a per-data subdirectory to avoid overwriting/mixing
    agg_out_for_data = Path(agg_output) / data_name
    agg_out_for_data.mkdir(parents=True, exist_ok=True)

    agg_cmd = [
        "uv", "run", "python", "-m", "ablation.aggregate_labels",
        "--csv_dir", out_dir_str,
        "--output", str(agg_out_for_data),
        "--name", name,
    ]

    print(f"[{data_name}] START aggregate", flush=True)
    # system snapshot before aggregate
    agg_sys_before = None
    if HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            agg_sys_before = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_total": int(getattr(vm, 'total', 0)),
                "mem_available": int(getattr(vm, 'available', 0)),
            }
        except Exception:
            agg_sys_before = None

    agg_res = run_command(agg_cmd, cwd=cwd, dry_run=dry_run)
    # system snapshot after aggregate
    agg_sys_after = None
    if HAS_PSUTIL:
        try:
            vm = psutil.virtual_memory()
            agg_sys_after = {
                "cpu_percent": psutil.cpu_percent(interval=None),
                "mem_total": int(getattr(vm, 'total', 0)),
                "mem_available": int(getattr(vm, 'available', 0)),
            }
        except Exception:
            agg_sys_after = None
    rc2 = agg_res.get("returncode", 1) if isinstance(agg_res, dict) else (agg_res or 1)
    try:
        if isinstance(agg_res, dict):
            agg_res['system_before'] = agg_sys_before
            agg_res['system_after'] = agg_sys_after
            if agg_sys_before and agg_sys_before.get('mem_total'):
                agg_res['system_before']['mem_total_human'] = human_bytes(agg_sys_before.get('mem_total'))
            if agg_sys_after and agg_sys_after.get('mem_total'):
                agg_res['system_after']['mem_total_human'] = human_bytes(agg_sys_after.get('mem_total'))
    except Exception:
        pass

    print(f"[{data_name}] aggregate rc={rc2} duration={agg_res.get('duration_hms')} max_cpu={agg_res.get('max_cpu_percent')} max_mem={agg_res.get('max_rss_human')}", flush=True)
    if rc2 != 0:
        try:
            if not dry_run:
                data_log_dir = Path(log_root) / data_name
                data_log_dir.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                with open(data_log_dir / f"aggregate_{ts}.json", "w", encoding="utf-8") as f:
                    json.dump(agg_res, f, indent=2)
        except Exception:
            pass
        return (data_name, rc2, "aggregate failed")

    # annotate aggregate result with per-run cost (number of runs processed)
    try:
        if isinstance(agg_res, dict):
            try:
                n_runs = sum(1 for p in out_dir.iterdir() if p.is_dir())
            except Exception:
                n_runs = 0
            if n_runs > 0:
                dur = agg_res.get('duration_s') or 0.0
                agg_res['n_runs'] = n_runs
                agg_res['secs_per_run'] = dur / n_runs if n_runs else None
    except Exception:
        pass

    # record aggregate timing/log
    try:
        if not dry_run:
            data_log_dir = Path(log_root) / data_name
            data_log_dir.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            with open(data_log_dir / f"aggregate_{ts}.json", "w", encoding="utf-8") as f:
                json.dump(agg_res, f, indent=2)
    except Exception:
        pass

    return (data_name, 0, "ok")


def parse_args():
    p = argparse.ArgumentParser(description="Run classifier+aggregate over multiple cache subdirs")
    p.add_argument('--cache-root', default='cache/diff_frame', help='Parent directory containing lastframe_* subdirs')
    p.add_argument('--pattern', default='lastframe_*', help='Pattern to match subdirectories under cache-root')
    p.add_argument('--output-root', default='output/period10/diff_frame', help='Base output root for classifier')
    p.add_argument('--agg-output', default='output/csv/final2.0/multi', help='Directory where aggregated CSVs will be written')
    p.add_argument('--period', type=float, default=10.0, help='Period threshold to use (default: 10)')
    p.add_argument('--ths', type=int, default=80, help='ths percentage to use for sign threshold (default: 80)')
    p.add_argument('--ignore-last-frames', type=int, default=3)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--name-template', default='256prompt_{period}thp_{ths}ths_{data}.csv',
                   help='Template for aggregated CSV filenames. Placeholders: {period}, {period_safe}, {ths}, {data}')
    p.add_argument('--dry-run', action='store_true', help='Print commands without running')
    p.add_argument('--log-root', default='logs', help='Directory where per-data logs will be written (subfolders per data)')
    return p.parse_args()


def main():
    args = parse_args()
    cwd = Path.cwd()
    cache_root = Path(args.cache_root)
    if not cache_root.exists():
        print(f"cache-root not found: {cache_root}")
        sys.exit(1)

    # collect matching subdirs
    subdirs = [p for p in sorted(cache_root.iterdir()) if p.is_dir() and fnmatch.fnmatch(p.name, args.pattern)]
    if not subdirs:
        print(f"No subdirectories matching {args.pattern} under {cache_root}")
        sys.exit(1)

    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    Path(args.agg_output).mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for d in subdirs:
            fut = ex.submit(process_data_dir, d, Path(args.output_root), Path(args.agg_output), args.period,
                             args.ths, args.ignore_last_frames, args.name_template, args.dry_run, cwd, Path(args.log_root))
            futures[fut] = d.name

        for fut in as_completed(futures):
            name = futures[fut]
            try:
                data_name, rc, msg = fut.result()
                if rc == 0:
                    print(f"{data_name}: completed successfully")
                else:
                    print(f"{data_name}: FAILED ({msg}, rc={rc})")
            except Exception as e:
                print(f"{name}: raised exception: {e}")

    # After all tasks complete, write a summary CSV into the log root
    try:
        import csv
        summary_path = Path(args.log_root) / 'summary.csv'
        rows = []
        header = [
            'data_name',
            'classifier_returncode', 'classifier_duration_s', 'classifier_duration_hms', 'classifier_max_cpu', 'classifier_avg_cpu', 'classifier_max_mem',
            'aggregate_returncode', 'aggregate_duration_s', 'aggregate_duration_hms', 'aggregate_max_cpu', 'aggregate_avg_cpu', 'aggregate_max_mem',
            'system_cpu_before', 'system_cpu_after', 'system_mem_total_before', 'system_mem_total_after'
        ]

        for d in subdirs:
            dn = d.name
            logdir = Path(args.log_root) / dn
            clf_vals = {}
            agg_vals = {}
            # find latest classifier_*.json and aggregate_*.json
            try:
                if logdir.exists():
                    clf_files = sorted(logdir.glob('classifier_*.json'))
                    agg_files = sorted(logdir.glob('aggregate_*.json'))
                    if clf_files:
                        with open(clf_files[-1], 'r', encoding='utf-8') as f:
                            clf_vals = json.load(f)
                    if agg_files:
                        with open(agg_files[-1], 'r', encoding='utf-8') as f:
                            agg_vals = json.load(f)
            except Exception:
                pass

            row = {
                'data_name': dn,
                'classifier_returncode': clf_vals.get('returncode') if isinstance(clf_vals, dict) else None,
                'classifier_duration_s': clf_vals.get('duration_s') if isinstance(clf_vals, dict) else None,
                'classifier_duration_hms': clf_vals.get('duration_hms') if isinstance(clf_vals, dict) else None,
                'classifier_max_cpu': clf_vals.get('max_cpu_percent') if isinstance(clf_vals, dict) else None,
                'classifier_avg_cpu': clf_vals.get('avg_cpu_percent') if isinstance(clf_vals, dict) else None,
                'classifier_max_mem': clf_vals.get('max_rss_human') if isinstance(clf_vals, dict) else None,
                'aggregate_returncode': agg_vals.get('returncode') if isinstance(agg_vals, dict) else None,
                'aggregate_duration_s': agg_vals.get('duration_s') if isinstance(agg_vals, dict) else None,
                'aggregate_duration_hms': agg_vals.get('duration_hms') if isinstance(agg_vals, dict) else None,
                'aggregate_max_cpu': agg_vals.get('max_cpu_percent') if isinstance(agg_vals, dict) else None,
                'aggregate_avg_cpu': agg_vals.get('avg_cpu_percent') if isinstance(agg_vals, dict) else None,
                'aggregate_max_mem': agg_vals.get('max_rss_human') if isinstance(agg_vals, dict) else None,
                'system_cpu_before': (clf_vals.get('system_before') or {}).get('cpu_percent') if isinstance(clf_vals, dict) else None,
                'system_cpu_after': (clf_vals.get('system_after') or {}).get('cpu_percent') if isinstance(clf_vals, dict) else None,
                'system_mem_total_before': (clf_vals.get('system_before') or {}).get('mem_total') if isinstance(clf_vals, dict) else None,
                'system_mem_total_after': (clf_vals.get('system_after') or {}).get('mem_total') if isinstance(clf_vals, dict) else None,
            }
            rows.append(row)

        # write CSV
        with open(summary_path, 'w', newline='', encoding='utf-8') as cf:
            writer = csv.DictWriter(cf, fieldnames=header)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"Wrote summary to {summary_path}")
    except Exception as e:
        print(f"Failed to write summary: {e}")

    print("All done.")


if __name__ == '__main__':
    main()
