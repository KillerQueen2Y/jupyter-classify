"""
Core classification utilities extracted from the notebook.
Provides functions to load attention, compute period spectra and a high-level
`classify_attention` function returning summarized results.
"""
from pathlib import Path
import json
import math

import torch
import numpy as np


def load_attention(attn_dir, layer, head):
    attn_path = Path(attn_dir) / f"layer{layer}.pt"
    payload = torch.load(attn_path, map_location="cpu", weights_only=False)

    per_head = payload["last_frame_attention_per_head"]
    if per_head.ndim != 2:
        raise ValueError(f"Expected per-head attention to be 2D, got shape {tuple(per_head.shape)}")
    if not 0 <= head < per_head.shape[0]:
        raise IndexError(f"HEAD={head} is out of range for {per_head.shape[0]} heads")

    attention = per_head[head].detach().cpu().float().numpy()
    if attention.ndim != 1:
        raise ValueError(f"Expected selected head attention to be 1D, got shape {attention.shape}")

    frame_idx = np.arange(attention.shape[0])
    return attention, frame_idx, str(attn_path)


def normalize_range(start, end, size):
    start = 0 if start is None else start
    end = size if end is None else end
    start_norm = start if start >= 0 else size + start
    end_norm = end if end >= 0 else size + end
    if start_norm < 0 or end_norm > size or start_norm >= end_norm:
        raise ValueError(f"Invalid FFT range {start}:{end} for {size} frames")
    return start, end, start_norm, end_norm


def softmax_1d(values, temperature=1.0):
    if temperature <= 0:
        raise ValueError("SOFTMAX_TEMPERATURE must be positive")
    scaled = values.astype(np.float64) / temperature
    scaled = scaled - np.max(scaled)
    exp_values = np.exp(scaled)
    return exp_values / exp_values.sum()


def local_softmax_1d(values, start, end, temperature=1.0):
    _, _, start_norm, end_norm = normalize_range(start, end, values.size)
    softmax_window = softmax_1d(values[start_norm:end_norm], temperature=temperature)
    softmax_full = np.full(values.shape, np.nan, dtype=np.float64)
    softmax_full[start_norm:end_norm] = softmax_window
    return softmax_full, softmax_window


def preprocess_signal(sequence, remove_dc=True, apply_window=True):
    if sequence.ndim != 1 or sequence.size < 2:
        raise ValueError(f"FFT input must be 1D with at least 2 frames, got shape {sequence.shape}")

    signal = sequence.astype(np.float64, copy=True)
    if remove_dc:
        signal = signal - signal.mean()
    if apply_window:
        signal = signal * np.hanning(signal.size)
    return signal


def compute_period_spectrum(
    sequence,
    period_min=2.0,
    period_max=None,
    response_period_min=4.0,
    response_period_max=18.0,
    min_response_cycles=3.0,
    remove_dc=True,
    apply_window=True,
):
    signal = preprocess_signal(sequence, remove_dc=remove_dc, apply_window=apply_window)
    freq = np.fft.rfftfreq(signal.size, d=1.0)
    amplitude = np.abs(np.fft.rfft(signal))
    nonzero_mask = freq > 0
    if not np.any(nonzero_mask):
        raise ValueError("FFT produced no nonzero frequency bins")

    period = 1.0 / freq[nonzero_mask]
    period_amplitude = amplitude[nonzero_mask]
    order = np.argsort(period)
    period = period[order]
    period_amplitude = period_amplitude[order]

    display_mask = np.ones_like(period, dtype=bool)
    if period_min is not None:
        display_mask &= period >= period_min
    if period_max is not None:
        display_mask &= period <= period_max
    if not np.any(display_mask):
        raise ValueError(f"Period display range [{period_min}, {period_max}] contains no FFT bins")

    global_idx = int(np.argmax(period_amplitude))
    cycles = signal.size / period
    response_mask = np.ones_like(period, dtype=bool)
    if response_period_min is not None:
        response_mask &= period >= response_period_min
    if response_period_max is not None:
        response_mask &= period <= response_period_max
    if min_response_cycles is not None:
        response_mask &= cycles >= min_response_cycles
    if np.any(response_mask):
        response_candidates = np.flatnonzero(response_mask)
        response_idx = int(response_candidates[np.argmax(period_amplitude[response_mask])])
    else:
        response_idx = None

    return {
        "period": period,
        "period_amplitude": period_amplitude,
        "display_mask": display_mask,
        "response_mask": response_mask,
        "global_period": float(period[global_idx]),
        "global_freq": float(1.0 / period[global_idx]),
        "global_amp": float(period_amplitude[global_idx]),
        "response_period": (math.nan if response_idx is None else float(period[response_idx])),
        "response_freq": (math.nan if response_idx is None else float(1.0 / period[response_idx])),
        "response_amp": (math.nan if response_idx is None else float(period_amplitude[response_idx])),
        "signal_size": int(signal.size),
    }


def analyze_fft_ranges(values, fft_ranges, **spectrum_kwargs):
    results = []
    for cfg in fft_ranges:
        start, end, start_norm, end_norm = normalize_range(cfg.get("start"), cfg.get("end"), values.size)
        spectrum = compute_period_spectrum(
            values[start:end],
            **spectrum_kwargs,
        )
        # Convert arrays to lists for JSON friendliness later
        spectrum["period"] = spectrum["period"].tolist()
        spectrum["period_amplitude"] = spectrum["period_amplitude"].tolist()
        spectrum["display_mask"] = spectrum["display_mask"].tolist()
        spectrum["response_mask"] = spectrum["response_mask"].tolist()
        results.append({**cfg, **spectrum, "start_norm": int(start_norm), "end_norm": int(end_norm)})
    return results


def classify_attention(
    attn_dir,
    layer,
    head,
    fft_ranges,
    softmax_range=(None, None),
    softmax_temperature=1.0,
    period_min=2.0,
    period_max=None,
    response_period_min=4.0,
    response_period_max=18.0,
    min_response_cycles=3.0,
    remove_dc=True,
    apply_window=True,
    ignore_last_frames: int = 3,
):
    attention, frame_idx, attn_path = load_attention(attn_dir, layer, head)
    # Optionally ignore the last N frames (e.g., to avoid self-attention spikes at the end)
    if ignore_last_frames and ignore_last_frames > 0:
        if attention.size - ignore_last_frames < 2:
            raise ValueError(f"Not enough frames after ignoring last {ignore_last_frames} frames")
        attention_used = attention[: attention.size - ignore_last_frames]
    else:
        attention_used = attention

    softmax_full, softmax_window = local_softmax_1d(
        attention_used, softmax_range[0], softmax_range[1], temperature=softmax_temperature
    )

    raw_results = analyze_fft_ranges(
        attention_used,
        fft_ranges,
        period_min=period_min,
        period_max=period_max,
        response_period_min=response_period_min,
        response_period_max=response_period_max,
        min_response_cycles=min_response_cycles,
        remove_dc=remove_dc,
        apply_window=apply_window,
    )

    softmax_results = analyze_fft_ranges(
        softmax_window,
        [ {"label": "softmax_window", "start": 0, "end": None} ],
        period_min=period_min,
        period_max=period_max,
        response_period_min=response_period_min,
        response_period_max=response_period_max,
        min_response_cycles=min_response_cycles,
        remove_dc=remove_dc,
        apply_window=apply_window,
    )

    return {
        "attn_path": attn_path,
        "layer": int(layer),
        "head": int(head),
        "frames": int(attention.size),
        "used_frames": int(attention_used.size),
        "ignored_tail": int(ignore_last_frames),
        "raw_results": raw_results,
        "softmax_results": softmax_results,
    }


def _example_usage():
    # Minimal example showing how to call classify_attention
    ATTN_DIR = Path("../lastframe_8/run_001")
    res = classify_attention(
        ATTN_DIR,
        layer=1,
        head=8,
        fft_ranges=[{"label": "0-68", "start": 0, "end": 69}],
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _example_usage()
