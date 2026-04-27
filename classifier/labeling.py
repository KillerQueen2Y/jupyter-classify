"""
Simple head-labeling utilities.

Rules implemented:
- If any detected period (global_period or response_period) is finite and
  less than `period_threshold` (default 6.0) -> 'Wave Head'
- Otherwise load the attention vector and compute its mean: mean > 0 ->
  'Anchor Head', else 'Veil Head'.

This module exposes `label_head_from_result(result, period_threshold=6.0)`.
"""
from pathlib import Path
import math
import torch
import numpy as np


def _has_short_period(result, threshold: float) -> bool:
    # Use raw_results' global_period only for Wave detection.
    for entry in result.get("raw_results", []):
        val = entry.get("global_period")
        if val is None:
            continue
        try:
            if not (isinstance(val, float) and math.isnan(val)) and float(val) <= float(threshold):
                return True
        except Exception:
            continue
    return False


def _load_attention_from_result(result):
    """Return 1D numpy attention vector given a classify_attention result dict.

    The `attn_path` in result points to the layer file; use that to load the
    per-head tensor and select the head specified in result['head'].
    """
    attn_path = Path(result["attn_path"])  # full path to layer file
    payload = torch.load(attn_path, map_location="cpu", weights_only=False)
    per_head = payload["last_frame_attention_per_head"]
    head = int(result["head"])
    arr = per_head[head].detach().cpu().float().numpy()
    return arr


def label_head_from_result(result: dict, period_threshold: float = 6.0) -> str:
    """Return label string for a head result: 'Wave Head'|'Anchor Head'|'Veil Head'."""
    if _has_short_period(result, period_threshold):
        return "Wave Head"

    try:
        attn = _load_attention_from_result(result)
        mean = float(np.mean(attn))
    except Exception:
        # If we cannot load attention, fallback to using global_amp sign if available
        amps = []
        for e in result.get("raw_results", []):
            try:
                amps.append(float(e.get("global_amp", 0.0)))
            except Exception:
                pass
        mean = float(np.mean(amps)) if amps else 0.0

    return "Anchor Head" if mean > 0 else "Veil Head"
