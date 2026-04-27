"""Simple head-labeling utilities.

Rules implemented (updated):
- First, if an attention vector can be loaded, compute the fraction of
    positive values (positive rate). If `positive_rate >= sign_threshold`
    -> 'Anchor Head'. If `(1 - positive_rate) >= sign_threshold` -> 'Veil Head'.
- Next, if any detected period (global_period or response_period) is finite
    and less than `period_threshold` -> 'Wave Head'.
- Otherwise compute the attention mean: mean > 0 -> 'Anchor Head', else
    'Veil Head'.

The main entrypoint is `label_head_from_result(result, period_threshold=6.0,
sign_threshold=0.9)`.
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


def label_head_from_result(result: dict, period_threshold: float = 6.0, sign_threshold: float = 0.9) -> str:
    """Return label string for a head result: 'Wave Head'|'Anchor Head'|'Veil Head'.

    Ordering:
    1. Try sign-rate based decision if attention vector is available.
    2. Wave detection via period threshold `period_threshold`.
    3. Fallback to attention mean (or global_amp sign) as before.

    `sign_threshold` should be in [0.0, 1.0].
    """
    attn = None
    try:
        attn = _load_attention_from_result(result)
    except Exception:
        attn = None

    # 1) Sign-rate based decision
    try:
        if attn is not None and sign_threshold is not None:
            if not (0.0 <= float(sign_threshold) <= 1.0):
                raise ValueError("sign_threshold must be in [0,1]")
            pos_rate = float(np.mean(attn > 0))
            neg_rate = 1.0 - pos_rate
            if pos_rate >= float(sign_threshold):
                return "Anchor Head"
            if neg_rate >= float(sign_threshold):
                return "Veil Head"
    except Exception:
        # Any failure here should not prevent further fallback checks
        pass

    # 2) Wave detection by short period
    if _has_short_period(result, period_threshold):
        return "Wave Head"

    # 3) Fallback to mean-based decision
    try:
        if attn is None:
            attn = _load_attention_from_result(result)
        mean = float(np.mean(attn))
    except Exception:
        amps = []
        for e in result.get("raw_results", []):
            try:
                amps.append(float(e.get("global_amp", 0.0)))
            except Exception:
                pass
        mean = float(np.mean(amps)) if amps else 0.0

    return "Anchor Head" if mean > 0 else "Veil Head"
