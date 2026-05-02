"""Tests for the first-difference + FFT + folded-top1 period pipeline merged
from spectral_analysis.ipynb into classifier.core, and for the labeling switch
to use folded_top1_period.
"""
import numpy as np
import pytest

from classifier.core import (
    preprocess_first_difference,
    compute_harmonic_folded_spectrum,
    compute_folded_top1_period,
    analyze_fft_ranges,
)
from classifier.labeling import label_head_from_result


# ---------- preprocess_first_difference ----------

def test_preprocess_first_difference_returns_n_minus_1_window_applied():
    rng = np.random.default_rng(123)
    seq = np.cumsum(rng.standard_normal(10))
    out = preprocess_first_difference(seq, remove_dc=True, apply_window=True)
    assert out.shape == (9,)
    # Hann window endpoints are 0
    assert out[0] == pytest.approx(0.0, abs=1e-12)
    assert out[-1] == pytest.approx(0.0, abs=1e-12)


def test_preprocess_first_difference_remove_dc_zero_means():
    rng = np.random.default_rng(0)
    seq = np.cumsum(rng.standard_normal(64))
    out = preprocess_first_difference(seq, remove_dc=True, apply_window=False)
    assert out.mean() == pytest.approx(0.0, abs=1e-12)


def test_preprocess_first_difference_constant_signal_returns_zeros():
    # Without z-score we no longer raise; constant input yields all-zero diff,
    # which the downstream pipeline must handle gracefully.
    seq = np.full(20, 3.0)
    out = preprocess_first_difference(seq, remove_dc=True, apply_window=True)
    np.testing.assert_allclose(out, 0.0)


def test_compute_folded_top1_period_returns_nan_for_constant_signal():
    seq = np.full(64, 0.7)
    assert np.isnan(compute_folded_top1_period(seq))


def test_compute_folded_top1_period_invariant_to_positive_scaling():
    rng = np.random.default_rng(7)
    seq = np.cumsum(rng.standard_normal(96))
    base = compute_folded_top1_period(seq)
    for scale in (1e-6, 0.5, 3.0, 1e6):
        assert compute_folded_top1_period(seq * scale) == base


# ---------- compute_harmonic_folded_spectrum ----------

def test_folded_spectrum_adds_decayed_harmonic_amp_to_fundamental():
    # period array sorted ascending; index 0 = period 4 (fundamental),
    # index 1 = period 8 (2× harmonic with amp 0.6, decay 0.5 -> +0.3)
    period = np.array([4.0, 8.0, 12.0])
    amplitude = np.array([1.0, 0.6, 0.0])
    folded = compute_harmonic_folded_spectrum(
        period, amplitude, signal_size=64,
        min_period=3.0, max_period_divisor=6.0,
        max_multiple=4, tol=0.2, decay=0.5,
    )
    assert folded[0] == pytest.approx(1.0 + 0.5 * 0.6)
    # Indices that are not fundamentals stay unchanged
    assert folded[1] == pytest.approx(0.6)
    assert folded[2] == pytest.approx(0.0)


def test_folded_spectrum_skips_when_no_matching_harmonic():
    period = np.array([4.0, 9.5, 13.0])  # 2*4=8 not within tol of 9.5/13
    amplitude = np.array([1.0, 0.6, 0.4])
    folded = compute_harmonic_folded_spectrum(
        period, amplitude, signal_size=64,
        min_period=3.0, max_period_divisor=6.0,
        max_multiple=4, tol=0.05, decay=0.5,
    )
    np.testing.assert_allclose(folded, amplitude)


# ---------- compute_folded_top1_period ----------

def test_folded_top1_period_pure_sine():
    n = 64
    period = 8.0
    t = np.arange(n)
    sig = np.sin(2 * np.pi * t / period)
    top1 = compute_folded_top1_period(sig)
    assert abs(top1 - period) <= 1.0


def test_folded_top1_period_with_folding_disabled_matches_raw_argmax():
    # With folding off, top1 == period at max FFT amp within display_mask of
    # the first-differenced + windowed signal. For a clean sine, this equals
    # the underlying period (modulo bin discretization).
    n = 64
    t = np.arange(n)
    sig = np.sin(2 * np.pi * t / 8.0)
    top1 = compute_folded_top1_period(sig, harmonic_folding_enabled=False)
    assert abs(top1 - 8.0) <= 1.0


def test_folded_top1_period_respects_period_min():
    # A short-period sine that would be filtered out by period_min should not
    # be returned; another component at longer period should win.
    n = 128
    t = np.arange(n)
    sig = 1.0 * np.sin(2 * np.pi * t / 3.0) + 0.4 * np.sin(2 * np.pi * t / 16.0)
    top1 = compute_folded_top1_period(sig, period_min=10.0, period_max=None)
    assert top1 >= 10.0


# ---------- analyze_fft_ranges adds folded_top1_period ----------

def test_analyze_fft_ranges_adds_folded_top1_field():
    n = 72
    t = np.arange(n)
    sig = np.sin(2 * np.pi * t / 8.0)
    results = analyze_fft_ranges(
        sig,
        [{"label": "0-68", "start": 0, "end": 69}],
    )
    assert len(results) == 1
    entry = results[0]
    assert "folded_top1_period" in entry
    assert isinstance(entry["folded_top1_period"], float)
    assert abs(entry["folded_top1_period"] - 8.0) <= 1.0


# ---------- labeling uses folded_top1_period ----------

def _fake_result(folded_top1_period, *, global_period=20.0):
    return {
        "attn_path": "/nonexistent/layer0.pt",  # forces sign-rate path to fail
        "layer": 0,
        "head": 0,
        "frames": 72,
        "used_frames": 69,
        "ignored_tail": 3,
        "raw_results": [
            {
                "label": "0-68",
                "start": 0,
                "end": 69,
                "global_period": global_period,
                "folded_top1_period": folded_top1_period,
            }
        ],
        "softmax_results": [],
    }


def test_label_wave_head_when_folded_top1_below_threshold():
    result = _fake_result(folded_top1_period=4.0, global_period=20.0)
    assert label_head_from_result(result, period_threshold=6.0, sign_threshold=0.9) == "Wave Head"


def test_label_ignores_old_global_period_for_wave_detection():
    # global_period is short but folded_top1_period is long -> not Wave
    result = _fake_result(folded_top1_period=20.0, global_period=4.0)
    label = label_head_from_result(result, period_threshold=6.0, sign_threshold=0.9)
    assert label != "Wave Head"


def test_label_no_wave_when_folded_top1_missing():
    result = _fake_result(folded_top1_period=20.0)
    # Drop the new field entirely; legacy results without folded_top1_period
    # must not be misclassified as Wave just because global_period is short.
    result["raw_results"][0].pop("folded_top1_period")
    result["raw_results"][0]["global_period"] = 4.0
    assert label_head_from_result(result, period_threshold=6.0, sign_threshold=0.9) != "Wave Head"
