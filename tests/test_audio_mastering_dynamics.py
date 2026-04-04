import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import signal as scipy_signal


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
DYNAMICS_MODULE = _load_module(
    "_test_audio_mastering_dynamics",
    ROOT / "src" / "definers" / "audio" / "mastering_dynamics.py",
)


def test_apply_true_peak_limiter_preserves_length_and_lowers_hot_peak():
    mastering = SimpleNamespace(resampling_target=1000)
    source = np.array([0.8, 2.0, 0.0], dtype=np.float32)
    signal_module = SimpleNamespace(
        resample_poly=lambda y, up, down, axis=-1: np.array(y, copy=True)
    )

    limited = DYNAMICS_MODULE.apply_true_peak_limiter(
        mastering,
        source,
        drive_db=0.0,
        ceil_db=0.0,
        os_factor=1,
        lookahead_ms=1.0,
        attack_ms=1.0,
        release_ms_min=1.0,
        release_ms_max=1.0,
        window_ms=1.0,
        signal_module=signal_module,
        maximum_filter1d_fn=lambda values, size, mode="constant": np.array(
            values, copy=True
        ),
        uniform_filter1d_fn=lambda values, size, mode="constant": np.array(
            values, copy=True
        ),
        limiter_smooth_env_fn=lambda values, atk_c, rel_c: np.array(
            values, copy=True
        ),
    )

    assert limited.shape == source.shape
    assert limited[1] < source[1]


def test_apply_true_peak_limiter_hard_catches_residual_overs_when_control_underestimates_peak():
    mastering = SimpleNamespace(resampling_target=1000)
    source = np.array([0.8, 2.0, 0.0], dtype=np.float32)
    signal_module = SimpleNamespace(
        resample_poly=lambda y, up, down, axis=-1: np.array(y, copy=True)
    )

    limited = DYNAMICS_MODULE.apply_true_peak_limiter(
        mastering,
        source,
        drive_db=0.0,
        ceil_db=0.0,
        os_factor=1,
        lookahead_ms=1.0,
        attack_ms=1.0,
        release_ms_min=1.0,
        release_ms_max=1.0,
        window_ms=1.0,
        signal_module=signal_module,
        maximum_filter1d_fn=lambda values, size, mode="constant": np.array(
            values, copy=True
        ),
        uniform_filter1d_fn=lambda values, size, mode="constant": np.array(
            values, copy=True
        ),
        limiter_smooth_env_fn=lambda values, atk_c, rel_c: (
            np.array(values, copy=True) * 0.5
        ),
    )

    assert float(np.max(np.abs(limited))) <= 1.0 + 1e-6


def test_apply_soft_clip_stage_reduces_peak_without_zeroing_signal():
    source = np.array([0.0, 0.9, 1.2, -1.2], dtype=np.float32)

    clipped = DYNAMICS_MODULE.apply_soft_clip_stage(
        source,
        ceil_db=0.0,
        soft_clip_ratio=0.2,
    )

    assert np.max(np.abs(clipped)) < np.max(np.abs(source))
    assert np.any(np.abs(clipped) > 0.0)


def test_apply_safety_clamp_respects_ceiling():
    source = np.array([0.0, 0.7, 1.4, -1.4], dtype=np.float32)

    clamped = DYNAMICS_MODULE.apply_safety_clamp(source, ceil_db=-1.0)

    assert np.max(np.abs(clamped)) == pytest.approx(10.0 ** (-1.0 / 20.0))


def test_apply_spatial_enhancement_adds_subtle_stereo_tonal_variation_to_high_band():
    mastering = SimpleNamespace(
        stereo_width=1.0,
        mono_bass_hz=140.0,
        low_cut=20.0,
        high_cut=3900.0,
        resampling_target=8000,
        stereo_tone_variation_db=0.6,
        stereo_tone_variation_cutoff_hz=1200.0,
        stereo_tone_variation_smoothing_ms=25.0,
        stereo_motion_mid_amount=0.6,
        stereo_motion_high_amount=1.0,
        stereo_motion_correlation_guard=1.0,
        stereo_motion_max_side_boost=0.16,
    )
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    shared_low = 0.5 * np.sin(2.0 * np.pi * 180.0 * time_axis)
    modulation = 1.0 + 0.35 * np.sin(2.0 * np.pi * 2.0 * time_axis)
    left = (
        shared_low
        + 0.18 * np.sin(2.0 * np.pi * 1700.0 * time_axis) * modulation
    )
    right = (
        shared_low
        + 0.18 * np.sin(2.0 * np.pi * 1700.0 * time_axis) / modulation
    )
    source = np.stack([left, right]).astype(np.float32)

    enhanced = DYNAMICS_MODULE.apply_spatial_enhancement(
        mastering,
        source,
        signal_module=scipy_signal,
    )

    high_sos = scipy_signal.butter(
        2, 1200.0 / (8000 / 2.0), btype="high", output="sos"
    )
    low_sos = scipy_signal.butter(
        2, 1200.0 / (8000 / 2.0), btype="low", output="sos"
    )
    before_high = scipy_signal.sosfiltfilt(high_sos, source, axis=-1)
    after_high = scipy_signal.sosfiltfilt(high_sos, enhanced, axis=-1)
    before_low = scipy_signal.sosfiltfilt(low_sos, source, axis=-1)
    after_low = scipy_signal.sosfiltfilt(low_sos, enhanced, axis=-1)
    high_side_delta = (after_high[0] - after_high[1]) - (
        before_high[0] - before_high[1]
    )
    low_mono_delta = np.mean(after_low, axis=0) - np.mean(before_low, axis=0)

    assert enhanced.shape == source.shape
    assert not np.allclose(enhanced, source)
    assert float(np.sqrt(np.mean(high_side_delta**2))) > 0.0
    assert (
        float(np.sqrt(np.mean(low_mono_delta**2)))
        < float(np.sqrt(np.mean(high_side_delta**2))) * 0.5
    )
    assert mastering.last_stereo_motion_activity > 0.0
    assert 0.0 < mastering.last_stereo_motion_correlation_guard <= 1.0


def test_apply_spatial_enhancement_restrains_motion_when_band_is_already_anti_phase():
    mastering = SimpleNamespace(
        stereo_width=1.0,
        mono_bass_hz=140.0,
        low_cut=20.0,
        high_cut=3900.0,
        resampling_target=8000,
        stereo_tone_variation_db=0.7,
        stereo_tone_variation_cutoff_hz=1000.0,
        stereo_tone_variation_smoothing_ms=20.0,
        stereo_motion_mid_amount=0.7,
        stereo_motion_high_amount=1.1,
        stereo_motion_correlation_guard=1.2,
        stereo_motion_max_side_boost=0.18,
    )
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    left = 0.2 * np.sin(2.0 * np.pi * 1800.0 * time_axis)
    right = -left
    source = np.stack([left, right]).astype(np.float32)

    enhanced = DYNAMICS_MODULE.apply_spatial_enhancement(
        mastering,
        source,
        signal_module=scipy_signal,
    )

    before_side_rms = float(np.sqrt(np.mean(np.square(source[0] - source[1]))))
    after_side_rms = float(
        np.sqrt(np.mean(np.square(enhanced[0] - enhanced[1])))
    )

    assert after_side_rms <= before_side_rms * 1.1
    assert mastering.last_stereo_motion_correlation_guard < 1.0
