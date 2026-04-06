from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from functools import cache, lru_cache
from typing import Any

import numpy as np
from scipy import signal

_ABSOLUTE_GATE_LUFS = -70.0
_LRA_RELATIVE_GATE_LU = -20.0
_LRA_SILENCE_PADDING_SECONDS = 1.5
_LRA_LOW_PERCENTILE = 10.0
_LRA_HIGH_PERCENTILE = 95.0
_LRA_SHORT_TERM_HOP_SECONDS = 0.09
_SHORT_TERM_WINDOW_SECONDS = 3.0
_MOMENTARY_WINDOW_SECONDS = 0.4
_HOP_SECONDS = 0.1
_SURROUND_CHANNEL_GAIN = 1.41
_DEMAN_HIGH_SHELF_GAIN_DB = 3.99984385397
_DEMAN_HIGH_SHELF_Q = 0.7071752369554193
_DEMAN_HIGH_SHELF_FC_HZ = 1681.9744509555319
_DEMAN_HIGH_PASS_Q = 0.5003270373253953
_DEMAN_HIGH_PASS_FC_HZ = 38.13547087613982
_REAL_SCIPY_SIGNAL_ATTRS: dict[str, Any] = {}


@dataclass(frozen=True, slots=True)
class MasteringLoudnessMetrics:
    integrated_lufs: float
    max_short_term_lufs: float
    max_momentary_lufs: float
    loudness_range_lu: float
    sample_peak_dbfs: float
    true_peak_dbfs: float
    crest_factor_db: float
    stereo_width_ratio: float
    low_end_mono_ratio: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _sanitize_audio(y: np.ndarray) -> np.ndarray:
    y_array = np.asarray(y, dtype=np.float32)
    if y_array.ndim == 0:
        y_array = y_array.reshape(1)
    return np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)


def _peak_to_dbfs(peak: float) -> float:
    if not np.isfinite(peak) or peak <= 0.0:
        return -120.0
    return float(20.0 * np.log10(peak))


def _rms(signal_values: np.ndarray) -> float:
    signal_array = np.asarray(signal_values, dtype=np.float32)
    if signal_array.size == 0:
        return 0.0
    return float(
        np.sqrt(np.mean(signal_array * signal_array, dtype=np.float32))
    )


def _loudness_from_power(mean_square: float) -> float:
    if not np.isfinite(mean_square) or mean_square <= 0.0:
        return _ABSOLUTE_GATE_LUFS
    return float(-0.691 + 10.0 * np.log10(np.maximum(mean_square, 1e-12)))


def _flatten_audio_channels(y: np.ndarray) -> np.ndarray:
    y_array = np.asarray(y, dtype=np.float64)
    if y_array.ndim == 1:
        return y_array[np.newaxis, :]
    return y_array.reshape(-1, y_array.shape[-1])


def _loudness_channel_weights(channel_count: int) -> np.ndarray:
    weights = np.ones(max(int(channel_count), 1), dtype=np.float64)
    if channel_count == 4:
        weights[2:] = _SURROUND_CHANNEL_GAIN
    elif channel_count >= 5:
        weights[3:5] = _SURROUND_CHANNEL_GAIN
    return weights


@cache
def _k_weighting_coefficients(
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rate = float(max(int(sample_rate), 1))

    high_shelf_k = np.tan(np.pi * _DEMAN_HIGH_SHELF_FC_HZ / rate)
    high_shelf_vh = np.power(10.0, _DEMAN_HIGH_SHELF_GAIN_DB / 20.0)
    high_shelf_vb = np.power(high_shelf_vh, 0.499666774155)
    high_shelf_a0 = (
        1.0 + high_shelf_k / _DEMAN_HIGH_SHELF_Q + high_shelf_k * high_shelf_k
    )
    high_shelf_b = np.array(
        [
            (
                high_shelf_vh
                + high_shelf_vb * high_shelf_k / _DEMAN_HIGH_SHELF_Q
                + high_shelf_k * high_shelf_k
            )
            / high_shelf_a0,
            2.0 * (high_shelf_k * high_shelf_k - high_shelf_vh) / high_shelf_a0,
            (
                high_shelf_vh
                - high_shelf_vb * high_shelf_k / _DEMAN_HIGH_SHELF_Q
                + high_shelf_k * high_shelf_k
            )
            / high_shelf_a0,
        ],
        dtype=np.float64,
    )
    high_shelf_a = np.array(
        [
            1.0,
            2.0 * (high_shelf_k * high_shelf_k - 1.0) / high_shelf_a0,
            (
                1.0
                - high_shelf_k / _DEMAN_HIGH_SHELF_Q
                + high_shelf_k * high_shelf_k
            )
            / high_shelf_a0,
        ],
        dtype=np.float64,
    )

    high_pass_k = np.tan(np.pi * _DEMAN_HIGH_PASS_FC_HZ / rate)
    high_pass_a0 = (
        1.0 + high_pass_k / _DEMAN_HIGH_PASS_Q + high_pass_k * high_pass_k
    )
    high_pass_b = np.array([1.0, -2.0, 1.0], dtype=np.float64)
    high_pass_a = np.array(
        [
            1.0,
            2.0 * (high_pass_k * high_pass_k - 1.0) / high_pass_a0,
            (1.0 - high_pass_k / _DEMAN_HIGH_PASS_Q + high_pass_k * high_pass_k)
            / high_pass_a0,
        ],
        dtype=np.float64,
    )

    return high_shelf_b, high_shelf_a, high_pass_b, high_pass_a


def _fallback_lfilter(
    b: np.ndarray,
    a: np.ndarray,
    values: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return np.array(x, copy=True)
    normalized_axis = axis if axis >= 0 else x.ndim + axis
    if normalized_axis != x.ndim - 1:
        x = np.moveaxis(x, normalized_axis, -1)
    b_array = np.asarray(b, dtype=np.float64)
    a_array = np.asarray(a, dtype=np.float64)
    if b_array.size == 0 or a_array.size == 0:
        return np.array(values, copy=True)
    a0 = float(a_array[0])
    if not np.isfinite(a0) or abs(a0) <= 1e-12:
        raise ValueError("Filter denominator must start with a finite value")
    if a0 != 1.0:
        b_array = b_array / a0
        a_array = a_array / a0
    order = max(len(a_array), len(b_array))
    if order == 1:
        filtered = b_array[0] * x
        if normalized_axis != x.ndim - 1:
            filtered = np.moveaxis(filtered, -1, normalized_axis)
        return filtered
    b_pad = np.pad(b_array, (0, order - len(b_array)))
    a_pad = np.pad(a_array, (0, order - len(a_array)))
    state = np.zeros(x.shape[:-1] + (order - 1,), dtype=np.float64)
    filtered = np.empty_like(x, dtype=np.float64)
    for sample_index in range(x.shape[-1]):
        x_n = x[..., sample_index]
        y_n = b_pad[0] * x_n + state[..., 0]
        filtered[..., sample_index] = y_n
        if order > 2:
            state[..., :-1] = (
                state[..., 1:]
                + b_pad[1:-1] * x_n[..., np.newaxis]
                - a_pad[1:-1] * y_n[..., np.newaxis]
            )
        state[..., -1] = b_pad[-1] * x_n - a_pad[-1] * y_n
    if normalized_axis != x.ndim - 1:
        filtered = np.moveaxis(filtered, -1, normalized_axis)
    return filtered


def _load_real_scipy_signal_attr(name: str):
    cached = _REAL_SCIPY_SIGNAL_ATTRS.get(name)
    if cached is not None:
        return cached
    original_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "scipy" or key.startswith("scipy.")
    }
    try:
        for key in tuple(original_modules):
            sys.modules.pop(key, None)
        real_signal = importlib.import_module("scipy.signal")
        resolved_attr = getattr(real_signal, name, None)
        if resolved_attr is not None:
            _REAL_SCIPY_SIGNAL_ATTRS[name] = resolved_attr
        return resolved_attr
    except Exception:
        return None
    finally:
        for key in tuple(sys.modules):
            if (
                key == "scipy" or key.startswith("scipy.")
            ) and key not in original_modules:
                sys.modules.pop(key, None)
        sys.modules.update(original_modules)


def _safe_lfilter(
    signal_module: Any,
    b: np.ndarray,
    a: np.ndarray,
    values: np.ndarray,
    *,
    axis: int = -1,
) -> np.ndarray:
    lfilter = getattr(signal_module, "lfilter", None)
    if callable(lfilter):
        return lfilter(b, a, values, axis=axis)
    real_lfilter = _load_real_scipy_signal_attr("lfilter")
    if callable(real_lfilter):
        return real_lfilter(b, a, values, axis=axis)
    return _fallback_lfilter(b, a, values, axis=axis)


def _weighted_power_series(
    y: np.ndarray,
    sr: int,
    *,
    signal_module: Any,
) -> np.ndarray:
    channels = _flatten_audio_channels(y)
    high_shelf_b, high_shelf_a, high_pass_b, high_pass_a = (
        _k_weighting_coefficients(sr)
    )
    weighted = _safe_lfilter(
        signal_module,
        high_shelf_b,
        high_shelf_a,
        channels,
        axis=-1,
    )
    weighted = _safe_lfilter(
        signal_module,
        high_pass_b,
        high_pass_a,
        weighted,
        axis=-1,
    )
    channel_weights = _loudness_channel_weights(weighted.shape[0])[
        :, np.newaxis
    ]
    return np.sum(
        channel_weights * np.square(weighted, dtype=np.float64),
        axis=0,
        dtype=np.float64,
    )


def _append_silence(
    y: np.ndarray, sr: int, duration_seconds: float
) -> np.ndarray:
    silence_samples = max(int(round(sr * duration_seconds)), 0)
    if silence_samples == 0:
        return np.array(y, copy=True)

    y_array = np.asarray(y)
    silence_shape = (*y_array.shape[:-1], silence_samples)
    silence = np.zeros(silence_shape, dtype=y_array.dtype)
    return np.concatenate([y_array, silence], axis=-1)


def _block_powers(
    y: np.ndarray,
    sr: int,
    window_seconds: float,
    hop_seconds: float,
    *,
    signal_module: Any,
) -> np.ndarray:
    if y.size == 0 or not np.isfinite(sr) or sr <= 0:
        return np.zeros(1, dtype=np.float64)

    y_square = _weighted_power_series(y, sr, signal_module=signal_module)

    return _block_powers_from_series(
        y_square,
        sr,
        window_seconds,
        hop_seconds,
    )


def _block_powers_from_series(
    y_square: np.ndarray,
    sr: int,
    window_seconds: float,
    hop_seconds: float,
) -> np.ndarray:
    if y_square.size == 0 or not np.isfinite(sr) or sr <= 0:
        return np.zeros(1, dtype=np.float64)

    window_size = max(int(round(sr * window_seconds)), 1)
    hop_size = max(int(round(sr * hop_seconds)), 1)

    if y_square.shape[-1] < window_size:
        return np.array(
            [float(np.mean(y_square, dtype=np.float64))],
            dtype=np.float64,
        )

    y_windows = np.lib.stride_tricks.sliding_window_view(y_square, window_size)[
        ::hop_size
    ]
    if y_windows.size == 0:
        return np.array(
            [float(np.mean(y_square, dtype=np.float64))],
            dtype=np.float64,
        )

    return np.mean(y_windows, axis=-1, dtype=np.float64)


def _gated_integrated_lufs(block_powers: np.ndarray) -> float:
    powers = np.nan_to_num(
        np.asarray(block_powers, dtype=np.float64),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if powers.size == 0:
        return _ABSOLUTE_GATE_LUFS

    abs_threshold = 10.0 ** ((_ABSOLUTE_GATE_LUFS + 0.691) / 10.0)
    gated = powers[powers >= abs_threshold]
    if gated.size == 0:
        return _ABSOLUTE_GATE_LUFS

    relative_gate = max(
        0.1 * float(np.mean(gated, dtype=np.float64)), abs_threshold
    )
    final = gated[gated > relative_gate]
    mean_square = float(
        np.mean(final if final.size > 0 else gated, dtype=np.float64)
    )
    return _loudness_from_power(mean_square)


def _loudness_series(block_powers: np.ndarray) -> np.ndarray:
    if block_powers.size == 0:
        return np.full(1, _ABSOLUTE_GATE_LUFS, dtype=np.float64)
    return np.array(
        [_loudness_from_power(float(value)) for value in block_powers],
        dtype=np.float64,
    )


def _loudness_range(short_term_lufs: np.ndarray) -> float:
    valid = np.asarray(short_term_lufs, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    valid = valid[valid >= _ABSOLUTE_GATE_LUFS]
    if valid.size == 0:
        return 0.0

    valid_power = float(np.mean(np.power(10.0, valid / 10.0), dtype=np.float64))
    if not np.isfinite(valid_power) or valid_power <= 0.0:
        return 0.0

    relative_gate = 10.0 * np.log10(valid_power) + _LRA_RELATIVE_GATE_LU
    valid = valid[valid >= relative_gate]
    if valid.size == 0:
        return 0.0

    p10 = float(np.percentile(valid, _LRA_LOW_PERCENTILE))
    p95 = float(np.percentile(valid, _LRA_HIGH_PERCENTILE))
    return max(0.0, p95 - p10)


def measure_sample_peak(y: np.ndarray) -> float:
    y_array = _sanitize_audio(y)
    peak = float(np.max(np.abs(y_array))) if y_array.size else 0.0
    return _peak_to_dbfs(peak)


def measure_true_peak(
    y: np.ndarray,
    sr: int,
    oversample_factor: int = 4,
    *,
    signal_module: Any = signal,
) -> float:
    y_array = _sanitize_audio(y)
    if y_array.size == 0 or not np.isfinite(sr) or sr <= 0:
        return -120.0

    safe_factor = max(int(oversample_factor), 1)
    if safe_factor > 1:
        oversampled = signal_module.resample_poly(
            y_array,
            safe_factor,
            1,
            axis=-1,
        )
    else:
        oversampled = y_array

    peak = float(np.max(np.abs(oversampled))) if oversampled.size else 0.0
    return _peak_to_dbfs(peak)


def measure_stereo_width(y: np.ndarray) -> float:
    y_array = _sanitize_audio(y)
    if y_array.ndim < 2 or y_array.shape[0] < 2:
        return 0.0

    left = y_array[0]
    right = y_array[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_rms = _rms(mid)
    side_rms = _rms(side)
    total = mid_rms + side_rms
    if total <= 1e-12:
        return 0.0

    return float(np.clip(side_rms / total, 0.0, 1.0))


def measure_low_end_mono_ratio(
    y: np.ndarray,
    sr: int,
    *,
    cutoff_hz: float = 160.0,
    signal_module: Any = signal,
) -> float:
    y_array = _sanitize_audio(y)
    if y_array.ndim < 2 or y_array.shape[0] < 2:
        return 1.0

    left = y_array[0]
    right = y_array[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_low = mid
    side_low = side

    has_low_pass = all(
        hasattr(signal_module, name) for name in ("butter", "sosfiltfilt")
    )
    safe_cutoff_hz = min(max(float(cutoff_hz), 20.0), max(sr / 2.0 - 1.0, 20.0))
    if has_low_pass and sr > 0 and safe_cutoff_hz < sr / 2.0 - 1.0:
        try:
            sos = signal_module.butter(
                4,
                safe_cutoff_hz / (sr / 2.0),
                btype="low",
                output="sos",
            )
            mid_low = signal_module.sosfiltfilt(sos, mid)
            side_low = signal_module.sosfiltfilt(sos, side)
        except Exception:
            mid_low = mid
            side_low = side

    mono_rms = _rms(mid_low)
    side_rms = _rms(side_low)
    total = mono_rms + side_rms
    if total <= 1e-12:
        return 1.0

    return float(np.clip(mono_rms / total, 0.0, 1.0))


def measure_mastering_loudness(
    y: np.ndarray,
    sr: int,
    *,
    true_peak_oversample_factor: int = 4,
    low_end_mono_cutoff_hz: float = 160.0,
    signal_module: Any = signal,
) -> MasteringLoudnessMetrics:
    y_array = _sanitize_audio(y)
    if y_array.size == 0 or not np.isfinite(sr) or sr <= 0:
        return MasteringLoudnessMetrics(
            integrated_lufs=_ABSOLUTE_GATE_LUFS,
            max_short_term_lufs=_ABSOLUTE_GATE_LUFS,
            max_momentary_lufs=_ABSOLUTE_GATE_LUFS,
            loudness_range_lu=0.0,
            sample_peak_dbfs=-120.0,
            true_peak_dbfs=-120.0,
            crest_factor_db=0.0,
            stereo_width_ratio=0.0,
            low_end_mono_ratio=1.0,
        )

    weighted_power = _weighted_power_series(
        y_array,
        sr,
        signal_module=signal_module,
    )
    momentary_powers = _block_powers_from_series(
        weighted_power,
        sr,
        _MOMENTARY_WINDOW_SECONDS,
        _HOP_SECONDS,
    )
    short_term_powers = _block_powers_from_series(
        weighted_power,
        sr,
        _SHORT_TERM_WINDOW_SECONDS,
        _HOP_SECONDS,
    )
    padded_weighted_power = _weighted_power_series(
        _append_silence(y_array, sr, _LRA_SILENCE_PADDING_SECONDS),
        sr,
        signal_module=signal_module,
    )

    integrated_lufs = _gated_integrated_lufs(momentary_powers)
    momentary_lufs = _loudness_series(momentary_powers)
    short_term_lufs = _loudness_series(short_term_powers)
    padded_short_term_lufs = _loudness_series(
        _block_powers_from_series(
            padded_weighted_power,
            sr,
            _SHORT_TERM_WINDOW_SECONDS,
            _LRA_SHORT_TERM_HOP_SECONDS,
        )
    )

    sample_peak_dbfs = measure_sample_peak(y_array)
    true_peak_dbfs = measure_true_peak(
        y_array,
        sr,
        oversample_factor=true_peak_oversample_factor,
        signal_module=signal_module,
    )

    if y_array.ndim > 1:
        rms = float(np.max(np.sqrt(np.mean(y_array**2, axis=1))))
    else:
        rms = float(np.sqrt(np.mean(y_array**2)))

    rms_dbfs = _peak_to_dbfs(rms)
    crest_factor_db = max(0.0, sample_peak_dbfs - rms_dbfs)
    stereo_width_ratio = measure_stereo_width(y_array)
    low_end_mono_ratio = measure_low_end_mono_ratio(
        y_array,
        sr,
        cutoff_hz=low_end_mono_cutoff_hz,
        signal_module=signal_module,
    )

    return MasteringLoudnessMetrics(
        integrated_lufs=integrated_lufs,
        max_short_term_lufs=float(np.max(short_term_lufs)),
        max_momentary_lufs=float(np.max(momentary_lufs)),
        loudness_range_lu=_loudness_range(padded_short_term_lufs),
        sample_peak_dbfs=sample_peak_dbfs,
        true_peak_dbfs=true_peak_dbfs,
        crest_factor_db=crest_factor_db,
        stereo_width_ratio=stereo_width_ratio,
        low_end_mono_ratio=low_end_mono_ratio,
    )


def get_lufs(y: np.ndarray, sr: int) -> float:
    return measure_mastering_loudness(y, sr).integrated_lufs


__all__ = [
    "MasteringLoudnessMetrics",
    "get_lufs",
    "measure_low_end_mono_ratio",
    "measure_mastering_loudness",
    "measure_sample_peak",
    "measure_stereo_width",
    "measure_true_peak",
]
