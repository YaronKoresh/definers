from __future__ import annotations

from collections import deque
from collections.abc import Callable

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()
from scipy import signal


def audio_eq(
    audio_data: np.ndarray,
    anchors: list[list[float]] | np.ndarray,
    sample_rate: int = 44100,
    nperseg: int = 8192,
) -> np.ndarray:
    anchors = sorted(anchors, key=lambda x: x[0])
    anchor_freqs = np.array([a[0] for a in anchors])
    anchor_gains_db = np.array([a[1] for a in anchors])

    unique_freqs, indices = np.unique(anchor_freqs, return_index=True)
    anchor_freqs = unique_freqs
    anchor_gains_db = anchor_gains_db[indices]

    f_axis, _times, stft_frames = signal.stft(
        audio_data,
        fs=sample_rate,
        nperseg=nperseg,
    )

    log_f = np.log10(f_axis + 1e-5)
    log_anchor_freqs = np.log10(anchor_freqs)

    interp_gains_db = np.interp(
        log_f,
        log_anchor_freqs,
        anchor_gains_db,
        left=anchor_gains_db[0],
        right=anchor_gains_db[-1],
    )

    gain_multipliers = 10 ** (interp_gains_db / 20)
    modified_frames = stft_frames * gain_multipliers[:, None]

    _reconstructed_times, output_audio = signal.istft(
        modified_frames,
        fs=sample_rate,
        nperseg=nperseg,
    )

    orig_len = audio_data.shape[-1]
    if output_audio.shape[-1] > orig_len:
        output_audio = output_audio[..., :orig_len]
    elif output_audio.shape[-1] < orig_len:
        pad_width = orig_len - output_audio.shape[-1]
        output_audio = (
            np.pad(output_audio, ((0, 0), (0, pad_width)))
            if output_audio.ndim > 1
            else np.pad(output_audio, (0, pad_width))
        )

    if np.issubdtype(audio_data.dtype, np.integer):
        info = np.iinfo(audio_data.dtype)
        return np.clip(output_audio, info.min, info.max).astype(
            audio_data.dtype
        )

    return output_audio.astype(np.float32)


def smooth_curve(
    self,
    curve: np.ndarray,
    f_axis: np.ndarray,
    smoothing_fraction: float | None = None,
) -> np.ndarray:
    if smoothing_fraction is None:
        return curve

    smoothed = np.copy(curve)

    for index, frequency_hz in enumerate(f_axis):
        bandwidth = frequency_hz * (
            2**smoothing_fraction - 2 ** (-smoothing_fraction)
        )
        low_f = frequency_hz - bandwidth / 2
        high_f = frequency_hz + bandwidth / 2
        mask = (f_axis >= low_f) & (f_axis <= high_f)

        if np.any(mask):
            smoothed[index] = np.mean(curve[mask])

    return smoothed


def _normalize_stem_role(stem_role: str | None) -> str:
    normalized_role = (
        "other" if stem_role is None else str(stem_role).strip().lower()
    )
    return normalized_role or "other"


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    if samples.size == 0 or window_size <= 1:
        return samples.astype(np.float32, copy=True)

    window_size = max(1, min(int(window_size), samples.size))
    if window_size <= 1:
        return samples.astype(np.float32, copy=True)

    cumulative = np.cumsum(samples, dtype=np.float64)
    smoothed = np.empty_like(samples)
    for index in range(samples.size):
        start = max(0, index - window_size + 1)
        total = cumulative[index]
        if start > 0:
            total -= cumulative[start - 1]
        smoothed[index] = total / float(index - start + 1)
    return smoothed.astype(np.float32, copy=False)


def _rolling_max(values: np.ndarray, window_size: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    if samples.size == 0 or window_size <= 1:
        return samples.astype(np.float32, copy=True)

    window_size = max(1, min(int(window_size), samples.size))
    if window_size <= 1:
        return samples.astype(np.float32, copy=True)

    rolling_max: deque[int] = deque()
    output = np.empty_like(samples)

    for index, value in enumerate(samples):
        while rolling_max and rolling_max[0] <= index - window_size:
            rolling_max.popleft()
        while rolling_max and samples[rolling_max[-1]] <= value:
            rolling_max.pop()
        rolling_max.append(index)
        output[index] = samples[rolling_max[0]]

    return output


def _restore_audio_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(values, info.min, info.max).astype(dtype)

    return values.astype(dtype, copy=False)


def _signal_rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float32)))


def _as_audio_channels(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim == 1:
        return array.reshape(1, -1)
    if array.ndim == 2 and array.shape[0] <= array.shape[-1]:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        return array.T.astype(np.float32, copy=False)
    raise ValueError("Unsupported audio shape")


def _align_audio_channels(signal: np.ndarray, target_length: int) -> np.ndarray:
    channels = _as_audio_channels(signal)
    current_length = int(channels.shape[-1])
    if current_length == target_length:
        return channels.astype(np.float32, copy=False)
    if current_length > target_length:
        return channels[:, :target_length].astype(np.float32, copy=False)
    return np.pad(
        channels,
        ((0, 0), (0, target_length - current_length)),
    ).astype(np.float32, copy=False)


def _restore_audio_layout(
    channels: np.ndarray,
    reference_signal: np.ndarray,
) -> np.ndarray:
    reference_array = np.asarray(reference_signal)
    if reference_array.ndim == 1:
        return channels[0].astype(np.float32, copy=False)
    if (
        reference_array.ndim == 2
        and reference_array.shape[0] > reference_array.shape[-1]
    ):
        return channels.T.astype(np.float32, copy=False)
    return channels.astype(np.float32, copy=False)


def _resolve_mask_window_samples(
    sample_count: int,
    sample_rate: int,
    milliseconds: float,
    *,
    max_fraction: float,
) -> int:
    requested = int(round(sample_rate * milliseconds / 1000.0))
    relative_cap = max(1, int(round(sample_count * max_fraction)))
    return max(1, min(sample_count, requested, relative_cap))


def _resolve_stem_cleanup_preservation_profile(
    stem_role: str | None,
) -> dict[str, float]:
    normalized_role = _normalize_stem_role(stem_role)
    if normalized_role == "drums":
        return {
            "min_full_rms_ratio": 0.28,
            "min_active_rms_ratio": 0.56,
            "min_preserved_coverage_ratio": 0.48,
            "restore_mix": 0.28,
        }
    if normalized_role == "bass":
        return {
            "min_full_rms_ratio": 0.56,
            "min_active_rms_ratio": 0.76,
            "min_preserved_coverage_ratio": 0.82,
            "restore_mix": 0.58,
        }
    if normalized_role == "vocals":
        return {
            "min_full_rms_ratio": 0.48,
            "min_active_rms_ratio": 0.7,
            "min_preserved_coverage_ratio": 0.78,
            "restore_mix": 0.54,
        }
    if normalized_role == "other":
        return {
            "min_full_rms_ratio": 0.42,
            "min_active_rms_ratio": 0.66,
            "min_preserved_coverage_ratio": 0.72,
            "restore_mix": 0.56,
        }
    return {
        "min_full_rms_ratio": 0.44,
        "min_active_rms_ratio": 0.68,
        "min_preserved_coverage_ratio": 0.74,
        "restore_mix": 0.5,
    }


def _measure_stem_cleanup_preservation(
    source_signal: np.ndarray,
    processed_signal: np.ndarray,
) -> dict[str, float]:
    source_channels = _as_audio_channels(source_signal)
    processed_channels = _as_audio_channels(processed_signal)
    target_length = max(
        int(source_channels.shape[-1]),
        int(processed_channels.shape[-1]),
    )
    source_channels = _align_audio_channels(source_channels, target_length)
    processed_channels = _align_audio_channels(
        processed_channels, target_length
    )

    source_energy = np.mean(np.abs(source_channels), axis=0, dtype=np.float32)
    processed_energy = np.mean(
        np.abs(processed_channels), axis=0, dtype=np.float32
    )
    source_rms = _signal_rms(source_energy)
    processed_rms = _signal_rms(processed_energy)
    if source_rms <= 1e-6:
        return {
            "full_rms_ratio": 1.0,
            "active_rms_ratio": 1.0,
            "preserved_coverage_ratio": 1.0,
        }

    source_peak = float(np.max(source_energy)) if source_energy.size else 0.0
    activity_threshold = float(
        max(
            np.percentile(source_energy, 66.0),
            np.mean(source_energy, dtype=np.float32) * 1.02,
            source_peak * 0.14,
            1e-6,
        )
    )
    active_mask = source_energy >= activity_threshold
    if not np.any(active_mask):
        active_mask = source_energy >= max(source_peak * 0.3, 1e-6)

    source_active = source_energy[active_mask]
    processed_active = processed_energy[active_mask]
    source_active_rms = _signal_rms(source_active)
    processed_active_rms = _signal_rms(processed_active)
    preserved_mask = processed_active >= np.maximum(
        source_active * 0.26,
        activity_threshold * 0.24,
    )
    return {
        "full_rms_ratio": float(processed_rms / max(source_rms, 1e-6)),
        "active_rms_ratio": float(
            processed_active_rms / max(source_active_rms, 1e-6)
        ),
        "preserved_coverage_ratio": float(
            np.mean(preserved_mask, dtype=np.float32)
        ),
    }


def _stem_cleanup_preservation_failed(
    metrics: dict[str, float],
    profile: dict[str, float],
) -> bool:
    active_failed = (
        metrics["active_rms_ratio"] < profile["min_active_rms_ratio"]
    )
    coverage_failed = (
        metrics["preserved_coverage_ratio"]
        < profile["min_preserved_coverage_ratio"]
    )
    full_failed = metrics["full_rms_ratio"] < profile["min_full_rms_ratio"]
    return active_failed or coverage_failed or (full_failed and coverage_failed)


def _build_stem_cleanup_restore_mask(source_signal: np.ndarray) -> np.ndarray:
    source_channels = _as_audio_channels(source_signal)
    source_energy = np.mean(np.abs(source_channels), axis=0, dtype=np.float32)
    if source_energy.size == 0:
        return np.zeros((1, 0), dtype=np.float32)

    source_peak = float(np.max(source_energy)) if source_energy.size else 0.0
    activity_threshold = float(
        max(
            np.percentile(source_energy, 66.0),
            np.mean(source_energy, dtype=np.float32) * 1.02,
            source_peak * 0.14,
            1e-6,
        )
    )
    active_mask = source_energy >= activity_threshold
    if not np.any(active_mask):
        active_mask = source_energy >= max(source_peak * 0.3, 1e-6)

    hold_samples = max(
        1,
        min(
            source_energy.size,
            max(4, source_energy.size // 36),
        ),
    )
    restore_curve = _rolling_max(active_mask.astype(np.float32), hold_samples)
    restore_curve = _moving_average(
        restore_curve,
        max(1, hold_samples // 2),
    )
    restore_curve = np.clip(restore_curve, 0.0, 1.0)
    return restore_curve.reshape(1, -1)


def _preserve_stem_cleanup_content(
    source_signal: np.ndarray,
    processed_signal: np.ndarray,
    *,
    stem_role: str | None,
) -> np.ndarray:
    profile = _resolve_stem_cleanup_preservation_profile(stem_role)
    source_array = np.asarray(source_signal, dtype=np.float32)
    processed_array = np.asarray(processed_signal, dtype=np.float32)
    metrics = _measure_stem_cleanup_preservation(source_array, processed_array)
    if not _stem_cleanup_preservation_failed(metrics, profile):
        return processed_array

    source_channels = _as_audio_channels(source_array)
    processed_channels = _align_audio_channels(
        processed_array,
        int(source_channels.shape[-1]),
    )
    restore_mix = float(np.clip(profile["restore_mix"], 0.0, 1.0))
    restore_mask = _build_stem_cleanup_restore_mask(source_array)
    blended_channels = processed_channels * (1.0 - restore_mix * restore_mask)
    blended_channels += source_channels * (restore_mix * restore_mask)
    restored = _restore_audio_layout(blended_channels, source_array)
    restored_metrics = _measure_stem_cleanup_preservation(
        source_array,
        restored,
    )
    if not _stem_cleanup_preservation_failed(restored_metrics, profile):
        return restored.astype(np.float32, copy=False)
    return _restore_audio_layout(source_channels, source_array)


def _resolve_stem_cleanup_pressure(
    stem_role: str | None,
    cleanup_pressure: float,
) -> float:
    normalized_role = _normalize_stem_role(stem_role)
    pressure = float(np.clip(cleanup_pressure, 0.0, 1.0))

    if normalized_role == "drums":
        return float(np.clip(pressure * 0.45, 0.0, 1.0))
    if normalized_role == "vocals":
        return float(np.clip(pressure * 0.68, 0.0, 1.0))
    if normalized_role == "bass":
        return float(np.clip(pressure * 0.74, 0.0, 1.0))
    return float(np.clip(pressure * 0.7, 0.0, 1.0))


def _resolve_stem_residual_profile(
    stem_role: str | None,
    cleanup_pressure: float,
) -> dict[str, float]:
    normalized_role = _normalize_stem_role(stem_role)

    if normalized_role == "drums":
        intensity = float(np.clip(0.62 + cleanup_pressure * 0.18, 0.0, 1.0))
        return {
            "fast_ms": 1.8,
            "slow_ms": 14.0,
            "hold_ms": 68.0,
            "release_ms": 14.0,
            "noise_percentile": 38.0,
            "suppression_floor": float(0.07 + (1.0 - intensity) * 0.05),
            "activity_exponent": 0.68,
            "activity_floor_scale": 0.94,
            "transient_blend": 0.9,
            "expansion_drive": float(0.42 + intensity * 0.36),
            "expansion_mix": float(0.16 + intensity * 0.18),
        }

    if normalized_role == "vocals":
        intensity = float(np.clip(0.52 + cleanup_pressure * 0.22, 0.0, 1.0))
        return {
            "fast_ms": 5.5,
            "slow_ms": 52.0,
            "hold_ms": 94.0,
            "release_ms": 34.0,
            "noise_percentile": 28.0,
            "suppression_floor": float(0.115 + (1.0 - intensity) * 0.08),
            "activity_exponent": 0.84,
            "activity_floor_scale": 0.44,
            "transient_blend": 0.16,
            "expansion_drive": float(0.46 + intensity * 0.24),
            "expansion_mix": float(0.18 + intensity * 0.14),
        }

    if normalized_role == "bass":
        intensity = float(np.clip(0.44 + cleanup_pressure * 0.18, 0.0, 1.0))
        return {
            "fast_ms": 7.0,
            "slow_ms": 76.0,
            "hold_ms": 126.0,
            "release_ms": 46.0,
            "noise_percentile": 24.0,
            "suppression_floor": float(0.18 + (1.0 - intensity) * 0.08),
            "activity_exponent": 0.96,
            "activity_floor_scale": 0.56,
            "transient_blend": 0.08,
            "expansion_drive": float(0.24 + intensity * 0.16),
            "expansion_mix": float(0.08 + intensity * 0.08),
        }

    intensity = float(np.clip(0.58 + cleanup_pressure * 0.22, 0.0, 1.0))
    return {
        "fast_ms": 4.5,
        "slow_ms": 42.0,
        "hold_ms": 86.0,
        "release_ms": 30.0,
        "noise_percentile": 30.0,
        "suppression_floor": float(0.1 + (1.0 - intensity) * 0.07),
        "activity_exponent": 0.8,
        "activity_floor_scale": 0.6,
        "transient_blend": 0.18,
        "expansion_drive": float(0.58 + intensity * 0.24),
        "expansion_mix": float(0.2 + intensity * 0.14),
    }


def _db_to_linear(level_db: float) -> float:
    if not np.isfinite(level_db):
        return 1.0
    return float(10.0 ** (float(level_db) / 20.0))


def _amplitude_to_db(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    return (20.0 * np.log10(np.maximum(array, 1e-6))).astype(np.float32)


def _measure_gate_sample_peak_dbfs(signal_values: np.ndarray) -> float:
    array = np.asarray(signal_values, dtype=np.float32)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak <= 0.0 or not np.isfinite(peak):
        return -120.0
    return float(20.0 * np.log10(peak))


def _measure_gate_true_peak_dbfs(
    signal_values: np.ndarray,
    sample_rate: int,
    oversample_factor: int,
) -> float:
    try:
        from .loudness import measure_true_peak as loudness_measure_true_peak
    except ImportError:
        from definers.audio.mastering.loudness import (
            measure_true_peak as loudness_measure_true_peak,
        )

    return float(
        loudness_measure_true_peak(
            signal_values,
            sample_rate,
            oversample_factor=max(int(oversample_factor), 1),
            signal_module=signal,
        )
    )


def _measure_gate_lufs(signal_values: np.ndarray, sample_rate: int) -> float:
    try:
        from .loudness import get_lufs as loudness_get_lufs
    except ImportError:
        from definers.audio.mastering.loudness import (
            get_lufs as loudness_get_lufs,
        )

    return float(loudness_get_lufs(signal_values, sample_rate))


def _resolve_gate_peak_oversampling(
    gate_profile: dict[str, float | bool | str | None],
) -> int:
    configured_factor = max(int(gate_profile["oversampling"]), 1)
    if bool(gate_profile["inter_sample_peak_awareness"]):
        return max(configured_factor, 4)
    return configured_factor


def _measure_gate_analysis_peak_dbfs(
    signal_values: np.ndarray,
    sample_rate: int,
    gate_profile: dict[str, float | bool | str | None],
) -> float:
    if bool(gate_profile["inter_sample_peak_awareness"]):
        return _measure_gate_true_peak_dbfs(
            signal_values,
            sample_rate,
            _resolve_gate_peak_oversampling(gate_profile),
        )
    return _measure_gate_sample_peak_dbfs(signal_values)


def _apply_gate_dc_offset_compensation(
    channels: np.ndarray,
    enabled: bool,
) -> np.ndarray:
    if not enabled:
        return np.asarray(channels, dtype=np.float32)
    channel_array = np.asarray(channels, dtype=np.float32)
    return (
        channel_array
        - np.mean(channel_array, axis=-1, keepdims=True, dtype=np.float32)
    ).astype(np.float32, copy=False)


def _normalize_gate_analysis_channels(
    channels: np.ndarray,
    sample_rate: int,
    gate_profile: dict[str, float | bool | str | None],
) -> np.ndarray:
    normalized = _apply_gate_dc_offset_compensation(
        channels,
        bool(gate_profile["dc_offset_compensation"]),
    )
    mode = str(gate_profile["normalization_mode"]).strip().lower()

    if mode == "lufs":
        current_lufs = _measure_gate_lufs(normalized, sample_rate)
        target_lufs = float(gate_profile["normalization_target_lufs"])
        if np.isfinite(current_lufs) and np.isfinite(target_lufs):
            normalized = normalized * _db_to_linear(target_lufs - current_lufs)
    elif mode == "dbfs":
        current_dbfs = _measure_gate_sample_peak_dbfs(normalized)
        target_dbfs = float(gate_profile["normalization_target_dbfs"])
        if (
            np.isfinite(current_dbfs)
            and current_dbfs > -120.0
            and np.isfinite(target_dbfs)
        ):
            normalized = normalized * _db_to_linear(target_dbfs - current_dbfs)

    current_peak_dbfs = _measure_gate_analysis_peak_dbfs(
        normalized,
        sample_rate,
        gate_profile,
    )
    target_peak_dbfs = float(gate_profile["analysis_peak_dbfs"])
    if (
        np.isfinite(current_peak_dbfs)
        and current_peak_dbfs > -120.0
        and np.isfinite(target_peak_dbfs)
    ):
        normalized = normalized * _db_to_linear(
            target_peak_dbfs - current_peak_dbfs
        )

    return np.asarray(normalized, dtype=np.float32)


def _resample_audio_channels(
    channels: np.ndarray,
    *,
    up: int,
    down: int,
) -> np.ndarray:
    source_channels = _as_audio_channels(channels)
    if max(int(up), 1) == max(int(down), 1):
        return source_channels.astype(np.float32, copy=False)
    resampled = signal.resample_poly(
        source_channels,
        max(int(up), 1),
        max(int(down), 1),
        axis=-1,
    )
    return _as_audio_channels(np.asarray(resampled, dtype=np.float32))


def _apply_first_order_lowpass(
    signal_values: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> np.ndarray:
    if cutoff_hz <= 0.0 or cutoff_hz >= sample_rate * 0.5:
        return np.asarray(signal_values, dtype=np.float32)

    dt = 1.0 / float(max(sample_rate, 1))
    rc = 1.0 / max(2.0 * np.pi * cutoff_hz, 1e-6)
    alpha = float(np.clip(dt / (rc + dt), 0.0, 1.0))
    source = np.asarray(signal_values, dtype=np.float32)
    filtered = np.empty_like(source)
    filtered[0] = source[0]
    for index in range(1, source.size):
        filtered[index] = filtered[index - 1] + alpha * (
            source[index] - filtered[index - 1]
        )
    return filtered.astype(np.float32, copy=False)


def _apply_first_order_highpass(
    signal_values: np.ndarray,
    sample_rate: int,
    cutoff_hz: float,
) -> np.ndarray:
    if cutoff_hz <= 0.0:
        return np.asarray(signal_values, dtype=np.float32)

    dt = 1.0 / float(max(sample_rate, 1))
    rc = 1.0 / max(2.0 * np.pi * cutoff_hz, 1e-6)
    alpha = float(np.clip(rc / (rc + dt), 0.0, 1.0))
    source = np.asarray(signal_values, dtype=np.float32)
    filtered = np.empty_like(source)
    filtered[0] = source[0]
    for index in range(1, source.size):
        filtered[index] = alpha * (
            filtered[index - 1] + source[index] - source[index - 1]
        )
    return filtered.astype(np.float32, copy=False)


def _apply_gate_sidechain_filters(
    channels: np.ndarray,
    sample_rate: int,
    gate_profile: dict[str, float | bool | str | None],
) -> np.ndarray:
    low_cut_hz = float(
        np.clip(gate_profile["sidechain_hpf_hz"], 0.0, sample_rate * 0.5)
    )
    high_cut_hz = float(
        np.clip(gate_profile["sidechain_lpf_hz"], 0.0, sample_rate * 0.5)
    )
    if low_cut_hz <= 0.0 and high_cut_hz <= 0.0:
        return np.asarray(channels, dtype=np.float32)

    source_channels = _as_audio_channels(channels)
    if int(source_channels.shape[-1]) == 0:
        return source_channels.astype(np.float32, copy=False)

    filtered_channels = []
    for channel in source_channels:
        filtered_channel = channel.astype(np.float32, copy=True)
        if high_cut_hz > 0.0:
            for _ in range(2):
                filtered_channel = _apply_first_order_lowpass(
                    filtered_channel,
                    sample_rate,
                    high_cut_hz,
                )
        if low_cut_hz > 0.0:
            for _ in range(4):
                filtered_channel = _apply_first_order_highpass(
                    filtered_channel,
                    sample_rate,
                    low_cut_hz,
                )
        filtered_channels.append(filtered_channel)

    return np.vstack(filtered_channels).astype(np.float32, copy=False)


def _compute_gate_rms_envelope(
    channels: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    source_channels = _as_audio_channels(channels)
    envelopes = []
    for channel in source_channels:
        channel_power = np.square(channel, dtype=np.float32)
        smoothed_power = _moving_average(channel_power, window_samples)
        envelopes.append(
            np.sqrt(np.maximum(smoothed_power, 1e-12)).astype(np.float32)
        )
    return np.vstack(envelopes).astype(np.float32)


def _link_gate_detector_envelope(
    detector_envelope: np.ndarray,
    stereo_link_percent: float,
) -> np.ndarray:
    detector_channels = _as_audio_channels(detector_envelope)
    if detector_channels.shape[0] <= 1:
        return detector_channels.astype(np.float32, copy=False)

    link_amount = float(np.clip(stereo_link_percent / 100.0, 0.0, 1.0))
    if link_amount <= 0.0:
        return detector_channels.astype(np.float32, copy=False)

    linked = np.max(detector_channels, axis=0, keepdims=True)
    return (
        detector_channels * (1.0 - link_amount) + linked * link_amount
    ).astype(np.float32, copy=False)


def _resolve_noise_gate_threshold_db(
    detector_db: np.ndarray,
    gate_profile: dict[str, float | bool | str | None],
) -> tuple[float, float, float]:
    finite_levels = detector_db[np.isfinite(detector_db)]
    if finite_levels.size == 0:
        return -60.0, -120.0, -60.0

    noise_floor_db = float(
        np.percentile(finite_levels, gate_profile["noise_percentile"])
    )
    peak_db = float(np.percentile(finite_levels, 99.8))
    threshold_override = gate_profile["threshold_db"]
    if threshold_override is None:
        dynamic_range_db = max(peak_db - noise_floor_db, 1.0)
        threshold_db = noise_floor_db + dynamic_range_db * float(
            gate_profile["threshold_ratio"]
        )
        threshold_db = float(
            np.clip(
                threshold_db,
                noise_floor_db + 0.5,
                max(peak_db - 0.25, noise_floor_db + 0.5),
            )
        )
    else:
        threshold_db = float(threshold_override)

    return threshold_db, noise_floor_db, peak_db


def _soft_gate_curve(
    level_db: float,
    close_threshold_db: float,
    full_open_db: float,
    knee_db: float,
) -> float:
    if full_open_db <= close_threshold_db:
        return 1.0 if level_db >= full_open_db else 0.0

    lower_bound = close_threshold_db - max(knee_db, 0.0) * 0.5
    upper_bound = full_open_db + max(knee_db, 0.0) * 0.5
    if upper_bound <= lower_bound:
        return 1.0 if level_db >= full_open_db else 0.0

    position = float(
        np.clip(
            (level_db - lower_bound) / (upper_bound - lower_bound),
            0.0,
            1.0,
        )
    )
    return float(position * position * (3.0 - 2.0 * position))


def _build_gate_gain_curve(
    detector_db: np.ndarray,
    *,
    sample_rate: int,
    gate_profile: dict[str, float | bool | str | None],
) -> tuple[np.ndarray, np.ndarray, float]:
    sample_count = int(detector_db.size)
    if sample_count == 0:
        return (
            np.ones(0, dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            -120.0,
        )

    attack_samples = _resolve_mask_window_samples(
        sample_count,
        sample_rate,
        float(gate_profile["attack_ms"]),
        max_fraction=0.08,
    )
    hold_samples = _resolve_mask_window_samples(
        sample_count,
        sample_rate,
        float(gate_profile["hold_ms"]),
        max_fraction=0.2,
    )
    release_samples = _resolve_mask_window_samples(
        sample_count,
        sample_rate,
        float(gate_profile["release_ms"]),
        max_fraction=0.24,
    )
    knee_db = float(np.clip(gate_profile["soft_knee_db"], 0.0, 24.0))
    threshold_db, noise_floor_db, peak_db = _resolve_noise_gate_threshold_db(
        detector_db,
        gate_profile,
    )
    close_threshold_db = threshold_db - float(
        np.clip(gate_profile["hysteresis_db"], 0.0, 24.0)
    )
    dynamic_range_db = max(peak_db - noise_floor_db, 1.0)
    full_open_db = float(
        min(
            peak_db,
            threshold_db
            + max(
                dynamic_range_db * float(gate_profile["full_open_ratio"]),
                max(knee_db * 0.5, 0.5),
            ),
        )
    )
    closed_gain = _db_to_linear(-float(gate_profile["reduction_range_db"]))
    adaptive_release_enabled = bool(gate_profile["adaptive_release_enabled"])
    adaptive_release_strength = float(
        np.clip(gate_profile["adaptive_release_strength"], 0.0, 1.5)
    )
    detector_activity = np.clip(
        (detector_db - noise_floor_db) / max(dynamic_range_db, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32)

    gain_curve = np.empty(sample_count, dtype=np.float32)
    state_curve = np.zeros(sample_count, dtype=np.float32)
    current_open = 0.0
    hold_remaining = 0
    gate_open = False

    for index, level_db in enumerate(detector_db):
        detector_level_db = (
            float(level_db) if np.isfinite(level_db) else noise_floor_db
        )
        if gate_open:
            if detector_level_db < close_threshold_db:
                if hold_remaining > 0:
                    hold_remaining -= 1
                else:
                    gate_open = False
            else:
                hold_remaining = hold_samples
        elif detector_level_db >= threshold_db:
            gate_open = True
            hold_remaining = hold_samples

        state_curve[index] = 1.0 if gate_open else 0.0
        target_open = _soft_gate_curve(
            detector_level_db,
            close_threshold_db,
            full_open_db,
            knee_db,
        )
        if gate_open and hold_remaining > 0:
            target_open = 1.0

        if target_open > current_open:
            smoothing_span = max(attack_samples, 1)
        else:
            release_multiplier = 1.0
            if adaptive_release_enabled:
                release_multiplier += adaptive_release_strength * float(
                    detector_activity[index]
                )
            smoothing_span = max(
                int(round(release_samples * release_multiplier)),
                1,
            )

        current_open += (target_open - current_open) / float(smoothing_span)
        current_open = float(np.clip(current_open, 0.0, 1.0))
        gain_curve[index] = closed_gain + (1.0 - closed_gain) * current_open

    return gain_curve, state_curve, threshold_db


def _find_nearest_zero_crossing(
    signal_values: np.ndarray,
    center_index: int,
    radius_samples: int,
) -> int:
    sample_count = int(signal_values.size)
    if sample_count <= 1 or radius_samples <= 0:
        return int(np.clip(center_index, 0, max(sample_count - 1, 0)))

    start_index = max(1, center_index - radius_samples)
    end_index = min(sample_count - 1, center_index + radius_samples)
    if end_index <= start_index:
        return int(np.clip(center_index, 0, sample_count - 1))

    segment = signal_values[start_index - 1 : end_index + 1]
    if segment.size <= 1:
        return int(np.clip(center_index, 0, sample_count - 1))

    sign_changes = np.where(
        np.signbit(segment[:-1]) != np.signbit(segment[1:])
    )[0]
    if sign_changes.size == 0:
        return int(np.clip(center_index, 0, sample_count - 1))

    crossing_positions = start_index + sign_changes
    return int(
        crossing_positions[
            int(np.argmin(np.abs(crossing_positions - center_index)))
        ]
    )


def _apply_zero_crossing_smoothing(
    program_channels: np.ndarray,
    gate_mask: np.ndarray,
    state_curves: np.ndarray,
    *,
    sample_rate: int,
    gate_profile: dict[str, float | bool | str | None],
) -> np.ndarray:
    if not bool(gate_profile["zero_crossing_enabled"]):
        return gate_mask.astype(np.float32, copy=False)

    source_channels = _as_audio_channels(program_channels)
    mask_channels = _as_audio_channels(gate_mask).astype(np.float32, copy=True)
    guide_channels = _as_audio_channels(state_curves)
    window_samples = _resolve_mask_window_samples(
        int(mask_channels.shape[-1]),
        sample_rate,
        float(gate_profile["zero_crossing_window_ms"]),
        max_fraction=0.02,
    )
    fade_samples = max(1, window_samples)

    for channel_index in range(mask_channels.shape[0]):
        guide_index = min(channel_index, guide_channels.shape[0] - 1)
        transition_indices = (
            np.where(np.abs(np.diff(guide_channels[guide_index])) > 0.5)[0] + 1
        )
        if transition_indices.size == 0:
            continue

        program_channel = source_channels[
            min(channel_index, source_channels.shape[0] - 1)
        ]
        for transition_index in transition_indices:
            zero_index = _find_nearest_zero_crossing(
                program_channel,
                int(transition_index),
                window_samples,
            )
            fade_start = max(0, zero_index - fade_samples)
            fade_end = min(
                mask_channels.shape[-1], zero_index + fade_samples + 1
            )
            if fade_end - fade_start <= 1:
                continue

            start_gain = float(mask_channels[channel_index, fade_start])
            end_gain = float(mask_channels[channel_index, fade_end - 1])
            mask_channels[channel_index, fade_start:fade_end] = np.linspace(
                start_gain,
                end_gain,
                fade_end - fade_start,
            ).astype(np.float32)

    return np.clip(mask_channels, 0.0, 1.0).astype(np.float32, copy=False)


def _delay_audio_channels(
    channels: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    source_channels = _as_audio_channels(channels)
    if delay_samples <= 0 or source_channels.size == 0:
        return source_channels.astype(np.float32, copy=False)
    return np.pad(
        source_channels,
        ((0, 0), (delay_samples, 0)),
    )[:, : source_channels.shape[-1]].astype(np.float32, copy=False)


def _compensate_audio_delay(
    channels: np.ndarray,
    delay_samples: int,
) -> np.ndarray:
    source_channels = _as_audio_channels(channels)
    if delay_samples <= 0 or source_channels.size == 0:
        return source_channels.astype(np.float32, copy=False)
    trimmed = source_channels[:, delay_samples:]
    return np.pad(
        trimmed,
        ((0, 0), (0, delay_samples)),
    ).astype(np.float32, copy=False)


def _resolve_stem_noise_gate_profile(
    stem_role: str | None,
    cleanup_pressure: float,
    gate_strength: float,
) -> dict[str, float | bool | str | None]:
    normalized_role = _normalize_stem_role(stem_role)
    strength = float(np.clip(gate_strength, 0.0, 1.5))

    if normalized_role == "drums":
        intensity = float(
            np.clip((0.74 + cleanup_pressure * 0.14) * strength, 0.0, 1.0)
        )
        return {
            "normalization_mode": "none",
            "normalization_target_lufs": -24.0,
            "normalization_target_dbfs": -6.0,
            "threshold_db": None,
            "hysteresis_db": 2.4,
            "reduction_range_db": float(18.0 + intensity * 7.0),
            "attack_ms": 1.2,
            "hold_ms": 52.0,
            "release_ms": 20.0,
            "lookahead_ms": 0.9,
            "soft_knee_db": 2.0,
            "oversampling": 1,
            "sidechain_hpf_hz": 28.0,
            "sidechain_lpf_hz": 0.0,
            "stereo_link_percent": 100.0,
            "zero_crossing_enabled": True,
            "zero_crossing_window_ms": 0.9,
            "adaptive_release_enabled": True,
            "adaptive_release_strength": 0.2,
            "rms_window_ms": 4.0,
            "dc_offset_compensation": True,
            "delay_compensation_enabled": True,
            "inter_sample_peak_awareness": True,
            "analysis_peak_dbfs": -1.0,
            "noise_percentile": 38.0,
            "threshold_ratio": float(
                np.clip(0.07 + intensity * 0.02, 0.045, 0.18)
            ),
            "full_open_ratio": 0.16,
        }

    if normalized_role == "vocals":
        intensity = float(
            np.clip((0.6 + cleanup_pressure * 0.18) * strength, 0.0, 1.0)
        )
        return {
            "normalization_mode": "none",
            "normalization_target_lufs": -23.0,
            "normalization_target_dbfs": -7.0,
            "threshold_db": None,
            "hysteresis_db": 4.8,
            "reduction_range_db": float(30.0 + intensity * 9.0),
            "attack_ms": 3.4,
            "hold_ms": 96.0,
            "release_ms": 42.0,
            "lookahead_ms": 2.0,
            "soft_knee_db": 4.5,
            "oversampling": 1,
            "sidechain_hpf_hz": 90.0,
            "sidechain_lpf_hz": 9000.0,
            "stereo_link_percent": 100.0,
            "zero_crossing_enabled": True,
            "zero_crossing_window_ms": 1.5,
            "adaptive_release_enabled": True,
            "adaptive_release_strength": 0.56,
            "rms_window_ms": 10.0,
            "dc_offset_compensation": True,
            "delay_compensation_enabled": True,
            "inter_sample_peak_awareness": True,
            "analysis_peak_dbfs": -1.0,
            "noise_percentile": 24.0,
            "threshold_ratio": float(
                np.clip(0.065 + intensity * 0.03, 0.045, 0.18)
            ),
            "full_open_ratio": 0.24,
        }

    if normalized_role == "bass":
        intensity = float(
            np.clip((0.5 + cleanup_pressure * 0.14) * strength, 0.0, 0.94)
        )
        return {
            "normalization_mode": "none",
            "normalization_target_lufs": -22.0,
            "normalization_target_dbfs": -6.0,
            "threshold_db": None,
            "hysteresis_db": 5.5,
            "reduction_range_db": float(10.0 + intensity * 8.0),
            "attack_ms": 5.4,
            "hold_ms": 126.0,
            "release_ms": 58.0,
            "lookahead_ms": 1.4,
            "soft_knee_db": 5.0,
            "oversampling": 1,
            "sidechain_hpf_hz": 0.0,
            "sidechain_lpf_hz": 1900.0,
            "stereo_link_percent": 100.0,
            "zero_crossing_enabled": True,
            "zero_crossing_window_ms": 1.8,
            "adaptive_release_enabled": True,
            "adaptive_release_strength": 0.68,
            "rms_window_ms": 16.0,
            "dc_offset_compensation": True,
            "delay_compensation_enabled": True,
            "inter_sample_peak_awareness": True,
            "analysis_peak_dbfs": -1.0,
            "noise_percentile": 18.0,
            "threshold_ratio": float(
                np.clip(0.11 + intensity * 0.025, 0.075, 0.22)
            ),
            "full_open_ratio": 0.28,
        }

    intensity = float(
        np.clip((0.58 + cleanup_pressure * 0.18) * strength, 0.0, 1.0)
    )
    return {
        "normalization_mode": "none",
        "normalization_target_lufs": -24.0,
        "normalization_target_dbfs": -6.0,
        "threshold_db": None,
        "hysteresis_db": 4.0,
        "reduction_range_db": float(22.0 + intensity * 8.0),
        "attack_ms": 3.2,
        "hold_ms": 84.0,
        "release_ms": 36.0,
        "lookahead_ms": 1.6,
        "soft_knee_db": 3.8,
        "oversampling": 1,
        "sidechain_hpf_hz": 45.0,
        "sidechain_lpf_hz": 14000.0,
        "stereo_link_percent": 100.0,
        "zero_crossing_enabled": True,
        "zero_crossing_window_ms": 1.4,
        "adaptive_release_enabled": True,
        "adaptive_release_strength": 0.42,
        "rms_window_ms": 8.0,
        "dc_offset_compensation": True,
        "delay_compensation_enabled": True,
        "inter_sample_peak_awareness": True,
        "analysis_peak_dbfs": -1.0,
        "noise_percentile": 24.0,
        "threshold_ratio": float(np.clip(0.075 + intensity * 0.03, 0.045, 0.2)),
        "full_open_ratio": 0.24,
    }


def _build_stem_activity_mask(
    mono_energy: np.ndarray,
    *,
    sample_rate: int,
    cleanup_profile: dict[str, float],
) -> np.ndarray:
    if mono_energy.size == 0:
        return np.ones(0, dtype=np.float32)

    fast_samples = _resolve_mask_window_samples(
        mono_energy.size,
        sample_rate,
        cleanup_profile["fast_ms"],
        max_fraction=0.08,
    )
    slow_samples = _resolve_mask_window_samples(
        mono_energy.size,
        sample_rate,
        cleanup_profile["slow_ms"],
        max_fraction=0.16,
    )
    hold_samples = _resolve_mask_window_samples(
        mono_energy.size,
        sample_rate,
        cleanup_profile["hold_ms"],
        max_fraction=0.14,
    )
    release_samples = _resolve_mask_window_samples(
        mono_energy.size,
        sample_rate,
        cleanup_profile["release_ms"],
        max_fraction=0.12,
    )

    fast_env = _moving_average(mono_energy, fast_samples)
    slow_env = _moving_average(mono_energy, slow_samples)

    noise_floor = float(
        np.percentile(slow_env, cleanup_profile["noise_percentile"])
    )
    peak_env = float(np.percentile(slow_env, 99.6))
    if peak_env <= noise_floor + 1e-6:
        return np.ones_like(mono_energy, dtype=np.float32)

    activity = np.clip(
        (slow_env - noise_floor) / (peak_env - noise_floor + 1e-6),
        0.0,
        1.0,
    )
    activity = np.power(activity, cleanup_profile["activity_exponent"])
    transient = np.clip(
        (fast_env / np.maximum(slow_env, 1e-6) - 1.0) / 0.85,
        0.0,
        1.0,
    )
    activity = np.clip(
        activity
        + transient * cleanup_profile["transient_blend"] * (1.0 - activity),
        0.0,
        1.0,
    )

    held_activity = _rolling_max(activity, hold_samples)
    smoothed_activity = _moving_average(held_activity, release_samples)
    smoothed_activity = np.maximum(
        smoothed_activity,
        held_activity * cleanup_profile["activity_floor_scale"],
    )
    suppression_floor = cleanup_profile["suppression_floor"]
    return suppression_floor + (1.0 - suppression_floor) * np.clip(
        smoothed_activity,
        0.0,
        1.0,
    )


def _build_stem_noise_gate_mask(
    detector_channels: np.ndarray,
    *,
    sample_rate: int,
    gate_profile: dict[str, float | bool | str | None],
) -> tuple[np.ndarray, np.ndarray, float]:
    source_channels = _as_audio_channels(detector_channels)
    if source_channels.size == 0:
        empty = np.ones_like(source_channels, dtype=np.float32)
        return empty, np.zeros_like(source_channels, dtype=np.float32), -120.0

    rms_window_samples = _resolve_mask_window_samples(
        int(source_channels.shape[-1]),
        sample_rate,
        float(gate_profile["rms_window_ms"]),
        max_fraction=0.12,
    )
    detector_envelope = _compute_gate_rms_envelope(
        source_channels,
        rms_window_samples,
    )
    linked_envelope = _link_gate_detector_envelope(
        detector_envelope,
        float(gate_profile["stereo_link_percent"]),
    )
    detector_db = _amplitude_to_db(linked_envelope)

    gate_masks: list[np.ndarray] = []
    state_curves: list[np.ndarray] = []
    thresholds: list[float] = []

    for channel_db in detector_db:
        channel_mask, channel_state, threshold_db = _build_gate_gain_curve(
            channel_db,
            sample_rate=sample_rate,
            gate_profile=gate_profile,
        )
        gate_masks.append(channel_mask)
        state_curves.append(channel_state)
        thresholds.append(threshold_db)

    threshold_value = float(np.mean(thresholds)) if thresholds else -120.0
    return (
        np.vstack(gate_masks).astype(np.float32, copy=False),
        np.vstack(state_curves).astype(np.float32, copy=False),
        threshold_value,
    )


def _apply_stem_residual_suppression(
    self,
    y: np.ndarray,
    *,
    stem_role: str | None,
    cleanup_pressure: float,
) -> np.ndarray:
    if y.size == 0:
        return y

    cleanup_profile = _resolve_stem_residual_profile(
        stem_role, cleanup_pressure
    )
    channels = y if y.ndim > 1 else y[np.newaxis, :]
    working_channels = np.asarray(channels, dtype=np.float32)
    mono_energy = np.mean(np.abs(working_channels), axis=0)
    if not np.any(mono_energy > 0.0):
        return y

    activity_mask = _build_stem_activity_mask(
        mono_energy,
        sample_rate=self.resampling_target,
        cleanup_profile=cleanup_profile,
    )
    expansion_drive = float(cleanup_profile["expansion_drive"])
    expansion_mix = float(cleanup_profile["expansion_mix"])
    cleaned_channels: list[np.ndarray] = []

    for channel in working_channels:
        cleaned_channel = channel * activity_mask
        peak = float(np.max(np.abs(cleaned_channel)))
        if peak > 1e-6 and expansion_mix > 0.0:
            normalized = np.clip(np.abs(cleaned_channel) / peak, 0.0, 1.0)
            expanded_channel = (
                np.sign(cleaned_channel)
                * peak
                * np.power(
                    normalized,
                    1.0 + expansion_drive,
                )
            )
            cleaned_channel = (
                cleaned_channel * (1.0 - expansion_mix)
                + expanded_channel * expansion_mix
            )
        cleaned_channels.append(cleaned_channel)

    cleaned = np.vstack(cleaned_channels) if y.ndim > 1 else cleaned_channels[0]
    return _restore_audio_dtype(cleaned, y.dtype)


def _apply_stem_noise_gate(
    self,
    y: np.ndarray,
    *,
    stem_role: str | None,
    cleanup_pressure: float,
) -> np.ndarray:
    self.last_stem_noise_gate_analysis_peak_dbfs = None
    self.last_stem_noise_gate_threshold_db = None
    self.last_stem_noise_gate_profile = None
    if y.size == 0:
        return y

    config = getattr(self, "config", None)
    if not bool(getattr(config, "stem_noise_gate_enabled", True)):
        return y

    gate_strength = float(
        np.clip(getattr(config, "stem_noise_gate_strength", 1.0), 0.0, 1.5)
    )
    if gate_strength <= 0.0:
        return y

    working_channels = _as_audio_channels(y)
    mono_energy = np.mean(np.abs(working_channels), axis=0)
    if not np.any(mono_energy > 0.0):
        return y

    gate_profile = _resolve_stem_noise_gate_profile(
        stem_role,
        cleanup_pressure,
        gate_strength,
    )
    gate_profile["normalization_mode"] = (
        str(getattr(config, "stem_noise_gate_normalization_mode", "none"))
        .strip()
        .lower()
    )
    gate_profile["normalization_target_lufs"] = float(
        getattr(config, "stem_noise_gate_normalization_target_lufs", -24.0)
    )
    gate_profile["normalization_target_dbfs"] = float(
        getattr(config, "stem_noise_gate_normalization_target_dbfs", -6.0)
    )
    gate_profile["threshold_db"] = getattr(
        config,
        "stem_noise_gate_threshold_db",
        None,
    )
    gate_profile["hysteresis_db"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_hysteresis_db", 4.5), 0.0, 24.0
        )
    )
    gate_profile["reduction_range_db"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_reduction_range_db", 28.0),
            0.0,
            96.0,
        )
    )
    gate_profile["attack_ms"] = float(
        np.clip(getattr(config, "stem_noise_gate_attack_ms", 4.0), 0.1, 80.0)
    )
    gate_profile["hold_ms"] = float(
        np.clip(getattr(config, "stem_noise_gate_hold_ms", 90.0), 0.0, 500.0)
    )
    gate_profile["release_ms"] = float(
        np.clip(getattr(config, "stem_noise_gate_release_ms", 42.0), 1.0, 500.0)
    )
    gate_profile["lookahead_ms"] = float(
        np.clip(getattr(config, "stem_noise_gate_lookahead_ms", 2.0), 0.0, 20.0)
    )
    gate_profile["soft_knee_db"] = float(
        np.clip(getattr(config, "stem_noise_gate_soft_knee_db", 4.0), 0.0, 24.0)
    )
    gate_profile["oversampling"] = int(
        np.clip(getattr(config, "stem_noise_gate_oversampling", 1), 1, 8)
    )
    gate_profile["sidechain_hpf_hz"] = float(
        max(getattr(config, "stem_noise_gate_sidechain_hpf_hz", 0.0), 0.0)
    )
    gate_profile["sidechain_lpf_hz"] = float(
        max(getattr(config, "stem_noise_gate_sidechain_lpf_hz", 0.0), 0.0)
    )
    gate_profile["stereo_link_percent"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_stereo_link_percent", 100.0),
            0.0,
            100.0,
        )
    )
    gate_profile["zero_crossing_enabled"] = bool(
        getattr(config, "stem_noise_gate_zero_crossing_enabled", True)
    )
    gate_profile["zero_crossing_window_ms"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_zero_crossing_window_ms", 1.5),
            0.0,
            25.0,
        )
    )
    gate_profile["adaptive_release_enabled"] = bool(
        getattr(config, "stem_noise_gate_adaptive_release_enabled", True)
    )
    gate_profile["adaptive_release_strength"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_adaptive_release_strength", 0.45),
            0.0,
            1.5,
        )
    )
    gate_profile["rms_window_ms"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_rms_window_ms", 10.0),
            0.5,
            200.0,
        )
    )
    gate_profile["dc_offset_compensation"] = bool(
        getattr(config, "stem_noise_gate_dc_offset_compensation", True)
    )
    gate_profile["delay_compensation_enabled"] = bool(
        getattr(config, "stem_noise_gate_delay_compensation_enabled", True)
    )
    gate_profile["inter_sample_peak_awareness"] = bool(
        getattr(config, "stem_noise_gate_inter_sample_peak_awareness", True)
    )
    gate_profile["analysis_peak_dbfs"] = float(
        np.clip(
            getattr(config, "stem_noise_gate_analysis_peak_dbfs", -1.0),
            -12.0,
            -0.1,
        )
    )
    self.last_stem_noise_gate_profile = dict(gate_profile)

    normalized_analysis = _normalize_gate_analysis_channels(
        working_channels,
        self.resampling_target,
        gate_profile,
    )
    self.last_stem_noise_gate_analysis_peak_dbfs = float(
        _measure_gate_analysis_peak_dbfs(
            normalized_analysis,
            self.resampling_target,
            gate_profile,
        )
    )

    oversampling = max(int(gate_profile["oversampling"]), 1)
    analysis_channels = normalized_analysis
    program_channels = working_channels.astype(np.float32, copy=False)
    if oversampling > 1:
        analysis_channels = _resample_audio_channels(
            analysis_channels,
            up=oversampling,
            down=1,
        )
        program_channels = _resample_audio_channels(
            program_channels,
            up=oversampling,
            down=1,
        )
    analysis_rate = int(self.resampling_target * oversampling)
    detector_channels = _apply_gate_sidechain_filters(
        analysis_channels,
        analysis_rate,
        gate_profile,
    )

    gate_mask, state_curves, threshold_db = _build_stem_noise_gate_mask(
        detector_channels,
        sample_rate=analysis_rate,
        gate_profile=gate_profile,
    )
    self.last_stem_noise_gate_threshold_db = float(threshold_db)
    lookahead_samples = _resolve_mask_window_samples(
        int(program_channels.shape[-1]),
        analysis_rate,
        float(gate_profile["lookahead_ms"]),
        max_fraction=0.06,
    )
    delayed_program = _delay_audio_channels(program_channels, lookahead_samples)
    gate_mask = _apply_zero_crossing_smoothing(
        delayed_program,
        gate_mask,
        state_curves,
        sample_rate=analysis_rate,
        gate_profile=gate_profile,
    )
    gated = delayed_program * gate_mask
    if bool(gate_profile["delay_compensation_enabled"]):
        gated = _compensate_audio_delay(gated, lookahead_samples)
    if oversampling > 1:
        gated = _resample_audio_channels(gated, up=1, down=oversampling)
    gated = _align_audio_channels(gated, int(working_channels.shape[-1]))
    output = _restore_audio_layout(gated, y)
    return _restore_audio_dtype(output, y.dtype)


def _resolve_stem_cleanup_anchors(
    self,
    *,
    stem_role: str | None,
    restoration_factor: float,
    air_restoration_factor: float,
    body_restoration_factor: float,
    closure_repair_factor: float,
) -> list[list[float]]:
    normalized_role = _normalize_stem_role(stem_role)
    mud_low_hz = self._fit_frequency(max(self.low_cut * 2.4, 180.0))
    mud_high_hz = self._fit_frequency(max(mud_low_hz * 1.8, 420.0))
    focus_hz = self._fit_frequency(max(self.bass_transition_hz * 1.8, 1150.0))
    presence_lift_hz = self._fit_frequency(
        max(self.treble_transition_hz * 0.62, 1700.0)
    )
    presence_cut_hz = self._fit_frequency(
        max(self.treble_transition_hz * 0.82, 2900.0)
    )
    air_low_hz = self._fit_frequency(
        max(self.treble_transition_hz * 1.28, 5200.0)
    )
    air_high_hz = self._fit_frequency(
        max(self.treble_transition_hz * 1.95, 9000.0)
    )
    harmonic_low_hz = self._fit_frequency(
        max(self.bass_transition_hz * 2.0, 850.0)
    )
    harmonic_high_hz = self._fit_frequency(
        max(self.bass_transition_hz * 3.1, 1800.0)
    )

    if normalized_role == "bass":
        mud_cut_db = float(
            np.clip(0.18 + body_restoration_factor * 0.75, 0.0, 1.45)
        )
        harmonic_boost_db = float(
            np.clip(
                0.08
                + closure_repair_factor * 0.28
                + air_restoration_factor * 0.14,
                0.0,
                0.95,
            )
        )
        return [
            [self.low_cut, 0.0],
            [mud_low_hz, -mud_cut_db],
            [mud_high_hz, -mud_cut_db * 0.72],
            [harmonic_low_hz, harmonic_boost_db * 0.42],
            [harmonic_high_hz, harmonic_boost_db],
            [self.high_cut, 0.0],
        ]

    if normalized_role == "vocals":
        mud_cut_db = float(
            np.clip(
                0.22
                + body_restoration_factor * 0.92
                + closure_repair_factor * 0.14,
                0.0,
                1.7,
            )
        )
        presence_lift_db = float(
            np.clip(0.12 + closure_repair_factor * 0.3, 0.0, 0.9)
        )
        presence_cut_db = float(
            np.clip(0.2 + closure_repair_factor * 0.98, 0.0, 1.9)
        )
        air_low_boost_db = float(
            np.clip(
                0.08
                + air_restoration_factor * 0.45
                + closure_repair_factor * 0.18,
                0.0,
                0.95,
            )
        )
        air_high_boost_db = float(
            np.clip(
                0.16
                + air_restoration_factor * 1.08
                + closure_repair_factor * 0.36,
                0.0,
                2.25,
            )
        )
        return [
            [self.low_cut, 0.0],
            [mud_low_hz, -mud_cut_db * 0.65],
            [mud_high_hz, -mud_cut_db],
            [presence_lift_hz, presence_lift_db],
            [presence_cut_hz, -presence_cut_db],
            [air_low_hz, air_low_boost_db],
            [air_high_hz, air_high_boost_db],
            [self.high_cut, 0.0],
        ]

    if normalized_role == "drums":
        mud_cut_db = float(
            np.clip(0.12 + body_restoration_factor * 0.42, 0.0, 0.85)
        )
        presence_cut_db = float(
            np.clip(0.08 + closure_repair_factor * 0.35, 0.0, 0.75)
        )
        air_high_boost_db = float(
            np.clip(0.1 + air_restoration_factor * 0.76, 0.0, 1.0)
        )
        return [
            [self.low_cut, 0.0],
            [mud_low_hz, -mud_cut_db],
            [presence_cut_hz, -presence_cut_db],
            [air_low_hz, air_high_boost_db * 0.45],
            [air_high_hz, air_high_boost_db],
            [self.high_cut, 0.0],
        ]

    mud_cut_db = float(
        np.clip(
            0.16 + body_restoration_factor * 0.58 + restoration_factor * 0.12,
            0.0,
            1.2,
        )
    )
    presence_lift_db = float(
        np.clip(0.08 + closure_repair_factor * 0.2, 0.0, 0.55)
    )
    presence_cut_db = float(
        np.clip(0.1 + closure_repair_factor * 0.52, 0.0, 1.0)
    )
    air_high_boost_db = float(
        np.clip(0.1 + air_restoration_factor * 0.82, 0.0, 1.3)
    )
    focus_boost_db = float(
        np.clip(0.06 + closure_repair_factor * 0.18, 0.0, 0.4)
    )
    return [
        [self.low_cut, 0.0],
        [mud_low_hz, -mud_cut_db * 0.72],
        [mud_high_hz, -mud_cut_db],
        [focus_hz, focus_boost_db],
        [presence_lift_hz, presence_lift_db],
        [presence_cut_hz, -presence_cut_db],
        [air_low_hz, air_high_boost_db * 0.5],
        [air_high_hz, air_high_boost_db],
        [self.high_cut, 0.0],
    ]


def apply_stem_cleanup(
    self,
    y: np.ndarray,
    *,
    stem_role: str | None,
    audio_eq_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    profile = getattr(self, "spectral_balance_profile", None)
    restoration_factor = float(
        np.clip(getattr(profile, "restoration_factor", 0.0), 0.0, 1.0)
    )
    air_restoration_factor = float(
        np.clip(getattr(profile, "air_restoration_factor", 0.0), 0.0, 1.0)
    )
    body_restoration_factor = float(
        np.clip(getattr(profile, "body_restoration_factor", 0.0), 0.0, 1.0)
    )
    closure_repair_factor = float(
        np.clip(
            getattr(
                profile,
                "closure_repair_factor",
                max(
                    air_restoration_factor * 0.85,
                    restoration_factor * 0.55,
                ),
            ),
            0.0,
            1.0,
        )
    )
    cleanup_pressure = float(
        np.clip(
            max(
                restoration_factor * 0.35,
                air_restoration_factor * 0.45,
                body_restoration_factor * 0.3,
                closure_repair_factor * 0.6,
            ),
            0.0,
            1.0,
        )
    )
    stem_cleanup_strength = float(
        np.clip(
            getattr(
                self,
                "stem_cleanup_strength",
                getattr(self.config, "stem_cleanup_strength", 1.0),
            ),
            0.0,
            1.5,
        )
    )
    cleanup_pressure = float(
        np.clip(cleanup_pressure * stem_cleanup_strength, 0.0, 1.0)
    )
    cleanup_pressure = _resolve_stem_cleanup_pressure(
        stem_role,
        cleanup_pressure,
    )
    cleaned = y
    if cleanup_pressure > 0.04:
        cleanup_anchors = _resolve_stem_cleanup_anchors(
            self,
            stem_role=stem_role,
            restoration_factor=restoration_factor,
            air_restoration_factor=air_restoration_factor,
            body_restoration_factor=body_restoration_factor,
            closure_repair_factor=closure_repair_factor,
        )

        def eq_channel(channel: np.ndarray) -> np.ndarray:
            return audio_eq_fn(
                audio_data=channel,
                anchors=cleanup_anchors,
                sample_rate=self.resampling_target,
                nperseg=self.analysis_nperseg,
            )

        cleaned = (
            np.vstack([eq_channel(channel) for channel in y])
            if y.ndim > 1
            else eq_channel(y)
        )

    cleaned = _apply_stem_noise_gate(
        self,
        cleaned,
        stem_role=stem_role,
        cleanup_pressure=cleanup_pressure,
    )
    cleaned = _apply_stem_residual_suppression(
        self,
        cleaned,
        stem_role=stem_role,
        cleanup_pressure=cleanup_pressure,
    )
    cleaned = _preserve_stem_cleanup_content(
        y,
        cleaned,
        stem_role=stem_role,
    )
    return _restore_audio_dtype(cleaned, y.dtype)


def apply_eq(
    self,
    y: np.ndarray,
    audio_eq_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

    input_db, f_axis = self.measure_spectrum(y_mono)
    input_db = self.smooth_curve(input_db, f_axis, self.smoothing_fraction)

    target_db = self.build_target_curve(f_axis)

    correction_db = target_db - input_db
    correction_db = np.nan_to_num(
        correction_db,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    self.spectral_balance_profile = self.build_spectral_balance_profile(
        correction_db,
        f_axis,
    )

    eq_stride = max(1, len(correction_db) // 192)

    correction_db = np.append(correction_db[:-1:eq_stride], correction_db[-1])
    f_axis = np.append(f_axis[:-1:eq_stride], f_axis[-1])

    reference_hz = float(
        np.sqrt(self.bass_transition_hz * self.treble_transition_hz)
    )
    reference_index = int(np.argmin(np.abs(f_axis - reference_hz)))
    restoration_factor = float(
        np.clip(
            getattr(self.spectral_balance_profile, "restoration_factor", 0.0),
            0.0,
            1.0,
        )
    )
    edge_baseline_db = float(np.average([correction_db[0], correction_db[-1]]))
    reference_baseline_db = float(correction_db[reference_index])
    baseline_db = float(
        edge_baseline_db * (1.0 - restoration_factor)
        + reference_baseline_db * restoration_factor
    )

    body_restoration_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "body_restoration_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    air_restoration_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "air_restoration_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    mud_cleanup_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "mud_cleanup_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    harshness_restraint_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "harshness_restraint_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    low_end_restraint_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "low_end_restraint_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    legacy_tonal_rebalance_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "legacy_tonal_rebalance_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    closed_top_end_repair_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "closed_top_end_repair_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    closure_repair_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "closure_repair_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    presence_start_hz = float(
        min(
            self.treble_transition_hz,
            max(reference_hz * 1.35, self.treble_transition_hz * 0.55),
        )
    )
    positive_correction_db = np.maximum(correction_db, 0.0)
    presence_mask = (f_axis >= presence_start_hz) & (
        f_axis < max(self.treble_transition_hz, presence_start_hz + 1.0)
    )
    high_band_mask = f_axis >= presence_start_hz
    negative_correction_db = np.maximum(-correction_db, 0.0)
    presence_deficit_db = (
        float(np.mean(positive_correction_db[presence_mask], dtype=np.float32))
        if np.any(presence_mask)
        else 0.0
    )
    high_band_deficit_db = (
        float(np.mean(positive_correction_db[high_band_mask], dtype=np.float32))
        if np.any(high_band_mask)
        else 0.0
    )
    if closure_repair_factor <= 0.0:
        closure_repair_factor = float(
            np.clip(
                max(presence_deficit_db * 0.9, high_band_deficit_db) / 9.0,
                0.0,
                1.0,
            )
        )
        closure_repair_factor = float(
            np.clip(
                closure_repair_factor
                * (
                    0.35
                    + restoration_factor * 0.35
                    + air_restoration_factor * 0.6
                ),
                0.0,
                1.0,
            )
        )
    mud_low_hz = float(
        min(self.high_cut, max(self.bass_transition_hz * 1.15, 160.0))
    )
    mud_high_hz = float(
        min(
            self.high_cut,
            max(
                mud_low_hz + 1.0,
                min(self.treble_transition_hz * 0.38, 550.0),
            ),
        )
    )
    mud_mask = (f_axis >= mud_low_hz) & (f_axis <= mud_high_hz)
    if mud_cleanup_factor <= 0.0 and np.any(mud_mask):
        mud_excess_db = float(
            np.mean(negative_correction_db[mud_mask], dtype=np.float32)
        )
        mud_peak_excess_db = float(
            np.percentile(negative_correction_db[mud_mask], 76.0)
        )
        mud_cleanup_factor = float(
            np.clip(
                (max(mud_excess_db, mud_peak_excess_db * 0.9) - 1.15) / 4.35,
                0.0,
                1.0,
            )
        )
        mud_cleanup_factor = float(
            np.clip(
                mud_cleanup_factor
                * (
                    0.45
                    + restoration_factor * 0.28
                    + air_restoration_factor * 0.18
                    + closure_repair_factor * 0.24
                ),
                0.0,
                1.0,
            )
        )
    low_end_focus_low_hz = float(
        min(
            self.high_cut,
            max(self.low_cut * 3.2, self.bass_transition_hz * 0.72, 75.0),
        )
    )
    low_end_focus_high_hz = float(
        min(
            self.high_cut,
            max(
                low_end_focus_low_hz + 1.0,
                min(self.bass_transition_hz * 1.9, 280.0),
            ),
        )
    )
    low_end_focus_mask = (f_axis >= low_end_focus_low_hz) & (
        f_axis <= low_end_focus_high_hz
    )
    if low_end_restraint_factor <= 0.0:
        bass_excess_db = (
            float(
                np.mean(
                    negative_correction_db[f_axis <= self.bass_transition_hz],
                    dtype=np.float32,
                )
            )
            if np.any(f_axis <= self.bass_transition_hz)
            else 0.0
        )
        bass_peak_excess_db = (
            float(
                np.percentile(
                    negative_correction_db[f_axis <= self.bass_transition_hz],
                    78.0,
                )
            )
            if np.any(f_axis <= self.bass_transition_hz)
            else 0.0
        )
        low_end_focus_excess_db = (
            float(
                np.mean(
                    negative_correction_db[low_end_focus_mask], dtype=np.float32
                )
            )
            if np.any(low_end_focus_mask)
            else 0.0
        )
        low_end_focus_peak_excess_db = (
            float(
                np.percentile(negative_correction_db[low_end_focus_mask], 80.0)
            )
            if np.any(low_end_focus_mask)
            else 0.0
        )
        low_end_restraint_factor = float(
            np.clip(
                (
                    max(
                        bass_excess_db * 0.82,
                        bass_peak_excess_db * 0.88,
                        low_end_focus_excess_db,
                        low_end_focus_peak_excess_db * 0.92,
                        mud_cleanup_factor * 3.2,
                    )
                    - 0.95
                )
                / 3.8,
                0.0,
                1.0,
            )
        )
        low_end_restraint_factor = float(
            np.clip(
                low_end_restraint_factor
                * (
                    0.16
                    + closure_repair_factor * 0.58
                    + mud_cleanup_factor * 0.34
                    + air_restoration_factor * 0.12
                ),
                0.0,
                1.0,
            )
        )
    harsh_low_hz = float(
        min(self.high_cut, max(presence_start_hz * 1.08, 2600.0))
    )
    harsh_high_hz = float(
        min(
            self.high_cut,
            max(
                harsh_low_hz + 1.0,
                min(max(self.treble_transition_hz * 1.18, 5200.0), 6200.0),
            ),
        )
    )
    harsh_mask = (f_axis >= harsh_low_hz) & (f_axis < harsh_high_hz)
    sibilance_low_hz = float(min(self.high_cut, max(harsh_high_hz, 5600.0)))
    sibilance_high_hz = float(
        min(
            self.high_cut,
            max(sibilance_low_hz + 1.0, min(self.high_cut, 9200.0)),
        )
    )
    sibilance_mask = (f_axis >= sibilance_low_hz) & (
        f_axis <= sibilance_high_hz
    )
    if harshness_restraint_factor <= 0.0:
        harsh_excess_db = (
            float(np.mean(negative_correction_db[harsh_mask], dtype=np.float32))
            if np.any(harsh_mask)
            else 0.0
        )
        harsh_peak_excess_db = (
            float(np.percentile(negative_correction_db[harsh_mask], 82.0))
            if np.any(harsh_mask)
            else 0.0
        )
        sibilance_peak_excess_db = (
            float(np.percentile(negative_correction_db[sibilance_mask], 78.0))
            if np.any(sibilance_mask)
            else 0.0
        )
        harshness_restraint_factor = float(
            np.clip(
                (
                    max(
                        harsh_excess_db,
                        harsh_peak_excess_db * 0.95,
                        sibilance_peak_excess_db * 0.86,
                    )
                    - 0.7
                )
                / 3.6,
                0.0,
                1.0,
            )
        )
        harshness_restraint_factor = float(
            np.clip(
                harshness_restraint_factor
                * (
                    0.12
                    + closure_repair_factor * 0.55
                    + air_restoration_factor * 0.35
                    + restoration_factor * 0.18
                ),
                0.0,
                1.0,
            )
        )
    baseline_db -= (
        max(reference_baseline_db, 0.0) * closure_repair_factor * 0.42
    )
    baseline_db -= max(reference_baseline_db, 0.0) * mud_cleanup_factor * 0.12
    correction_db -= baseline_db

    low_span_denominator = max(
        float(np.log2(self.bass_transition_hz / self.low_cut)),
        1e-6,
    )
    low_shape = np.clip(
        np.log2(self.bass_transition_hz / np.maximum(f_axis, self.low_cut))
        / low_span_denominator,
        0.0,
        1.0,
    )
    high_span_denominator = max(
        float(np.log2(self.high_cut / self.treble_transition_hz)),
        1e-6,
    )
    high_shape = np.clip(
        np.log2(
            np.maximum(f_axis, self.treble_transition_hz)
            / self.treble_transition_hz
        )
        / high_span_denominator,
        0.0,
        1.0,
    )
    presence_span_denominator = max(
        float(
            np.log2(
                max(self.treble_transition_hz, reference_hz + 1.0)
                / reference_hz
            )
        ),
        1e-6,
    )
    presence_shape = np.clip(
        np.log2(np.maximum(f_axis, reference_hz) / reference_hz)
        / presence_span_denominator,
        0.0,
        1.0,
    )
    presence_shape = np.clip(presence_shape - high_shape * 0.65, 0.0, 1.0)
    mud_center_hz = float(
        min(self.high_cut, max(self.bass_transition_hz * 1.9, 280.0))
    )
    mud_high_shape_hz = float(
        min(self.high_cut, max(mud_center_hz * 1.85, 620.0))
    )
    mud_rise_denominator = max(
        float(np.log2(max(mud_center_hz, mud_low_hz + 1.0) / mud_low_hz)),
        1e-6,
    )
    mud_fall_denominator = max(
        float(
            np.log2(max(mud_high_shape_hz, mud_center_hz + 1.0) / mud_center_hz)
        ),
        1e-6,
    )
    mud_rise_shape = np.clip(
        np.log2(np.maximum(f_axis, mud_low_hz) / mud_low_hz)
        / mud_rise_denominator,
        0.0,
        1.0,
    )
    mud_fall_shape = np.clip(
        np.log2(mud_high_shape_hz / np.maximum(f_axis, mud_center_hz))
        / mud_fall_denominator,
        0.0,
        1.0,
    )
    mud_shape = np.clip(
        np.minimum(mud_rise_shape, mud_fall_shape) * (1.0 - high_shape * 0.92),
        0.0,
        1.0,
    )
    low_end_focus_center_hz = float(
        min(
            self.high_cut,
            max(
                low_end_focus_low_hz * 1.28,
                self.bass_transition_hz * 1.05,
                135.0,
            ),
        )
    )
    low_end_focus_rise_denominator = max(
        float(
            np.log2(
                max(low_end_focus_center_hz, low_end_focus_low_hz + 1.0)
                / low_end_focus_low_hz
            )
        ),
        1e-6,
    )
    low_end_focus_fall_denominator = max(
        float(
            np.log2(
                max(low_end_focus_high_hz, low_end_focus_center_hz + 1.0)
                / low_end_focus_center_hz
            )
        ),
        1e-6,
    )
    low_end_focus_rise_shape = np.clip(
        np.log2(np.maximum(f_axis, low_end_focus_low_hz) / low_end_focus_low_hz)
        / low_end_focus_rise_denominator,
        0.0,
        1.0,
    )
    low_end_focus_fall_shape = np.clip(
        np.log2(
            low_end_focus_high_hz / np.maximum(f_axis, low_end_focus_center_hz)
        )
        / low_end_focus_fall_denominator,
        0.0,
        1.0,
    )
    low_end_focus_shape = np.clip(
        np.minimum(low_end_focus_rise_shape, low_end_focus_fall_shape)
        * (1.0 - presence_shape * 0.94)
        * (1.0 - high_shape * 0.96),
        0.0,
        1.0,
    )
    low_end_restraint_shape = np.clip(
        np.maximum(low_shape * 0.42, low_end_focus_shape + mud_shape * 0.02),
        0.0,
        1.0,
    )
    harsh_center_hz = float(
        min(
            self.high_cut,
            max(harsh_low_hz * 1.28, self.treble_transition_hz * 0.98, 3400.0),
        )
    )
    harsh_rise_denominator = max(
        float(np.log2(max(harsh_center_hz, harsh_low_hz + 1.0) / harsh_low_hz)),
        1e-6,
    )
    harsh_fall_denominator = max(
        float(
            np.log2(max(harsh_high_hz, harsh_center_hz + 1.0) / harsh_center_hz)
        ),
        1e-6,
    )
    harsh_rise_shape = np.clip(
        np.log2(np.maximum(f_axis, harsh_low_hz) / harsh_low_hz)
        / harsh_rise_denominator,
        0.0,
        1.0,
    )
    harsh_fall_shape = np.clip(
        np.log2(harsh_high_hz / np.maximum(f_axis, harsh_center_hz))
        / harsh_fall_denominator,
        0.0,
        1.0,
    )
    harshness_shape = np.clip(
        np.minimum(harsh_rise_shape, harsh_fall_shape)
        * (0.72 + presence_shape * 0.28)
        * (1.0 - high_shape * 0.38),
        0.0,
        1.0,
    )
    sibilance_shape = np.clip(
        np.log2(np.maximum(f_axis, sibilance_low_hz) / sibilance_low_hz)
        / max(
            float(
                np.log2(
                    max(sibilance_high_hz, sibilance_low_hz + 1.0)
                    / sibilance_low_hz
                )
            ),
            1e-6,
        ),
        0.0,
        1.0,
    )
    sibilance_shape = np.clip(
        sibilance_shape
        * (
            1.0
            - np.clip(
                (f_axis - sibilance_high_hz) / max(sibilance_high_hz, 1.0),
                0.0,
                1.0,
            )
        ),
        0.0,
        1.0,
    )
    closed_top_start_hz = float(
        min(
            self.high_cut,
            max(self.treble_transition_hz * 1.14, 4000.0),
        )
    )
    closed_top_span_denominator = max(
        float(
            np.log2(
                max(self.high_cut, closed_top_start_hz + 1.0)
                / closed_top_start_hz
            )
        ),
        1e-6,
    )
    closed_top_ramp = np.clip(
        np.log2(np.maximum(f_axis, closed_top_start_hz) / closed_top_start_hz)
        / closed_top_span_denominator,
        0.0,
        1.0,
    )
    closed_top_shape = np.where(
        f_axis >= closed_top_start_hz,
        0.3 + closed_top_ramp * 0.7,
        0.0,
    ).astype(np.float32, copy=False)
    treble_repair_factor = float(
        np.clip(
            max(
                closure_repair_factor,
                air_restoration_factor * 0.94,
                high_band_deficit_db / 8.5,
            ),
            0.0,
            1.0,
        )
    )
    air_shelf_shape = np.clip(high_shape + presence_shape * 0.42, 0.0, 1.0)
    correction_db += (
        low_shape
        * body_restoration_factor
        * (0.6 * (1.0 - low_end_restraint_factor * 0.9))
    )
    correction_db -= (
        low_end_restraint_shape
        * low_end_restraint_factor
        * (1.02 + mud_cleanup_factor * 0.5 + closure_repair_factor * 0.22)
    )
    correction_db -= (
        low_end_restraint_shape * legacy_tonal_rebalance_factor * 0.18
    )
    correction_db -= (
        mud_shape
        * mud_cleanup_factor
        * (1.55 + body_restoration_factor * 0.35 + closure_repair_factor * 0.3)
    )
    correction_db += presence_shape * closure_repair_factor * 0.95
    correction_db += high_shape * air_restoration_factor * 1.1
    correction_db += high_shape * closure_repair_factor * 0.55
    correction_db += air_shelf_shape * treble_repair_factor * 0.42
    correction_db += high_shape * legacy_tonal_rebalance_factor * 0.24
    correction_db += air_shelf_shape * legacy_tonal_rebalance_factor * 0.46
    correction_db += (
        closed_top_shape
        * closed_top_end_repair_factor
        * (0.88 + legacy_tonal_rebalance_factor * 0.16)
    )
    correction_db += air_shelf_shape * closed_top_end_repair_factor * 0.24
    correction_db -= (
        harshness_shape
        * harshness_restraint_factor
        * (0.9 + treble_repair_factor * 0.62 + closure_repair_factor * 0.22)
    )
    correction_db -= (
        sibilance_shape
        * harshness_restraint_factor
        * (0.18 + air_restoration_factor * 0.16)
    )

    correction_db[0], correction_db[-1] = 0.0, 0.0

    correction_db *= self.spectral_balance_profile.correction_strength
    correction_db = np.clip(
        correction_db,
        -self.spectral_balance_profile.max_cut_db,
        self.spectral_balance_profile.max_boost_db,
    )

    flat_anchors = np.column_stack((f_axis, correction_db))

    def eq_channel(channel: np.ndarray) -> np.ndarray:
        channel = audio_eq_fn(
            audio_data=channel,
            anchors=flat_anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )
        return audio_eq_fn(
            audio_data=channel,
            anchors=self.anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )

    if y.ndim > 1:
        return np.vstack([eq_channel(channel) for channel in y])

    return eq_channel(y)
