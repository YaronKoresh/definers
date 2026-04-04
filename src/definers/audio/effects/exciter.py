from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy.signal import butter, sosfiltfilt

from ...file_ops import catch, log
from ..dsp import remove_spectral_spikes, resample
from ..utils import get_rms
from .mixing import pad_audio


@dataclass(frozen=True, slots=True)
class ExciterConfig:
    min_cutoff_hz: float | None = None
    max_cutoff_hz: float | None = None
    min_frequency_hz: float | None = None
    default_cutoff_hz: float = 4000.0
    default_high_frequency_cutoff_hz: float = 6000.0
    min_adaptive_cutoff_hz: float = 1500.0
    min_sample_size: int = 128
    fft_max_size: int = 4096
    oversample_low_sr_threshold: int = 32000
    oversample_low_sr_factor: int = 4
    oversample_default_factor: int = 2
    highpass_order: int = 6
    smoothing_kernel_size: int = 5
    peak_limit_threshold: float | None = None
    high_frequency_cutoff: float | None = None
    min_drive: float = 0.75
    max_drive: float = 6.0
    max_spectral_gain: float = 2.5
    max_extrapolated_boost_db: float = 9.0
    band_rms_floor: float = 1e-9
    power_epsilon: float = 1e-24
    amplitude_floor: float = 1e-12


_DEFAULT_CONFIG = ExciterConfig()


def _sanitize_audio_array(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim == 0:
        array = array.reshape(1)
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def _move_samples_to_last(
    values: np.ndarray,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    if values.ndim <= 1:
        return values, lambda restored: restored

    sample_axis = int(np.argmax(values.shape))
    if sample_axis == values.ndim - 1:
        return values, lambda restored: restored

    return np.moveaxis(values, sample_axis, -1), lambda restored: np.moveaxis(
        restored, -1, sample_axis
    )


def _collapse_to_mono(values: np.ndarray) -> np.ndarray:
    if values.ndim == 1:
        return values
    return np.mean(
        values.reshape(-1, values.shape[-1]), axis=0, dtype=np.float32
    )


def _moving_average_last_axis(
    values: np.ndarray, window_size: int
) -> np.ndarray:
    if values.shape[-1] <= 2:
        return np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        )

    working = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )

    bounded_window = min(max(int(window_size), 1), working.shape[-1])
    if bounded_window % 2 == 0:
        bounded_window -= 1
    if bounded_window <= 1:
        return working

    pad_left = bounded_window // 2
    pad_right = bounded_window - 1 - pad_left
    pad_width = [(0, 0)] * working.ndim
    pad_width[-1] = (pad_left, pad_right)
    padded = np.pad(working, pad_width, mode="edge")
    cumulative = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative], axis=-1
    )
    averaged = (
        cumulative[..., bounded_window:] - cumulative[..., :-bounded_window]
    ) / bounded_window
    return np.asarray(averaged, dtype=working.dtype)


def _spectral_summary(
    mono_signal: np.ndarray,
    sample_rate: int,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if mono_signal.size < config.min_sample_size or sample_rate <= 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.0,
            0.0,
        )

    centered = mono_signal - np.mean(mono_signal, dtype=np.float32)
    if np.max(np.abs(centered)) <= config.amplitude_floor:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.0,
            0.0,
        )

    fft_size = min(
        config.fft_max_size, 1 << max(8, centered.size.bit_length() - 1)
    )
    segment = centered[-fft_size:]
    window = np.hanning(segment.size)
    spectrum = np.fft.rfft(segment * window)
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    power = np.square(np.abs(spectrum))

    nyquist = sample_rate / 2.0 - 1.0

    if config.min_frequency_hz is not None:
        valid = (
            (freqs >= config.min_frequency_hz)
            & (freqs <= nyquist)
            & np.isfinite(power)
        )
    else:
        valid = (freqs <= nyquist) & np.isfinite(power)

    if np.count_nonzero(valid) < 16:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.0,
            0.0,
        )

    freqs = freqs[valid]
    power = power[valid]
    smoothed_power = _moving_average_last_axis(
        power[np.newaxis, :],
        max(5, min(255, power.size // 48)),
    )[0]

    mean_power = float(np.mean(smoothed_power, dtype=np.float32))
    spectral_flatness = 0.0
    if mean_power > config.power_epsilon:
        spectral_flatness = float(
            np.exp(
                np.mean(
                    np.log(np.maximum(smoothed_power, config.power_epsilon)),
                    dtype=np.float32,
                )
            )
            / mean_power
        )

    total_power = float(np.sum(smoothed_power, dtype=np.float32))
    high_frequency_ratio = 0.0

    if total_power > config.power_epsilon:
        high_frequency_cutoff = _resolve_high_frequency_cutoff(
            sample_rate,
            freqs,
            config,
        )
        high_frequency_mask = freqs >= high_frequency_cutoff
        if np.any(high_frequency_mask):
            high_frequency_power = float(
                np.sum(smoothed_power[high_frequency_mask], dtype=np.float32)
            )
            high_frequency_ratio = float(high_frequency_power / total_power)

    return freqs, smoothed_power, spectral_flatness, high_frequency_ratio


def _resolve_high_frequency_cutoff(
    sample_rate: int,
    freqs: np.ndarray,
    config: ExciterConfig,
) -> float:
    if config.high_frequency_cutoff is not None:
        return float(config.high_frequency_cutoff)

    nyquist = max(
        sample_rate / 2.0 - 1.0, config.default_high_frequency_cutoff_hz
    )
    derived_cutoff = max(
        config.default_high_frequency_cutoff_hz,
        sample_rate * 0.18,
    )
    max_frequency = float(freqs[-1]) if freqs.size else nyquist
    return float(min(derived_cutoff, nyquist, max_frequency))


def _apply_adaptive_gate(
    values: np.ndarray,
    sample_rate: int,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    if values.size == 0:
        return values

    window_size = max(5, int(sample_rate * 0.004))
    envelope = np.sqrt(
        _moving_average_last_axis(np.square(values), window_size) + 1e-18
    )
    envelope_db = 20.0 * np.log10(np.maximum(envelope, config.amplitude_floor))
    gate_center = float(np.quantile(envelope_db, 0.35))
    gate_shape = 1.0 / (1.0 + np.exp(-(envelope_db - gate_center) / 6.0))
    return values * gate_shape


def _calculate_spectral_features(
    freqs: np.ndarray,
    power: np.ndarray,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> tuple[float, float, float]:
    if freqs.size < 16:
        return 0.0, 0.0, 0.0

    cumulative_power = np.cumsum(power)
    total_power = float(cumulative_power[-1])
    if total_power <= config.power_epsilon:
        return 0.0, 0.0, 0.0

    rolloff_index = int(
        np.searchsorted(cumulative_power, total_power * 0.95, side="left")
    )
    rolloff_index = min(rolloff_index, freqs.size - 1)

    spectral_rolloff = float(freqs[rolloff_index])
    spectral_centroid = float(
        np.sum(freqs * power, dtype=np.float32) / total_power
    )
    spectral_bandwidth = float(
        np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * power, dtype=np.float32)
            / total_power
        )
    )

    return spectral_rolloff, spectral_centroid, spectral_bandwidth


def _clip_cutoff(cutoff: float, config: ExciterConfig) -> float:
    min_limit = (
        config.min_cutoff_hz if config.min_cutoff_hz is not None else cutoff
    )
    max_limit = (
        config.max_cutoff_hz if config.max_cutoff_hz is not None else cutoff
    )
    return float(np.clip(cutoff, min_limit, max_limit))


def _clip_adaptive_cutoff(
    cutoff: float,
    sample_rate: int,
    config: ExciterConfig,
) -> float:
    nyquist = max(sample_rate / 2.0 - 1.0, 1.0)
    min_limit = (
        config.min_cutoff_hz
        if config.min_cutoff_hz is not None
        else min(config.min_adaptive_cutoff_hz, nyquist)
    )
    max_limit = (
        config.max_cutoff_hz if config.max_cutoff_hz is not None else nyquist
    )
    safe_min = min(min_limit, nyquist)
    safe_max = max(safe_min, min(max_limit, nyquist))
    return float(np.clip(cutoff, safe_min, safe_max))


def calculate_dynamic_cutoff(
    signal: np.ndarray,
    sample_rate: int,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> float:
    if sample_rate <= 0:
        return config.default_cutoff_hz

    sanitized = _sanitize_audio_array(signal)
    samples_last, _ = _move_samples_to_last(sanitized)
    mono_signal = _collapse_to_mono(samples_last)

    freqs, power, spectral_flatness, high_frequency_ratio = _spectral_summary(
        mono_signal,
        sample_rate,
        config,
    )

    if freqs.size < 16:
        return _clip_cutoff(config.default_cutoff_hz, config)

    spectral_rolloff, spectral_centroid, _ = _calculate_spectral_features(
        freqs, power, config
    )

    cutoff = (
        spectral_rolloff * 0.55
        + spectral_centroid * 0.3
        + spectral_rolloff * 0.15
    )
    cutoff *= 0.9 + 0.25 * float(np.sqrt(max(high_frequency_ratio, 0.0)))
    cutoff *= 1.0 - 0.08 * float(np.clip(spectral_flatness, 0.0, 1.0))

    return _clip_adaptive_cutoff(cutoff, sample_rate, config)


@dataclass(frozen=True, slots=True)
class ExciterAnalysis:
    cutoff_hz: float
    drive: float
    oversample_factor: int
    adaptive_mix: float
    band_rms: float
    spectral_flatness: float
    high_frequency_ratio: float
    transient_density: float = 0.0
    transient_ducking_depth: float = 0.0


def _measure_transient_density(
    values: np.ndarray,
    sample_rate: int,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> float:
    if values.size < 2 or sample_rate <= 0:
        return 0.0

    delta = np.abs(np.diff(values, prepend=values[..., :1], axis=-1))
    window_size = max(int(round(sample_rate * 0.008)), 1)
    smoothed = _moving_average_last_axis(delta, window_size)
    threshold = float(
        np.mean(smoothed, dtype=np.float32) + np.std(smoothed, dtype=np.float32)
    )
    if not np.isfinite(threshold) or threshold <= 0.0:
        return 0.0
    return float(np.mean(smoothed > threshold, dtype=np.float32))


def _build_transient_ducking_curve(
    signal: np.ndarray,
    sample_rate: int,
    depth: float,
) -> np.ndarray:
    sanitized = _sanitize_audio_array(signal)
    if sanitized.size == 0 or sample_rate <= 0 or depth <= 0.0:
        return np.ones_like(sanitized, dtype=np.float32)

    samples_last, _ = _move_samples_to_last(sanitized)
    linked = np.abs(_collapse_to_mono(samples_last))
    fast_window = max(int(round(sample_rate * 0.0025)), 1)
    slow_window = max(int(round(sample_rate * 0.02)), fast_window + 1)
    fast_env = _moving_average_last_axis(linked, fast_window)
    slow_env = _moving_average_last_axis(linked, slow_window)
    transient_mask = np.clip(
        (fast_env - slow_env) / np.maximum(fast_env, 1e-6),
        0.0,
        1.0,
    )
    transient_mask = _moving_average_last_axis(
        transient_mask,
        max(int(round(sample_rate * 0.004)), 1),
    )
    ducking_curve = 1.0 - transient_mask * float(np.clip(depth, 0.0, 0.9))
    if samples_last.ndim > 1:
        ducking_curve = ducking_curve[np.newaxis, :]
    return np.asarray(np.clip(ducking_curve, 0.1, 1.0), dtype=np.float32)


def analyze_exciter(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> ExciterAnalysis:
    sanitized = _sanitize_audio_array(signal)
    samples_last, _ = _move_samples_to_last(sanitized)

    if sample_rate <= 0 or samples_last.size == 0:
        return ExciterAnalysis(
            cutoff_hz=config.default_cutoff_hz,
            drive=0.8,
            oversample_factor=1,
            adaptive_mix=0.0,
            band_rms=0.0,
            spectral_flatness=0.0,
            high_frequency_ratio=0.0,
        )

    oversample_factor = (
        config.oversample_low_sr_factor
        if sample_rate <= config.oversample_low_sr_threshold
        else config.oversample_default_factor
    )
    oversampled_rate = sample_rate * oversample_factor
    oversampled_signal = (
        samples_last
        if oversample_factor == 1
        else resample(
            samples_last, sample_rate, target_sr=oversampled_rate
        ).astype(
            np.float32,
            copy=False,
        )
    )

    if cutoff_hz is None:
        resolved_cutoff = calculate_dynamic_cutoff(
            oversampled_signal,
            oversampled_rate,
            config,
        )
    else:
        min_limit = (
            config.min_cutoff_hz
            if config.min_cutoff_hz is not None
            else cutoff_hz
        )
        max_limit = (
            config.max_cutoff_hz
            if config.max_cutoff_hz is not None
            else sample_rate
        )
        resolved_cutoff = float(np.clip(cutoff_hz, min_limit, max_limit))

    highpass = butter(
        config.highpass_order,
        resolved_cutoff,
        "high",
        fs=oversampled_rate,
        output="sos",
    )
    high_band = sosfiltfilt(highpass, oversampled_signal, axis=-1)
    gated_band = _apply_adaptive_gate(high_band, oversampled_rate, config)
    band_rms = get_rms(gated_band)

    peak_level = float(np.max(np.abs(gated_band))) if gated_band.size else 0.0
    crest_factor = peak_level / max(band_rms, config.band_rms_floor)
    drive = float(
        (0.28 / max(band_rms, config.band_rms_floor))
        * float(crest_factor / 3.0)
    )
    drive_restraint = float(
        np.clip(1.02 - max(crest_factor - 3.0, 0.0) * 0.04, 0.72, 1.0)
    )

    mono_signal = _collapse_to_mono(oversampled_signal)
    _, _, spectral_flatness, high_frequency_ratio = _spectral_summary(
        mono_signal,
        oversampled_rate,
        config,
    )
    transient_density = _measure_transient_density(
        _collapse_to_mono(gated_band),
        oversampled_rate,
        config,
    )
    brightness_restraint = float(
        np.clip(1.0 - max(high_frequency_ratio - 0.12, 0.0) * 1.6, 0.6, 1.0)
    )
    transient_density_restraint = float(
        np.clip(1.02 - max(transient_density - 0.08, 0.0) * 1.8, 0.58, 1.0)
    )
    drive = float(
        np.clip(
            drive
            * drive_restraint
            * brightness_restraint
            * transient_density_restraint,
            config.min_drive,
            config.max_drive,
        )
    )
    transient_guard = float(
        np.clip(1.02 - max(crest_factor - 2.6, 0.0) * 0.08, 0.38, 1.0)
    )
    brightness_guard = float(
        np.clip(
            1.15 - 1.8 * high_frequency_ratio - 0.12 * spectral_flatness,
            0.22,
            1.0,
        )
    )
    density_guard = float(
        np.clip(1.04 - max(transient_density - 0.1, 0.0) * 1.15, 0.55, 1.0)
    )
    adaptive_mix = float(
        np.clip(mix, 0.0, 1.0)
        * brightness_guard
        * transient_guard
        * density_guard
    )
    transient_ducking_depth = float(
        np.clip(
            np.clip((crest_factor - 2.5) / 6.0, 0.0, 1.0) * 0.28
            + high_frequency_ratio * 0.4
            + transient_density * 1.2
            + spectral_flatness * 0.08,
            0.0,
            0.72,
        )
    )

    return ExciterAnalysis(
        cutoff_hz=resolved_cutoff,
        drive=drive,
        oversample_factor=oversample_factor,
        adaptive_mix=adaptive_mix,
        band_rms=band_rms,
        spectral_flatness=spectral_flatness,
        high_frequency_ratio=high_frequency_ratio,
        transient_density=transient_density,
        transient_ducking_depth=transient_ducking_depth,
    )


def _apply_exciter_core(
    samples_last: np.ndarray,
    sample_rate: int,
    analysis: ExciterAnalysis,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    if (
        sample_rate <= 0
        or samples_last.size == 0
        or analysis.band_rms <= config.band_rms_floor
    ):
        return samples_last

    oversampled_rate = sample_rate * analysis.oversample_factor
    oversampled_signal = (
        samples_last
        if analysis.oversample_factor == 1
        else resample(
            samples_last, sample_rate, target_sr=oversampled_rate
        ).astype(
            np.float32,
            copy=False,
        )
    )

    highpass = butter(
        config.highpass_order,
        analysis.cutoff_hz,
        "high",
        fs=oversampled_rate,
        output="sos",
    )
    high_band = sosfiltfilt(highpass, oversampled_signal, axis=-1)
    high_band = _apply_adaptive_gate(high_band, oversampled_rate, config)

    drive_normalizer = float(np.tanh(analysis.drive))
    if abs(drive_normalizer) <= config.amplitude_floor:
        drive_normalizer = 1.0

    symmetric_harmonics = np.tanh(high_band * analysis.drive) / drive_normalizer
    asymmetric_input = high_band + 0.225 * np.square(high_band)
    asymmetric_harmonics = (
        np.tanh(asymmetric_input * analysis.drive) / drive_normalizer
    )
    harmonic_residual = (
        symmetric_harmonics * 0.58 + asymmetric_harmonics * 0.42
    ) - high_band
    harmonic_residual = sosfiltfilt(highpass, harmonic_residual, axis=-1)

    sample_count = harmonic_residual.shape[-1]
    if sample_count < 16:
        return samples_last

    signal_spectrum = np.fft.rfft(oversampled_signal, axis=-1)
    residual_spectrum = np.fft.rfft(harmonic_residual, axis=-1)

    smoothing_bins = max(7, min(255, signal_spectrum.shape[-1] // 48))
    signal_power = _moving_average_last_axis(
        np.square(np.abs(signal_spectrum)),
        smoothing_bins,
    )
    residual_power = _moving_average_last_axis(
        np.square(np.abs(residual_spectrum)),
        smoothing_bins,
    )

    mean_signal_power = np.mean(
        signal_power.reshape(-1, signal_power.shape[-1]),
        axis=0,
        dtype=np.float32,
    )
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / oversampled_rate)

    reference_low = max(250.0, analysis.cutoff_hz * 0.55)
    reference_high = min(freqs[-1], analysis.cutoff_hz * 0.98)
    reference_mask = (freqs >= reference_low) & (freqs <= reference_high)
    extension_mask = freqs >= analysis.cutoff_hz

    target_power = np.nan_to_num(
        mean_signal_power.copy(),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if np.count_nonzero(reference_mask) >= 8:
        reference_freqs = np.maximum(freqs[reference_mask], 1.0)
        reference_db = 10.0 * np.log10(
            np.maximum(mean_signal_power[reference_mask], config.power_epsilon)
        )
        slope, intercept = np.polyfit(np.log(reference_freqs), reference_db, 1)
        predicted_db = intercept + slope * np.log(
            np.maximum(freqs, reference_freqs[0])
        )
        predicted_db = np.nan_to_num(
            predicted_db,
            nan=float(np.max(reference_db)),
            posinf=float(np.max(reference_db)),
            neginf=float(np.min(reference_db)),
        )
        predicted_db = np.clip(
            predicted_db,
            float(np.min(reference_db)) - 48.0,
            float(np.max(reference_db)) + config.max_extrapolated_boost_db,
        )
        air_decay = np.exp(
            -np.maximum(freqs - analysis.cutoff_hz, 0.0)
            / max(analysis.cutoff_hz * 1.8, 2500.0)
        )
        fitted_power = np.power(
            10.0, predicted_db.astype(np.float64) / 10.0
        ) * (0.6 + 0.4 * air_decay)
        fitted_power = np.nan_to_num(
            fitted_power,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)
        target_power[extension_mask] = np.maximum(
            target_power[extension_mask],
            fitted_power[extension_mask],
        )
        target_power = np.nan_to_num(
            target_power,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
            copy=False,
        )

    transition_width = max(analysis.cutoff_hz * 0.08, 400.0)
    transition = np.clip(
        (freqs - analysis.cutoff_hz) / transition_width, 0.0, 1.0
    )
    transition = 0.5 - 0.5 * np.cos(np.pi * transition)

    desired_wet_power = (
        np.maximum(target_power - mean_signal_power, 0.0) * transition
    )
    dampening = np.exp(
        -np.maximum(freqs - analysis.cutoff_hz, 0.0)
        / max(analysis.cutoff_hz * 2.5, 5000.0)
    )
    desired_wet_power *= 0.75 + 0.25 * dampening
    desired_wet_power = np.nan_to_num(
        desired_wet_power,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
        copy=False,
    )

    desired_shape = desired_wet_power.reshape(
        (1,) * (residual_power.ndim - 1) + (-1,)
    )
    spectral_gain = np.sqrt(
        desired_shape / np.maximum(residual_power, config.power_epsilon)
    )
    spectral_gain = _moving_average_last_axis(spectral_gain, smoothing_bins)
    spectral_gain = np.clip(spectral_gain, 0.0, config.max_spectral_gain)
    spectral_gain = remove_spectral_spikes(spectral_gain)

    residual_spectrum *= spectral_gain
    wet_oversampled = np.fft.irfft(residual_spectrum, n=sample_count, axis=-1)

    if get_rms(wet_oversampled) <= config.amplitude_floor:
        return samples_last

    wet_signal = (
        wet_oversampled
        if analysis.oversample_factor == 1
        else resample(wet_oversampled, oversampled_rate, target_sr=sample_rate)
    )

    dry_aligned, wet_aligned = pad_audio(samples_last, wet_signal)
    wet_aligned = wet_aligned * _build_transient_ducking_curve(
        dry_aligned,
        sample_rate,
        analysis.transient_ducking_depth,
    )
    output = dry_aligned + wet_aligned * analysis.adaptive_mix

    if config.peak_limit_threshold is not None:
        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if np.isfinite(peak) and peak > config.peak_limit_threshold:
            output = output / peak * config.peak_limit_threshold

    return _sanitize_audio_array(output)


def apply_exciter(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
    max_drive: float | None = None,
    high_frequency_cutoff_hz: float | None = None,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> np.ndarray:
    sanitized = _sanitize_audio_array(signal)
    if sanitized.size == 0 or sample_rate <= 0:
        return sanitized

    if max_drive is not None or high_frequency_cutoff_hz is not None:
        config = replace(
            config,
            max_drive=(
                config.max_drive if max_drive is None else float(max_drive)
            ),
            high_frequency_cutoff=(
                config.high_frequency_cutoff
                if high_frequency_cutoff_hz is None
                else float(high_frequency_cutoff_hz)
            ),
        )

    samples_last, restore_layout = _move_samples_to_last(sanitized)

    try:
        analysis = analyze_exciter(
            samples_last,
            sample_rate,
            cutoff_hz=cutoff_hz,
            mix=mix,
            config=config,
        )
        log(
            "Exciter",
            f"Adaptive cutoff={analysis.cutoff_hz:.2f}Hz drive={analysis.drive:.2f}dB mix={analysis.adaptive_mix:.2f} oversample=x{analysis.oversample_factor}",
        )
        processed = _apply_exciter_core(
            samples_last, sample_rate, analysis, config
        )
    except Exception as e:
        catch(
            e,
            "Error during exciter processing",
        )
        return sanitized

    return restore_layout(_sanitize_audio_array(processed))


def apply_exciter_with_analysis(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
    max_drive: float | None = None,
    high_frequency_cutoff_hz: float | None = None,
    config: ExciterConfig = _DEFAULT_CONFIG,
) -> tuple[np.ndarray, ExciterAnalysis]:
    sanitized = _sanitize_audio_array(signal)
    if max_drive is not None or high_frequency_cutoff_hz is not None:
        config = replace(
            config,
            max_drive=(
                config.max_drive if max_drive is None else float(max_drive)
            ),
            high_frequency_cutoff=(
                config.high_frequency_cutoff
                if high_frequency_cutoff_hz is None
                else float(high_frequency_cutoff_hz)
            ),
        )
    samples_last, restore_layout = _move_samples_to_last(sanitized)
    analysis = analyze_exciter(
        samples_last, sample_rate, cutoff_hz=cutoff_hz, mix=mix, config=config
    )
    processed = _apply_exciter_core(samples_last, sample_rate, analysis, config)
    return restore_layout(_sanitize_audio_array(processed)), analysis
