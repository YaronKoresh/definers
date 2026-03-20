from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from ...file_ops import catch, log
from ..dsp import remove_spectral_spikes, resample
from ..utils import get_rms, loudness_maximizer
from .mixing import pad_audio


def _baseline_calculate_dynamic_cutoff(
    y: np.ndarray, sr: int, rolloff_percent: float = 0.95
) -> float:
    if sr <= 0:
        return 4000.0

    y_array = np.asarray(y, dtype=np.float64)

    if y_array.ndim == 0:
        return 4000.0

    if y_array.ndim == 1:
        y_mono = y_array
    elif y_array.shape[0] <= 8:
        y_mono = np.mean(y_array, axis=0, dtype=np.float64)
    elif y_array.shape[-1] <= 8:
        y_mono = np.mean(y_array, axis=-1, dtype=np.float64)
    else:
        y_mono = np.mean(y_array, axis=-2, dtype=np.float64)

    y_mono = np.ravel(np.nan_to_num(y_mono, nan=0.0, posinf=0.0, neginf=0.0))

    nyquist_limited_cutoff = 200.0
    default_cutoff = float(min(4000.0, nyquist_limited_cutoff))

    if y_mono.size < 32:
        return default_cutoff

    y_mono = y_mono - np.mean(y_mono)

    if np.max(np.abs(y_mono)) <= 1e-12:
        return default_cutoff

    max_segment = min(y_mono.size, 262144)
    nperseg = max(256, 1 << (max_segment.bit_length() - 1))
    nperseg = min(nperseg, max_segment)

    freqs, psd = welch(
        y_mono,
        fs=sr,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="spectrum",
    )

    valid = np.isfinite(freqs) & np.isfinite(psd) & (freqs > 20.0)
    if not np.any(valid):
        return default_cutoff

    freqs = freqs[valid]
    psd = psd[valid]

    smoothing_kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64)
    psd = np.convolve(
        psd, smoothing_kernel / smoothing_kernel.sum(), mode="same"
    )

    noise_floor = float(np.median(psd))
    psd = np.clip(psd - (noise_floor * 0.25), 0.0, None)

    high_bound = float(min(16000.0, nyquist_limited_cutoff))
    low_bound = float(min(1200.0, high_bound))
    analysis_floor = min(250.0, high_bound * 0.25)

    analysis_mask = (freqs >= analysis_floor) & (freqs <= high_bound)
    if np.any(analysis_mask):
        freqs = freqs[analysis_mask]
        psd = psd[analysis_mask]

    total_energy = float(np.sum(psd))
    if total_energy <= 1e-20:
        return default_cutoff

    bounded_rolloff_percent = float(np.clip(rolloff_percent, 0.05, 0.95))
    cumulative_energy = np.cumsum(psd)
    rolloff_index = int(
        np.searchsorted(
            cumulative_energy,
            total_energy * bounded_rolloff_percent,
            side="left",
        )
    )
    rolloff_index = min(rolloff_index, freqs.size - 1)

    spectral_rolloff = float(freqs[rolloff_index])
    spectral_centroid = float(np.sum(freqs * psd) / total_energy)
    spectral_bandwidth = float(
        np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / total_energy)
    )

    cutoff_hz = (
        spectral_rolloff * 0.55
        + spectral_centroid * 0.3
        + (spectral_centroid + spectral_bandwidth * 0.5) * 0.15
    )

    return float(np.clip(cutoff_hz, low_bound, high_bound))


def _baseline_apply_exciter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
) -> np.ndarray:
    def smooth_last_axis(values: np.ndarray, kernel_size: int) -> np.ndarray:
        if values.shape[-1] <= 2:
            return values

        bounded_kernel_size = min(max(kernel_size, 3), values.shape[-1])
        if bounded_kernel_size % 2 == 0:
            bounded_kernel_size -= 1
        if bounded_kernel_size <= 1:
            return values

        kernel = (
            np.ones(bounded_kernel_size, dtype=np.float64) / bounded_kernel_size
        )
        return np.apply_along_axis(
            lambda channel: np.convolve(channel, kernel, mode="same"),
            axis=-1,
            arr=values,
        )

    y_array = np.asarray(y, dtype=np.float64)
    if y_array.size == 0 or sr <= 0:
        return y_array

    oversample = 2
    sr_up = sr * oversample
    y_up = resample(y_array, sr, target_sr=sr_up).astype(np.float64, copy=False)

    if cutoff_hz is None:
        cutoff_hz = _baseline_calculate_dynamic_cutoff(y_up, sr_up)

    cutoff_hz = float(np.clip(cutoff_hz, 1500.0, sr))
    mix = float(np.clip(mix, 0.0, 1.0))

    log(
        "Exciter",
        f"Calculated dynamic cutoff frequency: {cutoff_hz:.2f} Hz based on spectral rolloff",
    )

    sos = butter(4, cutoff_hz, "high", fs=sr_up, output="sos")
    high_band = sosfiltfilt(sos, y_up, axis=-1)

    band_rms = get_rms(high_band)
    if not np.isfinite(band_rms) or band_rms <= 1e-9:
        return y_array

    drive = float(3.0 / (band_rms + 1e-9))
    log(
        "Exciter",
        f"Calculated drive amount: {drive:.2f} based on high-band rms {band_rms:.6f}",
    )

    nonlinear = np.tanh(high_band * drive) / np.tanh(drive)
    harmonic_residual = sosfiltfilt(sos, nonlinear - high_band, axis=-1)

    sample_count = y_up.shape[-1]
    if sample_count < 16:
        return y_array

    signal_spectrum = np.fft.rfft(y_up, axis=-1)
    residual_spectrum = np.fft.rfft(harmonic_residual, axis=-1)

    signal_power = np.square(np.abs(signal_spectrum))
    residual_power = np.square(np.abs(residual_spectrum))

    smoothing_bins = max(9, min(257, signal_power.shape[-1] // 32))
    signal_power_smooth = smooth_last_axis(signal_power, smoothing_bins)
    residual_power_smooth = smooth_last_axis(residual_power, smoothing_bins)

    mean_signal_power = np.mean(
        signal_power_smooth.reshape(-1, signal_power_smooth.shape[-1]),
        axis=0,
        dtype=np.float64,
    )
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / sr_up)

    reference_low = max(200.0, cutoff_hz * 0.45)
    reference_high = min(cutoff_hz * 0.98, freqs[-1])
    reference_mask = (freqs >= reference_low) & (freqs <= reference_high)

    extension_start = min(cutoff_hz, freqs[-1])
    transition_stop = min(freqs[-1], cutoff_hz * 1.05)
    extension_mask = freqs >= extension_start
    transition = np.zeros_like(freqs, dtype=np.float64)

    if transition_stop > extension_start:
        ramp_mask = (freqs >= extension_start) & (freqs <= transition_stop)
        ramp = (freqs[ramp_mask] - extension_start) / (
            transition_stop - extension_start
        )
        transition[ramp_mask] = 0.5 - 0.5 * np.cos(
            np.pi * np.clip(ramp, 0.0, 1.0)
        )
    transition[freqs > transition_stop] = 1.0

    target_power = mean_signal_power.copy()
    if np.count_nonzero(reference_mask) >= 8:
        reference_freqs = freqs[reference_mask]
        reference_db = 10.0 * np.log10(
            np.maximum(mean_signal_power[reference_mask], 1e-24)
        )
        slope, intercept = np.polyfit(np.log(reference_freqs), reference_db, 1)

        fitted_db = intercept + slope * np.log(
            np.maximum(freqs, reference_freqs[0])
        )

        anchor_mask = (freqs >= reference_high * 0.9) & (
            freqs <= reference_high
        )
        if np.count_nonzero(anchor_mask) >= 4:
            anchor_actual_db = float(
                np.mean(
                    10.0
                    * np.log10(
                        np.maximum(mean_signal_power[anchor_mask], 1e-24)
                    ),
                    dtype=np.float64,
                )
            )
            anchor_fitted_db = float(
                np.mean(fitted_db[anchor_mask], dtype=np.float64)
            )
            fitted_db += anchor_actual_db - anchor_fitted_db

        fitted_power = np.power(10.0, fitted_db / 10.0)
        target_power[extension_mask] = fitted_power[extension_mask]

    target_power = smooth_last_axis(target_power, smoothing_bins)
    desired_wet_power = (
        np.maximum(target_power - mean_signal_power, 0.0) * transition
    )

    gain_shape = desired_wet_power.reshape(
        (1,) * (residual_power_smooth.ndim - 1) + (-1,)
    )
    spectral_gain = np.sqrt(
        gain_shape / np.maximum(residual_power_smooth, 1e-24)
    )
    spectral_gain = remove_spectral_spikes(spectral_gain)
    spectral_gain = smooth_last_axis(spectral_gain, smoothing_bins)

    extension_shape = extension_mask.reshape(
        (1,) * (spectral_gain.ndim - 1) + (-1,)
    )
    spectral_gain *= extension_shape

    wet_spectrum = residual_spectrum * spectral_gain
    wet_up = np.fft.irfft(
        wet_spectrum,
        n=sample_count,
        axis=-1,
    )

    wet_rms = get_rms(wet_up)
    if not np.isfinite(wet_rms) or wet_rms <= 1e-12:
        return y_array

    highs = resample(wet_up, sr_up, target_sr=sr)
    dry_final, highs_final = pad_audio(y_array, highs)

    final = dry_final + highs_final * mix

    final = remove_spectral_spikes(final)

    return loudness_maximizer(final)


@dataclass(frozen=True, slots=True)
class ExciterAnalysis:
    cutoff_hz: float
    drive: float
    oversample_factor: int
    adaptive_mix: float
    band_rms: float
    spectral_flatness: float
    high_frequency_ratio: float


def _sanitize_audio_array(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float64)
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
        values.reshape(-1, values.shape[-1]), axis=0, dtype=np.float64
    )


def _moving_average_last_axis(
    values: np.ndarray, window_size: int
) -> np.ndarray:
    if values.shape[-1] <= 2:
        return values

    bounded_window = min(max(int(window_size), 1), values.shape[-1])
    if bounded_window % 2 == 0:
        bounded_window -= 1
    if bounded_window <= 1:
        return values

    pad_left = bounded_window // 2
    pad_right = bounded_window - 1 - pad_left
    pad_width = [(0, 0)] * values.ndim
    pad_width[-1] = (pad_left, pad_right)
    padded = np.pad(values, pad_width, mode="edge")
    cumulative = np.cumsum(padded, axis=-1, dtype=np.float64)
    cumulative = np.concatenate(
        [np.zeros_like(cumulative[..., :1]), cumulative], axis=-1
    )
    return (
        cumulative[..., bounded_window:] - cumulative[..., :-bounded_window]
    ) / bounded_window


def _spectral_summary(
    mono_signal: np.ndarray,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    if mono_signal.size < 128 or sample_rate <= 0:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
            0.0,
        )

    centered = mono_signal - np.mean(mono_signal, dtype=np.float64)
    if np.max(np.abs(centered)) <= 1e-12:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
            0.0,
        )

    fft_size = min(262144, 1 << max(8, centered.size.bit_length() - 1))
    segment = centered[-fft_size:]
    window = np.hanning(segment.size)
    spectrum = np.fft.rfft(segment * window)
    freqs = np.fft.rfftfreq(segment.size, d=1.0 / sample_rate)
    power = np.square(np.abs(spectrum))

    valid = (freqs >= 20.0) & (freqs <= sample_rate * 0.48) & np.isfinite(power)
    if np.count_nonzero(valid) < 16:
        return (
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            0.0,
            0.0,
        )

    freqs = freqs[valid]
    power = power[valid]
    smoothed_power = _moving_average_last_axis(
        power[np.newaxis, :],
        max(5, min(255, power.size // 48)),
    )[0]

    epsilon = 1e-24
    mean_power = float(np.mean(smoothed_power, dtype=np.float64))
    spectral_flatness = 0.0
    if mean_power > epsilon:
        spectral_flatness = float(
            np.exp(
                np.mean(
                    np.log(np.maximum(smoothed_power, epsilon)),
                    dtype=np.float64,
                )
            )
            / mean_power
        )

    total_power = float(np.sum(smoothed_power, dtype=np.float64))
    high_frequency_ratio = 0.0
    if total_power > epsilon:
        high_frequency_ratio = float(
            np.sum(smoothed_power[freqs >= 6000.0], dtype=np.float64)
            / total_power
        )

    return freqs, smoothed_power, spectral_flatness, high_frequency_ratio


def _apply_adaptive_gate(values: np.ndarray, sample_rate: int) -> np.ndarray:
    if values.size == 0:
        return values

    window_size = max(5, int(sample_rate * 0.004))
    envelope = np.sqrt(
        _moving_average_last_axis(np.square(values), window_size) + 1e-18
    )
    envelope_db = 20.0 * np.log10(np.maximum(envelope, 1e-12))
    gate_center = float(np.quantile(envelope_db, 0.35))
    gate_shape = 1.0 / (1.0 + np.exp(-(envelope_db - gate_center) / 6.0))
    return values * gate_shape


def calculate_dynamic_cutoff(
    signal: np.ndarray,
    sample_rate: int,
    rolloff_percent: float = 0.95,
) -> float:
    if sample_rate <= 0:
        return 4000.0

    sanitized = _sanitize_audio_array(signal)
    baseline_cutoff = float(
        _baseline_calculate_dynamic_cutoff(
            sanitized, sample_rate, rolloff_percent
        )
    )

    samples_last, _ = _move_samples_to_last(sanitized)
    mono_signal = _collapse_to_mono(samples_last)
    freqs, power, spectral_flatness, high_frequency_ratio = _spectral_summary(
        mono_signal,
        sample_rate,
    )

    upper_cutoff = float(sample_rate)
    lower_cutoff = float(min(max(1200.0, sample_rate * 0.02), upper_cutoff))
    if freqs.size < 16:
        log(
            "Exciter",
            "Insufficient spectral data for dynamic cutoff calculation, using baseline cutoff",
        )
        return float(np.clip(baseline_cutoff, lower_cutoff, upper_cutoff))

    bounded_rolloff = float(np.clip(rolloff_percent, 0.05, 0.99))
    cumulative_power = np.cumsum(power)
    total_power = float(cumulative_power[-1])
    if total_power <= 1e-24:
        log(
            "Exciter",
            "Spectral power too low for dynamic cutoff calculation, using baseline cutoff",
        )
        return float(np.clip(baseline_cutoff, lower_cutoff, upper_cutoff))

    rolloff_index = int(
        np.searchsorted(
            cumulative_power, total_power * bounded_rolloff, side="left"
        )
    )
    rolloff_index = min(rolloff_index, freqs.size - 1)

    spectral_rolloff = float(freqs[rolloff_index])
    spectral_centroid = float(
        np.sum(freqs * power, dtype=np.float64) / total_power
    )
    enhanced_cutoff = (
        baseline_cutoff * 0.5 + spectral_rolloff * 0.3 + spectral_centroid * 0.2
    )
    enhanced_cutoff *= 0.9 + 0.25 * float(
        np.sqrt(max(high_frequency_ratio, 0.0))
    )
    enhanced_cutoff *= 1.0 - 0.08 * float(np.clip(spectral_flatness, 0.0, 1.0))

    return float(np.clip(enhanced_cutoff, lower_cutoff, upper_cutoff))


def analyze_exciter(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
) -> ExciterAnalysis:
    sanitized = _sanitize_audio_array(signal)
    samples_last, _ = _move_samples_to_last(sanitized)

    if sample_rate <= 0 or samples_last.size == 0:
        return ExciterAnalysis(
            cutoff_hz=4000.0,
            drive=0.8,
            oversample_factor=1,
            adaptive_mix=0.0,
            band_rms=0.0,
            spectral_flatness=0.0,
            high_frequency_ratio=0.0,
        )

    oversample_factor = 4 if sample_rate <= 32000 else 2
    oversampled_rate = sample_rate * oversample_factor
    oversampled_signal = (
        samples_last
        if oversample_factor == 1
        else resample(
            samples_last, sample_rate, target_sr=oversampled_rate
        ).astype(
            np.float64,
            copy=False,
        )
    )

    resolved_cutoff = (
        calculate_dynamic_cutoff(oversampled_signal, sample_rate)
        if cutoff_hz is None
        else float(np.clip(cutoff_hz, 1500.0, sample_rate))
    )

    highpass = butter(
        6, resolved_cutoff, "high", fs=oversampled_rate, output="sos"
    )
    high_band = sosfiltfilt(highpass, oversampled_signal, axis=-1)
    gated_band = _apply_adaptive_gate(high_band, oversampled_rate)
    band_rms = get_rms(gated_band)

    peak_level = float(np.max(np.abs(gated_band))) if gated_band.size else 0.0
    crest_factor = peak_level / max(band_rms, 1e-9)
    drive = float(
        (0.28 / max(band_rms, 1e-9))
        * float(np.clip(crest_factor / 3.0, 0.75, 1.35))
    )

    mono_signal = _collapse_to_mono(oversampled_signal)
    _, _, spectral_flatness, high_frequency_ratio = _spectral_summary(
        mono_signal,
        oversampled_rate,
    )
    adaptive_mix = float(
        np.clip(mix, 0.0, 1.0)
        * float(np.clip(1.2 - 1.4 * high_frequency_ratio, 0.35, 1.0))
    )

    return ExciterAnalysis(
        cutoff_hz=resolved_cutoff,
        drive=drive,
        oversample_factor=oversample_factor,
        adaptive_mix=adaptive_mix,
        band_rms=band_rms,
        spectral_flatness=spectral_flatness,
        high_frequency_ratio=high_frequency_ratio,
    )


def _apply_exciter_core(
    samples_last: np.ndarray,
    sample_rate: int,
    analysis: ExciterAnalysis,
) -> np.ndarray:
    if sample_rate <= 0 or samples_last.size == 0 or analysis.band_rms <= 1e-9:
        return samples_last

    oversampled_rate = sample_rate * analysis.oversample_factor
    oversampled_signal = (
        samples_last
        if analysis.oversample_factor == 1
        else resample(
            samples_last, sample_rate, target_sr=oversampled_rate
        ).astype(
            np.float64,
            copy=False,
        )
    )

    highpass = butter(
        6, analysis.cutoff_hz, "high", fs=oversampled_rate, output="sos"
    )
    high_band = sosfiltfilt(highpass, oversampled_signal, axis=-1)
    high_band = _apply_adaptive_gate(high_band, oversampled_rate)

    drive_normalizer = float(np.tanh(analysis.drive))
    if abs(drive_normalizer) <= 1e-12:
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
        dtype=np.float64,
    )
    freqs = np.fft.rfftfreq(sample_count, d=1.0 / oversampled_rate)

    reference_low = max(250.0, analysis.cutoff_hz * 0.55)
    reference_high = min(freqs[-1], analysis.cutoff_hz * 0.98)
    reference_mask = (freqs >= reference_low) & (freqs <= reference_high)
    extension_mask = freqs >= analysis.cutoff_hz

    target_power = mean_signal_power.copy()
    if np.count_nonzero(reference_mask) >= 8:
        reference_freqs = freqs[reference_mask]
        reference_db = 10.0 * np.log10(
            np.maximum(mean_signal_power[reference_mask], 1e-24)
        )
        slope, intercept = np.polyfit(np.log(reference_freqs), reference_db, 1)
        predicted_db = intercept + slope * np.log(
            np.maximum(freqs, reference_freqs[0])
        )
        air_decay = np.exp(
            -np.maximum(freqs - analysis.cutoff_hz, 0.0)
            / max(analysis.cutoff_hz * 1.8, 2500.0)
        )
        fitted_power = np.power(10.0, predicted_db / 10.0) * (
            0.6 + 0.4 * air_decay
        )
        target_power[extension_mask] = np.maximum(
            target_power[extension_mask],
            fitted_power[extension_mask],
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

    desired_shape = desired_wet_power.reshape(
        (1,) * (residual_power.ndim - 1) + (-1,)
    )
    spectral_gain = np.sqrt(desired_shape / np.maximum(residual_power, 1e-24))
    spectral_gain = remove_spectral_spikes(spectral_gain)
    spectral_gain = _moving_average_last_axis(spectral_gain, smoothing_bins)

    residual_spectrum *= spectral_gain
    wet_oversampled = np.fft.irfft(residual_spectrum, n=sample_count, axis=-1)

    if get_rms(wet_oversampled) <= 1e-12:
        return samples_last

    wet_signal = (
        wet_oversampled
        if analysis.oversample_factor == 1
        else resample(wet_oversampled, oversampled_rate, target_sr=sample_rate)
    )

    dry_aligned, wet_aligned = pad_audio(samples_last, wet_signal)
    output = dry_aligned + wet_aligned * analysis.adaptive_mix

    peak = float(np.max(np.abs(output))) if output.size else 0.0
    if np.isfinite(peak) and peak > 1.25:
        output = output / peak * 1.25

    return _sanitize_audio_array(loudness_maximizer(output))


def apply_exciter(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
) -> np.ndarray:
    sanitized = _sanitize_audio_array(signal)
    if sanitized.size == 0 or sample_rate <= 0:
        return sanitized

    samples_last, restore_layout = _move_samples_to_last(sanitized)

    try:
        analysis = analyze_exciter(
            samples_last, sample_rate, cutoff_hz=cutoff_hz, mix=mix
        )
        log(
            "Exciter",
            f"Adaptive cutoff={analysis.cutoff_hz:.2f}Hz drive={analysis.drive:.2f}dB mix={analysis.adaptive_mix:.2f} oversample=x{analysis.oversample_factor}",
        )
        processed = _apply_exciter_core(samples_last, sample_rate, analysis)
    except Exception as e:
        catch(
            e,
            "Error during exciter processing. Falling back to baseline method",
        )
        processed = _baseline_apply_exciter(
            samples_last,
            sample_rate,
            cutoff_hz=cutoff_hz,
            mix=mix,
        )

    return restore_layout(_sanitize_audio_array(processed))


def apply_exciter_with_analysis(
    signal: np.ndarray,
    sample_rate: int,
    cutoff_hz: float | None = None,
    mix: float = 1.0,
) -> tuple[np.ndarray, ExciterAnalysis]:
    sanitized = _sanitize_audio_array(signal)
    samples_last, restore_layout = _move_samples_to_last(sanitized)
    analysis = analyze_exciter(
        samples_last, sample_rate, cutoff_hz=cutoff_hz, mix=mix
    )
    processed = _apply_exciter_core(samples_last, sample_rate, analysis)
    return restore_layout(_sanitize_audio_array(processed)), analysis
