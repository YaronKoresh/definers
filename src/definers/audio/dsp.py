from __future__ import annotations

import numpy as np
from numba import njit
from scipy.ndimage import median_filter

from ..file_ops import log

_FLOAT_EPSILON = np.finfo(np.float32).eps
_ROBUST_SIGMA_SCALE = 1.4826
_AUDIO_WINDOWS_CACHE: dict[int, tuple[int, ...]] = {}
_SPECTRAL_WINDOWS_CACHE: dict[int, tuple[int, ...]] = {}


def _make_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _max_supported_odd_window(length: int) -> int:
    return max(0, length if length % 2 else length - 1)


def _unique_odd_windows(
    candidates: list[int], maximum: int, minimum: int = 3
) -> tuple[int, ...]:
    windows: list[int] = []
    seen: set[int] = set()
    for candidate in candidates:
        window = _make_odd(min(candidate, maximum), minimum=minimum)
        if minimum <= window <= maximum and window not in seen:
            seen.add(window)
            windows.append(window)
    windows.sort()
    return tuple(windows)


def _audio_windows(length: int) -> tuple[int, ...]:
    cached = _AUDIO_WINDOWS_CACHE.get(length)
    if cached is not None:
        return cached

    maximum = _max_supported_odd_window(length)
    windows = _unique_odd_windows(
        [
            max(21, length // 256),
            max(321, length // 16),
            max(1021, length // 4),
        ],
        maximum=maximum,
        minimum=3,
    )

    log("Despiker", f"Selected audio despiking windows: {windows}")

    _AUDIO_WINDOWS_CACHE[length] = windows
    return windows


def _spectral_windows(length: int) -> tuple[int, ...]:
    cached = _SPECTRAL_WINDOWS_CACHE.get(length)
    if cached is not None:
        return cached

    maximum = _max_supported_odd_window(length)
    windows = (13,) if maximum >= 13 else ()

    log("Despiker", f"Selected spectral despiking windows: {windows}")

    _SPECTRAL_WINDOWS_CACHE[length] = windows
    return windows


def _sanitize_nonfinite_channel(channel: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(channel)
    if finite_mask.all():
        return channel
    if not finite_mask.any():
        return np.zeros_like(channel)
    finite_indices = np.flatnonzero(finite_mask)
    finite_values = channel[finite_indices]
    if finite_indices.size == 1:
        return np.full_like(channel, finite_values[0])
    full_index = np.arange(channel.size, dtype=np.float32)
    return np.interp(
        full_index, finite_indices.astype(np.float32), finite_values
    )


def _filter_size(signal: np.ndarray, window: int) -> int | tuple[int, int]:
    return window if signal.ndim == 1 else (1, window)


def _interp_last_axis(
    values: np.ndarray, hop: int, output_length: int
) -> np.ndarray:
    if values.shape[-1] == 1:
        return np.broadcast_to(
            values[..., :1], values.shape[:-1] + (output_length,)
        ).copy()

    full_index = np.arange(output_length, dtype=np.float32)
    decimated_index = np.arange(values.shape[-1], dtype=np.float32) * hop

    if values.ndim == 1:
        return np.interp(full_index, decimated_index, values)

    interpolated = np.empty(
        values.shape[:-1] + (output_length,), dtype=values.dtype
    )
    flat_values = values.reshape(-1, values.shape[-1])
    flat_output = interpolated.reshape(-1, output_length)
    for row_index in range(flat_values.shape[0]):
        flat_output[row_index] = np.interp(
            full_index, decimated_index, flat_values[row_index]
        )
    return interpolated


def _windowed_median_and_mad(
    signal: np.ndarray, window: int, mode: str
) -> tuple[np.ndarray, np.ndarray]:
    signal_length = signal.shape[-1]
    maximum = _max_supported_odd_window(signal_length)
    if maximum < 3:
        return signal.copy(), np.zeros_like(signal)

    window = _make_odd(min(window, maximum))
    if window <= 3 or signal_length <= 64:
        size = _filter_size(signal, window)
        median = median_filter(signal, size=size, mode=mode)
        mad = median_filter(np.abs(signal - median), size=size, mode=mode)
        return median, mad

    hop = (
        1
        if window <= 63
        else max(1, min(window // 16, max(1, signal_length // 2048)))
    )
    if hop == 1:
        size = _filter_size(signal, window)
        median = median_filter(signal, size=size, mode=mode)
        mad = median_filter(np.abs(signal - median), size=size, mode=mode)
        return median, mad

    decimated = signal[..., ::hop]
    small_window = _make_odd(max(3, window // hop))
    small_size = _filter_size(decimated, small_window)

    decimated_median = median_filter(decimated, size=small_size, mode=mode)
    decimated_mad = median_filter(
        np.abs(decimated - decimated_median), size=small_size, mode=mode
    )

    median = _interp_last_axis(decimated_median, hop, signal_length)
    mad = _interp_last_axis(decimated_mad, hop, signal_length)
    return median, mad


def _infer_audio_domain(data: np.ndarray) -> bool:
    magnitude_reference = max(
        float(np.quantile(np.abs(data), 0.99)), _FLOAT_EPSILON
    )
    negative_floor = max(1e-9, magnitude_reference * 1e-6)
    return bool(np.any(data < -negative_floor))


def _axis_neighbors(
    values: np.ndarray, step: int
) -> tuple[np.ndarray, np.ndarray]:
    step = max(1, int(step))
    length = values.shape[-1]
    if length <= step:
        left = np.broadcast_to(values[..., :1], values.shape)
        right = np.broadcast_to(values[..., -1:], values.shape)
        return left, right

    left = np.concatenate(
        (np.repeat(values[..., :1], step, axis=-1), values[..., :-step]),
        axis=-1,
    )
    right = np.concatenate(
        (values[..., step:], np.repeat(values[..., -1:], step, axis=-1)),
        axis=-1,
    )
    return left, right


def _despike_audio(data: np.ndarray, windows: tuple[int, ...]) -> np.ndarray:
    if not windows:
        return data

    left_1, right_1 = _axis_neighbors(data, 1)
    left_2, right_2 = _axis_neighbors(data, 2)

    upper_reference = 0.5 * (
        np.maximum(left_1, right_1) + np.maximum(left_2, right_2)
    )
    lower_reference = 0.5 * (
        np.minimum(left_1, right_1) + np.minimum(left_2, right_2)
    )

    upper_value_sum = np.zeros_like(data)
    upper_weight_sum = np.zeros_like(data)
    lower_value_sum = np.zeros_like(data)
    lower_weight_sum = np.zeros_like(data)
    vote_count = np.zeros(data.shape, dtype=np.uint8)

    for scale_index, window in enumerate(windows):
        median, mad = _windowed_median_and_mad(data, window, mode="reflect")
        sigma = _ROBUST_SIGMA_SCALE * np.maximum(mad, 1e-9)

        threshold = 2.75 + (0.25 * scale_index)
        prominence_threshold = (0.35 + (0.07 * scale_index)) * sigma

        upper_limit = median + (threshold * sigma)
        lower_limit = median - (threshold * sigma)

        upper_excess = data - upper_limit
        lower_excess = lower_limit - data

        upper_prominence = data - upper_reference
        lower_prominence = lower_reference - data

        upper_flag = (upper_excess > 0.0) & (
            upper_prominence > prominence_threshold
        )
        lower_flag = (lower_excess > 0.0) & (
            lower_prominence > prominence_threshold
        )
        vote_count += upper_flag | lower_flag

        scale_weight = 1.5 / np.sqrt(float(window))
        upper_strength = (1.0 + (upper_excess / sigma)) * (
            1.0 + (upper_prominence / sigma)
        )
        lower_strength = (1.0 + (lower_excess / sigma)) * (
            1.0 + (lower_prominence / sigma)
        )

        upper_weight = np.where(upper_flag, scale_weight * upper_strength, 0.0)
        lower_weight = np.where(lower_flag, scale_weight * lower_strength, 0.0)

        upper_target = np.minimum(
            upper_limit,
            np.maximum(median, upper_reference) + (0.2 * sigma),
        )
        lower_target = np.maximum(
            lower_limit,
            np.minimum(median, lower_reference) - (0.2 * sigma),
        )

        upper_value_sum += upper_weight * upper_target
        upper_weight_sum += upper_weight
        lower_value_sum += lower_weight * lower_target
        lower_weight_sum += lower_weight

    consensus = 1 if len(windows) <= 2 else 2 if len(windows) <= 5 else 3
    corrected = data.copy()

    upper_mask = (vote_count >= consensus) & (upper_weight_sum > 0.0)
    lower_mask = (vote_count >= consensus) & (lower_weight_sum > 0.0)

    if np.any(upper_mask):
        corrected[upper_mask] = (
            upper_value_sum[upper_mask] / upper_weight_sum[upper_mask]
        )

    if np.any(lower_mask):
        corrected[lower_mask] = (
            lower_value_sum[lower_mask] / lower_weight_sum[lower_mask]
        )

    return corrected


def _suppress_spectral_peaks(
    data: np.ndarray, windows: tuple[int, ...]
) -> np.ndarray:
    if not windows:
        log(
            "Despiker",
            "No spectral windows available, applying non-negativity clamp",
        )
        return np.maximum(data, 0.0)

    nonnegative = np.maximum(data, 0.0)
    log_data = np.log1p(nonnegative)

    left_1, right_1 = _axis_neighbors(log_data, 1)
    left_2, right_2 = _axis_neighbors(log_data, 2)

    ridge_reference = 0.5 * (
        np.maximum(left_1, right_1) + np.maximum(left_2, right_2)
    )

    upper_value_sum = np.zeros_like(log_data)
    upper_weight_sum = np.zeros_like(log_data)
    vote_count = np.zeros(log_data.shape, dtype=np.uint8)

    for scale_index, window in enumerate(windows):
        median, mad = _windowed_median_and_mad(log_data, window, mode="nearest")
        sigma = _ROBUST_SIGMA_SCALE * np.maximum(mad, 1e-9)

        threshold = 3.5 + (0.35 * scale_index)
        prominence_threshold = (0.75 + (0.12 * scale_index)) * sigma

        upper_limit = median + (threshold * sigma)
        excess = log_data - upper_limit
        prominence = log_data - ridge_reference

        flag = (excess > 0.0) & (prominence > prominence_threshold)
        vote_count += flag

        scale_weight = 1.5 / np.sqrt(float(window))
        strength = (1.0 + (excess / sigma)) * (1.0 + (prominence / sigma))
        weight = np.where(flag, scale_weight * strength, 0.0)

        upper_target = np.minimum(
            upper_limit,
            np.maximum(median, ridge_reference) + (0.2 * sigma),
        )

        upper_value_sum += weight * upper_target
        upper_weight_sum += weight

    consensus = max(2, int(np.ceil(len(windows) * 0.5)))

    mask = (vote_count >= consensus) & (upper_weight_sum > 0.0)

    max_reduction_db = 1.5
    max_reduction_ln = max_reduction_db * np.log(10.0) / 20.0
    target = upper_value_sum[mask] / upper_weight_sum[mask]

    corrected_log = log_data.copy()
    corrected_log[mask] = np.maximum(target, log_data[mask] - max_reduction_ln)

    if np.any(mask):
        corrected_log[mask] = upper_value_sum[mask] / upper_weight_sum[mask]

    return np.maximum(np.expm1(corrected_log), 0.0)


def remove_spectral_spikes(data: np.ndarray) -> np.ndarray:
    if not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray")
    if data.ndim not in (1, 2):
        raise ValueError("data must be 1D or 2D")
    if np.iscomplexobj(data):
        raise TypeError("complex-valued input is not supported")
    if data.dtype == np.bool_:
        raise TypeError("boolean input is not supported")
    if data.size == 0:
        return np.array(data, copy=True)

    output_dtype = (
        data.dtype
        if np.issubdtype(data.dtype, np.floating) and data.dtype.itemsize >= 4
        else np.float32
    )
    working = np.ascontiguousarray(data, dtype=np.float32)
    channels = working[np.newaxis, :] if working.ndim == 1 else working

    if not np.isfinite(channels).all():
        channels = channels.copy()
        for channel_index in range(channels.shape[0]):
            channels[channel_index] = _sanitize_nonfinite_channel(
                channels[channel_index]
            )

    is_audio = _infer_audio_domain(channels)

    if is_audio:
        log(
            "Despiker",
            "Detected audio waveform. Running adaptive multi-scale robust despiker.",
        )
        windows = _audio_windows(channels.shape[-1])
        if windows:
            channels = _despike_audio(channels, windows)
    else:
        log(
            "Despiker",
            "Detected spectral data. Running log-domain adaptive peak suppressor.",
        )
        windows = _spectral_windows(channels.shape[-1])
        if windows:
            channels = _suppress_spectral_peaks(channels, windows)
        else:
            channels = np.maximum(channels, 0.0)

    result = channels[0] if data.ndim == 1 else channels
    return np.asarray(result, dtype=output_dtype)


def resample(
    y: np.ndarray,
    sr: int,
    target_sr: int = 44100,
    *,
    allow_brickwall: bool = True,
    method: str | None = None,
) -> np.ndarray:
    import librosa

    if sr == target_sr:
        return y

    y = np.ascontiguousarray(y)

    if np.issubdtype(y.dtype, np.integer):
        y = librosa.util.buf_to_float(y, n_bytes=y.itemsize)
    else:
        y = y.astype(np.float32, copy=False)

    if sr > target_sr and sr % target_sr == 0 and allow_brickwall:
        log(
            "Resampling",
            "Downsampling using FFT-based Perfect Brickwall Filter method for integer ratio",
        )

        factor = sr // target_sr
        is_mono = y.ndim == 1
        if is_mono:
            y = y[np.newaxis, :]

        num_samples = y.shape[-1]
        new_len = num_samples // factor
        cutoff_bin = num_samples // (2 * factor)

        spectrum = np.fft.rfft(y, axis=-1)
        if cutoff_bin + 1 < spectrum.shape[-1]:
            spectrum[..., cutoff_bin + 1 :] = 0

        filtered = np.fft.irfft(spectrum, n=num_samples, axis=-1)
        y_downsampled = np.ascontiguousarray(
            filtered[..., : new_len * factor : factor]
        )

        return y_downsampled[0] if is_mono else y_downsampled

    if method is None:
        method = "fft" if target_sr < sr else "soxr_vhq"

    return librosa.resample(
        y,
        orig_sr=float(sr),
        target_sr=float(target_sr),
        res_type=method,
        fix=False,
        axis=-1,
    )


@njit(cache=True)
def decoupled_envelope(
    env_db: np.ndarray, attack_coef: float, release_coef: float
):
    out = np.empty_like(env_db)
    prev = env_db[0]
    for i in range(len(env_db)):
        coef = attack_coef if env_db[i] > prev else release_coef
        prev = coef * prev + (1.0 - coef) * env_db[i]
        out[i] = prev
    return out


@njit(cache=True)
def limiter_smooth_env(
    env: np.ndarray, attack_coef: float, release_coef: float
):
    out = np.empty_like(env)
    prev = env[0]

    is_adaptive = release_coef.ndim > 0

    for i in range(len(env)):
        rc = release_coef[i] if is_adaptive else release_coef

        coef = attack_coef if env[i] > prev else rc

        prev = coef * prev + (1.0 - coef) * env[i]
        out[i] = prev

    return out


def process_audio_chunks(fn, data, chunk_size, overlap=0):

    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    (_num_channels, audio_length) = data.shape
    step = chunk_size - overlap
    final_result = np.zeros_like(data, dtype=np.float32)
    window_sum = np.zeros_like(data, dtype=np.float32)
    window = np.hanning(chunk_size)
    window = window[np.newaxis, :]

    start = 0
    while start < audio_length:
        end = min(start + chunk_size, audio_length)
        current_chunk_size = end - start
        chunk = data[:, start:end]
        if current_chunk_size < chunk_size:
            padding_size = chunk_size - current_chunk_size
            chunk = np.pad(chunk, ((0, 0), (0, padding_size)), "constant")

        processed_chunk = fn(chunk)
        if processed_chunk.ndim == 1:
            processed_chunk = processed_chunk[np.newaxis, :]

        final_result[:, start:end] += (
            processed_chunk[:, :current_chunk_size]
            * window[:, :current_chunk_size]
        )
        window_sum[:, start:end] += window[:, :current_chunk_size] ** 2

        if end == audio_length:
            break
        start += step

    window_sum[window_sum == 0] = 1.0
    final_result /= window_sum
    return final_result
