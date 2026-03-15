from __future__ import annotations

import numpy as np
from numba import njit


def resample(y: np.ndarray, sr: int, target_sr: int = 44100) -> np.ndarray:
    import librosa

    if sr == target_sr:
        return y

    y = np.ascontiguousarray(y)

    if np.issubdtype(y.dtype, np.integer):
        y = librosa.util.buf_to_float(y, n_bytes=y.itemsize)
    else:
        y = y.astype(np.float64, copy=False)

    method = "fft" if target_sr < sr else "soxr_vhq"
    return librosa.resample(
        y, orig_sr=sr, target_sr=target_sr, res_type=method, axis=-1
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

    (num_channels, audio_length) = data.shape
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
