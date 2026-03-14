
from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import butter, sosfilt, resample_poly, welch

from .mixing import pad_audio


def calculate_dynamic_cutoff(
    y: np.ndarray, sr: int, rolloff_percent: float
) -> float:
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

    freqs, psd = welch(y_mono, fs=sr, nperseg=4096)

    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]

    rolloff_index = np.argmax(cumulative_energy >= (rolloff_percent * total_energy))
    optimal_cutoff_hz = freqs[rolloff_index]

    optimal_cutoff_hz = np.clip(optimal_cutoff_hz, 4000.0, 12000.0)

    return float(optimal_cutoff_hz)


def apply_exciter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float | None = None,
    drive: float | None = None,
    oversample: int = 2,
    mix: float = 1.0,
) -> np.ndarray:
    nyquist = sr / 2.0

    if cutoff_hz is None:
        cutoff_hz = calculate_dynamic_cutoff(y, sr, rolloff_percent=0.7)

    sos = butter(4, cutoff_hz / nyquist * oversample, btype="high", output="sos")
    high_band = sosfilt(sos, y)

    if drive is None:
        peak_highs = np.max(np.abs(high_band))
        drive = np.clip(4.0 / (peak_highs + 1e-6), 1.0, 50.0)

    y_up = resample_poly(high_band, up=oversample, down=1, axis=-1)

    distorted_up = np.tanh(y_up * drive)
    distorted_up *= peak_highs / (np.max(np.abs(distorted_up)) + 1e-6)

    distorted_highs = resample_poly(distorted_up, up=1, down=oversample, axis=-1)

    if distorted_highs.shape[-1] > high_band.shape[-1]:
        distorted_highs = distorted_highs[..., : high_band.shape[-1]]

    pure_air = sosfilt(sos, distorted_highs)

    y, pure_air = pad_audio(y, pure_air)
    y_enhanced = y + (pure_air * mix)

    return y_enhanced
