from __future__ import annotations

import librosa
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from ..dsp import resample
from .mixing import pad_audio


def calculate_dynamic_cutoff(
    y: np.ndarray, sr: int, rolloff_percent: float
) -> float:
    y_mono = np.mean(y, axis=-2) if y.ndim > 1 else y
    freqs, psd = welch(y_mono, fs=sr, nperseg=4096)
    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]
    rolloff_index = np.argmax(
        cumulative_energy >= (rolloff_percent * total_energy)
    )
    optimal_cutoff_hz = np.clip(freqs[rolloff_index], 4000.0, 12000.0)
    return float(optimal_cutoff_hz)


def apply_exciter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float | None = None,
    drive: float | None = None,
    oversample: int = 2,
    mix: float = 1.0,
) -> np.ndarray:
    if cutoff_hz is None:
        cutoff_hz = calculate_dynamic_cutoff(y, sr, rolloff_percent=1.0)

    sr_up = sr * oversample
    y_up = resample(y, sr, target_sr=sr_up)
    nyquist_up = sr_up / 2.0

    sos_pre = butter(4, cutoff_hz / nyquist_up, btype="high", output="sos")
    high_band_orig_up = sosfiltfilt(sos_pre, y_up)

    peak_val = np.max(np.abs(high_band_orig_up)) + 1e-9
    high_band_norm = high_band_orig_up / peak_val

    if drive is None:
        drive = np.clip(15.0 / (peak_val * 2 + 1e-6), 2.0, 50.0)

    distorted_up = np.arctan(high_band_norm * drive)

    sos_post = butter(4, cutoff_hz / nyquist_up, btype="high", output="sos")
    distorted_up = sosfiltfilt(sos_post, distorted_up)

    distorted_highs = resample(distorted_up, sr_up, target_sr=sr)

    min_len = min(y.shape[-1], distorted_highs.shape[-1])
    y_final = y[..., :min_len].copy()
    highs_final = distorted_highs[..., :min_len]

    sos_orig = butter(4, cutoff_hz / (sr / 2), btype="high", output="sos")
    y_highs_only = sosfiltfilt(sos_orig, y_final)
    target_std = np.std(y_highs_only) + 1e-9
    current_std = np.std(highs_final) + 1e-9

    highs_final = highs_final * (target_std / current_std)

    y_final, highs_final = pad_audio(y_final, highs_final)

    final = y_final + (highs_final * mix)

    return final
