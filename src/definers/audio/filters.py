
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def freq_cut(y: np.ndarray, sr: int, low_cut: int | None, high_cut: int | None) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    original_length = y.shape[-1]

    if y.ndim > 1:
        for i in range(y.shape[0]):
            y[i] -= np.mean(y[i])
    else:
        y -= np.mean(y)

    original_length = y.shape[-1]

    if high_cut:
        target_high_sr = high_cut * 2
        from scipy.signal import resample

        y_high_cut_resampled = resample(y, target_high_sr)
        y_main = resample(y_high_cut_resampled, sr)

        if y_main.shape[-1] > original_length:
            y = y_main[..., :original_length]
        elif y_main.shape[-1] < original_length:
            pad_width = original_length - y_main.shape[-1]
            y = np.pad(y_main, (*((0, 0),) * (y_main.ndim - 1), (0, pad_width)))

    if low_cut:
        target_low_sr = low_cut * 2
        from scipy.signal import resample

        y_low_only_resampled = resample(y, target_low_sr)
        y_low_reconstructed = resample(y_low_only_resampled, sr)

        if y_low_reconstructed.shape[-1] > original_length:
            y_low_reconstructed = y_low_reconstructed[..., :original_length]
        elif y_low_reconstructed.shape[-1] < original_length:
            pad_width = original_length - y_low_reconstructed.shape[-1]
            y_low_reconstructed = np.pad(
                y_low_reconstructed,
                (*((0, 0),) * (y_low_reconstructed.ndim - 1), (0, pad_width)),
            )

        y -= y_low_reconstructed

    return y
