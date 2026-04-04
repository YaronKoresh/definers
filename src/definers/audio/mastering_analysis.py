from __future__ import annotations

from typing import Any

import numpy as np


def measure_spectrum(
    self,
    y_mono: np.ndarray,
    signal_module: Any,
) -> tuple[np.ndarray, np.ndarray]:
    floor_db = -120.0
    ceiling_db = 20.0

    if len(y_mono) < self.analysis_nperseg:
        y_mono_padded = np.pad(
            y_mono,
            (0, self.analysis_nperseg - len(y_mono)),
        )
    else:
        y_mono_padded = y_mono

    f_axis, psd = signal_module.welch(
        y_mono_padded,
        fs=self.resampling_target,
        nperseg=self.analysis_nperseg,
        nfft=self.fft_n,
        noverlap=int(self.analysis_nperseg * 0.875),
        window=("kaiser", 18.0),
        scaling="density",
        average="median",
        detrend="constant",
    )

    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
    psd_db = np.clip(psd_db, floor_db, ceiling_db)

    f_axis = np.clip(f_axis, self.low_cut, self.high_cut)

    return psd_db, f_axis
