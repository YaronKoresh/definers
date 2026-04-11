from __future__ import annotations

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from ..file_ops import log


def freq_cut(
    y: np.ndarray, sr: int, low_cut: float | None, high_cut: float | None
) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)

    if low_cut is None and high_cut is None:
        return np.ascontiguousarray(y)

    y = y - np.mean(y, axis=-1, keepdims=True)

    n = y.shape[-1]
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    y_fft = np.fft.rfft(y, axis=-1)

    mask = np.ones_like(freqs)

    if low_cut is not None and low_cut > 0.0:
        mask[freqs < low_cut] = 0.0

    if high_cut is not None and high_cut < freqs[-1]:
        mask[freqs > high_cut] = 0.0

    if np.all(mask == 1.0):
        return np.ascontiguousarray(y)

    y_filtered_fft = y_fft * mask

    y_res = np.fft.irfft(y_filtered_fft, n=n, axis=-1)

    return np.ascontiguousarray(y_res)
