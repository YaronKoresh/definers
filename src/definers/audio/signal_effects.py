from __future__ import annotations

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()


def stereo_widen(y: np.ndarray, width: float = 1.5) -> np.ndarray:
    if y.ndim < 2 or y.shape[0] != 2:
        return y
    coef = np.sqrt(2.0 / (1.0 + width**2))
    mid = (y[0] + y[1]) * 0.5
    side = (y[0] - y[1]) * 0.5 * width

    return np.stack([(mid + side) * coef, (mid - side) * coef])


def compute_gain_envelope(
    signal: np.ndarray, attack: float = 0.01, release: float = 0.1
) -> np.ndarray:
    envelope = np.abs(signal)
    window = int(attack * 44100)
    if window < 1:
        window = 1
    return np.convolve(envelope, np.ones(window) / window, mode="same")


def loudness_maximizer(
    signal: np.ndarray, threshold: float = -0.1
) -> np.ndarray:
    max_value = np.max(np.abs(signal))
    if max_value == 0:
        return signal
    gain = 10 ** (threshold / 20.0) / max_value
    return np.clip(signal * gain, -1.0, 1.0)


def apply_compressor(
    signal: np.ndarray, threshold: float = -20.0, ratio: float = 4.0
) -> np.ndarray:
    db_signal = 20 * np.log10(np.maximum(np.abs(signal), 1e-12))
    over_threshold = db_signal > threshold
    gain_reduction = np.zeros_like(db_signal)
    gain_reduction[over_threshold] = (db_signal[over_threshold] - threshold) * (
        1 - 1 / ratio
    )
    linear_gain = 10 ** (-gain_reduction / 20.0)
    return signal * linear_gain


def riaa_filter(input_filename: str, bass_factor: float = 1.0) -> np.ndarray:
    import soundfile as sf
    from scipy.signal import butter, lfilter

    y, sr = sf.read(input_filename)
    nyq = 0.5 * sr
    low = 20 / nyq
    high = 20000 / nyq
    b, a = butter(2, [low, high], btype="band")
    filtered = lfilter(b, a, y)
    return filtered * bass_factor
