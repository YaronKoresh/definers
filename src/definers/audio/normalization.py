from __future__ import annotations

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from .dependencies import librosa_module


def get_rms(y: np.ndarray) -> float:
    current_rms = (
        np.max(np.sqrt(np.mean(y**2, axis=1)))
        if y.ndim > 1
        else np.sqrt(np.mean(y**2))
    )

    return float(current_rms)


def apply_rms(y: np.ndarray, rms: float):
    y *= rms / get_rms(y) + 1e-12

    return y


def get_lufs(y: np.ndarray, sr: int) -> float:
    from .mastering.loudness import get_lufs as _get_lufs

    return _get_lufs(y, sr)


def adjust_final_output_lufs(
    y: np.ndarray, sr: int, target_lufs: float
) -> np.ndarray:
    y_array = np.asarray(y)
    if y_array.ndim == 0:
        y_array = y_array.reshape(1)

    finite_y = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
    if finite_y.size == 0 or not np.isfinite(sr) or sr <= 0:
        raise ValueError(
            "Invalid audio data or sample rate, cannot adjust LUFS."
        )

    if not np.isfinite(target_lufs):
        raise ValueError("Target LUFS is not finite, cannot adjust.")

    current_lufs = get_lufs(finite_y, sr)
    if not np.isfinite(current_lufs):
        raise ValueError("Current LUFS is not finite, cannot adjust.")

    gain_db = float(np.clip(target_lufs - current_lufs, -60.0, 60.0))
    gain_lin = float(10.0 ** (gain_db / 20.0))
    if not np.isfinite(gain_lin):
        raise ValueError("Calculated gain is not finite, cannot adjust.")

    out = finite_y * gain_lin

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def apply_lufs(y: np.ndarray, sr: int, target_lufs: float) -> np.ndarray:
    return adjust_final_output_lufs(y, sr, target_lufs)


def calculate_active_rms(
    y: np.ndarray,
    sr: int,
    top_db: float = 40,
    frame_length: int = 1024,
    hop_length: int = 256,
) -> float:
    librosa = librosa_module()
    non_silent_intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    if len(non_silent_intervals) == 0:
        return 1e-06
    active_audio = np.concatenate(
        [y[start:end] for (start, end) in non_silent_intervals]
    )
    if len(active_audio) == 0:
        return 1e-06
    return float(np.sqrt(np.mean(active_audio**2)))
