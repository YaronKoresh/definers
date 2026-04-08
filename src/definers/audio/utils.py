from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from definers.audio.dependencies import librosa_module
from definers.logger import init_logger
from definers.system import catch, log, run, tmp

_logger = init_logger()


def stereo_widen(y: np.ndarray, width: float = 1.5) -> np.ndarray:
    if y.ndim < 2 or y.shape[0] != 2:
        return y
    coef = np.sqrt(2.0 / (1.0 + width**2))
    mid = (y[0] + y[1]) * 0.5
    side = (y[0] - y[1]) * 0.5 * width

    return np.stack([(mid + side) * coef, (mid - side) * coef])


def generate_bands(
    start_freq: float, end_freq: float, num_bands: int
) -> list[float]:
    if num_bands < 2:
        return [start_freq]

    bands = []
    factor = (end_freq / start_freq) ** (1 / (num_bands - 1))

    for i in range(num_bands):
        freq = start_freq * (factor**i)
        bands.append(freq)

    return bands


def subdivide_beats(beat_times: np.ndarray, subdivision: int) -> np.ndarray:
    if subdivision <= 1 or len(beat_times) < 2:
        return np.array(beat_times)
    new_beats: list[float] = []
    for i in range(len(beat_times) - 1):
        start_beat = beat_times[i]
        end_beat = beat_times[i + 1]
        interval = (end_beat - start_beat) / subdivision
        for j in range(subdivision):
            new_beats.append(start_beat + j * interval)
    new_beats.append(beat_times[-1])
    return np.array(sorted(list(set(new_beats))))


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
    try:
        from .mastering_loudness import get_lufs as _get_lufs
    except ImportError:
        from definers.audio.mastering_loudness import get_lufs as _get_lufs

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

    log(
        "LUFS Normalization",
        f"Current LUFS: {current_lufs:.2f} dB, Target LUFS: {target_lufs:.2f} dB, Gain: {gain_db:.2f} dB",
    )

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


def normalize_audio_to_peak(
    input_path: str, target_level: float = 0.9, format: str | None = None
) -> str | None:
    from pydub import AudioSegment

    if not 0.0 < target_level <= 1.0:
        catch("target_level must be between 0.0 and 1.0")
        return None

    if format is None:
        format = Path(input_path).suffix.lstrip(".") or "wav"

    output_path = tmp(format)

    try:
        audio = AudioSegment.from_file(input_path)
    except FileNotFoundError:
        catch(f"Input file not found at {input_path}")
        return None

    if target_level == 0.0 or audio.max_dBFS == -float("inf"):
        silent_audio = AudioSegment.silent(duration=len(audio))
        silent_audio.export(output_path, format=format)
        log(
            "Exported silent file"
            f"Silent audio detected or target level is 0. Saved silent file to '{output_path}'"
        )
        return output_path

    target_dbfs = 20 * math.log10(target_level)
    gain_to_apply = target_dbfs - audio.max_dBFS
    normalized_audio = audio.apply_gain(gain_to_apply)
    normalized_audio.export(output_path, format=format)
    log(
        f"Successfully normalized '{input_path}' to a peak of {target_dbfs:.2f} dBFS."
    )
    _logger.info("Saved result to '%s'", output_path)
    return output_path


def stretch_audio(
    input_path: str, output_path: str | None = None, speed_factor: float = 0.85
) -> str | None:
    if output_path is None:
        output_path = tmp("wav")

    command = [
        "rubberband",
        "--formant",
        "--tempo",
        str(speed_factor),
        "-q",
        input_path,
        output_path,
    ]

    try:
        run(command)
        return normalize_audio_to_peak(output_path)
    except Exception as e:
        catch(f"Error during audio stretching with rubberband: {e}")
        return None


def get_scale_notes(
    key: str = "C",
    scale: str = "major",
    start_octave: int = 1,
    end_octave: int = 9,
) -> np.ndarray:
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
    start_note_midi = (start_octave - 1) * 12 + NOTES.index(key.upper())
    scale_intervals = SCALES.get(scale.lower(), SCALES["major"])
    scale_notes: list[int] = []
    for i in range((end_octave - start_octave) * 12):
        if i % 12 in scale_intervals:
            scale_notes.append(start_note_midi + i)
    return np.array(scale_notes)


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


def create_sample_audio(duration_s: float, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)


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
