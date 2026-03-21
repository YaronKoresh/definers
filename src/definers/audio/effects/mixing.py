from __future__ import annotations

from pathlib import Path

import numpy as np

from definers.logger import init_logger
from definers.platform.filesystem import delete
from definers.platform.paths import tmp

from ..io import save_audio

_logger = init_logger()


def stereo(*y_s, lr: bool = True, sr: int = 44100) -> np.ndarray:
    stereo_result = None

    if lr and len(y_s) == 2:
        a, b = y_s
        a = stereo(a)
        b = stereo(b)
        _l, _r = pad_audio(a, b)
        y_s = [np.vstack([_l[0], _r[1]])]

    for y in y_s:
        if y.ndim == 1:
            y = y[np.newaxis, :]
            y = np.vstack([y, y])
        if y.ndim > 2:
            y = np.vstack([y[0], y[1]])
        if y.shape[1] < y.shape[0]:
            y = y.T

        y = np.asarray(y, dtype=np.float32)

        if stereo_result is None:
            stereo_result = y
        else:
            stereo_result = mix_audio(stereo_result, y, sr, weight=0.5)

    return stereo_result


def pad_audio(y1: np.ndarray, y2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y1 = stereo(y1)
    y2 = stereo(y2)

    max_len = max(y1.shape[1], y2.shape[1])

    y1_padded = np.zeros((y1.shape[0], max_len), dtype=np.float32)
    y2_padded = np.zeros((y2.shape[0], max_len), dtype=np.float32)
    y1_padded[:, : y1.shape[1]] = y1
    y2_padded[:, : y2.shape[1]] = y2

    return y1_padded, y2_padded


def mix_audio(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    weight: float = 0.3,
    fade_duration: float = 2.0,
    duck_depth: float = 0.3,
    cutoff_hz: int | None = None,
) -> np.ndarray:
    y1 = np.asarray(y1, dtype=np.float32)
    y2 = np.asarray(y2, dtype=np.float32)

    y1, y2 = stereo(y1), stereo(y2)
    y1_padded, y2_padded = pad_audio(y1, y2)

    y1_padded = y1_padded.T
    y2_padded = y2_padded.T

    if cutoff_hz is None:
        y1_filtered = y1_padded
    else:
        from scipy.signal import butter, lfilter

        nyquist = sr / 2.0 - 1.0
        normal_cutoff = cutoff_hz / nyquist
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        y1_filtered = lfilter(b, a, y1_padded, axis=0)

    window_size = int(sr * 0.1)
    envelope = np.abs(y2_padded.mean(axis=1))
    envelope = np.convolve(
        envelope, np.ones(window_size) / window_size, mode="same"
    )

    if np.max(envelope) > 0:
        envelope = envelope / np.max(envelope)
    envelope = envelope[:, np.newaxis]

    y1_processed = (y1_padded * (1 - envelope)) + (y1_filtered * envelope)

    duck_mask = 1.0 - (envelope * (1.0 - duck_depth))

    y_length = len(y1_padded)

    fade_samples = min(int(fade_duration * sr), y_length)
    t = np.linspace(0, np.pi / 2, fade_samples).reshape(-1, 1)
    fade_in, fade_out = np.sin(t), np.cos(t)

    mask1 = np.ones((y_length, 1)) * weight
    mask1[-fade_samples:] *= fade_out

    mask2 = np.zeros((y_length, 1))
    mask2[:fade_samples] = (1.0 - weight) * fade_in
    mask2[fade_samples:] = 1.0 - weight

    y1_weighted = y1_processed * mask1 * duck_mask
    y2_weighted = y2_padded * mask2

    y_mixed = y1_weighted + y2_weighted

    max_val = np.max(np.abs(y_mixed))
    if max_val > 1.0:
        y_mixed /= max_val

    if y_mixed.shape[0] > y_mixed.shape[1]:
        y_mixed = y_mixed.T

    return y_mixed


def dj_mix(
    files: list[str],
    mix_type: str | None = None,
    target_bpm: float | None = None,
    transition_sec: float = 5,
    format_choice: str = "mp3",
) -> str | None:

    import madmom
    import pydub

    if not files or len(files) < 2:
        _logger.warning(
            "Please provide at least two audio files for DJ mixing."
        )
        return None

    transition_ms = int(transition_sec * 1000)
    processed_tracks = []

    if target_bpm is None or target_bpm == 0:
        all_bpms: list[float] = []
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        beat_processor = madmom.features.beats.RNNBeatProcessor()
        _logger.info("Analyzing BPM for all tracks to determine average BPM")
        for file in files:
            try:
                act = beat_processor(str(file))
                bpm = np.median(60 / np.diff(proc(act)))
                if bpm > 0:
                    all_bpms.append(bpm)
            except Exception as e:
                _logger.warning(
                    "Could not analyze BPM for %s; skipping this track. Error: %s",
                    Path(str(file)).name,
                    e,
                )
                continue
        if all_bpms:
            target_bpm = float(np.mean(all_bpms))
            _logger.info("Average target BPM calculated as: %.2f", target_bpm)
        else:
            _logger.warning(
                "Could not determine BPM for any track. Beatmatching will be skipped."
            )
            target_bpm = 0.0

    for file in files:
        try:
            temp_stretched_path = None
            current_path = str(file)
            if (
                mix_type is not None
                and "beatmatched" in mix_type.lower()
                and (target_bpm is not None and target_bpm > 0)
            ):
                proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                act = madmom.features.beats.RNNBeatProcessor()(current_path)
                original_bpm = np.median(60 / np.diff(proc(act)))
                if original_bpm > 0 and target_bpm > 0:
                    speed_factor = target_bpm / original_bpm
                    temp_stretched_path = tmp(Path(current_path).suffix)
                    from definers.audio import stretch_audio

                    stretch_audio(
                        current_path, temp_stretched_path, speed_factor
                    )
                    current_path = temp_stretched_path

            track_segment = pydub.AudioSegment.from_file(current_path)
            processed_tracks.append(track_segment)
            if temp_stretched_path:
                delete(temp_stretched_path)
        except Exception as e:
            _logger.warning(
                "Could not process track %s, skipping. Error: %s",
                Path(file).name,
                e,
            )
            continue

    if not processed_tracks:
        _logger.warning("No tracks could be processed.")
        return None

    final_mix = processed_tracks[0]
    for i in range(1, len(processed_tracks)):
        final_mix = final_mix.append(
            processed_tracks[i], crossfade=transition_ms
        )

    output_stem = tmp("dj_mix", keep=False)
    final_output_path = save_audio(
        destination_path=output_stem,
        audio_signal=final_mix,
        output_format=format_choice,
    )
    return final_output_path
