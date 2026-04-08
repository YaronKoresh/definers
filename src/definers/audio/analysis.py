from __future__ import annotations

import numpy as np

from definers.constants import MADMOM_AVAILABLE
from definers.image.helpers import get_max_resolution
from definers.logger import init_logger
from definers.system import cores
from definers.system.paths import tmp

from .dependencies import librosa_module

_logger = init_logger()


def _normalize_series(values: np.ndarray) -> np.ndarray:
    value_min = np.min(values)
    value_max = np.max(values)
    if value_max - value_min == 0:
        return np.zeros_like(values)
    return (values - value_min) / (value_max - value_min)


def _resolve_tempo_and_beats(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    audio_path: str,
    duration: float | None,
):
    librosa = librosa_module()
    if MADMOM_AVAILABLE and (duration is None or duration > 10):
        try:
            import madmom

            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            act = madmom.features.beats.RNNBeatProcessor()(audio_path)
            beat_times = proc(act)
            beat_frames = librosa.time_to_frames(
                beat_times, sr=sr, hop_length=hop_length
            )
            if len(beat_times) > 1:
                bpm = int(round(float(60.0 / np.mean(np.diff(beat_times)))))
                return (bpm, beat_frames)
        except Exception:
            pass
    tempo, beat_frames = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    return (int(round(tempo)), beat_frames)


def detect_silence_mask(
    audio_data: np.ndarray,
    sample_rate: int,
    threshold_db: float = -16,
    min_silence_len: float = 0.1,
) -> np.ndarray:
    librosa = librosa_module()

    threshold_amplitude = librosa.db_to_amplitude(threshold_db)
    frame_length = int(0.02 * sample_rate)
    hop_length = frame_length // 4

    rms = librosa.feature.rms(
        y=audio_data, frame_length=frame_length, hop_length=hop_length
    )[0]
    silence_mask_rms = rms < threshold_amplitude
    silence_mask = np.repeat(silence_mask_rms, hop_length)

    if len(silence_mask) > len(audio_data):
        silence_mask = silence_mask[: len(audio_data)]
    elif len(silence_mask) < len(audio_data):
        padding = np.ones(len(audio_data) - len(silence_mask), dtype=bool)
        silence_mask = np.concatenate((silence_mask, padding))

    min_silence_samples = int(min_silence_len * sample_rate)
    silence_mask_filtered = silence_mask.copy()

    silence_regions = librosa.effects.split(
        silence_mask.astype(float), top_db=0.5
    )
    for start, end in silence_regions:
        if end - start < min_silence_samples:
            silence_mask_filtered[start:end] = False

    return silence_mask_filtered


def get_active_audio_timeline(
    audio_file: str,
    threshold_db: float = -16,
    min_silence_len: float = 0.1,
) -> list[tuple[float, float]]:
    librosa = librosa_module()

    (audio_data, sample_rate) = librosa.load(audio_file, sr=32000)
    silence_mask = detect_silence_mask(
        audio_data, sample_rate, threshold_db, min_silence_len
    )
    active_regions = librosa.effects.split(
        np.logical_not(silence_mask).astype(float), frame_length=1, hop_length=1
    )
    return [
        (start.item() / int(sample_rate), end.item() / int(sample_rate))
        for (start, end) in active_regions
    ]


def _build_audio_analysis_payload(
    y: np.ndarray,
    sr: int,
    audio_path: str,
    hop_length: int,
    actual_duration: float,
    duration: float | None,
) -> dict[str, object]:
    librosa = librosa_module()
    stft = librosa.stft(y, hop_length=hop_length)
    (mag, _) = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)

    low_mask = (freqs >= 20) & (freqs < 250)
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000

    rms_all = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_low = np.mean(mag[low_mask, :], axis=0) if np.any(low_mask) else rms_all
    rms_mid = np.mean(mag[mid_mask, :], axis=0) if np.any(mid_mask) else rms_all
    rms_high = (
        np.mean(mag[high_mask, :], axis=0) if np.any(high_mask) else rms_all
    )
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    bpm, beat_frames = _resolve_tempo_and_beats(
        y=y,
        sr=sr,
        hop_length=hop_length,
        audio_path=audio_path,
        duration=duration,
    )

    return {
        "y": y,
        "sr": sr,
        "hop_length": hop_length,
        "duration": actual_duration,
        "bpm": bpm,
        "beat_frames": beat_frames,
        "spectral_centroid": spectral_centroid,
        "stft": stft,
        "stft_db": stft_db,
        "rms": rms_all,
        "rms_low": rms_low,
        "rms_mid": rms_mid,
        "rms_high": rms_high,
        "normalize": _normalize_series,
    }


def analyze_audio(
    audio_path: str,
    hop_length: int = 1024,
    duration: float | None = None,
    offset: float = 0.0,
) -> dict[str, object]:
    librosa = librosa_module()

    (y, sr) = librosa.load(
        audio_path, sr=None, duration=duration, offset=offset
    )
    actual_duration = librosa.get_duration(y=y, sr=sr)
    return _build_audio_analysis_payload(
        y=y,
        sr=sr,
        audio_path=audio_path,
        hop_length=hop_length,
        actual_duration=actual_duration,
        duration=duration,
    )


def analyze_audio_features(audio_path: str, txt: bool = True):
    librosa = librosa_module()

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception:
        _logger.exception("Failed to load audio for feature analysis")
        return None

    try:
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    except Exception:
        tempo = 0.0

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_sum = np.mean(chroma, axis=1)

    NOTES = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    major_profile = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
    minor_profile = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)

    best_score = -1.0
    best_key = "C"
    best_mode = "major"

    for root in range(12):
        major_rot = np.roll(major_profile, root)
        minor_rot = np.roll(minor_profile, root)
        score_major = float(np.dot(chroma_sum, major_rot))
        score_minor = float(np.dot(chroma_sum, minor_rot))
        if score_major > best_score:
            best_score = score_major
            best_key = NOTES[root]
            best_mode = "major"
        if score_minor > best_score:
            best_score = score_minor
            best_key = NOTES[root]
            best_mode = "minor"

    if txt:
        return f"{best_key} {best_mode} ({int(round(tempo))} bpm)"
    return (best_key, best_mode, float(tempo))


def beat_visualizer(
    image_path: str,
    audio_path: str,
    image_effect: str,
    animation_style: str,
    scale_intensity: float,
) -> str:
    librosa = librosa_module()

    from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip
    from PIL import Image, ImageFilter

    img = Image.open(image_path)
    (w, h) = get_max_resolution(*img.size)
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    (W, H) = img.size

    effect_map = {
        "Blur": ImageFilter.BLUR,
        "Sharpen": ImageFilter.SHARPEN,
        "Contour": ImageFilter.CONTOUR,
        "Emboss": ImageFilter.EMBOSS,
    }
    if image_effect in effect_map:
        img = img.filter(effect_map[image_effect])

    output_path = tmp(".mp4")

    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    (y, sr) = librosa.load(audio_path, sr=None)
    hop_length = 512
    effect_strength = scale_intensity - 1.0
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-07)
    rms_scales = 1.0 + rms_normalized * effect_strength * 0.5
    (tempo, beat_frames) = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    beat_impulses = np.zeros_like(rms_normalized)
    decay_rate = 0.75
    for beat_frame in beat_frames:
        frame = beat_frame
        impulse = 1.0
        while frame < len(beat_impulses) and impulse > 0.01:
            beat_impulses[frame] = max(beat_impulses[frame], impulse)
            impulse *= decay_rate
            frame += 1
    beat_scales = 1.0 + beat_impulses * effect_strength

    def base_animation_func(t: float) -> float:
        if animation_style == "Zoom In":
            return 1 + 0.1 * (t / duration)
        elif animation_style == "Zoom Out":
            return 1.1 - 0.1 * (t / duration)
        return 1.0

    def final_scale_func(t: float) -> float:
        frame_index = int(t * sr / hop_length)
        frame_index = min(frame_index, len(rms_scales) - 1)
        return (
            base_animation_func(t)
            * rms_scales[frame_index]
            * beat_scales[frame_index]
        )

    image_clip = ImageClip(np.array(img), duration=duration)
    animated_image = image_clip.with_position(("center", "center")).resized(
        final_scale_func
    )
    background = ColorClip(size=(W, H), color=(0, 0, 0), duration=duration)
    final_clip = CompositeVideoClip([background, animated_image])
    final_clip = final_clip.with_audio(audio_clip)
    final_clip.write_videofile(
        output_path,
        fps=20,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=cores(),
    )
    return output_path
