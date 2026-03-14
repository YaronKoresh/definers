
from __future__ import annotations

import os

import numpy as np

import definers.file_ops as file_ops
from definers.platform.filesystem import exist
from definers.platform.paths import full_path, tmp


def _is_audio_segment(audio_signal) -> bool:
    return hasattr(audio_signal, "export") and hasattr(audio_signal, "frame_rate")


def _run_ffmpeg(command: list[str]):
    import subprocess

    try:
        subprocess.run(command, check=True)
        return command[-1]
    except subprocess.CalledProcessError as exception:
        file_ops.catch(exception)
        return None


def read_audio(audio_file, normalized: bool = False) -> tuple[int, np.ndarray]:
    import pydub

    audio_segment = pydub.AudioSegment.from_file(audio_file)
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        audio_data = samples.reshape((-1, 2)).T
    else:
        audio_data = samples.reshape((1, -1))
    if normalized:
        return (audio_segment.frame_rate, np.float32(audio_data) / 32768.0)
    else:
        return (audio_segment.frame_rate, audio_data)


def write_mp3(file_path: str, sr: int, audio_data: np.ndarray) -> None:
    import pydub

    if audio_data.ndim == 1:
        channels = 1
    else:
        channels = audio_data.shape[0]
    y = np.int8(audio_data * 128.0 / 2 + 128.0 + (audio_data * 128.0 / 2 - 128.0))
    interleaved_data = np.ascontiguousarray(y.T)
    song = pydub.AudioSegment(
        interleaved_data.tobytes(),
        frame_rate=sr,
        sample_width=1,
        channels=channels,
    )
    song.export(file_path, format="mp3", bitrate="320k")


def save_audio(
    destination_path: str,
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    output_format: str = "mp3",
    *,
    audio_bit_depth: int = 32,
    audio_bitrate: int = 320,
) -> None:
    import lameenc
    import soundfile as sf

    base_path = os.path.splitext(str(destination_path))[0]
    final_path = f"{base_path}.mp3" if output_format.lower() == "mp3" else f"{base_path}.wav"

    if _is_audio_segment(audio_signal):
        export_kwargs = {"format": output_format.lower()}
        if output_format.lower() == "mp3":
            export_kwargs["bitrate"] = f"{audio_bitrate}k"
        audio_signal.export(final_path, **export_kwargs)
        return final_path

    ceiling_db = -0.1
    lin_amp = 10 ** (ceiling_db / 20.0)

    audio_signal = np.clip(audio_signal * lin_amp, -lin_amp, lin_amp)

    if output_format.lower() == "mp3":
        INT16_MAX = np.iinfo(np.int16).max
        y_scaled = (audio_signal * INT16_MAX).astype(np.int16)

        if y_scaled.ndim == 1:
            channels = 1
            y_interleaved = y_scaled
        else:
            y_interleaved = np.ascontiguousarray(
                y_scaled.T if y_scaled.shape[0] < y_scaled.shape[1] else y_scaled
            )
            channels = y_interleaved.shape[1]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(audio_bitrate)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(channels)
        encoder.set_quality(2)

        mp3_data = encoder.encode(y_interleaved.tobytes())
        mp3_data += encoder.flush()

        with open(final_path, "wb") as f:
            f.write(mp3_data)
        return final_path

    if audio_bit_depth == 16:
        INT16_MIN = np.iinfo(np.int16).min
        INT16_MAX = np.iinfo(np.int16).max
        y_scaled = audio_signal * float(INT16_MAX)

        tpdf = (
            np.random.uniform(-1.0, 1.0, y_scaled.shape)
            + np.random.uniform(-1.0, 1.0, y_scaled.shape)
        ) * 0.5

        y_dithered = y_scaled + tpdf
        audio_signal = np.clip(np.rint(y_dithered), INT16_MIN, INT16_MAX).astype(
            np.int16
        )

    elif audio_bit_depth == 32:
        audio_signal = audio_signal.astype(np.float32)

    elif audio_bit_depth == 64:
        audio_signal = audio_signal.astype(np.float64)

    else:
        raise ValueError(f"Unsupported bit depth: {audio_bit_depth}. Use 16, 32, or 64.")

    if audio_signal.ndim == 2 and audio_signal.shape[0] < audio_signal.shape[1]:
        audio_signal = audio_signal.T

    subtype_map = {16: "PCM_16", 32: "FLOAT", 64: "DOUBLE"}
    sf.write(
        final_path,
        audio_signal,
        sample_rate,
        subtype=subtype_map[audio_bit_depth],
        format="RF64",
    )
    return final_path


def split_audio(
    audio_file_path: str,
    chunk_duration: float = 5.0,
    audio_format: str = "mp3",
    *,
    chunks_limit: int | None = None,
    skip_time: float = 0.0,
    target_sample_rate: int | None = None,
    output_folder: str | None = None,
) -> list[str]:
    import math
    import os
    import pydub

    audio_file_path = full_path(audio_file_path)
    if not exist(audio_file_path):
        return []

    try:
        audio_segment = pydub.AudioSegment.from_file(audio_file_path)
    except Exception:
        return []

    total_ms = len(audio_segment)
    skip_ms = int(skip_time * 1000.0)
    if skip_ms >= total_ms:
        return []

    duration_ms = int(chunk_duration * 1000.0)
    if duration_ms <= 0:
        return []

    remaining_ms = total_ms - skip_ms
    max_chunks = math.ceil(remaining_ms / duration_ms)
    num_chunks = min(chunks_limit, max_chunks) if chunks_limit is not None else max_chunks

    if output_folder:
        output_folder = full_path(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = tmp(dir=True)

    res_paths: list[str] = []
    for i in range(num_chunks):
        start_ms = skip_ms + i * duration_ms
        if start_ms >= total_ms:
            break
        end_ms = min(start_ms + duration_ms, total_ms)
        chunk = audio_segment[start_ms:end_ms]
        if target_sample_rate:
            chunk = chunk.set_frame_rate(target_sample_rate)
        chunk_path = full_path(output_folder, f"chunk_{i:04d}.{audio_format}")
        export_kwargs = {}
        if audio_format.lower() == "mp3":
            export_kwargs["bitrate"] = "192k"
        chunk.export(chunk_path, format=audio_format, **export_kwargs)
        res_paths.append(chunk_path)

    return res_paths


def remove_silence(input_file: str, output_file: str):
    return _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ac",
            "2",
            "-af",
            "silenceremove=stop_duration=0.1:stop_threshold=-32dB",
            output_file,
        ]
    )


def compact_audio(input_file: str, output_file: str):
    return _run_ffmpeg(
        [
            "ffmpeg",
            "-y",
            "-i",
            input_file,
            "-ar",
            "32000",
            "-ab",
            "96k",
            "-ac",
            "1",
            output_file,
        ]
    )


def export_to_pkl(model, pkl_path: str):
    import pickle

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
