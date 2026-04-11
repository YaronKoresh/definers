from __future__ import annotations

import os

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

import definers.file_ops as file_ops
from definers.system import get_ext
from definers.system.filesystem import exist
from definers.system.paths import full_path, tmp


def is_audio_segment(audio_signal) -> bool:
    return hasattr(audio_signal, "export") and hasattr(
        audio_signal, "frame_rate"
    )


def _resolve_wav_subtype(bit_depth: int) -> str:
    normalized_bit_depth = int(bit_depth)
    if normalized_bit_depth >= 64:
        return "DOUBLE"
    if normalized_bit_depth >= 32:
        return "FLOAT"
    if normalized_bit_depth >= 24:
        return "PCM_24"
    if normalized_bit_depth >= 16:
        return "PCM_16"
    return "PCM_U8"


def _read_audio_with_soundfile(
    audio_file: str,
) -> tuple[int, np.ndarray] | None:
    import soundfile as sf

    extension = str(get_ext(audio_file)).strip().lower()
    if extension not in {"wav", "flac", "aif", "aiff"}:
        return None
    audio_data, sample_rate = sf.read(
        audio_file,
        dtype="float32",
        always_2d=True,
    )
    return int(sample_rate), np.asarray(audio_data, dtype=np.float32).T


def _write_wav_audio(
    soundfile_module,
    destination_path: str,
    audio_signal: np.ndarray,
    sample_rate: int,
    *,
    subtype: str,
) -> None:
    try:
        soundfile_module.write(
            destination_path,
            audio_signal,
            sample_rate,
            subtype=subtype,
            format="WAV",
        )
    except Exception:
        soundfile_module.write(
            destination_path,
            audio_signal,
            sample_rate,
            subtype=subtype,
            format="RF64",
        )


def run_ffmpeg(command: list[str]):
    import subprocess

    from definers.system import install_ffmpeg

    install_ffmpeg()

    try:
        subprocess.run(command, check=True)
        return command[-1]
    except subprocess.CalledProcessError as exception:
        file_ops.catch(exception)
        return None


def read_audio(audio_file: str) -> tuple[int, np.ndarray]:
    audio_file = full_path(audio_file)
    if not exist(audio_file):
        raise FileNotFoundError(audio_file)

    try:
        soundfile_result = _read_audio_with_soundfile(audio_file)
    except Exception:
        soundfile_result = None
    if soundfile_result is not None:
        return soundfile_result

    import pydub

    from definers.system import install_ffmpeg

    install_ffmpeg()

    audio_segment = pydub.AudioSegment.from_file(audio_file)
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    audio_data = samples.reshape((-1, audio_segment.channels)).T

    sample_width = max(int(getattr(audio_segment, "sample_width", 2)), 1)
    normalization_scale = float(1 << (sample_width * 8 - 1))

    return audio_segment.frame_rate, audio_data / normalization_scale


def save_audio(
    destination_path: str,
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    *,
    bit_depth: int = 32,
    bitrate: int = 320,
    compression_level: int = 9,
) -> None:
    import pydub
    import soundfile as sf

    from definers.system import install_ffmpeg

    install_ffmpeg()

    base_path = os.path.splitext(str(destination_path))[0]
    ext = get_ext(destination_path)
    final_path = f"{base_path}.{ext}"

    lossy_params = [
        "-compression_level",
        str(compression_level),
        "-b:a",
        f"{bitrate}k",
    ]

    if is_audio_segment(audio_signal):
        if ext == "wav":
            audio_signal.export(final_path, format=ext)
        else:
            audio_signal.export(
                final_path,
                format=ext,
                bitrate=f"{bitrate}k",
                parameters=lossy_params,
            )
        return final_path

    if audio_signal.ndim == 2 and audio_signal.shape[0] < 10:
        audio_signal = audio_signal.T

    if ext == "wav":
        _write_wav_audio(
            sf,
            final_path,
            audio_signal,
            sample_rate,
            subtype=_resolve_wav_subtype(bit_depth),
        )

    else:
        import io

        buffer = io.BytesIO()
        _write_wav_audio(
            sf,
            buffer,
            audio_signal,
            sample_rate,
            subtype=_resolve_wav_subtype(bit_depth),
        )
        buffer.seek(0)
        song = pydub.AudioSegment.from_wav(buffer)
        song.export(
            final_path,
            format=ext,
            bitrate=f"{bitrate}k",
            parameters=lossy_params,
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

    from definers.system import install_ffmpeg

    install_ffmpeg()

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
    num_chunks = (
        min(chunks_limit, max_chunks)
        if chunks_limit is not None
        else max_chunks
    )

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
    return run_ffmpeg(
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
    return run_ffmpeg(
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
