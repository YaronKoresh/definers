from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from definers.logger import init_logger
from definers.system import catch, log, run, tmp

_logger = init_logger()


def normalize_audio_to_peak(
    input_path: str,
    target_level: float = 0.9,
    format: str | None = None,
    output_path: str | None = None,
) -> str | None:
    from pydub import AudioSegment

    if not 0.0 < target_level <= 1.0:
        catch("target_level must be between 0.0 and 1.0")
        return None

    if format is None:
        format = Path(input_path).suffix.lstrip(".") or "wav"

    resolved_output_path = output_path or tmp(format)

    try:
        audio = AudioSegment.from_file(input_path)
    except FileNotFoundError:
        catch(f"Input file not found at {input_path}")
        return None

    if target_level == 0.0 or audio.max_dBFS == -float("inf"):
        silent_audio = AudioSegment.silent(duration=len(audio))
        silent_audio.export(resolved_output_path, format=format)
        log(
            "Exported silent file"
            f"Silent audio detected or target level is 0. Saved silent file to '{resolved_output_path}'"
        )
        return resolved_output_path

    target_dbfs = 20 * math.log10(target_level)
    gain_to_apply = target_dbfs - audio.max_dBFS
    normalized_audio = audio.apply_gain(gain_to_apply)
    normalized_audio.export(resolved_output_path, format=format)
    log(
        f"Successfully normalized '{input_path}' to a peak of {target_dbfs:.2f} dBFS."
    )
    _logger.info("Saved result to '%s'", resolved_output_path)
    return resolved_output_path


def stretch_audio(
    input_path: str,
    output_path: str | None = None,
    speed_factor: float = 0.85,
    *,
    normalize_audio_to_peak_func: Callable[[str], str | None] | None = None,
    run_command: Callable[[list[str]], object] | None = None,
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

    normalize_audio = normalize_audio_to_peak_func or normalize_audio_to_peak
    execute_command = run_command or run

    try:
        execute_command(command)
        return normalize_audio(output_path)
    except Exception as error:
        catch(f"Error during audio stretching with rubberband: {error}")
        return None
