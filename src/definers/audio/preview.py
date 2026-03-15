from __future__ import annotations

from definers.logger import init_logger
from definers.system import catch

from .analysis import get_active_audio_timeline
from .io import read_audio, split_audio

_logger = init_logger()


def get_audio_duration(file_path: str) -> float | None:
    try:
        sr, audio = read_audio(file_path)
        return len(audio) / sr
    except Exception as error:
        catch(error)
        return None


def audio_preview(file_path: str, max_duration: float = 30) -> str | None:
    from definers.system import exist, full_path

    file_path = full_path(file_path)
    if not exist(file_path):
        catch(f"Error: Audio file not found at {file_path}")
        return None
    if max_duration <= 0:
        catch("Error: max_duration must be positive.")
        return None

    try:
        total_duration = get_audio_duration(file_path)
        if total_duration is None:
            catch(f"Error: Could not get duration for {file_path}")
            return None

        _logger.debug("Total audio duration: %s seconds", total_duration)

        if total_duration <= max_duration:
            _logger.debug(
                "Audio duration <= max_duration: returning original copy"
            )
            preview_paths = split_audio(
                file_path,
                chunk_duration=total_duration,
                chunks_limit=1,
                skip_time=0,
            )
            return preview_paths[0] if preview_paths else None

        start_time = 0.0
        timeline = get_active_audio_timeline(
            file_path, threshold_db=-25, min_silence_len=0.5
        )
        if timeline:
            longest_segment_duration = 0.0
            longest_segment_center = 0.0
            for start, end in timeline:
                duration = end - start
                if duration > longest_segment_duration:
                    longest_segment_duration = duration
                    longest_segment_center = start + duration / 2.0
            _logger.debug(
                "Longest active segment duration: %s, center: %s",
                longest_segment_duration,
                longest_segment_center,
            )
            ideal_start = longest_segment_center - max_duration / 2.0
            start_time = max(0.0, ideal_start)
            start_time = min(start_time, total_duration - max_duration)
            _logger.debug(
                "Calculated preview start time: %s seconds", start_time
            )
        else:
            start_time = min(
                total_duration * 0.1, total_duration - max_duration
            )
            start_time = max(0.0, start_time)
            _logger.debug(
                "No significant active segments found; defaulting preview start time to %s seconds",
                start_time,
            )

        _logger.debug(
            "Extracting preview chunk from %s to %s seconds",
            start_time,
            start_time + max_duration,
        )
        preview_paths = split_audio(
            file_path,
            chunk_duration=max_duration,
            chunks_limit=1,
            skip_time=start_time,
        )
        if preview_paths:
            _logger.debug("Preview extraction successful: %s", preview_paths[0])
            return preview_paths[0]
        else:
            catch(
                "Error: split_audio did not return any paths for the preview."
            )
            return None
    except Exception as e:
        catch(f"An unexpected error occurred in audio_preview: {e}")
        return None
