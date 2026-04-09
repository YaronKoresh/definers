from __future__ import annotations

from pathlib import Path

from definers.system import catch, delete, tmp

from .io import save_audio
from .utils import stretch_audio


def change_audio_speed(
    audio_path: str,
    speed_factor: float,
    preserve_pitch: bool,
    format_choice: str,
):
    import pydub

    from definers.system.output_paths import managed_output_path

    sound_out = None
    if preserve_pitch:
        audio_path_out = tmp(Path(audio_path).suffix)
        stretched = stretch_audio(audio_path, audio_path_out, speed_factor)
        if stretched:
            sound_out = pydub.AudioSegment.from_file(audio_path_out)
            delete(audio_path_out)
        else:
            catch("Failed to stretch audio while preserving pitch.")
            return None
    else:
        sound = pydub.AudioSegment.from_file(audio_path)
        new_frame_rate = int(sound.frame_rate * speed_factor)
        sound_out = sound._spawn(
            sound.raw_data,
            overrides={"frame_rate": new_frame_rate},
        ).set_frame_rate(sound.frame_rate)
    if sound_out:
        output_stem = managed_output_path(
            format_choice,
            section="audio",
            stem=f"{Path(audio_path).stem}_speed_{speed_factor}x",
        )
        return save_audio(
            destination_path=output_stem,
            audio_signal=sound_out,
            sample_rate=24000,
        )
    catch("Could not process audio speed change.")
    return None
