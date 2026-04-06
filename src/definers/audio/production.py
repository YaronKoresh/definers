from __future__ import annotations

from .editing import change_audio_speed
from .music_generation import (
    audio_to_midi,
    extend_audio,
    generate_music,
    midi_to_audio,
)
from .spectrum_visualization import create_spectrum_visualization
from .stems import identify_instruments, separate_stems, stem_mixer
from .voice import (
    autotune_song,
    generate_voice,
    humanize_vocals,
    pitch_shift_vocals,
    transcribe_audio,
    value_to_keys,
)

__all__ = [
    "audio_to_midi",
    "autotune_song",
    "change_audio_speed",
    "create_spectrum_visualization",
    "extend_audio",
    "generate_music",
    "generate_voice",
    "humanize_vocals",
    "identify_instruments",
    "midi_to_audio",
    "pitch_shift_vocals",
    "separate_stems",
    "stem_mixer",
    "transcribe_audio",
    "value_to_keys",
]
