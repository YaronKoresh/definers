from __future__ import annotations

from definers.system import run

try:
    from .file_processing import (
        normalize_audio_to_peak as _normalize_audio_to_peak,
        stretch_audio as _stretch_audio,
    )
    from .music_theory import (
        create_sample_audio,
        generate_bands,
        get_scale_notes,
        subdivide_beats,
    )
    from .normalization import (
        adjust_final_output_lufs,
        apply_lufs,
        apply_rms,
        calculate_active_rms,
        get_lufs,
        get_rms,
    )
    from .signal_effects import (
        apply_compressor,
        compute_gain_envelope,
        loudness_maximizer,
        riaa_filter,
        stereo_widen,
    )
except ImportError:
    from definers.audio.file_processing import (
        normalize_audio_to_peak as _normalize_audio_to_peak,
        stretch_audio as _stretch_audio,
    )
    from definers.audio.music_theory import (
        create_sample_audio,
        generate_bands,
        get_scale_notes,
        subdivide_beats,
    )
    from definers.audio.normalization import (
        adjust_final_output_lufs,
        apply_lufs,
        apply_rms,
        calculate_active_rms,
        get_lufs,
        get_rms,
    )
    from definers.audio.signal_effects import (
        apply_compressor,
        compute_gain_envelope,
        loudness_maximizer,
        riaa_filter,
        stereo_widen,
    )

normalize_audio_to_peak = _normalize_audio_to_peak


def stretch_audio(
    input_path: str, output_path: str | None = None, speed_factor: float = 0.85
) -> str | None:
    return _stretch_audio(
        input_path,
        output_path=output_path,
        speed_factor=speed_factor,
        normalize_audio_to_peak_func=normalize_audio_to_peak,
        run_command=run,
    )


__all__ = [
    "adjust_final_output_lufs",
    "apply_compressor",
    "apply_lufs",
    "apply_rms",
    "calculate_active_rms",
    "compute_gain_envelope",
    "create_sample_audio",
    "generate_bands",
    "get_lufs",
    "get_rms",
    "get_scale_notes",
    "loudness_maximizer",
    "normalize_audio_to_peak",
    "riaa_filter",
    "stereo_widen",
    "stretch_audio",
    "subdivide_beats",
]
