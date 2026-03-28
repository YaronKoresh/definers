from __future__ import annotations


def _module_exports(module_name: str, *names: str) -> dict[str, str]:
    return {name: module_name for name in names}


AUDIO_EXPORTS = {
    **_module_exports(
        "analysis",
        "analyze_audio",
        "analyze_audio_features",
        "beat_visualizer",
        "detect_silence_mask",
        "get_active_audio_timeline",
    ),
    **_module_exports("config", "SmartMasteringConfig"),
    **_module_exports(
        "dsp",
        "decoupled_envelope",
        "limiter_smooth_env",
        "process_audio_chunks",
        "remove_spectral_spikes",
        "resample",
    ),
    **_module_exports("effects.exciter", "apply_exciter"),
    **_module_exports(
        "effects.mixing",
        "dj_mix",
        "mix_audio",
        "pad_audio",
        "stereo",
    ),
    **_module_exports(
        "features",
        "extract_audio_features",
        "features_to_audio",
        "predict_audio",
    ),
    **_module_exports(
        "feedback",
        "get_audio_feedback",
        "get_color_palette",
    ),
    **_module_exports("filters", "freq_cut"),
    **_module_exports(
        "io",
        "compact_audio",
        "export_to_pkl",
        "is_audio_segment",
        "read_audio",
        "remove_silence",
        "save_audio",
        "split_audio",
    ),
    **_module_exports(
        "mastering",
        "SmartMastering",
        "audio_eq",
        "master",
    ),
    **_module_exports("preview", "audio_preview", "get_audio_duration"),
    **_module_exports(
        "production",
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
    ),
    **_module_exports("sharing", "create_share_links"),
    **_module_exports(
        "utils",
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
    ),
}

__all__ = tuple(AUDIO_EXPORTS)