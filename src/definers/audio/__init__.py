
from __future__ import annotations

from definers.system import catch, cores, delete, exist, full_path, get_ext, log, run, tmp

from .analysis import (
    analyze_audio,
    analyze_audio_features,
    beat_visualizer,
    detect_silence_mask,
    get_active_audio_timeline,
)
from .dsp import decoupled_envelope, limiter_smooth_env, process_audio_chunks, resample
from .effects.exciter import apply_exciter, calculate_dynamic_cutoff
from .effects.mixing import dj_mix, mix_audio, pad_audio, stereo
from .filters import freq_cut
from .features import (
    extract_audio_features,
    features_to_audio,
    predict_audio,
)
from .io import (
    compact_audio,
    export_to_pkl,
    read_audio,
    remove_silence,
    save_audio,
    split_audio,
    write_mp3,
)
from .config import SmartMasteringConfig
from .mastering import SmartMastering, generate_bands, master
from .preview import audio_preview, get_audio_duration
from .production import (
    audio_to_midi,
    autotune_song,
    change_audio_speed,
    create_spectrum_visualization,
    extend_audio,
    generate_music,
    generate_voice,
    humanize_vocals,
    identify_instruments,
    midi_to_audio,
    pitch_shift_vocals,
    separate_stems,
    stem_mixer,
    transcribe_audio,
    value_to_keys,
)
from .sharing import create_share_links
from .utils import (
    apply_compressor,
    calculate_active_rms,
    compute_gain_envelope,
    create_sample_audio,
    get_scale_notes,
    loudness_maximizer,
    normalize_audio_to_peak,
    riaa_filter,
    stretch_audio,
    subdivide_beats,
)
from .feedback import get_audio_feedback, get_color_palette

__all__ = [
    "catch",
    "cores",
    "delete",
    "exist",
    "full_path",
    "get_ext",
    "log",
    "run",
    "tmp",
    "analyze_audio",
    "analyze_audio_features",
    "beat_visualizer",
    "apply_exciter",
    "calculate_dynamic_cutoff",
    "decoupled_envelope",
    "detect_silence_mask",
    "dj_mix",
    "mix_audio",
    "pad_audio",
    "stereo",
    "limiter_smooth_env",
    "extract_audio_features",
    "freq_cut",
    "extend_audio",
    "features_to_audio",
    "get_active_audio_timeline",
    "audio_preview",
    "get_audio_duration",
    "get_scale_notes",
    "generate_bands",
    "generate_music",
    "generate_voice",
    "get_audio_feedback",
    "get_color_palette",
    "humanize_vocals",
    "loudness_maximizer",
    "identify_instruments",
    "master",
    "normalize_audio_to_peak",
    "predict_audio",
    "pitch_shift_vocals",
    "process_audio_chunks",
    "read_audio",
    "remove_silence",
    "resample",
    "riaa_filter",
    "save_audio",
    "separate_stems",
    "split_audio",
    "stem_mixer",
    "stretch_audio",
    "transcribe_audio",
    "autotune_song",
    "change_audio_speed",
    "compact_audio",
    "compute_gain_envelope",
    "calculate_active_rms",
    "create_sample_audio",
    "create_share_links",
    "create_spectrum_visualization",
    "export_to_pkl",
    "midi_to_audio",
    "apply_compressor",
    "value_to_keys",
    "subdivide_beats",
    "write_mp3",
    "SmartMasteringConfig",
    "SmartMastering",
]
