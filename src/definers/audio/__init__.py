from __future__ import annotations

from .analysis import (
    analyze_audio,
    analyze_audio_features,
    beat_visualizer,
    detect_silence_mask,
    get_active_audio_timeline,
)
from .config import SmartMasteringConfig
from .dsp import (
    decoupled_envelope,
    limiter_smooth_env,
    process_audio_chunks,
    remove_spectral_spikes,
    resample,
)
from .editing import change_audio_speed
from .effects.exciter import apply_exciter
from .effects.mixing import dj_mix, mix_audio, pad_audio, stereo
from .features import extract_audio_features, features_to_audio, predict_audio
from .feedback import get_audio_feedback, get_color_palette
from .file_processing import normalize_audio_to_peak, stretch_audio
from .filters import freq_cut
from .io import (
    compact_audio,
    export_to_pkl,
    is_audio_segment,
    read_audio,
    remove_silence,
    save_audio,
    split_audio,
)
from .mastering.character import (
    LimiterRecoverySettings,
    apply_low_end_mono_tightening,
    apply_micro_dynamics_finish,
    resolve_limiter_recovery_settings,
    resolve_low_end_mono_tightening_amount,
)
from .mastering.contract import (
    MasteringContract,
    MasteringContractAssessment,
    assess_mastering_contract,
    resolve_mastering_contract,
)
from .mastering.delivery import (
    DeliveryProfile,
    DeliveryVerificationResult,
    resolve_delivery_profile,
    save_verified_audio,
    verify_delivery_export,
)
from .mastering.engine import SmartMastering, master
from .mastering.eq import audio_eq
from .mastering.finalization import (
    CharacterStageDecision,
    FinalizationAction,
    PeakCatchEvent,
    apply_delivery_trim,
    apply_pre_limiter_saturation,
    apply_stereo_width_restraint,
    compute_dynamic_drive,
    compute_primary_soft_clip_ratio,
    plan_follow_up_action,
    resolve_final_true_peak_target,
)
from .mastering.loudness import (
    MasteringLoudnessMetrics,
    measure_low_end_mono_ratio,
    measure_mastering_loudness,
    measure_sample_peak,
    measure_stereo_width,
    measure_true_peak,
)
from .mastering.metrics import (
    MasteringReport,
    generate_mastering_report,
    write_mastering_report,
)
from .mastering.presets import (
    MasteringPresets,
    balanced,
    edm,
    mastering_preset,
    vocal,
)
from .mastering.reference import (
    ReferenceAnalysis,
    ReferenceMatchAssist,
    analyze_reference,
    measure_spectral_tilt,
    measure_transient_density,
    reference_match_assist,
)
from .mastering.stems import (
    StemMasteringPlan,
    mix_stem_layers,
    process_stem_layers,
    resolve_stem_mastering_plan,
)
from .music_generation import (
    audio_to_midi,
    extend_audio,
    generate_music,
    midi_to_audio,
)
from .music_theory import (
    create_sample_audio,
    generate_bands,
    get_scale_notes,
    subdivide_beats,
)
from .normalization import (
    apply_lufs,
    apply_rms,
    calculate_active_rms,
    get_lufs,
    get_rms,
)
from .preview import audio_preview, get_audio_duration
from .sharing import create_share_links
from .signal_effects import (
    apply_compressor,
    compute_gain_envelope,
    loudness_maximizer,
    riaa_filter,
    stereo_widen,
)
from .spectrum_visualization import create_spectrum_visualization
from .stems import (
    identify_instruments,
    separate_stem_layers,
    separate_stems,
    stem_mixer,
)
from .utils import adjust_final_output_lufs
from .voice import (
    autotune_song,
    generate_voice,
    humanize_vocals,
    pitch_shift_vocals,
    transcribe_audio,
    value_to_keys,
)

__all__ = [glb for glb in globals() if not glb.startswith("_")]
