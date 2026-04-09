import definers.text.system_messages as system_messages
import definers.text.text_transforms as text_transforms
import definers.text.translation as translation
from definers import ml as ml_facade
from definers.audio import (
    editing,
    music_generation,
    production,
    spectrum_visualization,
    stems,
    voice,
)
from definers.audio.mastering import master
from definers.audio.mastering.character import (
    LimiterRecoverySettings,
    apply_low_end_mono_tightening,
    apply_micro_dynamics_finish,
    resolve_limiter_recovery_settings,
)
from definers.audio.mastering.contract import (
    MasteringContract,
    MasteringContractAssessment,
    assess_mastering_contract,
    resolve_mastering_contract,
)
from definers.audio.mastering.delivery import (
    DeliveryProfile,
    DeliveryVerificationResult,
    resolve_delivery_profile,
    save_verified_audio,
    verify_delivery_export,
)
from definers.audio.mastering.finalization import (
    FinalizationAction,
    apply_delivery_trim,
    apply_pre_limiter_saturation,
    apply_stereo_width_restraint,
    compute_dynamic_drive,
    compute_primary_soft_clip_ratio,
    plan_follow_up_action,
    resolve_final_true_peak_target,
)
from definers.audio.mastering.loudness import (
    MasteringLoudnessMetrics,
    measure_low_end_mono_ratio,
    measure_mastering_loudness,
    measure_stereo_width,
    measure_true_peak,
)
from definers.audio.mastering.metrics import (
    MasteringReport,
    generate_mastering_report,
)
from definers.audio.mastering.presets import (
    MasteringPresets,
    balanced,
    edm,
    vocal,
)
from definers.audio.mastering.profile import SpectralBalanceProfile
from definers.audio.mastering.reference import (
    ReferenceAnalysis,
    ReferenceMatchAssist,
    analyze_reference,
    reference_match_assist,
)
from definers.audio.mastering.stems import (
    StemMasteringPlan,
    mix_stem_layers,
    process_stem_layers,
    resolve_stem_mastering_plan,
)
from definers.ml.health_api import get_ml_health_snapshot, ml_health_markdown
from definers.ml.regression_api import (
    linear_regression,
    predict_linear_regression,
)
from definers.ml.text.api import map_reduce_summary, optimize_prompt_realism
from definers.system import compress, extract, get_ext, secure_path
from definers.system.installation import apt_install, install_ffmpeg


def test_audio_production_facade_reexports_specific_modules():
    assert production.value_to_keys is voice.value_to_keys
    assert production.generate_music is music_generation.generate_music
    assert production.change_audio_speed is editing.change_audio_speed
    assert production.separate_stems is stems.separate_stems
    assert (
        production.create_spectrum_visualization
        is spectrum_visualization.create_spectrum_visualization
    )


def test_audio_mastering_facade_reexports_reporting_modules():
    import definers.audio as audio_facade

    assert audio_facade.measure_mastering_loudness is measure_mastering_loudness
    assert audio_facade.measure_true_peak is measure_true_peak
    assert audio_facade.generate_mastering_report is generate_mastering_report
    assert audio_facade.balanced is balanced
    assert audio_facade.edm is edm
    assert audio_facade.vocal is vocal
    assert audio_facade.resolve_delivery_profile is resolve_delivery_profile
    assert audio_facade.save_verified_audio is save_verified_audio
    assert audio_facade.compute_dynamic_drive is compute_dynamic_drive
    assert audio_facade.apply_delivery_trim is apply_delivery_trim
    assert audio_facade.resolve_mastering_contract is resolve_mastering_contract
    assert audio_facade.measure_stereo_width is measure_stereo_width
    assert audio_facade.reference_match_assist is reference_match_assist
    assert audio_facade.master is master
    assert (
        audio_facade.apply_micro_dynamics_finish is apply_micro_dynamics_finish
    )


def test_application_text_modules_expose_direct_exports():
    assert callable(translation.ai_translate)
    assert callable(text_transforms.simple_text)
    assert callable(text_transforms.camel_case)
    assert callable(system_messages.set_system_message)
    assert callable(translation.translate_with_code_using)


def test_ml_facade_reexports_specific_modules():
    assert ml_facade.map_reduce_summary is map_reduce_summary
    assert ml_facade.optimize_prompt_realism is optimize_prompt_realism
    assert ml_facade.linear_regression is linear_regression
    assert ml_facade.predict_linear_regression is predict_linear_regression
    assert ml_facade.get_ml_health_snapshot is get_ml_health_snapshot
    assert ml_facade.ml_health_markdown is ml_health_markdown


def test_split_modules_are_directly_importable():
    profile = SpectralBalanceProfile(
        rescue_factor=0.1,
        correction_strength=0.2,
        max_boost_db=1.0,
        max_cut_db=0.5,
        band_intensity=1.0,
    )

    assert profile.rescue_factor == 0.1
    assert callable(compress)
    assert callable(extract)
    assert callable(get_ext)
    assert callable(secure_path)
    assert callable(install_ffmpeg)
    assert callable(apt_install)
    assert callable(map_reduce_summary)
    assert callable(linear_regression)
    assert callable(get_ml_health_snapshot)
    assert MasteringLoudnessMetrics.__name__ == "MasteringLoudnessMetrics"
    assert MasteringContract.__name__ == "MasteringContract"
    assert MasteringContractAssessment.__name__ == "MasteringContractAssessment"
    assert LimiterRecoverySettings.__name__ == "LimiterRecoverySettings"
    assert MasteringReport.__name__ == "MasteringReport"
    assert FinalizationAction.__name__ == "FinalizationAction"
    assert ReferenceAnalysis.__name__ == "ReferenceAnalysis"
    assert ReferenceMatchAssist.__name__ == "ReferenceMatchAssist"
    assert StemMasteringPlan.__name__ == "StemMasteringPlan"
    assert DeliveryProfile.__name__ == "DeliveryProfile"
    assert DeliveryVerificationResult.__name__ == "DeliveryVerificationResult"
    assert callable(measure_mastering_loudness)
    assert callable(measure_low_end_mono_ratio)
    assert callable(measure_stereo_width)
    assert callable(generate_mastering_report)
    assert callable(MasteringPresets.balanced)
    assert callable(MasteringPresets.edm)
    assert callable(MasteringPresets.vocal)
    assert callable(verify_delivery_export)
    assert callable(assess_mastering_contract)
    assert callable(resolve_limiter_recovery_settings)
    assert callable(apply_low_end_mono_tightening)
    assert callable(apply_micro_dynamics_finish)
    assert callable(apply_pre_limiter_saturation)
    assert callable(apply_stereo_width_restraint)
    assert callable(compute_primary_soft_clip_ratio)
    assert callable(plan_follow_up_action)
    assert callable(resolve_final_true_peak_target)
    assert callable(analyze_reference)
    assert callable(reference_match_assist)
    assert callable(resolve_stem_mastering_plan)
    assert callable(process_stem_layers)
    assert callable(mix_stem_layers)
