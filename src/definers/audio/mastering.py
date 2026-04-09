from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from definers.system import get_ext, tmp

from ..file_ops import log
from .config import SmartMasteringConfig
from .dsp import (
    decoupled_envelope,
    limiter_smooth_env,
    resample,
)
from .effects.exciter import apply_exciter
from .effects.mixing import stereo
from .filters import freq_cut
from .mastering_analysis import measure_spectrum as _measure_spectrum
from .mastering_character import (
    LimiterRecoverySettings,
    apply_low_end_mono_tightening as _apply_low_end_mono_tightening,
    apply_micro_dynamics_finish as _apply_micro_dynamics_finish,
    resolve_limiter_recovery_settings as _resolve_limiter_recovery_settings,
)
from .mastering_contract import (
    MasteringContract,
    assess_mastering_contract as _assess_mastering_contract,
    resolve_mastering_contract as _resolve_mastering_contract,
)
from .mastering_delivery import save_verified_audio
from .mastering_dynamics import (
    apply_limiter as _apply_limiter,
    apply_pre_limiter_saturation as _apply_pre_limiter_saturation,
    apply_safety_clamp as _apply_safety_clamp,
    apply_spatial_enhancement as _apply_spatial_enhancement,
    multiband_compress as _multiband_compress,
)
from .mastering_eq import (
    apply_eq as _apply_eq,
    apply_stem_cleanup as _apply_stem_cleanup,
    audio_eq,
    smooth_curve as _smooth_curve,
)
from .mastering_finalization import (
    apply_delivery_trim as _apply_delivery_trim,
    apply_final_headroom_recovery as _apply_final_headroom_recovery,
    apply_stereo_width_restraint as _apply_stereo_width_restraint,
    compute_dynamic_drive as _compute_dynamic_drive,
    compute_primary_soft_clip_ratio as _compute_primary_soft_clip_ratio,
    plan_follow_up_action as _plan_follow_up_action,
    resolve_final_true_peak_target as _resolve_final_true_peak_target,
)
from .mastering_loudness import (
    measure_mastering_loudness as _measure_mastering_loudness,
    measure_true_peak as _measure_true_peak,
)
from .mastering_metrics import (
    MasteringReport,
    write_mastering_report as _write_mastering_report,
)
from .mastering_pipeline import (
    process as _process,
    process_stem as _process_stem,
)
from .mastering_profile import (
    SpectralBalanceProfile,
    build_spectral_balance_profile as _build_spectral_balance_profile,
    build_target_curve as _build_target_curve,
    fit_frequency as _fit_frequency,
    plan_follow_up_drives as _plan_follow_up_drives,
    update_bands as _update_bands,
    update_profile as _update_profile,
)
from .mastering_reference import (
    ReferenceAnalysis,
    ReferenceMatchAssist,
    analyze_reference as _analyze_reference,
    measure_spectral_tilt as _measure_spectral_tilt,
    measure_stereo_motion as _measure_stereo_motion,
    measure_transient_density as _measure_transient_density,
    reference_match_assist as _reference_match_assist,
)
from .mastering_state import configure_runtime_state
from .utils import get_lufs


def _sanitize_audio_for_preset_selection(
    signal_to_analyze: np.ndarray,
) -> np.ndarray:
    signal_array = np.nan_to_num(
        np.asarray(signal_to_analyze, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal_array.ndim == 0:
        return signal_array.reshape(1)
    return signal_array


def _measure_preset_band_profile(
    signal_to_analyze: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    signal_array = _sanitize_audio_for_preset_selection(signal_to_analyze)
    mono = (
        np.mean(signal_array, axis=0, dtype=np.float32)
        if signal_array.ndim > 1
        else signal_array
    )
    if mono.size < 32 or sample_rate <= 0:
        return {
            "bass_share": 0.0,
            "low_mid_share": 0.0,
            "presence_share": 0.0,
            "air_share": 0.0,
        }

    analysis_size = int(
        min(
            max(1024, 2 ** int(np.ceil(np.log2(max(mono.size, 64))))),
            65536,
        )
    )
    if mono.size < analysis_size:
        analyzed = np.pad(mono, (0, analysis_size - mono.size))
    else:
        analyzed = mono[:analysis_size]

    window = np.hanning(analyzed.size).astype(np.float32)
    spectrum = np.fft.rfft(analyzed * window)
    power = np.square(np.abs(spectrum), dtype=np.float64)
    freqs = np.fft.rfftfreq(analyzed.size, d=1.0 / float(sample_rate))
    high_limit_hz = float(min(sample_rate / 2.0 - 1.0, 16000.0))
    valid_mask = (freqs >= 35.0) & (freqs <= high_limit_hz)
    total_energy = float(np.sum(power[valid_mask], dtype=np.float64))
    if total_energy <= 1e-12:
        return {
            "bass_share": 0.0,
            "low_mid_share": 0.0,
            "presence_share": 0.0,
            "air_share": 0.0,
        }

    def band_share(low_hz: float, high_hz: float) -> float:
        mask = (freqs >= low_hz) & (freqs < min(high_hz, high_limit_hz))
        if not np.any(mask):
            return 0.0
        return float(np.sum(power[mask], dtype=np.float64) / total_energy)

    return {
        "bass_share": band_share(35.0, 180.0),
        "low_mid_share": band_share(180.0, 900.0),
        "presence_share": band_share(1000.0, 4200.0),
        "air_share": band_share(6000.0, 14000.0),
    }


@dataclass(frozen=True, slots=True)
class MasteringInputMetrics:
    integrated_lufs: float
    crest_factor_db: float
    stereo_width_ratio: float
    low_end_mono_ratio: float
    spectral_tilt: float
    transient_density: float
    stereo_motion: float
    bass_share: float
    low_mid_share: float
    presence_share: float
    air_share: float


@dataclass(frozen=True, slots=True)
class MasteringInputAnalysis:
    preset_name: str
    quality_flags: tuple[str, ...]
    target_sample_rate: int
    metrics: MasteringInputMetrics | None


def _resolve_mastering_processing_sample_rate(input_sample_rate: int) -> int:
    if int(input_sample_rate) >= 48000:
        return 48000
    return 44100


def _collect_mastering_input_metrics(
    signal_to_analyze: np.ndarray,
    sample_rate: int,
) -> MasteringInputMetrics | None:
    signal_array = _sanitize_audio_for_preset_selection(signal_to_analyze)
    if signal_array.size == 0 or sample_rate <= 0:
        return None

    try:
        loudness_metrics = _measure_mastering_loudness(
            signal_array,
            int(sample_rate),
        )
        spectral_tilt = _measure_spectral_tilt(signal_array, int(sample_rate))
        transient_density = _measure_transient_density(
            signal_array,
            int(sample_rate),
        )
        stereo_motion = _measure_stereo_motion(signal_array, int(sample_rate))
        band_profile = _measure_preset_band_profile(
            signal_array, int(sample_rate)
        )
    except Exception:
        return None

    return MasteringInputMetrics(
        integrated_lufs=float(loudness_metrics.integrated_lufs),
        crest_factor_db=float(loudness_metrics.crest_factor_db),
        stereo_width_ratio=float(loudness_metrics.stereo_width_ratio),
        low_end_mono_ratio=float(loudness_metrics.low_end_mono_ratio),
        spectral_tilt=float(spectral_tilt),
        transient_density=float(transient_density),
        stereo_motion=float(stereo_motion),
        bass_share=float(band_profile["bass_share"]),
        low_mid_share=float(band_profile["low_mid_share"]),
        presence_share=float(band_profile["presence_share"]),
        air_share=float(band_profile["air_share"]),
    )


def _select_mastering_preset_from_metrics(
    metrics: MasteringInputMetrics,
) -> str:
    edm_score = float(
        0.3 * np.clip((metrics.bass_share - 0.22) / 0.18, 0.0, 1.0)
        + 0.22 * np.clip((10.0 - metrics.crest_factor_db) / 5.0, 0.0, 1.0)
        + 0.18 * np.clip((metrics.integrated_lufs + 16.0) / 9.0, 0.0, 1.0)
        + 0.15 * np.clip((metrics.transient_density - 0.055) / 0.12, 0.0, 1.0)
        + 0.08 * np.clip((metrics.low_end_mono_ratio - 0.7) / 0.25, 0.0, 1.0)
        + 0.07 * np.clip((0.33 - metrics.stereo_width_ratio) / 0.23, 0.0, 1.0)
    )
    vocal_score = float(
        0.34
        * np.clip(
            (metrics.presence_share - metrics.bass_share * 0.72 - 0.06) / 0.22,
            0.0,
            1.0,
        )
        + 0.18
        * np.clip(
            (metrics.air_share - metrics.bass_share * 0.35 - 0.02) / 0.16,
            0.0,
            1.0,
        )
        + 0.18 * np.clip((metrics.crest_factor_db - 11.5) / 6.0, 0.0, 1.0)
        + 0.12 * np.clip((metrics.stereo_width_ratio - 0.22) / 0.35, 0.0, 1.0)
        + 0.1 * np.clip((metrics.stereo_motion - 0.02) / 0.16, 0.0, 1.0)
        + 0.08 * np.clip((0.095 - metrics.transient_density) / 0.08, 0.0, 1.0)
        + 0.0 * np.clip((metrics.spectral_tilt + 8.0) / 8.0, 0.0, 1.0)
    )
    legacy_vocal_score = float(
        0.24
        * np.clip(
            (
                metrics.low_mid_share
                + metrics.bass_share * 0.45
                - metrics.presence_share
                - metrics.air_share * 0.6
                - 0.12
            )
            / 0.26,
            0.0,
            1.0,
        )
        + 0.2 * np.clip(((-metrics.spectral_tilt) - 5.8) / 4.6, 0.0, 1.0)
        + 0.14 * np.clip((0.075 - metrics.air_share) / 0.065, 0.0, 1.0)
        + 0.12 * np.clip((0.24 - metrics.stereo_width_ratio) / 0.18, 0.0, 1.0)
        + 0.1 * np.clip((0.055 - metrics.stereo_motion) / 0.055, 0.0, 1.0)
        + 0.08 * np.clip((metrics.crest_factor_db - 10.5) / 4.5, 0.0, 1.0)
        + 0.06 * np.clip((0.08 - metrics.transient_density) / 0.065, 0.0, 1.0)
        + 0.06 * np.clip((metrics.low_end_mono_ratio - 0.84) / 0.12, 0.0, 1.0)
    )

    if edm_score >= 0.56 and edm_score >= vocal_score + 0.08:
        return "edm"
    if vocal_score >= 0.56 and vocal_score >= edm_score + 0.08:
        return "vocal"
    if legacy_vocal_score >= 0.54:
        return "vocal"
    return "balanced"


def _resolve_mastering_quality_flags(
    metrics: MasteringInputMetrics,
    input_sample_rate: int,
) -> tuple[str, ...]:
    old_recording_score = float(
        0.24
        * np.clip(
            (
                metrics.low_mid_share
                + metrics.bass_share * 0.45
                - metrics.presence_share
                - metrics.air_share * 0.7
                - 0.12
            )
            / 0.26,
            0.0,
            1.0,
        )
        + 0.2 * np.clip(((-metrics.spectral_tilt) - 5.8) / 4.6, 0.0, 1.0)
        + 0.14 * np.clip((0.075 - metrics.air_share) / 0.065, 0.0, 1.0)
        + 0.14 * np.clip((0.24 - metrics.stereo_width_ratio) / 0.18, 0.0, 1.0)
        + 0.1 * np.clip((0.055 - metrics.stereo_motion) / 0.055, 0.0, 1.0)
        + 0.1 * np.clip((metrics.low_end_mono_ratio - 0.84) / 0.12, 0.0, 1.0)
        + 0.08 * np.clip((metrics.crest_factor_db - 10.5) / 4.5, 0.0, 1.0)
    )
    low_quality_score = float(
        0.2 * np.clip((44100.0 - float(input_sample_rate)) / 22050.0, 0.0, 1.0)
        + 0.18 * np.clip((0.12 - metrics.presence_share) / 0.1, 0.0, 1.0)
        + 0.16 * np.clip((0.045 - metrics.air_share) / 0.045, 0.0, 1.0)
        + 0.15 * np.clip((0.2 - metrics.stereo_width_ratio) / 0.18, 0.0, 1.0)
        + 0.11 * np.clip((0.045 - metrics.stereo_motion) / 0.045, 0.0, 1.0)
        + 0.1 * np.clip((9.0 - metrics.crest_factor_db) / 5.0, 0.0, 1.0)
        + 0.1 * np.clip((metrics.low_mid_share - 0.36) / 0.18, 0.0, 1.0)
    )

    quality_flags: list[str] = []
    if old_recording_score >= 0.54:
        quality_flags.append("Old-Recording")
    if low_quality_score >= 0.52 or int(input_sample_rate) < 44100:
        quality_flags.append("Low-Quality")
    return tuple(quality_flags)


def _analyze_mastering_input(
    signal_to_analyze: np.ndarray,
    sample_rate: int,
) -> MasteringInputAnalysis:
    metrics = _collect_mastering_input_metrics(signal_to_analyze, sample_rate)
    target_sample_rate = _resolve_mastering_processing_sample_rate(sample_rate)
    if metrics is None:
        return MasteringInputAnalysis(
            preset_name="balanced",
            quality_flags=(),
            target_sample_rate=target_sample_rate,
            metrics=None,
        )

    return MasteringInputAnalysis(
        preset_name=_select_mastering_preset_from_metrics(metrics),
        quality_flags=_resolve_mastering_quality_flags(metrics, sample_rate),
        target_sample_rate=target_sample_rate,
        metrics=metrics,
    )


def _select_mastering_preset(
    signal_to_analyze: np.ndarray,
    sample_rate: int,
) -> str:
    metrics = _collect_mastering_input_metrics(signal_to_analyze, sample_rate)
    if metrics is None:
        return "balanced"
    return _select_mastering_preset_from_metrics(metrics)


def _resolve_explicit_preset_name(
    mastering_kwargs: dict[str, object],
) -> str | None:
    for key in ("preset", "preset_name"):
        value = mastering_kwargs.get(key)
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized and normalized != "auto":
            return normalized
    return None


def _resolve_mastering_kwargs_for_input(
    input_signal: np.ndarray,
    input_sample_rate: int,
    mastering_kwargs: dict[str, object],
    *,
    input_analysis: MasteringInputAnalysis | None = None,
) -> dict[str, object]:
    resolved_kwargs = dict(mastering_kwargs)
    explicit_preset_name = _resolve_explicit_preset_name(resolved_kwargs)
    if explicit_preset_name is not None:
        return resolved_kwargs

    if input_analysis is None:
        selected_preset = _select_mastering_preset(
            input_signal,
            input_sample_rate,
        )
    else:
        selected_preset = input_analysis.preset_name
    resolved_kwargs["preset"] = selected_preset
    preset_name_value = resolved_kwargs.get("preset_name")
    if (
        preset_name_value is None
        or str(preset_name_value).strip().lower() == "auto"
    ):
        resolved_kwargs.pop("preset_name", None)
    return resolved_kwargs


def _resolve_mastered_stems_output_dir(output_path: str) -> str:
    resolved_output_path = Path(str(output_path))
    return str(
        resolved_output_path.parent / f"{resolved_output_path.stem}_stems"
    )


class SmartMastering:
    @property
    def slope_db(self) -> float:
        return self._slope_db

    @slope_db.setter
    def slope_db(self, value: float) -> None:
        self._slope_db = float(value)
        self.config.bass_boost_db_per_oct = self._slope_db
        self.update_profile()
        self.update_bands()

    _fit_frequency = _fit_frequency
    build_target_curve = _build_target_curve
    update_profile = _update_profile
    update_bands = _update_bands
    build_spectral_balance_profile = _build_spectral_balance_profile
    plan_follow_up_drives = _plan_follow_up_drives
    smooth_curve = _smooth_curve
    compute_dynamic_drive = _compute_dynamic_drive
    compute_primary_soft_clip_ratio = _compute_primary_soft_clip_ratio
    plan_follow_up_action = _plan_follow_up_action
    resolve_final_true_peak_target = _resolve_final_true_peak_target
    assess_mastering_contract = _assess_mastering_contract
    resolve_limiter_recovery_settings = _resolve_limiter_recovery_settings

    def __init__(
        self,
        sr: int,
        **config: object,
    ) -> None:
        preset_name = config.pop("preset", None)
        if preset_name is None:
            preset_name = config.get("preset_name")

        cfg = SmartMasteringConfig.from_preset(preset_name)

        for key, val in config.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)

        self.config = cfg
        configure_runtime_state(self, sr, cfg)

    def measure_spectrum(
        self,
        y_mono: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return _measure_spectrum(self, y_mono, signal_module=signal)

    def apply_limiter(
        self,
        y: np.ndarray,
        drive_db: float = 0.0,
        ceil_db: float = -0.1,
        os_factor: int = 4,
        lookahead_ms: float = 2.0,
        attack_ms: float = 1.0,
        release_ms_min: float = 30.0,
        release_ms_max: float = 130.0,
        soft_clip_ratio: float = 0.2,
        window_ms: float = 4.0,
    ) -> np.ndarray:
        recovery_settings = _resolve_limiter_recovery_settings(
            self,
            attack_ms=attack_ms,
            release_ms_min=release_ms_min,
            release_ms_max=release_ms_max,
            window_ms=window_ms,
        )
        return _apply_limiter(
            self,
            y,
            drive_db=drive_db,
            ceil_db=ceil_db,
            os_factor=os_factor,
            lookahead_ms=lookahead_ms,
            attack_ms=recovery_settings.attack_ms,
            release_ms_min=recovery_settings.release_ms_min,
            release_ms_max=recovery_settings.release_ms_max,
            soft_clip_ratio=soft_clip_ratio,
            window_ms=recovery_settings.window_ms,
            signal_module=signal,
            maximum_filter1d_fn=maximum_filter1d,
            uniform_filter1d_fn=uniform_filter1d,
            limiter_smooth_env_fn=limiter_smooth_env,
        )

    def apply_eq(self, y: np.ndarray) -> np.ndarray:
        return _apply_eq(self, y, audio_eq_fn=audio_eq)

    def apply_stem_cleanup(
        self,
        y: np.ndarray,
        *,
        stem_role: str | None = None,
    ) -> np.ndarray:
        return _apply_stem_cleanup(
            self,
            y,
            stem_role=stem_role,
            audio_eq_fn=audio_eq,
        )

    def apply_spatial_enhancement(self, y: np.ndarray) -> np.ndarray:
        return _apply_spatial_enhancement(self, y, signal_module=signal)

    def apply_safety_clamp(
        self, y: np.ndarray, *, ceil_db: float = -0.1
    ) -> np.ndarray:
        return _apply_safety_clamp(y, ceil_db=ceil_db)

    def apply_pre_limiter_saturation(
        self,
        y: np.ndarray,
        *,
        dynamic_drive_db: float = 0.0,
    ) -> np.ndarray:
        return _apply_pre_limiter_saturation(
            self,
            y,
            dynamic_drive_db=dynamic_drive_db,
        )

    def apply_stereo_width_restraint(
        self,
        y: np.ndarray,
        *,
        stereo_width_scale: float = 1.0,
    ) -> np.ndarray:
        return _apply_stereo_width_restraint(
            y,
            stereo_width_scale=stereo_width_scale,
        )

    def apply_low_end_mono_tightening(self, y: np.ndarray) -> np.ndarray:
        return _apply_low_end_mono_tightening(
            self,
            y,
            sample_rate=self.resampling_target,
            cutoff_hz=self.contract_low_end_mono_cutoff_hz,
        )

    def apply_micro_dynamics_finish(self, y: np.ndarray) -> np.ndarray:
        return _apply_micro_dynamics_finish(
            self,
            y,
            sample_rate=self.resampling_target,
        )

    def resolve_mastering_contract(self) -> MasteringContract:
        return _resolve_mastering_contract(
            self.preset_name,
            target_lufs=self.target_lufs,
            ceil_db=self.ceil_db,
            target_lufs_tolerance_db=self.contract_target_lufs_tolerance_db,
            max_short_term_lufs=self.contract_max_short_term_lufs,
            max_momentary_lufs=self.contract_max_momentary_lufs,
            min_crest_factor_db=self.contract_min_crest_factor_db,
            max_crest_factor_db=self.contract_max_crest_factor_db,
            max_stereo_width_ratio=self.contract_max_stereo_width_ratio,
            min_low_end_mono_ratio=self.contract_min_low_end_mono_ratio,
            low_end_mono_cutoff_hz=self.contract_low_end_mono_cutoff_hz,
        )

    def analyze_reference(
        self,
        reference_signal: np.ndarray,
        candidate_signal: np.ndarray,
    ) -> ReferenceAnalysis:
        analysis = _analyze_reference(
            reference_signal,
            candidate_signal,
            self.resampling_target,
            low_end_mono_cutoff_hz=self.contract_low_end_mono_cutoff_hz,
            true_peak_oversample_factor=self.true_peak_oversample_factor,
        )
        self.last_reference_analysis = analysis
        return analysis

    def reference_match_assist(
        self,
        reference_signal: np.ndarray,
        candidate_signal: np.ndarray,
        *,
        match_amount: float | None = None,
    ) -> ReferenceMatchAssist:
        assist = _reference_match_assist(
            reference_signal,
            candidate_signal,
            self.resampling_target,
            current_config=self.config,
            match_amount=match_amount,
            low_end_mono_cutoff_hz=self.contract_low_end_mono_cutoff_hz,
            true_peak_oversample_factor=self.true_peak_oversample_factor,
        )
        self.last_reference_match_assist = assist
        return assist

    def apply_delivery_trim(self, y: np.ndarray) -> np.ndarray:
        return _apply_delivery_trim(
            self,
            y,
            sample_rate=self.resampling_target,
            measure_true_peak_fn=_measure_true_peak,
        )

    def apply_final_headroom_recovery(self, y: np.ndarray) -> np.ndarray:
        return _apply_final_headroom_recovery(
            self,
            y,
            sample_rate=self.resampling_target,
            measure_true_peak_fn=_measure_true_peak,
        )

    def multiband_compress(self, y: np.ndarray) -> np.ndarray:
        return _multiband_compress(
            self,
            y,
            signal_module=signal,
            decoupled_envelope_fn=decoupled_envelope,
        )

    def process(
        self,
        y: np.ndarray,
        sr: int | None = None,
    ) -> tuple[int, np.ndarray]:
        return _process(
            self,
            y,
            sr=sr,
            resample_fn=resample,
            stereo_fn=stereo,
            freq_cut_fn=freq_cut,
            apply_exciter_fn=apply_exciter,
            get_lufs_fn=get_lufs,
            measure_mastering_loudness_fn=_measure_mastering_loudness,
            log_fn=log,
        )

    def process_stem(
        self,
        y: np.ndarray,
        sr: int | None = None,
        *,
        stem_role: str | None = None,
    ) -> tuple[int, np.ndarray]:
        return _process_stem(
            self,
            y,
            sr=sr,
            stem_role=stem_role,
            resample_fn=resample,
            stereo_fn=stereo,
            freq_cut_fn=freq_cut,
            apply_exciter_fn=apply_exciter,
            log_fn=log,
        )


def master(
    input_path: str,
    output_path: str | None = None,
    **kwargs: object,
) -> tuple[str | None, MasteringReport | None]:

    stem_mastering = bool(kwargs.pop("stem_mastering", True))
    raise_on_error = kwargs.pop("raise_on_error", None)
    internal_kwargs = dict(kwargs)
    if raise_on_error is not None:
        internal_kwargs["raise_on_error"] = bool(raise_on_error)
    return _master_internal(
        input_path,
        output_path=output_path,
        stem_mastering=stem_mastering,
        **internal_kwargs,
    )


def _master_internal(
    input_path: str,
    *,
    output_path: str | None,
    stem_mastering: bool,
    raise_on_error: bool = False,
    **kwargs: object,
) -> tuple[str | None, MasteringReport | None]:
    try:
        from .io import read_audio, save_audio

        report_path = kwargs.pop(
            "report_path", f"{output_path}.report.json" if output_path else None
        )
        report_indent = int(kwargs.pop("report_indent", 2))
        bit_depth = int(kwargs.pop("bit_depth", 32))
        bitrate = int(kwargs.pop("bitrate", 320))
        compression_level = int(kwargs.pop("compression_level", 9))
        stem_model_name = str(kwargs.pop("stem_model_name", "mastering"))
        stem_shifts = int(kwargs.pop("stem_shifts", 2))
        stem_mix_headroom_db = float(kwargs.pop("stem_mix_headroom_db", 6.0))
        save_mastered_stems = bool(
            kwargs.pop("save_mastered_stems", stem_mastering)
        )
        mastered_stems_format = (
            str(kwargs.pop("mastered_stems_format", "wav"))
            .strip()
            .lower()
            .lstrip(".")
            or "wav"
        )

        input_sample_rate, input_signal = read_audio(input_path)
        resolved_output_path = output_path or tmp(
            get_ext(input_path), keep=False
        )
        processing_sample_rate = input_sample_rate
        processing_signal = np.array(input_signal, dtype=np.float32, copy=True)
        input_analysis = _analyze_mastering_input(
            processing_signal,
            input_sample_rate,
        )
        resolved_mastering_kwargs = _resolve_mastering_kwargs_for_input(
            processing_signal,
            input_sample_rate,
            dict(kwargs),
            input_analysis=input_analysis,
        )
        resolved_mastering_kwargs.setdefault(
            "resampling_target",
            input_analysis.target_sample_rate,
        )
        if _resolve_explicit_preset_name(dict(kwargs)) is None:
            log(
                "Auto-selected mastering preset",
                resolved_mastering_kwargs["preset"],
            )
        if input_analysis.quality_flags:
            log(
                "Auto-detected mastering input quality flags",
                ", ".join(input_analysis.quality_flags),
            )

        if stem_mastering:
            from definers.system import delete

            from .mastering_stems import process_stem_layers
            from .stems import separate_stem_layers

            base_mastering = SmartMastering(
                input_sample_rate,
                **resolved_mastering_kwargs,
            )
            processing_sample_rate, processing_signal = process_stem_layers(
                input_path,
                base_config=base_mastering.config,
                base_mastering_kwargs=resolved_mastering_kwargs,
                process_stem_fn=_process_stem_signal,
                separate_stems_fn=separate_stem_layers,
                read_audio_fn=read_audio,
                delete_fn=delete,
                model_name=stem_model_name,
                shifts=stem_shifts,
                quality_flags=input_analysis.quality_flags,
                mix_headroom_db=stem_mix_headroom_db,
                save_mastered_stems=save_mastered_stems,
                mastered_stems_output_dir=(
                    _resolve_mastered_stems_output_dir(resolved_output_path)
                    if save_mastered_stems
                    else None
                ),
                save_audio_fn=save_audio,
                mastered_stems_format=mastered_stems_format,
                mastered_stems_bit_depth=bit_depth,
                mastered_stems_bitrate=bitrate,
                mastered_stems_compression_level=compression_level,
            )

        return _render_master_output(
            input_path,
            input_signal=input_signal,
            processing_signal=processing_signal,
            processing_sample_rate=processing_sample_rate,
            output_path=resolved_output_path,
            report_path=report_path,
            report_indent=report_indent,
            bit_depth=bit_depth,
            bitrate=bitrate,
            compression_level=compression_level,
            read_audio_fn=read_audio,
            save_audio_fn=save_audio,
            **resolved_mastering_kwargs,
        )
    except Exception as error:
        from definers.system import catch

        catch(error)
        if raise_on_error:
            raise
        return None, None


def _process_stem_signal(
    signal_to_process: np.ndarray,
    sample_rate: int,
    mastering_kwargs: dict[str, object],
) -> tuple[int, np.ndarray]:
    resolved_mastering_kwargs = dict(mastering_kwargs)
    stem_role = resolved_mastering_kwargs.pop("stem_role", None)
    mastering = SmartMastering(sample_rate, **resolved_mastering_kwargs)
    process_stem = getattr(mastering, "process_stem", None)
    if callable(process_stem):
        return process_stem(
            signal_to_process,
            sample_rate,
            stem_role=None if stem_role is None else str(stem_role),
        )
    return mastering.process(signal_to_process, sample_rate)


def _render_master_output(
    input_path: str,
    *,
    input_signal: np.ndarray,
    processing_signal: np.ndarray,
    processing_sample_rate: int,
    output_path: str | None,
    report_path: str | None,
    report_indent: int,
    bit_depth: int,
    bitrate: int,
    compression_level: int,
    read_audio_fn,
    save_audio_fn,
    **kwargs: object,
) -> tuple[str | None, MasteringReport | None]:

    from definers.system import tmp

    mastering = SmartMastering(processing_sample_rate, **kwargs)
    sr_mastered, y_mastered = mastering.process(
        processing_signal,
        processing_sample_rate,
    )
    resolved_true_peak_target_dbfs = getattr(
        mastering,
        "last_resolved_final_true_peak_target_dbfs",
        None,
    )
    if resolved_true_peak_target_dbfs is None:
        resolve_final_true_peak_target = getattr(
            mastering,
            "resolve_final_true_peak_target",
            None,
        )
        if callable(resolve_final_true_peak_target):
            resolved_true_peak_target_dbfs = float(
                resolve_final_true_peak_target()
            )
        else:
            resolved_true_peak_target_dbfs = float(mastering.ceil_db)

    resolved_output_path = output_path or tmp(get_ext(input_path), keep=False)

    final_output_path, _final_signal, verification = save_verified_audio(
        destination_path=resolved_output_path,
        audio_signal=y_mastered,
        sample_rate=sr_mastered,
        input_signal=input_signal,
        post_eq_signal=getattr(mastering, "last_stage_signals", {}).get(
            "post_eq"
        ),
        post_spatial_signal=getattr(mastering, "last_stage_signals", {}).get(
            "post_spatial"
        ),
        post_limiter_signal=getattr(mastering, "last_stage_signals", {}).get(
            "post_limiter"
        ),
        post_character_signal=getattr(mastering, "last_stage_signals", {}).get(
            "post_character"
        ),
        post_peak_catch_signal=getattr(mastering, "last_stage_signals", {}).get(
            "post_peak_catch"
        ),
        post_delivery_trim_signal=getattr(
            mastering, "last_stage_signals", {}
        ).get("post_delivery_trim"),
        post_clamp_signal=getattr(mastering, "last_stage_signals", {}).get(
            "post_clamp"
        ),
        save_audio_fn=save_audio_fn,
        read_audio_fn=read_audio_fn,
        target_lufs=mastering.target_lufs,
        ceil_db=mastering.ceil_db,
        preset_name=mastering.preset_name,
        contract=getattr(mastering, "last_mastering_contract", None)
        or mastering.resolve_mastering_contract(),
        delivery_profile_name=mastering.delivery_profile,
        character_stage_decision=getattr(
            mastering, "last_character_stage_decision", None
        ),
        peak_catch_events=getattr(mastering, "last_peak_catch_events", ()),
        resolved_true_peak_target_dbfs=resolved_true_peak_target_dbfs,
        stereo_motion_activity=getattr(
            mastering, "last_stereo_motion_activity", None
        ),
        stereo_motion_correlation_guard=getattr(
            mastering,
            "last_stereo_motion_correlation_guard",
            None,
        ),
        delivery_trim_attenuation_db=float(
            getattr(mastering, "last_delivery_trim_attenuation_db", 0.0)
        ),
        delivery_trim_input_true_peak_dbfs=getattr(
            mastering,
            "last_delivery_trim_input_true_peak_dbfs",
            None,
        ),
        delivery_trim_target_dbfs=getattr(
            mastering,
            "last_delivery_trim_target_dbfs",
            None,
        ),
        delivery_trim_output_true_peak_dbfs=getattr(
            mastering,
            "last_delivery_trim_output_true_peak_dbfs",
            None,
        ),
        post_clamp_true_peak_dbfs=getattr(
            mastering,
            "last_post_clamp_true_peak_dbfs",
            None,
        ),
        post_clamp_true_peak_delta_db=getattr(
            mastering,
            "last_post_clamp_true_peak_delta_db",
            None,
        ),
        headroom_recovery_gain_db=float(
            getattr(mastering, "last_headroom_recovery_gain_db", 0.0)
        ),
        headroom_recovery_input_true_peak_dbfs=getattr(
            mastering,
            "last_headroom_recovery_input_true_peak_dbfs",
            None,
        ),
        headroom_recovery_output_true_peak_dbfs=getattr(
            mastering,
            "last_headroom_recovery_output_true_peak_dbfs",
            None,
        ),
        headroom_recovery_failure_reasons=getattr(
            mastering,
            "last_headroom_recovery_failure_reasons",
            (),
        ),
        headroom_recovery_mode=getattr(
            mastering,
            "last_headroom_recovery_mode",
            None,
        ),
        headroom_recovery_integrated_gap_db=getattr(
            mastering,
            "last_headroom_recovery_integrated_gap_db",
            None,
        ),
        headroom_recovery_transient_density=getattr(
            mastering,
            "last_headroom_recovery_transient_density",
            None,
        ),
        headroom_recovery_closed_margin_db=getattr(
            mastering,
            "last_headroom_recovery_closed_margin_db",
            None,
        ),
        headroom_recovery_unused_margin_db=getattr(
            mastering,
            "last_headroom_recovery_unused_margin_db",
            None,
        ),
        decoded_true_peak_dbfs=(
            None
            if mastering.delivery_decoded_true_peak_dbfs is None
            else mastering.delivery_decoded_true_peak_dbfs
            - mastering.codec_headroom_margin_db
        ),
        decoded_lufs_tolerance_db=mastering.delivery_lufs_tolerance_db,
        true_peak_oversample_factor=mastering.true_peak_oversample_factor,
        bit_depth=bit_depth,
        bitrate=bitrate
        if mastering.delivery_bitrate is None
        else mastering.delivery_bitrate,
        compression_level=compression_level,
    )

    if report_path is not None:
        _write_mastering_report(
            verification.report,
            str(report_path),
            indent=report_indent,
        )

    return final_output_path, verification.report
