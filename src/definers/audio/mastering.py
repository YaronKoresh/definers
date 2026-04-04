from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from definers.system import get_ext

from ..file_ops import log
from .config import SmartMasteringConfig
from .dsp import (
    decoupled_envelope,
    limiter_smooth_env,
    resample,
)
from .effects import apply_exciter, stereo
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
from .mastering_pipeline import process as _process
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
    reference_match_assist as _reference_match_assist,
)
from .mastering_state import configure_runtime_state
from .utils import get_lufs


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


def master(
    input_path: str,
    output_path: str | None = None,
    **kwargs: object,
) -> tuple[str | None, MasteringReport | None]:

    from definers.system import tmp

    try:
        from .io import read_audio, save_audio

        report_path = kwargs.pop("report_path", None)
        report_indent = int(kwargs.pop("report_indent", 2))
        bit_depth = int(kwargs.pop("bit_depth", 32))
        bitrate = int(kwargs.pop("bitrate", 320))
        compression_level = int(kwargs.pop("compression_level", 9))

        sr, y = read_audio(input_path)

        mastering = SmartMastering(sr, **kwargs)
        sr_mastered, y_mastered = mastering.process(y, sr)
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

        output_path = output_path or tmp(get_ext(input_path), keep=False)

        final_output_path, _final_signal, verification = save_verified_audio(
            destination_path=output_path,
            audio_signal=y_mastered,
            sample_rate=sr_mastered,
            input_signal=y,
            post_eq_signal=getattr(mastering, "last_stage_signals", {}).get(
                "post_eq"
            ),
            post_spatial_signal=getattr(
                mastering, "last_stage_signals", {}
            ).get("post_spatial"),
            post_limiter_signal=getattr(
                mastering, "last_stage_signals", {}
            ).get("post_limiter"),
            post_character_signal=getattr(
                mastering, "last_stage_signals", {}
            ).get("post_character"),
            post_peak_catch_signal=getattr(
                mastering, "last_stage_signals", {}
            ).get("post_peak_catch"),
            post_delivery_trim_signal=getattr(
                mastering, "last_stage_signals", {}
            ).get("post_delivery_trim"),
            post_clamp_signal=getattr(mastering, "last_stage_signals", {}).get(
                "post_clamp"
            ),
            save_audio_fn=save_audio,
            read_audio_fn=read_audio,
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

    except Exception as e:
        from definers.system import catch

        catch(e)
        return None, None
