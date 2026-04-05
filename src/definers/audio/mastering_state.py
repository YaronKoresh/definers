from __future__ import annotations

import numpy as np

from .config import SmartMasteringConfig
from .mastering_profile import SpectralBalanceProfile


def configure_runtime_state(
    self,
    sr: int,
    cfg: SmartMasteringConfig,
) -> None:
    self.resampling_target = cfg.resampling_target

    nyquist_hz = self.resampling_target / 2.0 - 1.0
    low_cut_ceiling = max(1.0, nyquist_hz - 1.0)
    analysis_low_hz = float(np.clip(cfg.analysis_low_hz, 1.0, low_cut_ceiling))
    self.filter_low_cut = (
        None
        if cfg.low_cut is None
        else float(np.clip(cfg.low_cut, 1.0, low_cut_ceiling))
    )
    self.low_cut = (
        analysis_low_hz if self.filter_low_cut is None else self.filter_low_cut
    )
    self.filter_high_cut = (
        None
        if cfg.high_cut is None
        else float(np.clip(cfg.high_cut, self.low_cut + 1.0, nyquist_hz))
    )
    self.high_cut = (
        nyquist_hz if self.filter_high_cut is None else self.filter_high_cut
    )

    self.bass_transition_hz = float(
        np.clip(cfg.stop_bass_boost_hz, self.low_cut, self.high_cut)
    )
    min_treble_start_hz = min(
        self.high_cut,
        max(self.bass_transition_hz * 1.5, self.low_cut * 2.0),
    )
    self.treble_transition_hz = float(
        np.clip(
            cfg.start_treble_boost_hz,
            min_treble_start_hz,
            self.high_cut,
        )
    )
    self._slope_db = float(cfg.bass_boost_db_per_oct)
    self.treble_boost_db_per_oct = float(cfg.treble_boost_db_per_oct)
    self.anchor_curve_strength = float(max(cfg.anchor_curve_strength, 0.0))

    self.sr = sr
    self.drive_db = float(cfg.drive_db)
    self.ceil_db = float(cfg.ceil_db)
    self.num_bands = cfg.num_bands
    self.smoothing_fraction = cfg.smoothing_fraction
    self.target_lufs = float(cfg.target_lufs)
    self.correction_strength = float(cfg.correction_strength)
    self.mid_slope = float(cfg.mid_slope)
    self.max_spectrum_boost_db = float(cfg.max_spectrum_boost_db)
    self.max_spectrum_cut_db = float(cfg.max_spectrum_cut_db)
    self.stereo_width = float(max(cfg.stereo_width, 0.0))
    self.mono_bass_hz = float(
        np.clip(cfg.mono_bass_hz, self.low_cut, self.high_cut)
    )
    self.final_lufs_tolerance = float(max(cfg.final_lufs_tolerance, 0.0))
    self.max_final_boost_db = float(max(cfg.max_final_boost_db, 0.0))
    self.max_follow_up_passes = max(int(cfg.max_follow_up_passes), 1)
    self.follow_up_soft_clip_ratio_step = float(
        max(cfg.follow_up_soft_clip_ratio_step, 0.0)
    )
    self.limiter_oversample_factor = max(int(cfg.limiter_oversample_factor), 1)
    self.limiter_soft_clip_ratio = float(
        np.clip(cfg.limiter_soft_clip_ratio, 0.0, 0.95)
    )
    self.limiter_recovery_style = (
        str(cfg.limiter_recovery_style).strip().lower()
    )
    self.true_peak_oversample_factor = max(
        int(cfg.true_peak_oversample_factor), 1
    )
    self.pre_limiter_saturation_ratio = float(
        np.clip(cfg.pre_limiter_saturation_ratio, 0.0, 1.0)
    )
    self.low_end_mono_tightening = (
        str(cfg.low_end_mono_tightening).strip().lower()
    )
    self.low_end_mono_tightening_amount = float(
        np.clip(cfg.low_end_mono_tightening_amount, 0.0, 1.0)
    )
    self.codec_headroom_margin_db = float(
        max(cfg.codec_headroom_margin_db, 0.0)
    )
    self.reference_match_amount = float(
        np.clip(cfg.reference_match_amount, 0.0, 1.0)
    )
    self.micro_dynamics_strength = float(
        np.clip(cfg.micro_dynamics_strength, 0.0, 1.0)
    )
    self.micro_dynamics_fast_window_ms = float(
        max(cfg.micro_dynamics_fast_window_ms, 1.0)
    )
    self.micro_dynamics_slow_window_ms = float(
        max(
            cfg.micro_dynamics_slow_window_ms,
            cfg.micro_dynamics_fast_window_ms + 1.0,
        )
    )
    self.micro_dynamics_transient_bias = float(
        np.clip(cfg.micro_dynamics_transient_bias, 0.0, 1.0)
    )
    self.preset_name = cfg.preset_name
    self.delivery_profile = cfg.delivery_profile
    self.delivery_decoded_true_peak_dbfs = cfg.delivery_decoded_true_peak_dbfs
    self.delivery_lufs_tolerance_db = float(
        max(cfg.delivery_lufs_tolerance_db, 0.0)
    )
    self.contract_target_lufs_tolerance_db = float(
        max(cfg.contract_target_lufs_tolerance_db, 0.0)
    )
    self.contract_max_short_term_lufs = (
        None
        if cfg.contract_max_short_term_lufs is None
        else float(cfg.contract_max_short_term_lufs)
    )
    self.contract_max_momentary_lufs = (
        None
        if cfg.contract_max_momentary_lufs is None
        else float(cfg.contract_max_momentary_lufs)
    )
    self.contract_min_crest_factor_db = (
        None
        if cfg.contract_min_crest_factor_db is None
        else float(cfg.contract_min_crest_factor_db)
    )
    self.contract_max_crest_factor_db = (
        None
        if cfg.contract_max_crest_factor_db is None
        else float(cfg.contract_max_crest_factor_db)
    )
    self.contract_max_stereo_width_ratio = (
        None
        if cfg.contract_max_stereo_width_ratio is None
        else float(cfg.contract_max_stereo_width_ratio)
    )
    self.contract_min_low_end_mono_ratio = (
        None
        if cfg.contract_min_low_end_mono_ratio is None
        else float(cfg.contract_min_low_end_mono_ratio)
    )
    self.contract_low_end_mono_cutoff_hz = float(
        max(cfg.contract_low_end_mono_cutoff_hz, 20.0)
    )
    self.delivery_bitrate = (
        None
        if cfg.delivery_bitrate is None
        else max(int(cfg.delivery_bitrate), 32)
    )
    self.spectral_rescue_strength = float(
        max(cfg.spectral_rescue_strength, 0.0)
    )
    self.spectral_rescue_boost_db = float(
        max(cfg.spectral_rescue_boost_db, 0.0)
    )
    self.spectral_rescue_cut_db = float(max(cfg.spectral_rescue_cut_db, 0.0))
    self.spectral_rescue_band_intensity = float(
        max(cfg.spectral_rescue_band_intensity, 0.0)
    )
    self.spectral_drive_bias_db = float(max(cfg.spectral_drive_bias_db, 0.0))
    self.exciter_mix = float(np.clip(cfg.exciter_mix, 0.0, 1.0))
    self.exciter_cutoff_hz = (
        None
        if cfg.exciter_cutoff_hz is None
        else float(max(cfg.exciter_cutoff_hz, 20.0))
    )
    self.exciter_max_drive = float(max(cfg.exciter_max_drive, 0.5))
    self.exciter_high_frequency_cutoff_hz = (
        None
        if cfg.exciter_high_frequency_cutoff_hz is None
        else float(max(cfg.exciter_high_frequency_cutoff_hz, 1000.0))
    )

    self.spectral_balance_profile = SpectralBalanceProfile(
        rescue_factor=0.0,
        correction_strength=self.correction_strength,
        max_boost_db=self.max_spectrum_boost_db,
        max_cut_db=self.max_spectrum_cut_db,
        band_intensity=float(np.clip(cfg.intensity, 0.25, 2.5)),
        closure_repair_factor=0.0,
        mud_cleanup_factor=0.0,
        harshness_restraint_factor=0.0,
        low_end_restraint_factor=0.0,
    )
    self.stereo_tone_variation_db = float(
        np.clip(cfg.stereo_tone_variation_db, 0.0, 1.5)
    )
    self.stereo_tone_variation_cutoff_hz = float(
        max(cfg.stereo_tone_variation_cutoff_hz, 250.0)
    )
    self.stereo_tone_variation_smoothing_ms = float(
        max(cfg.stereo_tone_variation_smoothing_ms, 10.0)
    )
    self.stereo_motion_mid_amount = float(
        np.clip(cfg.stereo_motion_mid_amount, 0.0, 1.5)
    )
    self.stereo_motion_high_amount = float(
        np.clip(cfg.stereo_motion_high_amount, 0.0, 1.5)
    )
    self.stereo_motion_correlation_guard = float(
        np.clip(cfg.stereo_motion_correlation_guard, 0.0, 1.5)
    )
    self.stereo_motion_max_side_boost = float(
        np.clip(cfg.stereo_motion_max_side_boost, 0.0, 0.3)
    )
    self.last_stage_signals = {}
    self.last_finalization_actions = ()
    self.last_mastering_contract = None
    self.last_reference_analysis = None
    self.last_reference_match_assist = None
    self.last_character_stage_decision = None
    self.last_peak_catch_events = ()
    self.last_delivery_trim_attenuation_db = 0.0
    self.last_delivery_trim_input_true_peak_dbfs = None
    self.last_delivery_trim_target_dbfs = None
    self.last_delivery_trim_output_true_peak_dbfs = None
    self.last_post_clamp_true_peak_dbfs = None
    self.last_post_clamp_true_peak_delta_db = None
    self.last_post_clamp_metrics = None
    self.last_stereo_motion_activity = 0.0
    self.last_stereo_motion_correlation_guard = 1.0
    self.last_headroom_recovery_gain_db = 0.0
    self.last_headroom_recovery_input_true_peak_dbfs = None
    self.last_headroom_recovery_output_true_peak_dbfs = None
    self.last_headroom_recovery_failure_reasons = ()
    self.last_headroom_recovery_mode = None
    self.last_headroom_recovery_integrated_gap_db = None
    self.last_headroom_recovery_transient_density = None
    self.last_headroom_recovery_closed_margin_db = None
    self.last_headroom_recovery_unused_margin_db = None
    self.last_resolved_final_true_peak_target_dbfs = None

    self.update_profile()
    self.update_bands()

    target_analysis_size = max(1024.0, self.resampling_target * 0.35)
    self.analysis_nperseg = max(
        1024,
        int(2 ** np.ceil(np.log2(target_analysis_size))),
    )
    self.fft_n = self.analysis_nperseg * 2
