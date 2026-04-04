from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmartMasteringConfig:
    num_bands: int = 6
    intensity: float = 1.0
    preset_name: str | None = None
    delivery_profile: str | None = None
    delivery_decoded_true_peak_dbfs: float | None = None
    delivery_lufs_tolerance_db: float = 0.75
    delivery_bitrate: int | None = None
    contract_target_lufs_tolerance_db: float = 0.5
    contract_max_short_term_lufs: float | None = None
    contract_max_momentary_lufs: float | None = None
    contract_min_crest_factor_db: float | None = 4.0
    contract_max_crest_factor_db: float | None = 12.0
    contract_max_stereo_width_ratio: float | None = 0.55
    contract_min_low_end_mono_ratio: float | None = 0.85
    contract_low_end_mono_cutoff_hz: float = 160.0

    bass_knee_db: float = 10.0
    treb_knee_db: float = 2.0

    bass_ratio: float = 3.5
    bass_attack_ms: float = 30.0
    bass_release_ms: float = 130.0
    bass_threshold_db: float = -19.5

    treb_ratio: float = 2.3
    treb_attack_ms: float = 4.0
    treb_release_ms: float = 100.0
    treb_threshold_db: float = -30.0

    resampling_target: int = 44100

    target_lufs: float = -5.5

    stop_bass_boost_hz: float = 140.0
    start_treble_boost_hz: float = 4500.0

    bass_boost_db_per_oct: float = 1.3
    mid_slope: float = -1.65
    treble_boost_db_per_oct: float = 1.25
    anchor_curve_strength: float = 0.6

    smoothing_fraction: float | None = 0.2

    correction_strength: float = 0.95

    analysis_low_hz: float = 10.0
    low_cut: float | None = None
    high_cut: float | None = None

    drive_db: float = 1.25
    ceil_db: float = -0.05

    max_spectrum_boost_db: float = 6.5
    max_spectrum_cut_db: float = 7.0
    spectral_rescue_strength: float = 0.24
    spectral_rescue_boost_db: float = 2.5
    spectral_rescue_cut_db: float = 1.25
    spectral_rescue_band_intensity: float = 0.8
    spectral_drive_bias_db: float = 1.0
    exciter_mix: float = 1.0
    exciter_cutoff_hz: float | None = None
    exciter_max_drive: float = 6.0
    exciter_high_frequency_cutoff_hz: float | None = None

    stereo_width: float = 1.25
    mono_bass_hz: float = 140.0
    stereo_tone_variation_db: float = 0.0
    stereo_tone_variation_cutoff_hz: float = 1800.0
    stereo_tone_variation_smoothing_ms: float = 180.0
    stereo_motion_mid_amount: float = 0.55
    stereo_motion_high_amount: float = 1.0
    stereo_motion_correlation_guard: float = 1.0
    stereo_motion_max_side_boost: float = 0.16

    final_lufs_tolerance: float = 0.2
    max_final_boost_db: float = 5.5
    max_follow_up_passes: int = 4
    follow_up_soft_clip_ratio_step: float = 0.06

    limiter_oversample_factor: int = 8
    limiter_soft_clip_ratio: float = 0.24
    limiter_recovery_style: str = "balanced"
    true_peak_oversample_factor: int = 4
    pre_limiter_saturation_ratio: float = 0.0
    low_end_mono_tightening: str = "balanced"
    low_end_mono_tightening_amount: float = 1.0
    codec_headroom_margin_db: float = 0.0
    reference_match_amount: float = 0.35
    micro_dynamics_strength: float = 0.0
    micro_dynamics_fast_window_ms: float = 8.0
    micro_dynamics_slow_window_ms: float = 45.0
    micro_dynamics_transient_bias: float = 0.75

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        return (
            "edm",
            "pop",
            "flat",
            "safe",
        )

    @classmethod
    def edm(cls) -> SmartMasteringConfig:
        return cls(
            preset_name="edm",
            delivery_profile="lossless",
            delivery_decoded_true_peak_dbfs=-0.1,
            delivery_lufs_tolerance_db=0.35,
            delivery_bitrate=320,
            contract_target_lufs_tolerance_db=0.45,
            contract_max_short_term_lufs=-3.7,
            contract_max_momentary_lufs=-2.6,
            contract_min_crest_factor_db=4.0,
            contract_max_crest_factor_db=9.5,
            contract_max_stereo_width_ratio=0.58,
            contract_min_low_end_mono_ratio=0.88,
            contract_low_end_mono_cutoff_hz=160.0,
            target_lufs=-3.0,
            bass_ratio=4.0,
            bass_threshold_db=-20.5,
            treb_ratio=2.45,
            treb_threshold_db=-25.5,
            treb_attack_ms=5.5,
            start_treble_boost_hz=5200.0,
            treble_boost_db_per_oct=0.58,
            anchor_curve_strength=0.52,
            drive_db=2.25,
            ceil_db=-0.1,
            exciter_mix=0.58,
            exciter_max_drive=3.3,
            exciter_high_frequency_cutoff_hz=6800.0,
            stereo_width=1.5,
            mono_bass_hz=150.0,
            stereo_tone_variation_db=1.25,
            stereo_tone_variation_cutoff_hz=1550.0,
            stereo_tone_variation_smoothing_ms=260.0,
            stereo_motion_mid_amount=1.0,
            stereo_motion_high_amount=1.5,
            stereo_motion_correlation_guard=1.5,
            stereo_motion_max_side_boost=0.5,
            max_final_boost_db=6.5,
            limiter_soft_clip_ratio=0.5,
            limiter_recovery_style="tight",
            pre_limiter_saturation_ratio=0.3,
            low_end_mono_tightening="firm",
            low_end_mono_tightening_amount=1.0,
            codec_headroom_margin_db=0.0,
            reference_match_amount=0.45,
            micro_dynamics_strength=0.035,
            micro_dynamics_fast_window_ms=7.0,
            micro_dynamics_slow_window_ms=38.0,
            micro_dynamics_transient_bias=0.76,
            spectral_drive_bias_db=1.75,
            true_peak_oversample_factor=8,
        )

    @classmethod
    def pop(cls) -> SmartMasteringConfig:
        return cls(
            preset_name="pop",
            delivery_profile="lossless",
            delivery_decoded_true_peak_dbfs=-0.6,
            delivery_lufs_tolerance_db=0.75,
            delivery_bitrate=320,
            contract_target_lufs_tolerance_db=0.5,
            contract_max_short_term_lufs=-5.1,
            contract_max_momentary_lufs=-4.0,
            contract_min_crest_factor_db=4.8,
            contract_max_crest_factor_db=11.5,
            contract_max_stereo_width_ratio=0.54,
            contract_min_low_end_mono_ratio=0.87,
            contract_low_end_mono_cutoff_hz=160.0,
            target_lufs=-7.0,
            bass_ratio=3.2,
            treb_ratio=2.4,
            drive_db=1.0,
            ceil_db=-0.25,
            stereo_width=1.1,
            max_final_boost_db=4.5,
            limiter_soft_clip_ratio=0.26,
            limiter_recovery_style="balanced",
            pre_limiter_saturation_ratio=0.08,
            low_end_mono_tightening="balanced",
            low_end_mono_tightening_amount=0.82,
            codec_headroom_margin_db=0.1,
            reference_match_amount=0.35,
            micro_dynamics_strength=0.08,
            micro_dynamics_fast_window_ms=8.0,
            micro_dynamics_slow_window_ms=42.0,
            micro_dynamics_transient_bias=0.76,
            true_peak_oversample_factor=8,
        )

    @classmethod
    def flat(cls) -> SmartMasteringConfig:
        return cls(
            preset_name="flat",
            delivery_profile="analysis",
            delivery_decoded_true_peak_dbfs=None,
            delivery_lufs_tolerance_db=1.25,
            delivery_bitrate=320,
            contract_target_lufs_tolerance_db=1.0,
            contract_max_short_term_lufs=-11.5,
            contract_max_momentary_lufs=-10.0,
            contract_min_crest_factor_db=8.0,
            contract_max_crest_factor_db=18.0,
            contract_max_stereo_width_ratio=0.45,
            contract_min_low_end_mono_ratio=0.94,
            contract_low_end_mono_cutoff_hz=160.0,
            target_lufs=-14.0,
            bass_boost_db_per_oct=0.0,
            mid_slope=0.0,
            treble_boost_db_per_oct=0.0,
            anchor_curve_strength=0.0,
            correction_strength=0.0,
            bass_ratio=1.15,
            treb_ratio=1.05,
            drive_db=0.0,
            ceil_db=-1.0,
            max_spectrum_boost_db=0.0,
            max_spectrum_cut_db=0.0,
            spectral_rescue_strength=0.0,
            spectral_rescue_boost_db=0.0,
            spectral_rescue_cut_db=0.0,
            spectral_rescue_band_intensity=0.0,
            spectral_drive_bias_db=0.0,
            stereo_width=1.0,
            max_final_boost_db=0.0,
            limiter_soft_clip_ratio=0.0,
            limiter_recovery_style="glue",
            pre_limiter_saturation_ratio=0.0,
            low_end_mono_tightening="gentle",
            low_end_mono_tightening_amount=0.35,
            codec_headroom_margin_db=0.0,
            reference_match_amount=0.15,
            micro_dynamics_strength=0.0,
            micro_dynamics_fast_window_ms=8.0,
            micro_dynamics_slow_window_ms=45.0,
            micro_dynamics_transient_bias=0.7,
            true_peak_oversample_factor=4,
        )

    @classmethod
    def safe(cls) -> SmartMasteringConfig:
        return cls(
            preset_name="safe",
            delivery_profile="streaming_lossy",
            delivery_decoded_true_peak_dbfs=-1.0,
            delivery_lufs_tolerance_db=1.0,
            delivery_bitrate=256,
            contract_target_lufs_tolerance_db=0.75,
            contract_max_short_term_lufs=-8.0,
            contract_max_momentary_lufs=-6.7,
            contract_min_crest_factor_db=5.5,
            contract_max_crest_factor_db=13.0,
            contract_max_stereo_width_ratio=0.5,
            contract_min_low_end_mono_ratio=0.9,
            contract_low_end_mono_cutoff_hz=150.0,
            target_lufs=-10.5,
            drive_db=0.75,
            ceil_db=-1.0,
            stereo_width=1.08,
            max_final_boost_db=2.5,
            limiter_soft_clip_ratio=0.16,
            limiter_recovery_style="glue",
            pre_limiter_saturation_ratio=0.04,
            low_end_mono_tightening="balanced",
            low_end_mono_tightening_amount=0.75,
            codec_headroom_margin_db=0.2,
            reference_match_amount=0.3,
            micro_dynamics_strength=0.04,
            micro_dynamics_fast_window_ms=8.0,
            micro_dynamics_slow_window_ms=45.0,
            micro_dynamics_transient_bias=0.72,
            true_peak_oversample_factor=8,
        )

    @classmethod
    def from_preset(cls, name: str | None) -> SmartMasteringConfig:
        if name is None:
            return cls()

        normalized = str(name).strip().lower()
        presets = {
            "edm": cls.edm,
            "pop": cls.pop,
            "flat": cls.flat,
            "safe": cls.safe,
        }
        preset_factory = presets.get(normalized)
        if preset_factory is None:
            raise ValueError(f"Unknown mastering preset: {name}")
        return preset_factory()

    @classmethod
    def make_bands_from_fcs(
        cls, fcs: list[float], freq_min: float, freq_max: float
    ) -> list[dict]:
        return cls().build_bands_from_fcs(fcs, freq_min, freq_max)

    def build_bands_from_fcs(
        self, fcs: list[float], freq_min: float, freq_max: float
    ) -> list[dict]:
        if not fcs:
            return []

        safe_min = float(max(freq_min, 1e-3))
        safe_max = float(max(freq_max, safe_min * 1.01))
        fcs_arr = np.array(fcs, dtype=float)
        ref_min = np.log2(safe_min)
        ref_max = np.log2(safe_max)
        fcs_safe = np.clip(fcs_arr, safe_min, safe_max)
        fcs_log = np.log2(fcs_safe)

        positions = (fcs_log - ref_min) / (ref_max - ref_min)

        knees = (
            self.bass_knee_db
            + (self.treb_knee_db - self.bass_knee_db) * positions
        )

        base_thr = (
            self.bass_threshold_db
            + (self.treb_threshold_db - self.bass_threshold_db) * positions
        )

        ratios = (
            self.bass_ratio + (self.treb_ratio - self.bass_ratio) * positions
        )

        attacks = (
            self.bass_attack_ms
            + (self.treb_attack_ms - self.bass_attack_ms) * positions
        )

        releases = (
            self.bass_release_ms
            + (self.treb_release_ms - self.bass_release_ms) * positions
        )

        intensity = float(np.clip(self.intensity, 0.25, 2.5))
        thr_center = float(np.mean(base_thr))
        thr = thr_center + (base_thr - thr_center) * intensity
        ratios = np.maximum(1.0, 1.0 + (ratios - 1.0) * intensity)

        timing_scale = 1.0 / np.sqrt(intensity)
        attacks = np.maximum(attacks * timing_scale, 1.0)
        releases = np.maximum(releases * timing_scale, attacks)

        knees *= np.clip(0.85 + 0.15 * intensity, 0.7, 1.3)

        makeups = np.maximum(
            0.0,
            np.abs(thr) * (1.0 - 1.0 / ratios) * 0.35,
        )

        return [
            {
                "fc": float(fcs_arr[i]),
                "base_threshold": float(thr[i]),
                "ratio": float(ratios[i]),
                "attack_ms": float(attacks[i]),
                "release_ms": float(releases[i]),
                "makeup_db": float(makeups[i]),
                "knee_db": float(knees[i]),
            }
            for i in range(len(fcs_arr))
        ]
