from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmartMasteringConfig:
    num_bands: int = 6
    intensity: float = 1.0
    preset_name: str | None = "balanced"
    delivery_profile: str | None = "lossless"
    delivery_decoded_true_peak_dbfs: float | None = None
    delivery_lufs_tolerance_db: float = 0.6
    delivery_bitrate: int | None = 320
    contract_target_lufs_tolerance_db: float = 0.55
    contract_max_short_term_lufs: float | None = -6.1
    contract_max_momentary_lufs: float | None = -4.7
    contract_min_crest_factor_db: float | None = 5.0
    contract_max_crest_factor_db: float | None = 12.5
    contract_max_stereo_width_ratio: float | None = 0.68
    contract_min_low_end_mono_ratio: float | None = 0.84
    contract_low_end_mono_cutoff_hz: float = 145.0

    bass_knee_db: float = 10.0
    treb_knee_db: float = 2.0

    bass_ratio: float = 3.0
    bass_attack_ms: float = 36.0
    bass_release_ms: float = 155.0
    bass_threshold_db: float = -18.5

    treb_ratio: float = 1.95
    treb_attack_ms: float = 6.5
    treb_release_ms: float = 115.0
    treb_threshold_db: float = -27.5

    resampling_target: int = 44100

    target_lufs: float = -6.8

    stop_bass_boost_hz: float = 145.0
    start_treble_boost_hz: float = 3650.0

    bass_boost_db_per_oct: float = 1.18
    mid_slope: float = -0.82
    treble_boost_db_per_oct: float = 1.16
    anchor_curve_strength: float = 0.5

    smoothing_fraction: float | None = 0.2

    correction_strength: float = 1.0

    analysis_low_hz: float = 10.0
    low_cut: float | None = None
    high_cut: float | None = None

    drive_db: float = 1.35
    ceil_db: float = -0.3

    max_spectrum_boost_db: float = 6.0
    max_spectrum_cut_db: float = 6.0
    spectral_rescue_strength: float = 0.24
    spectral_rescue_boost_db: float = 2.3
    spectral_rescue_cut_db: float = 1.0
    spectral_rescue_band_intensity: float = 0.68
    spectral_drive_bias_db: float = 1.05
    exciter_mix: float = 0.52
    exciter_cutoff_hz: float | None = None
    exciter_max_drive: float = 3.4
    exciter_high_frequency_cutoff_hz: float | None = 6800.0

    stereo_width: float = 1.36
    mono_bass_hz: float = 120.0
    stereo_tone_variation_db: float = 1.28
    stereo_tone_variation_cutoff_hz: float = 1375.0
    stereo_tone_variation_smoothing_ms: float = 128.0
    stereo_motion_mid_amount: float = 1.12
    stereo_motion_high_amount: float = 1.4
    stereo_motion_correlation_guard: float = 0.84
    stereo_motion_max_side_boost: float = 0.28

    final_lufs_tolerance: float = 0.25
    max_final_boost_db: float = 5.0
    max_follow_up_passes: int = 4
    follow_up_soft_clip_ratio_step: float = 0.05

    limiter_oversample_factor: int = 8
    limiter_soft_clip_ratio: float = 0.32
    limiter_recovery_style: str = "balanced"
    true_peak_oversample_factor: int = 8
    pre_limiter_saturation_ratio: float = 0.14
    low_end_mono_tightening: str = "balanced"
    low_end_mono_tightening_amount: float = 0.76
    codec_headroom_margin_db: float = 0.1
    reference_match_amount: float = 0.44
    micro_dynamics_strength: float = 0.14
    micro_dynamics_fast_window_ms: float = 8.0
    micro_dynamics_slow_window_ms: float = 58.0
    micro_dynamics_transient_bias: float = 0.73

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        return (
            "balanced",
            "edm",
            "vocal",
        )

    @classmethod
    def balanced(cls) -> SmartMasteringConfig:
        return cls()

    @classmethod
    def edm(cls) -> SmartMasteringConfig:
        return cls(
            preset_name="edm",
            intensity=1.12,
            delivery_profile="lossless",
            delivery_decoded_true_peak_dbfs=-0.15,
            delivery_lufs_tolerance_db=0.4,
            delivery_bitrate=320,
            contract_target_lufs_tolerance_db=0.4,
            contract_max_short_term_lufs=-4.4,
            contract_max_momentary_lufs=-3.2,
            contract_min_crest_factor_db=4.0,
            contract_max_crest_factor_db=9.5,
            contract_max_stereo_width_ratio=0.62,
            contract_min_low_end_mono_ratio=0.86,
            contract_low_end_mono_cutoff_hz=150.0,
            target_lufs=-4.7,
            bass_ratio=3.6,
            bass_attack_ms=32.0,
            bass_release_ms=135.0,
            bass_threshold_db=-19.3,
            treb_ratio=2.15,
            treb_threshold_db=-26.2,
            treb_attack_ms=5.5,
            treb_release_ms=105.0,
            stop_bass_boost_hz=150.0,
            start_treble_boost_hz=4350.0,
            bass_boost_db_per_oct=1.36,
            mid_slope=-1.02,
            treble_boost_db_per_oct=1.0,
            anchor_curve_strength=0.58,
            correction_strength=1.0,
            drive_db=2.25,
            ceil_db=-0.18,
            max_spectrum_boost_db=6.3,
            max_spectrum_cut_db=6.4,
            spectral_rescue_strength=0.28,
            spectral_rescue_boost_db=2.6,
            spectral_rescue_cut_db=1.1,
            spectral_rescue_band_intensity=0.76,
            spectral_drive_bias_db=1.45,
            exciter_mix=0.58,
            exciter_max_drive=3.9,
            exciter_high_frequency_cutoff_hz=6400.0,
            stereo_width=1.33,
            mono_bass_hz=128.0,
            stereo_tone_variation_db=1.08,
            stereo_tone_variation_cutoff_hz=1460.0,
            stereo_tone_variation_smoothing_ms=160.0,
            stereo_motion_mid_amount=1.0,
            stereo_motion_high_amount=1.24,
            stereo_motion_correlation_guard=0.94,
            stereo_motion_max_side_boost=0.24,
            max_final_boost_db=6.1,
            limiter_soft_clip_ratio=0.42,
            limiter_recovery_style="tight",
            pre_limiter_saturation_ratio=0.24,
            low_end_mono_tightening="balanced",
            low_end_mono_tightening_amount=0.92,
            codec_headroom_margin_db=0.05,
            reference_match_amount=0.36,
            micro_dynamics_strength=0.07,
            micro_dynamics_fast_window_ms=7.5,
            micro_dynamics_slow_window_ms=40.0,
            micro_dynamics_transient_bias=0.7,
            true_peak_oversample_factor=8,
        )

    @classmethod
    def vocal(cls) -> SmartMasteringConfig:
        return cls(
            preset_name="vocal",
            intensity=0.94,
            delivery_profile="lossless",
            delivery_decoded_true_peak_dbfs=None,
            delivery_lufs_tolerance_db=0.8,
            delivery_bitrate=320,
            contract_target_lufs_tolerance_db=0.6,
            contract_max_short_term_lufs=-7.4,
            contract_max_momentary_lufs=-6.0,
            contract_min_crest_factor_db=6.2,
            contract_max_crest_factor_db=14.5,
            contract_max_stereo_width_ratio=0.72,
            contract_min_low_end_mono_ratio=0.82,
            contract_low_end_mono_cutoff_hz=140.0,
            target_lufs=-8.9,
            bass_ratio=2.3,
            bass_attack_ms=42.0,
            bass_release_ms=175.0,
            bass_threshold_db=-17.0,
            treb_ratio=1.6,
            treb_attack_ms=8.0,
            treb_release_ms=135.0,
            treb_threshold_db=-24.6,
            stop_bass_boost_hz=128.0,
            start_treble_boost_hz=3400.0,
            bass_boost_db_per_oct=0.88,
            mid_slope=-0.5,
            treble_boost_db_per_oct=1.06,
            anchor_curve_strength=0.34,
            correction_strength=1.0,
            drive_db=0.9,
            ceil_db=-0.65,
            max_spectrum_boost_db=5.0,
            max_spectrum_cut_db=5.7,
            spectral_rescue_strength=0.19,
            spectral_rescue_boost_db=1.7,
            spectral_rescue_cut_db=0.8,
            spectral_rescue_band_intensity=0.58,
            spectral_drive_bias_db=0.65,
            exciter_mix=0.38,
            exciter_max_drive=2.6,
            exciter_high_frequency_cutoff_hz=7600.0,
            stereo_width=1.42,
            mono_bass_hz=102.0,
            stereo_tone_variation_db=1.42,
            stereo_tone_variation_cutoff_hz=1280.0,
            stereo_tone_variation_smoothing_ms=112.0,
            stereo_motion_mid_amount=1.22,
            stereo_motion_high_amount=1.46,
            stereo_motion_correlation_guard=0.78,
            stereo_motion_max_side_boost=0.29,
            final_lufs_tolerance=0.3,
            max_final_boost_db=3.8,
            follow_up_soft_clip_ratio_step=0.04,
            limiter_soft_clip_ratio=0.21,
            limiter_recovery_style="glue",
            pre_limiter_saturation_ratio=0.08,
            low_end_mono_tightening="gentle",
            low_end_mono_tightening_amount=0.86,
            codec_headroom_margin_db=0.15,
            reference_match_amount=0.52,
            micro_dynamics_strength=0.2,
            micro_dynamics_fast_window_ms=10.0,
            micro_dynamics_slow_window_ms=64.0,
            micro_dynamics_transient_bias=0.79,
            true_peak_oversample_factor=8,
        )

    @classmethod
    def from_preset(cls, name: str | None) -> SmartMasteringConfig:
        if name is None:
            return cls.balanced()

        normalized = str(name).strip().lower()
        presets = {
            "balanced": cls.balanced,
            "edm": cls.edm,
            "vocal": cls.vocal,
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
