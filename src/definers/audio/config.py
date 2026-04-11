from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

_PRESET_MACRO_DEFAULTS: dict[str, dict[str, float]] = {
    "balanced": {
        "bass": 0.5,
        "volume": 0.5,
        "effects": 1.0,
    },
    "edm": {
        "bass": 1.0,
        "volume": 1.0,
        "effects": 1.0,
    },
    "vocal": {
        "bass": 0.25,
        "volume": 0.25,
        "effects": 1.0,
    },
}

_LEGACY_MACRO_ALIASES: dict[str, tuple[str, ...]] = {
    "bass": ("tone",),
    "volume": ("loudness", "compression"),
    "effects": ("space", "detail", "control"),
}

_DERIVED_FIELD_SPECS: dict[str, dict[str, Any]] = {
    "intensity": {
        "base": 1.0,
        "coeffs": {
            "bass": 0.03,
            "volume": 0.1,
            "effects": -0.005,
        },
        "clip": (0.25, 2.5),
    },
    "delivery_lufs_tolerance_db": {
        "base": 0.6,
        "coeffs": {"volume": -0.15, "effects": 0.1},
        "clip": (0.2, 1.5),
    },
    "delivery_bitrate": {
        "base": 320,
        "kind": "int",
        "clip": (32, 640),
    },
    "contract_target_lufs_tolerance_db": {
        "base": 0.55,
        "coeffs": {"volume": -0.1, "effects": 0.08},
        "clip": (0.2, 1.5),
    },
    "contract_max_short_term_lufs": {
        "base": -6.1,
        "coeffs": {"volume": 1.6, "effects": -0.2},
        "clip": (-12.0, -3.0),
    },
    "contract_max_momentary_lufs": {
        "base": -4.7,
        "coeffs": {"volume": 1.5, "effects": -0.25},
        "clip": (-10.0, -2.0),
    },
    "contract_min_crest_factor_db": {
        "base": 5.0,
        "coeffs": {"volume": -0.7, "effects": 0.5},
        "clip": (2.0, 10.0),
    },
    "contract_max_crest_factor_db": {
        "base": 12.5,
        "coeffs": {"volume": -1.5, "effects": 1.5},
        "clip": (5.0, 18.0),
    },
    "contract_max_stereo_width_ratio": {
        "base": 0.68,
        "coeffs": {
            "bass": -0.01,
            "volume": -0.02,
            "effects": 0.06,
        },
        "clip": (0.45, 0.85),
    },
    "contract_min_low_end_mono_ratio": {
        "base": 0.84,
        "coeffs": {
            "bass": 0.03,
            "volume": 0.02,
            "effects": -0.02,
        },
        "clip": (0.7, 1.0),
    },
    "contract_low_end_mono_cutoff_hz": {
        "base": 145.0,
        "coeffs": {
            "bass": 8.0,
            "volume": 2.0,
            "effects": -4.0,
        },
        "clip": (80.0, 220.0),
    },
    "bass_knee_db": {
        "base": 10.0,
        "coeffs": {
            "bass": 0.2,
            "volume": 0.5,
            "effects": -0.1,
        },
        "clip": (1.0, 18.0),
    },
    "treb_knee_db": {
        "base": 2.0,
        "coeffs": {
            "bass": -0.1,
            "volume": 0.12,
            "effects": -0.05,
        },
        "clip": (0.5, 6.0),
    },
    "bass_ratio": {
        "base": 3.0,
        "coeffs": {
            "bass": 0.25,
            "volume": 0.5,
            "effects": -0.15,
        },
        "clip": (1.2, 6.0),
    },
    "bass_attack_ms": {
        "base": 36.0,
        "coeffs": {
            "bass": 2.0,
            "volume": -4.0,
            "effects": 1.0,
        },
        "clip": (1.0, 80.0),
    },
    "bass_release_ms": {
        "base": 155.0,
        "coeffs": {
            "bass": 6.0,
            "volume": -18.0,
            "effects": 3.0,
        },
        "clip": (20.0, 260.0),
    },
    "bass_threshold_db": {
        "base": -18.5,
        "coeffs": {
            "bass": -0.4,
            "volume": -1.0,
            "effects": 0.2,
        },
        "clip": (-40.0, -6.0),
    },
    "treb_ratio": {
        "base": 1.95,
        "coeffs": {
            "bass": -0.08,
            "volume": 0.28,
            "effects": -0.3,
        },
        "clip": (1.0, 4.0),
    },
    "treb_attack_ms": {
        "base": 6.5,
        "coeffs": {
            "bass": -0.2,
            "volume": -0.8,
            "effects": 0.6,
        },
        "clip": (1.0, 20.0),
    },
    "treb_release_ms": {
        "base": 115.0,
        "coeffs": {
            "bass": -2.0,
            "volume": -9.0,
            "effects": 6.0,
        },
        "clip": (40.0, 200.0),
    },
    "treb_threshold_db": {
        "base": -27.5,
        "coeffs": {
            "bass": 0.8,
            "volume": -1.0,
            "effects": 0.2,
        },
        "clip": (-40.0, -12.0),
    },
    "target_lufs": {
        "base": -9.0,
        "coeffs": {"volume": 3.0, "effects": 0.2},
        "clip": (-14.0, -4.5),
    },
    "stop_bass_boost_hz": {
        "base": 145.0,
        "coeffs": {
            "bass": 12.0,
            "volume": 3.0,
            "effects": -4.0,
        },
        "clip": (90.0, 220.0),
    },
    "start_treble_boost_hz": {
        "base": 3650.0,
        "coeffs": {
            "bass": 450.0,
            "volume": 50.0,
            "effects": -200.0,
        },
        "clip": (2200.0, 5500.0),
    },
    "bass_boost_db_per_oct": {
        "base": 1.18,
        "coeffs": {
            "bass": 0.22,
            "volume": 0.06,
            "effects": -0.04,
        },
        "clip": (0.4, 1.8),
    },
    "mid_slope": {
        "base": -0.82,
        "coeffs": {
            "bass": -0.18,
            "volume": -0.08,
            "effects": 0.08,
        },
        "clip": (-1.6, 0.2),
    },
    "treble_boost_db_per_oct": {
        "base": 1.16,
        "coeffs": {
            "bass": -0.12,
            "volume": -0.04,
            "effects": 0.08,
        },
        "clip": (0.5, 1.6),
    },
    "anchor_curve_strength": {
        "base": 0.5,
        "coeffs": {
            "bass": 0.05,
            "volume": 0.08,
            "effects": -0.1,
        },
        "clip": (0.1, 0.8),
    },
    "drive_db": {
        "base": 1.35,
        "coeffs": {
            "bass": 0.1,
            "volume": 0.9,
            "effects": -0.05,
        },
        "clip": (0.0, 3.5),
    },
    "ceil_db": {
        "base": -0.3,
        "coeffs": {"volume": 0.18, "effects": -0.05},
        "clip": (-1.2, -0.1),
    },
    "max_spectrum_boost_db": {
        "base": 6.0,
        "coeffs": {
            "bass": -0.1,
            "volume": 0.4,
            "effects": 0.2,
        },
        "clip": (3.0, 8.0),
    },
    "max_spectrum_cut_db": {
        "base": 6.0,
        "coeffs": {"volume": 0.4, "effects": -0.1},
        "clip": (3.0, 8.0),
    },
    "spectral_rescue_strength": {
        "base": 1.0,
        "coeffs": {
            "bass": 0.05,
            "volume": -0.45,
            "effects": -0.35,
        },
        "clip": (0.0, 1.5),
    },
    "spectral_rescue_boost_db": {
        "base": 2.3,
        "coeffs": {"volume": 0.25, "effects": -0.1},
        "clip": (0.5, 4.0),
    },
    "spectral_rescue_cut_db": {
        "base": 1.0,
        "coeffs": {"volume": 0.12, "effects": -0.08},
        "clip": (0.2, 2.0),
    },
    "spectral_rescue_band_intensity": {
        "base": 1.0,
        "coeffs": {"volume": -0.2, "effects": -0.2},
        "clip": (0.2, 1.6),
    },
    "spectral_drive_bias_db": {
        "base": 1.05,
        "coeffs": {
            "bass": 0.05,
            "volume": 0.35,
            "effects": -0.15,
        },
        "clip": (0.0, 2.0),
    },
    "exciter_mix": {
        "base": 0.86,
        "coeffs": {"volume": -0.03, "effects": 0.12},
        "clip": (0.0, 1.0),
    },
    "exciter_cutoff_hz": {"base": None},
    "exciter_max_drive": {
        "base": 3.4,
        "coeffs": {
            "bass": -0.15,
            "volume": 0.45,
            "effects": -0.1,
        },
        "clip": (0.5, 5.0),
    },
    "exciter_high_frequency_cutoff_hz": {
        "base": 6800.0,
        "coeffs": {
            "bass": -500.0,
            "volume": -250.0,
            "effects": 350.0,
        },
        "clip": (3500.0, 9000.0),
    },
    "stereo_width": {
        "base": 1.36,
        "coeffs": {
            "bass": -0.04,
            "volume": -0.02,
            "effects": 0.09,
        },
        "clip": (0.8, 1.6),
    },
    "mono_bass_hz": {
        "base": 120.0,
        "coeffs": {
            "bass": 16.0,
            "volume": 2.0,
            "effects": -6.0,
        },
        "clip": (80.0, 170.0),
    },
    "stereo_tone_variation_db": {
        "base": 1.28,
        "coeffs": {
            "bass": -0.04,
            "volume": -0.06,
            "effects": 0.16,
        },
        "clip": (0.0, 1.5),
    },
    "stereo_tone_variation_cutoff_hz": {
        "base": 1375.0,
        "coeffs": {
            "bass": 90.0,
            "volume": 40.0,
            "effects": -100.0,
        },
        "clip": (800.0, 1800.0),
    },
    "stereo_tone_variation_smoothing_ms": {
        "base": 128.0,
        "coeffs": {"volume": 18.0, "effects": -16.0},
        "clip": (20.0, 220.0),
    },
    "stereo_motion_mid_amount": {
        "base": 1.12,
        "coeffs": {"volume": -0.03, "effects": 0.12},
        "clip": (0.0, 1.6),
    },
    "stereo_motion_high_amount": {
        "base": 1.4,
        "coeffs": {
            "bass": -0.02,
            "volume": -0.05,
            "effects": 0.15,
        },
        "clip": (0.0, 1.7),
    },
    "stereo_motion_correlation_guard": {
        "base": 0.84,
        "coeffs": {
            "bass": 0.01,
            "volume": 0.06,
            "effects": -0.06,
        },
        "clip": (0.5, 1.1),
    },
    "stereo_motion_max_side_boost": {
        "base": 0.28,
        "coeffs": {"volume": -0.03, "effects": 0.03},
        "clip": (0.0, 0.3),
    },
    "final_lufs_tolerance": {
        "base": 0.25,
        "coeffs": {"volume": -0.02, "effects": 0.03},
        "clip": (0.1, 0.5),
    },
    "max_final_boost_db": {
        "base": 3.5,
        "coeffs": {"volume": 0.9, "effects": -0.1},
        "clip": (1.0, 5.0),
    },
    "max_follow_up_passes": {
        "base": 2,
        "kind": "int",
        "clip": (1, 4),
    },
    "follow_up_soft_clip_ratio_step": {
        "base": 0.015,
        "coeffs": {"volume": 0.004},
        "clip": (0.005, 0.03),
    },
    "limiter_oversample_factor": {
        "base": 8,
        "kind": "int",
        "clip": (1, 16),
    },
    "limiter_soft_clip_ratio": {
        "base": 0.18,
        "coeffs": {"volume": 0.08, "effects": -0.01},
        "clip": (0.0, 0.5),
    },
    "true_peak_oversample_factor": {
        "base": 8,
        "kind": "int",
        "clip": (1, 16),
    },
    "pre_limiter_saturation_ratio": {
        "base": 0.06,
        "coeffs": {
            "bass": 0.01,
            "volume": 0.06,
            "effects": -0.015,
        },
        "clip": (0.0, 0.25),
    },
    "low_end_mono_tightening_amount": {
        "base": 0.76,
        "coeffs": {
            "bass": 0.06,
            "volume": 0.1,
            "effects": -0.05,
        },
        "clip": (0.0, 1.0),
    },
    "codec_headroom_margin_db": {
        "base": 0.1,
        "coeffs": {"volume": -0.03, "effects": 0.04},
        "clip": (0.0, 0.3),
    },
    "stem_noise_gate_enabled": {"base": True, "kind": "bool"},
    "stem_noise_gate_strength": {
        "base": 1.0,
        "coeffs": {"volume": 0.05, "effects": -0.02},
        "clip": (0.0, 1.5),
    },
    "stem_cleanup_strength": {
        "base": 1.0,
        "clip": (0.0, 1.5),
    },
    "stem_tone_enrichment_enabled": {"base": True, "kind": "bool"},
    "stem_tone_enrichment_mix": {
        "base": 0.14,
        "coeffs": {"volume": -0.01, "effects": 0.02},
        "clip": (0.0, 0.3),
    },
    "stem_glue_reverb_amount": {
        "base": 1.0,
        "clip": (0.0, 1.5),
    },
    "stem_drum_edge_amount": {
        "base": 1.0,
        "clip": (0.0, 1.5),
    },
    "stem_vocal_pullback_db": {
        "base": 0.0,
        "clip": (0.0, 3.0),
    },
    "reference_match_amount": {
        "base": 0.44,
        "coeffs": {"volume": -0.04, "effects": 0.1},
        "clip": (0.0, 1.0),
    },
    "micro_dynamics_strength": {
        "base": 0.1,
        "coeffs": {"volume": -0.04, "effects": 0.05},
        "clip": (0.0, 0.3),
    },
    "micro_dynamics_fast_window_ms": {
        "base": 8.0,
        "coeffs": {
            "bass": 0.2,
            "volume": -0.6,
            "effects": 2.0,
        },
        "clip": (1.0, 20.0),
    },
    "micro_dynamics_slow_window_ms": {
        "base": 58.0,
        "coeffs": {
            "bass": 2.0,
            "volume": -10.0,
            "effects": 12.0,
        },
        "clip": (10.0, 120.0),
    },
    "micro_dynamics_transient_bias": {
        "base": 0.73,
        "coeffs": {"volume": -0.03, "effects": 0.07},
        "clip": (0.0, 1.0),
    },
}

_DERIVED_FIELDS: set[str] = set(_DERIVED_FIELD_SPECS) | {
    "delivery_decoded_true_peak_dbfs",
    "limiter_recovery_style",
    "low_end_mono_tightening",
}


def _normalize_macro_value(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _coerce_direct_macro(
    value: float | None,
    default: float,
) -> float:
    if value is None:
        return default
    return _normalize_macro_value(value)


def _coerce_legacy_macro(value: float) -> float:
    return float((np.clip(float(value), -1.0, 1.0) + 1.0) * 0.5)


def _signed_macro_value(value: float) -> float:
    return float(_normalize_macro_value(value) * 2.0 - 1.0)


def _apply_numeric_spec(
    spec: dict[str, Any],
    bass: float,
    volume: float,
    effects: float,
) -> Any:
    kind = str(spec.get("kind", "float"))
    base_value = spec["base"]
    if kind == "bool":
        return bool(base_value)
    if base_value is None:
        return None

    coeffs = spec.get("coeffs", {})
    resolved = float(base_value)
    resolved += bass * float(coeffs.get("bass", 0.0))
    resolved += volume * float(coeffs.get("volume", 0.0))
    resolved += effects * float(coeffs.get("effects", 0.0))

    clip_range = spec.get("clip")
    if clip_range is not None:
        resolved = float(np.clip(resolved, clip_range[0], clip_range[1]))

    if kind == "int":
        return int(round(resolved))
    return resolved


@dataclass
class SmartMasteringConfig:
    num_bands: int = 6
    preset_name: str | None = "balanced"
    delivery_profile: str | None = "lossless"
    resampling_target: int = 44100
    smoothing_fraction: float | None = 0.2
    correction_strength: float = 1.0
    analysis_low_hz: float = 10.0
    low_cut: float | None = None
    high_cut: float | None = None
    bass: float = 0.5
    volume: float = 0.5
    effects: float = 0.5
    _detail_overrides: dict[str, Any] = field(
        default_factory=dict,
        repr=False,
        compare=False,
    )
    _band_intensity_override: float | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    _preset_macro_defaults: ClassVar[dict[str, dict[str, float]]] = (
        _PRESET_MACRO_DEFAULTS
    )
    _legacy_macro_aliases: ClassVar[dict[str, tuple[str, ...]]] = (
        _LEGACY_MACRO_ALIASES
    )
    _derived_field_specs: ClassVar[dict[str, dict[str, Any]]] = (
        _DERIVED_FIELD_SPECS
    )
    _derived_fields: ClassVar[set[str]] = _DERIVED_FIELDS

    def __init__(
        self,
        num_bands: int = 6,
        preset_name: str | None = "balanced",
        delivery_profile: str | None = "lossless",
        resampling_target: int = 44100,
        smoothing_fraction: float | None = 0.2,
        correction_strength: float = 1.0,
        analysis_low_hz: float = 10.0,
        low_cut: float | None = None,
        high_cut: float | None = None,
        bass: float | None = None,
        volume: float | None = None,
        effects: float | None = None,
        _detail_overrides: dict[str, Any] | None = None,
        _band_intensity_override: float | None = None,
        **legacy_overrides: Any,
    ) -> None:
        normalized_preset = self._normalize_preset_name(preset_name)
        preset_defaults = self._preset_macro_defaults[normalized_preset]
        direct_macros = {
            "bass": bass,
            "volume": volume,
            "effects": effects,
        }
        resolved_macros = {
            macro_name: self._resolve_macro_value(
                macro_name,
                direct_value=direct_macros[macro_name],
                default_value=preset_defaults[macro_name],
                legacy_overrides=legacy_overrides,
            )
            for macro_name in ("bass", "volume", "effects")
        }

        object.__setattr__(self, "num_bands", int(num_bands))
        object.__setattr__(self, "preset_name", normalized_preset)
        object.__setattr__(self, "delivery_profile", delivery_profile)
        object.__setattr__(self, "resampling_target", int(resampling_target))
        object.__setattr__(self, "smoothing_fraction", smoothing_fraction)
        object.__setattr__(
            self, "correction_strength", float(correction_strength)
        )
        object.__setattr__(self, "analysis_low_hz", float(analysis_low_hz))
        object.__setattr__(
            self,
            "low_cut",
            None if low_cut is None else float(low_cut),
        )
        object.__setattr__(
            self,
            "high_cut",
            None if high_cut is None else float(high_cut),
        )
        object.__setattr__(
            self,
            "bass",
            resolved_macros["bass"],
        )
        object.__setattr__(self, "volume", resolved_macros["volume"])
        object.__setattr__(self, "effects", resolved_macros["effects"])
        object.__setattr__(
            self,
            "_detail_overrides",
            {} if _detail_overrides is None else dict(_detail_overrides),
        )
        object.__setattr__(
            self,
            "_band_intensity_override",
            None
            if _band_intensity_override is None
            else float(_band_intensity_override),
        )

        for key, value in legacy_overrides.items():
            if key in self._iter_legacy_macro_aliases():
                continue
            if key not in self._derived_fields:
                raise TypeError(
                    f"SmartMasteringConfig.__init__() got an unexpected keyword argument '{key}'"
                )
            self._detail_overrides[key] = value

    def __getattr__(self, name: str) -> Any:
        if name in self._iter_legacy_macro_aliases():
            return self._legacy_macro_value(name)
        if name in self._derived_fields:
            if name in self._detail_overrides:
                return self._detail_overrides[name]
            return self._resolve_derived_value(name)
        raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_") or "_detail_overrides" not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        if name in self._derived_fields:
            self._detail_overrides[name] = value
            return
        if name in self._iter_legacy_macro_aliases():
            self._set_legacy_macro_value(name, value)
            return
        if name in {"bass", "volume", "effects"}:
            object.__setattr__(self, name, _normalize_macro_value(value))
            return
        object.__setattr__(self, name, value)

    @classmethod
    def preset_names(cls) -> tuple[str, ...]:
        return (
            "balanced",
            "edm",
            "vocal",
        )

    @classmethod
    def balanced(cls) -> SmartMasteringConfig:
        return cls(preset_name="balanced")

    @classmethod
    def edm(cls) -> SmartMasteringConfig:
        return cls(preset_name="edm")

    @classmethod
    def vocal(cls) -> SmartMasteringConfig:
        return cls(preset_name="vocal")

    @classmethod
    def from_preset(cls, name: str | None) -> SmartMasteringConfig:
        return cls(preset_name=cls._normalize_preset_name(name))

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

    @classmethod
    def _normalize_preset_name(cls, name: str | None) -> str:
        if name is None:
            return "balanced"
        normalized = str(name).strip().lower()
        if not normalized or normalized == "auto":
            return "balanced"
        if normalized not in cls._preset_macro_defaults:
            raise ValueError(f"Unknown mastering preset: {name}")
        return normalized

    @classmethod
    def _iter_legacy_macro_aliases(cls) -> set[str]:
        aliases: set[str] = set()
        for values in cls._legacy_macro_aliases.values():
            aliases.update(values)
        return aliases

    @classmethod
    def _resolve_macro_value(
        cls,
        macro_name: str,
        *,
        direct_value: float | None,
        default_value: float,
        legacy_overrides: dict[str, Any],
    ) -> float:
        if direct_value is not None:
            return _coerce_direct_macro(direct_value, default_value)

        alias_values = [
            _coerce_legacy_macro(legacy_overrides[alias_name])
            for alias_name in cls._legacy_macro_aliases[macro_name]
            if alias_name in legacy_overrides
        ]
        if alias_values:
            return float(np.mean(alias_values, dtype=np.float64))
        return default_value

    def _macro_vector(self) -> tuple[float, float, float]:
        return (
            _signed_macro_value(self.bass),
            _signed_macro_value(self.volume),
            _signed_macro_value(self.effects),
        )

    def _legacy_macro_value(self, alias_name: str) -> float:
        if alias_name == "tone":
            return _signed_macro_value(self.bass)
        if alias_name in {"loudness", "compression"}:
            return _signed_macro_value(self.volume)
        return _signed_macro_value(self.effects)

    def _set_legacy_macro_value(self, alias_name: str, value: Any) -> None:
        normalized = _coerce_legacy_macro(value)
        if alias_name == "tone":
            object.__setattr__(self, "bass", normalized)
            return
        if alias_name in {"loudness", "compression"}:
            object.__setattr__(self, "volume", normalized)
            return
        object.__setattr__(self, "effects", normalized)

    def _resolve_derived_value(self, field_name: str) -> Any:
        if field_name == "delivery_decoded_true_peak_dbfs":
            return self._resolve_delivery_decoded_true_peak_dbfs()
        if field_name == "limiter_recovery_style":
            return self._resolve_limiter_recovery_style()
        if field_name == "low_end_mono_tightening":
            return self._resolve_low_end_mono_tightening()

        spec = self._derived_field_specs[field_name]
        if (
            field_name == "intensity"
            and self._band_intensity_override is not None
        ):
            return float(np.clip(self._band_intensity_override, 0.25, 2.5))

        bass, volume, effects = self._macro_vector()
        return _apply_numeric_spec(spec, bass, volume, effects)

    def _resolve_delivery_decoded_true_peak_dbfs(self) -> float | None:
        if self.volume < 0.9 or self.effects > 0.3 or self.bass < 0.75:
            return None
        loudness_pressure = float(np.clip((self.volume - 0.9) / 0.1, 0.0, 1.0))
        bass_pressure = float(np.clip((self.bass - 0.75) / 0.25, 0.0, 1.0))
        effect_restraint = float(np.clip((0.3 - self.effects) / 0.3, 0.0, 1.0))
        resolved = -0.3 - loudness_pressure * 0.45
        resolved -= bass_pressure * 0.1 + effect_restraint * 0.15
        return float(np.clip(resolved, -1.0, -0.3))

    def _resolve_limiter_recovery_style(self) -> str:
        volume = _signed_macro_value(self.volume)
        effects = _signed_macro_value(self.effects)
        if volume >= 0.45:
            return "tight"
        if effects >= 0.35 and volume <= 0.0:
            return "glue"
        return "balanced"

    def _resolve_low_end_mono_tightening(self) -> str:
        bass, volume, effects = self._macro_vector()
        pressure = volume * 0.7 + bass * 0.4 - effects * 0.3
        if pressure >= 0.75:
            return "firm"
        if pressure <= -0.25:
            return "gentle"
        return "balanced"
