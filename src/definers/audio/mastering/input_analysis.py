from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from .loudness import measure_mastering_loudness as _measure_mastering_loudness
from .reference import (
    measure_spectral_tilt as _measure_spectral_tilt,
    measure_stereo_motion as _measure_stereo_motion,
    measure_transient_density as _measure_transient_density,
)


def _resolve_package_symbol(name: str, fallback):
    try:
        module = import_module(__package__)
    except Exception:
        return fallback
    return getattr(module, name, fallback)


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

    measure_mastering_loudness_fn = _resolve_package_symbol(
        "_measure_mastering_loudness",
        _measure_mastering_loudness,
    )
    measure_spectral_tilt_fn = _resolve_package_symbol(
        "_measure_spectral_tilt",
        _measure_spectral_tilt,
    )
    measure_transient_density_fn = _resolve_package_symbol(
        "_measure_transient_density",
        _measure_transient_density,
    )
    measure_stereo_motion_fn = _resolve_package_symbol(
        "_measure_stereo_motion",
        _measure_stereo_motion,
    )
    measure_preset_band_profile_fn = _resolve_package_symbol(
        "_measure_preset_band_profile",
        _measure_preset_band_profile,
    )

    try:
        loudness_metrics = measure_mastering_loudness_fn(
            signal_array,
            int(sample_rate),
        )
        spectral_tilt = measure_spectral_tilt_fn(signal_array, int(sample_rate))
        transient_density = measure_transient_density_fn(
            signal_array,
            int(sample_rate),
        )
        stereo_motion = measure_stereo_motion_fn(
            signal_array,
            int(sample_rate),
        )
        band_profile = measure_preset_band_profile_fn(
            signal_array,
            int(sample_rate),
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
    collect_mastering_input_metrics = _resolve_package_symbol(
        "_collect_mastering_input_metrics",
        _collect_mastering_input_metrics,
    )
    select_mastering_preset_from_metrics = _resolve_package_symbol(
        "_select_mastering_preset_from_metrics",
        _select_mastering_preset_from_metrics,
    )
    resolve_mastering_quality_flags = _resolve_package_symbol(
        "_resolve_mastering_quality_flags",
        _resolve_mastering_quality_flags,
    )
    metrics = collect_mastering_input_metrics(signal_to_analyze, sample_rate)
    target_sample_rate = _resolve_mastering_processing_sample_rate(sample_rate)
    if metrics is None:
        return MasteringInputAnalysis(
            preset_name="balanced",
            quality_flags=(),
            target_sample_rate=target_sample_rate,
            metrics=None,
        )

    return MasteringInputAnalysis(
        preset_name=select_mastering_preset_from_metrics(metrics),
        quality_flags=resolve_mastering_quality_flags(metrics, sample_rate),
        target_sample_rate=target_sample_rate,
        metrics=metrics,
    )


def _select_mastering_preset(
    signal_to_analyze: np.ndarray,
    sample_rate: int,
) -> str:
    collect_mastering_input_metrics = _resolve_package_symbol(
        "_collect_mastering_input_metrics",
        _collect_mastering_input_metrics,
    )
    select_mastering_preset_from_metrics = _resolve_package_symbol(
        "_select_mastering_preset_from_metrics",
        _select_mastering_preset_from_metrics,
    )
    metrics = collect_mastering_input_metrics(signal_to_analyze, sample_rate)
    if metrics is None:
        return "balanced"
    return select_mastering_preset_from_metrics(metrics)


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
    resolve_explicit_preset_name = _resolve_package_symbol(
        "_resolve_explicit_preset_name",
        _resolve_explicit_preset_name,
    )
    select_mastering_preset = _resolve_package_symbol(
        "_select_mastering_preset",
        _select_mastering_preset,
    )
    resolved_kwargs = dict(mastering_kwargs)
    explicit_preset_name = resolve_explicit_preset_name(resolved_kwargs)
    if explicit_preset_name is not None:
        return resolved_kwargs

    if input_analysis is None:
        selected_preset = select_mastering_preset(
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
