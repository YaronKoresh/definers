from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .config import SmartMasteringConfig


@dataclass(frozen=True, slots=True)
class StemMasteringPlan:
    stem_name: str
    mix_gain_db: float
    overrides: dict[str, float | str | None]


def _stem_mix_gain_linear(mix_gain_db: float) -> float:
    return float(10.0 ** (float(mix_gain_db) / 20.0))


def _level_to_dbfs(level: float) -> float:
    safe_level = float(level)
    if not np.isfinite(safe_level) or safe_level <= 0.0:
        return -120.0
    return float(20.0 * np.log10(safe_level))


def _as_stereo(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim == 1:
        return np.stack([array, array], axis=0)
    if array.ndim == 2 and array.shape[0] <= array.shape[-1]:
        return array.astype(np.float32, copy=False)
    if array.ndim == 2:
        return array.T.astype(np.float32, copy=False)
    raise ValueError("Unsupported stem shape")


def _match_signal_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32).reshape(-1)
    if array.shape[-1] > target_length:
        return array[:target_length]
    if array.shape[-1] < target_length:
        return np.pad(array, (0, target_length - array.shape[-1]))
    return array


def _resolve_stem_mix_target_dbfs(stem_name: str) -> float:
    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return -12.8
    if normalized_name == "bass":
        return -15.0
    if normalized_name == "vocals":
        return -17.6
    if normalized_name == "guitar":
        return -18.6
    if normalized_name == "piano":
        return -18.9
    return -19.8


def _resolve_stem_mix_balance_profile(
    stem_name: str,
) -> tuple[float, float, float]:
    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return 0.78, -4.0, 4.6
    if normalized_name == "bass":
        return 0.64, -4.4, 3.8
    if normalized_name == "vocals":
        return 0.52, -4.5, 2.6
    if normalized_name == "guitar":
        return 0.56, -4.5, 2.8
    if normalized_name == "piano":
        return 0.54, -4.5, 2.8
    return 0.58, -4.5, 3.0


def _measure_active_stem_level_dbfs(signal: np.ndarray) -> float:
    stereo_signal = _as_stereo(signal)
    if stereo_signal.size == 0:
        return -120.0

    mono_energy = np.mean(np.abs(stereo_signal), axis=0, dtype=np.float32)
    if not np.any(mono_energy > 0.0):
        return -120.0

    activity_threshold = float(
        max(
            np.percentile(mono_energy, 65.0),
            np.mean(mono_energy, dtype=np.float32) * 0.9,
        )
    )
    active_mask = mono_energy >= activity_threshold
    active_signal = stereo_signal[:, active_mask] if np.any(active_mask) else stereo_signal
    rms_level = float(
        np.sqrt(
            np.mean(
                np.square(np.asarray(active_signal, dtype=np.float64)),
                dtype=np.float64,
            )
        )
    )
    return _level_to_dbfs(rms_level)


def _apply_stem_mix_balance(
    signal: np.ndarray,
    stem_name: str,
    base_mix_gain_db: float,
) -> tuple[np.ndarray, float]:
    stereo_signal = _as_stereo(signal)
    measured_dbfs = _measure_active_stem_level_dbfs(stereo_signal)
    target_dbfs = _resolve_stem_mix_target_dbfs(stem_name)
    correction_weight, minimum_gain_db, maximum_gain_db = (
        _resolve_stem_mix_balance_profile(stem_name)
    )
    corrective_gain_db = float(np.clip(target_dbfs - measured_dbfs, -4.0, 4.0))
    applied_gain_db = float(
        np.clip(
            base_mix_gain_db + corrective_gain_db * correction_weight,
            minimum_gain_db,
            maximum_gain_db,
        )
    )
    balanced_signal = stereo_signal * _stem_mix_gain_linear(applied_gain_db)

    peak = float(np.max(np.abs(balanced_signal))) if balanced_signal.size else 0.0
    if peak > 0.98:
        balanced_signal = balanced_signal * (0.98 / peak)

    return balanced_signal.astype(np.float32, copy=False), applied_gain_db


def _prepare_mixed_stem_layers(
    stem_layers: Mapping[str, tuple[int, np.ndarray]],
    *,
    mix_headroom_db: float = 6.0,
    target_sample_rate: int | None = None,
    resample_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> tuple[int, dict[str, tuple[int, np.ndarray]], np.ndarray]:
    if not stem_layers:
        raise ValueError("No stem layers to mix")

    resolved_sample_rate = int(
        target_sample_rate
        or max(int(sample_rate) for sample_rate, _signal in stem_layers.values())
    )
    prepared_layers: dict[str, tuple[int, np.ndarray]] = {}
    max_length = 0
    channel_count = 2

    for stem_name, (sample_rate, signal) in stem_layers.items():
        stem_signal = _as_stereo(signal)
        if int(sample_rate) != resolved_sample_rate:
            if resample_fn is None:
                from .dsp import resample as _resample

                resample_fn = _resample
            stem_signal = _as_stereo(
                resample_fn(stem_signal, int(sample_rate), resolved_sample_rate)
            )
        prepared_layers[stem_name] = (
            resolved_sample_rate,
            stem_signal.astype(np.float32, copy=False),
        )
        max_length = max(max_length, stem_signal.shape[-1])
        channel_count = stem_signal.shape[0]

    mixed = np.zeros((channel_count, max_length), dtype=np.float32)
    aligned_layers: dict[str, tuple[int, np.ndarray]] = {}

    for stem_name, (sample_rate, signal) in prepared_layers.items():
        aligned_signal = signal
        if aligned_signal.shape[-1] < max_length:
            aligned_signal = np.pad(
                aligned_signal,
                ((0, 0), (0, max_length - aligned_signal.shape[-1])),
            )
        aligned_signal = aligned_signal.astype(np.float32, copy=False)
        aligned_layers[stem_name] = (sample_rate, aligned_signal)
        mixed = mixed + aligned_signal

    safe_headroom_db = float(max(mix_headroom_db, 0.0))
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 0.0:
        target_peak = float(10.0 ** (-safe_headroom_db / 20.0))
        if peak > target_peak:
            mix_scale = target_peak / peak
            mixed = mixed * mix_scale
            aligned_layers = {
                stem_name: (sample_rate, signal * mix_scale)
                for stem_name, (sample_rate, signal) in aligned_layers.items()
            }

    return resolved_sample_rate, aligned_layers, mixed.astype(np.float32, copy=False)


def _save_mastered_stem_layers(
    stem_layers: Mapping[str, tuple[int, np.ndarray]],
    *,
    output_dir: str,
    save_audio_fn: Callable[..., str | None],
    output_format: str,
    bit_depth: int,
    bitrate: int,
    compression_level: int,
) -> None:
    resolved_output_dir = Path(output_dir)
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    normalized_format = str(output_format).strip().lower().lstrip(".") or "wav"

    for stem_name, (sample_rate, signal) in stem_layers.items():
        save_audio_fn(
            destination_path=str(
                resolved_output_dir / f"{stem_name}_mastered.{normalized_format}"
            ),
            audio_signal=_as_stereo(signal),
            sample_rate=int(sample_rate),
            bit_depth=bit_depth,
            bitrate=bitrate,
            compression_level=compression_level,
        )


def _resolve_stem_tone_layers(
    stem_name: str,
    base_config: SmartMasteringConfig,
) -> tuple[tuple[float, float], ...]:
    if not bool(getattr(base_config, "stem_tone_enrichment_enabled", True)):
        return ()

    mix_amount = float(
        np.clip(getattr(base_config, "stem_tone_enrichment_mix", 0.08), 0.0, 0.24)
    )
    if mix_amount <= 0.0:
        return ()

    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return ()
    if normalized_name == "bass":
        return (
            (-0.07, mix_amount * 0.08),
            (12.0, mix_amount * 0.28),
        )
    if normalized_name == "vocals":
        return (
            (-12.0, mix_amount * 0.24),
            (-0.18, mix_amount * 0.16),
            (0.18, mix_amount * 0.14),
            (12.0, mix_amount * 0.18),
        )
    if normalized_name == "guitar":
        return (
            (-12.0, mix_amount * 0.2),
            (-0.12, mix_amount * 0.12),
            (0.12, mix_amount * 0.1),
            (12.0, mix_amount * 0.16),
        )
    if normalized_name == "piano":
        return (
            (-12.0, mix_amount * 0.16),
            (-0.08, mix_amount * 0.1),
            (0.08, mix_amount * 0.1),
            (12.0, mix_amount * 0.18),
        )
    return (
        (-12.0, mix_amount * 0.18),
        (-0.1, mix_amount * 0.1),
        (0.1, mix_amount * 0.09),
        (12.0, mix_amount * 0.15),
    )


def _resolve_parallel_stem_workers(stem_count: int) -> int:
    if stem_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(stem_count, cpu_count, 3))


def _load_pitch_shift_fn() -> Callable[[np.ndarray, int, float], np.ndarray] | None:
    try:
        import librosa
    except Exception:
        return None

    return lambda channel, sample_rate, semitones: np.asarray(
        librosa.effects.pitch_shift(
            y=np.asarray(channel, dtype=np.float32),
            sr=int(sample_rate),
            n_steps=float(semitones),
        ),
        dtype=np.float32,
    )


def _apply_stem_tone_enrichment(
    signal: np.ndarray,
    sample_rate: int,
    stem_name: str,
    base_config: SmartMasteringConfig,
    *,
    pitch_shift_fn: Callable[[np.ndarray, int, float], np.ndarray] | None = None,
) -> np.ndarray:
    stereo_signal = _as_stereo(signal)
    if stereo_signal.shape[-1] < 128:
        return stereo_signal
    tone_layers = _resolve_stem_tone_layers(stem_name, base_config)
    if not tone_layers:
        return stereo_signal

    resolved_pitch_shift_fn = pitch_shift_fn or _load_pitch_shift_fn()
    if resolved_pitch_shift_fn is None:
        return stereo_signal

    enriched = stereo_signal.astype(np.float32, copy=True)
    dry_peak = float(np.max(np.abs(stereo_signal))) if stereo_signal.size else 0.0

    for semitones, mix in tone_layers:
        if mix <= 0.0:
            continue
        shifted_channels = []
        for channel in stereo_signal:
            shifted_channel = resolved_pitch_shift_fn(channel, sample_rate, semitones)
            shifted_channels.append(
                _match_signal_length(shifted_channel, stereo_signal.shape[-1])
            )
        shifted = np.vstack(shifted_channels).astype(np.float32, copy=False)
        shifted_peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
        if dry_peak > 1e-6 and shifted_peak > dry_peak:
            shifted = shifted * (dry_peak / shifted_peak)
        enriched = enriched + shifted * float(mix)

    peak = float(np.max(np.abs(enriched))) if enriched.size else 0.0
    allowed_peak = dry_peak * 1.18 if dry_peak > 1e-6 else 1.0
    if peak > allowed_peak and allowed_peak > 1e-6:
        enriched = enriched * (allowed_peak / peak)

    return enriched.astype(np.float32, copy=False)


def resolve_stem_mastering_plan(
    stem_name: str,
    base_config: SmartMasteringConfig,
) -> StemMasteringPlan:
    normalized_name = str(stem_name).strip().lower() or "other"
    shared_target_lufs = float(base_config.target_lufs - 3.2)
    shared_ceil_db = float(min(base_config.ceil_db, -1.2))
    shared_drive_db = float(max(base_config.drive_db * 0.78, 0.35))
    shared_soft_clip_ratio = float(
        np.clip(base_config.limiter_soft_clip_ratio * 0.72, 0.0, 0.5)
    )
    shared_saturation_ratio = float(
        np.clip(base_config.pre_limiter_saturation_ratio * 0.68, 0.0, 0.4)
    )
    shared_max_final_boost_db = float(
        max(base_config.max_final_boost_db * 0.55, 1.4)
    )
    shared_width = float(np.clip(base_config.stereo_width, 0.92, 1.18))
    shared_mix_gain_db = 0.0
    shared_overrides: dict[str, float | str | None] = {
        "target_lufs": shared_target_lufs,
        "ceil_db": shared_ceil_db,
        "drive_db": shared_drive_db,
        "max_final_boost_db": shared_max_final_boost_db,
        "limiter_soft_clip_ratio": shared_soft_clip_ratio,
        "pre_limiter_saturation_ratio": shared_saturation_ratio,
        "stereo_width": shared_width,
        "contract_max_short_term_lufs": None,
        "contract_max_momentary_lufs": None,
        "contract_min_crest_factor_db": None,
        "contract_max_crest_factor_db": None,
        "contract_max_stereo_width_ratio": None,
        "contract_min_low_end_mono_ratio": None,
    }

    if normalized_name == "drums":
        shared_mix_gain_db = 1.35
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.35,
                "bass_boost_db_per_oct": base_config.bass_boost_db_per_oct + 0.22,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct + 0.18,
                "exciter_mix": float(np.clip(base_config.exciter_mix + 0.08, 0.0, 1.0)),
                "exciter_max_drive": base_config.exciter_max_drive + 0.45,
                "exciter_high_frequency_cutoff_hz": None
                if base_config.exciter_high_frequency_cutoff_hz is None
                else max(base_config.exciter_high_frequency_cutoff_hz - 700.0, 3500.0),
                "limiter_recovery_style": "tight",
                "low_end_mono_tightening": "firm",
                "low_end_mono_tightening_amount": 0.95,
                "micro_dynamics_strength": float(
                    np.clip(base_config.micro_dynamics_strength * 0.7, 0.0, 0.18)
                ),
                "stereo_width": float(np.clip(shared_width, 0.98, 1.1)),
            }
        )
    elif normalized_name == "bass":
        shared_mix_gain_db = 0.2
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.12,
                "bass_boost_db_per_oct": base_config.bass_boost_db_per_oct + 0.35,
                "treble_boost_db_per_oct": max(base_config.treble_boost_db_per_oct - 0.4, 0.0),
                "exciter_mix": float(np.clip(base_config.exciter_mix * 0.62, 0.0, 1.0)),
                "exciter_max_drive": max(base_config.exciter_max_drive * 0.82, 0.5),
                "stereo_width": 0.94,
                "mono_bass_hz": max(base_config.mono_bass_hz, 145.0),
                "stereo_tone_variation_db": 0.0,
                "stereo_motion_mid_amount": 0.0,
                "stereo_motion_high_amount": 0.0,
                "low_end_mono_tightening": "firm",
                "low_end_mono_tightening_amount": 1.0,
                "micro_dynamics_strength": float(
                    np.clip(base_config.micro_dynamics_strength * 0.55, 0.0, 0.12)
                ),
            }
        )
    elif normalized_name == "vocals":
        shared_mix_gain_db = -0.3
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.08,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct + 0.2,
                "bass_boost_db_per_oct": max(base_config.bass_boost_db_per_oct - 0.12, 0.0),
                "exciter_mix": float(np.clip(base_config.exciter_mix + 0.1, 0.0, 1.0)),
                "exciter_max_drive": base_config.exciter_max_drive + 0.2,
                "stereo_width": float(np.clip(shared_width + 0.1, 1.0, 1.3)),
                "micro_dynamics_strength": float(
                    np.clip(base_config.micro_dynamics_strength + 0.05, 0.0, 0.32)
                ),
                "start_treble_boost_hz": max(base_config.start_treble_boost_hz - 250.0, 2200.0),
                "low_end_mono_tightening": "gentle",
                "low_end_mono_tightening_amount": 0.45,
            }
        )
    elif normalized_name == "guitar":
        shared_mix_gain_db = -0.18
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.05,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct + 0.12,
                "exciter_mix": float(np.clip(base_config.exciter_mix + 0.05, 0.0, 1.0)),
                "stereo_width": float(np.clip(shared_width + 0.05, 0.96, 1.24)),
                "micro_dynamics_strength": float(
                    np.clip(base_config.micro_dynamics_strength + 0.02, 0.0, 0.25)
                ),
                "low_end_mono_tightening": "gentle",
                "low_end_mono_tightening_amount": 0.35,
            }
        )
    elif normalized_name == "piano":
        shared_mix_gain_db = -0.22
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.03,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct + 0.1,
                "exciter_mix": float(np.clip(base_config.exciter_mix + 0.04, 0.0, 1.0)),
                "stereo_width": float(np.clip(shared_width + 0.04, 0.98, 1.24)),
                "micro_dynamics_strength": float(
                    np.clip(base_config.micro_dynamics_strength + 0.03, 0.0, 0.28)
                ),
                "low_end_mono_tightening": "gentle",
                "low_end_mono_tightening_amount": 0.3,
            }
        )
    else:
        shared_mix_gain_db = -0.48
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.05,
                "bass_boost_db_per_oct": base_config.bass_boost_db_per_oct + 0.08,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct + 0.12,
                "exciter_mix": float(np.clip(base_config.exciter_mix + 0.05, 0.0, 1.0)),
                "stereo_width": float(np.clip(shared_width + 0.08, 1.0, 1.28)),
                "micro_dynamics_strength": float(
                    np.clip(base_config.micro_dynamics_strength + 0.02, 0.0, 0.25)
                ),
                "low_end_mono_tightening": "balanced",
                "low_end_mono_tightening_amount": 0.55,
            }
        )

    return StemMasteringPlan(
        stem_name=normalized_name,
        mix_gain_db=shared_mix_gain_db,
        overrides=shared_overrides,
    )


def mix_stem_layers(
    stem_layers: Mapping[str, tuple[int, np.ndarray]],
    *,
    mix_headroom_db: float = 6.0,
    target_sample_rate: int | None = None,
    resample_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
) -> tuple[int, np.ndarray]:
    resolved_sample_rate, _aligned_layers, mixed = _prepare_mixed_stem_layers(
        stem_layers,
        mix_headroom_db=mix_headroom_db,
        target_sample_rate=target_sample_rate,
        resample_fn=resample_fn,
    )
    return resolved_sample_rate, mixed


def process_stem_layers(
    audio_path: str,
    *,
    base_config: SmartMasteringConfig,
    base_mastering_kwargs: Mapping[str, object],
    process_stem_fn: Callable[[np.ndarray, int, dict[str, object]], tuple[int, np.ndarray]],
    separate_stems_fn: Callable[..., tuple[dict[str, str], str]],
    read_audio_fn: Callable[[str], tuple[int, np.ndarray]],
    delete_fn: Callable[[str], object],
    model_name: str = "mastering",
    shifts: int = 2,
    quality_flags: Sequence[str] = (),
    mix_headroom_db: float = 6.0,
    resample_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    pitch_shift_fn: Callable[[np.ndarray, int, float], np.ndarray] | None = None,
    save_mastered_stems: bool = True,
    mastered_stems_output_dir: str | None = None,
    save_audio_fn: Callable[..., str | None] | None = None,
    mastered_stems_format: str = "wav",
    mastered_stems_bit_depth: int = 32,
    mastered_stems_bitrate: int = 320,
    mastered_stems_compression_level: int = 9,
) -> tuple[int, np.ndarray]:
    separated_output_dir: str | None = None

    def process_single_stem(
        stem_name: str,
        stem_path: str,
    ) -> tuple[str, tuple[int, np.ndarray]]:
        plan = resolve_stem_mastering_plan(stem_name, base_config)
        stem_sample_rate, stem_signal = read_audio_fn(stem_path)
        mastering_kwargs = dict(base_mastering_kwargs)
        mastering_kwargs.update(plan.overrides)
        mastering_kwargs["stem_role"] = plan.stem_name
        processed_sample_rate, processed_signal = process_stem_fn(
            stem_signal,
            stem_sample_rate,
            mastering_kwargs,
        )
        enriched_signal = _apply_stem_tone_enrichment(
            processed_signal,
            processed_sample_rate,
            plan.stem_name,
            base_config,
            pitch_shift_fn=pitch_shift_fn,
        )
        balanced_signal, _applied_mix_gain_db = _apply_stem_mix_balance(
            enriched_signal,
            plan.stem_name,
            plan.mix_gain_db,
        )
        return plan.stem_name, (
            processed_sample_rate,
            balanced_signal,
        )

    try:
        stem_paths, separated_output_dir = separate_stems_fn(
            audio_path,
            model_name=model_name,
            shifts=shifts,
            quality_flags=tuple(
                str(flag).strip()
                for flag in quality_flags
                if str(flag).strip()
            ),
        )
        if not stem_paths:
            raise ValueError("No mastering stems were produced")

        mastered_layers: dict[str, tuple[int, np.ndarray]] = {}
        worker_count = _resolve_parallel_stem_workers(len(stem_paths))
        if worker_count <= 1:
            for stem_name, stem_path in stem_paths.items():
                normalized_name, processed_result = process_single_stem(
                    stem_name,
                    stem_path,
                )
                mastered_layers[normalized_name] = processed_result
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_by_stem = {
                    stem_name: executor.submit(
                        process_single_stem,
                        stem_name,
                        stem_path,
                    )
                    for stem_name, stem_path in stem_paths.items()
                }
                for stem_name in stem_paths:
                    normalized_name, processed_result = future_by_stem[
                        stem_name
                    ].result()
                    mastered_layers[normalized_name] = processed_result

        mixed_sample_rate, aligned_layers, mixed_signal = _prepare_mixed_stem_layers(
            mastered_layers,
            mix_headroom_db=mix_headroom_db,
            resample_fn=resample_fn,
        )
        if (
            save_mastered_stems
            and mastered_stems_output_dir is not None
            and save_audio_fn is not None
        ):
            _save_mastered_stem_layers(
                aligned_layers,
                output_dir=mastered_stems_output_dir,
                save_audio_fn=save_audio_fn,
                output_format=mastered_stems_format,
                bit_depth=mastered_stems_bit_depth,
                bitrate=mastered_stems_bitrate,
                compression_level=mastered_stems_compression_level,
            )

        return mixed_sample_rate, mixed_signal
    finally:
        if separated_output_dir is not None:
            delete_fn(separated_output_dir)


__all__ = (
    "StemMasteringPlan",
    "mix_stem_layers",
    "process_stem_layers",
    "resolve_stem_mastering_plan",
)