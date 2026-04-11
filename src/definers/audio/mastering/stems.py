from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from ..config import SmartMasteringConfig


@dataclass(frozen=True, slots=True)
class StemMasteringPlan:
    stem_name: str
    mix_gain_db: float
    overrides: dict[str, float | str | None]


_STEM_SUM_ATTENUATION_LINEAR = float(10.0 ** (-6.0 / 20.0))
_STEM_SAVED_PEAK_CEILING_LINEAR = 0.98
_STEM_TONE_ENRICHMENT_PEAK_GROWTH_LIMIT = 1.08


def _moving_average(signal: np.ndarray, window_size: int) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    resolved_window = max(int(window_size), 1)
    if array.size == 0 or resolved_window <= 1:
        return array.astype(np.float32, copy=False)
    if array.ndim == 1:
        padding = resolved_window // 2
        padded = np.pad(
            array,
            (padding, resolved_window - 1 - padding),
            mode="edge",
        )
        cumulative = np.cumsum(
            np.insert(padded.astype(np.float64, copy=False), 0, 0.0)
        )
        averaged = (
            cumulative[resolved_window:] - cumulative[:-resolved_window]
        ) / float(resolved_window)
        return averaged.astype(np.float32, copy=False)
    return np.vstack(
        [
            _moving_average(channel, resolved_window)
            for channel in np.asarray(array, dtype=np.float32)
        ]
    ).astype(np.float32, copy=False)


def _stem_mix_gain_linear(mix_gain_db: float) -> float:
    return float(10.0 ** (float(mix_gain_db) / 20.0))


def _sanitize_stem_signal(
    signal: np.ndarray,
    *,
    peak_ceiling: float = _STEM_SAVED_PEAK_CEILING_LINEAR,
) -> np.ndarray:
    stabilized = np.nan_to_num(
        np.asarray(signal, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    peak = float(np.max(np.abs(stabilized))) if stabilized.size else 0.0
    if peak > float(peak_ceiling) > 0.0:
        stabilized = stabilized * (float(peak_ceiling) / peak)
    return stabilized.astype(np.float32, copy=False)


def _constrain_stem_peak_growth(
    signal: np.ndarray,
    *,
    reference_signal: np.ndarray | None,
    peak_growth_limit: float = _STEM_TONE_ENRICHMENT_PEAK_GROWTH_LIMIT,
    absolute_peak_ceiling: float = _STEM_SAVED_PEAK_CEILING_LINEAR,
) -> np.ndarray:
    stabilized = _sanitize_stem_signal(
        signal,
        peak_ceiling=absolute_peak_ceiling,
    )
    if reference_signal is None:
        return stabilized

    reference_array = np.asarray(reference_signal, dtype=np.float32)
    reference_peak = (
        float(np.max(np.abs(reference_array))) if reference_array.size else 0.0
    )
    if reference_peak <= 1e-6:
        return stabilized

    allowed_peak = min(
        float(absolute_peak_ceiling),
        max(reference_peak, reference_peak * float(peak_growth_limit)),
    )
    stabilized_peak = (
        float(np.max(np.abs(stabilized))) if stabilized.size else 0.0
    )
    if stabilized_peak > allowed_peak > 0.0:
        stabilized = stabilized * (allowed_peak / stabilized_peak)
    return stabilized.astype(np.float32, copy=False)


def _level_to_dbfs(level: float) -> float:
    safe_level = float(level)
    if not np.isfinite(safe_level) or safe_level <= 0.0:
        return -120.0
    return float(20.0 * np.log10(safe_level))


def _signal_rms(signal: np.ndarray) -> float:
    array = np.asarray(signal, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float32)))


def _measure_stem_stereo_width_ratio(signal: np.ndarray) -> float:
    stereo_signal = _as_stereo(signal)
    mid = 0.5 * (stereo_signal[0] + stereo_signal[1])
    side = 0.5 * (stereo_signal[0] - stereo_signal[1])
    mid_rms = _signal_rms(mid)
    side_rms = _signal_rms(side)
    if mid_rms + side_rms <= 1e-6:
        return 0.0
    return float(np.clip(side_rms / (mid_rms + side_rms), 0.0, 1.0))


def _resolve_stem_override_float(
    stem_overrides: Mapping[str, float | str | None] | None,
    key: str,
    fallback: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    value = float(fallback)
    if stem_overrides is not None:
        override_value = stem_overrides.get(key)
        if isinstance(override_value, (int, float)):
            value = float(override_value)
    if minimum is not None or maximum is not None:
        lower_bound = minimum if minimum is not None else value
        upper_bound = maximum if maximum is not None else value
        value = float(np.clip(value, lower_bound, upper_bound))
    return value


def _as_stereo(signal: np.ndarray) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32)
    if array.ndim == 1:
        return np.stack([array, array], axis=0)
    if array.ndim == 2:
        oriented = (
            array if array.shape[0] <= array.shape[-1] else array.T
        ).astype(np.float32, copy=False)
        if oriented.shape[0] == 1:
            return np.repeat(oriented, 2, axis=0)
        if oriented.shape[0] >= 2:
            return oriented[:2]
    raise ValueError("Unsupported stem shape")


def _synchronize_stem_mix_shape(
    signal: np.ndarray,
    *,
    target_channels: int,
    target_length: int,
) -> np.ndarray:
    synchronized = _as_stereo(signal)
    if synchronized.shape[0] > target_channels:
        synchronized = synchronized[:target_channels]
    elif synchronized.shape[0] < target_channels:
        synchronized = np.pad(
            synchronized,
            ((0, target_channels - synchronized.shape[0]), (0, 0)),
        )
    if synchronized.shape[-1] > target_length:
        synchronized = synchronized[:, :target_length]
    elif synchronized.shape[-1] < target_length:
        synchronized = np.pad(
            synchronized,
            ((0, 0), (0, target_length - synchronized.shape[-1])),
        )
    return synchronized.astype(np.float32, copy=False)


def _match_signal_length(signal: np.ndarray, target_length: int) -> np.ndarray:
    array = np.asarray(signal, dtype=np.float32).reshape(-1)
    if array.shape[-1] > target_length:
        return array[:target_length]
    if array.shape[-1] < target_length:
        return np.pad(array, (0, target_length - array.shape[-1]))
    return array


def _resolve_stem_mix_target_dbfs(
    stem_name: str,
    *,
    vocal_pullback_db: float = 0.0,
) -> float:
    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return -11.9
    if normalized_name == "bass":
        return -13.6
    if normalized_name == "vocals":
        return -19.2 - float(np.clip(vocal_pullback_db, 0.0, 3.0))
    if normalized_name == "guitar":
        return -18.6
    if normalized_name == "piano":
        return -18.9
    return -19.0


def _resolve_stem_mix_balance_profile(
    stem_name: str,
    *,
    vocal_pullback_db: float = 0.0,
) -> tuple[float, float, float]:
    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return 0.88, -3.6, 5.4
    if normalized_name == "bass":
        return 0.82, -3.6, 5.2
    if normalized_name == "vocals":
        resolved_pullback_db = float(np.clip(vocal_pullback_db, 0.0, 3.0))
        return (
            0.46,
            -5.4 - resolved_pullback_db,
            1.8 - resolved_pullback_db * 0.65,
        )
    if normalized_name == "guitar":
        return 0.56, -4.5, 2.8
    if normalized_name == "piano":
        return 0.54, -4.5, 2.8
    return 0.64, -4.0, 3.4


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
    active_signal = (
        stereo_signal[:, active_mask] if np.any(active_mask) else stereo_signal
    )
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
    *,
    vocal_pullback_db: float = 0.0,
) -> tuple[np.ndarray, float]:
    stereo_signal = _as_stereo(signal)
    measured_dbfs = _measure_active_stem_level_dbfs(stereo_signal)
    target_dbfs = _resolve_stem_mix_target_dbfs(
        stem_name,
        vocal_pullback_db=vocal_pullback_db,
    )
    correction_weight, minimum_gain_db, maximum_gain_db = (
        _resolve_stem_mix_balance_profile(
            stem_name,
            vocal_pullback_db=vocal_pullback_db,
        )
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

    return (
        _sanitize_stem_signal(
            balanced_signal,
            peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
        ),
        applied_gain_db,
    )


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
        or max(
            int(sample_rate) for sample_rate, _signal in stem_layers.values()
        )
    )
    prepared_layers: dict[str, tuple[int, np.ndarray]] = {}
    max_length = 0
    channel_count = 2

    for stem_name, (sample_rate, signal) in stem_layers.items():
        stem_signal = _as_stereo(signal)
        if int(sample_rate) != resolved_sample_rate:
            if resample_fn is None:
                from ..dsp import resample as _resample

                resample_fn = _resample
            stem_signal = _as_stereo(
                resample_fn(stem_signal, int(sample_rate), resolved_sample_rate)
            )
        prepared_layers[stem_name] = (
            resolved_sample_rate,
            stem_signal.astype(np.float32, copy=False),
        )
        max_length = max(max_length, stem_signal.shape[-1])
        channel_count = max(channel_count, stem_signal.shape[0])

    mixed = np.zeros((channel_count, max_length), dtype=np.float32)
    aligned_layers: dict[str, tuple[int, np.ndarray]] = {}

    for stem_name, (sample_rate, signal) in prepared_layers.items():
        aligned_signal = _synchronize_stem_mix_shape(
            signal,
            target_channels=channel_count,
            target_length=max_length,
        )
        attenuated_signal = np.asarray(
            aligned_signal * _STEM_SUM_ATTENUATION_LINEAR,
            dtype=np.float32,
        )
        aligned_layers[stem_name] = (sample_rate, attenuated_signal)
        mixed += attenuated_signal

    mixed = np.nan_to_num(
        mixed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

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

    return (
        resolved_sample_rate,
        aligned_layers,
        mixed.astype(np.float32, copy=False),
    )


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
                resolved_output_dir
                / f"{stem_name}_mastered.{normalized_format}"
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
        np.clip(
            getattr(base_config, "stem_tone_enrichment_mix", 0.08), 0.0, 0.24
        )
    )
    if mix_amount <= 0.0:
        return ()

    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return ()
    if normalized_name == "bass":
        return (
            (-0.07, mix_amount * 0.16),
            (12.0, mix_amount * 0.42),
        )
    return ()


def _resolve_parallel_stem_workers(stem_count: int) -> int:
    if stem_count <= 1:
        return 1
    cpu_count = os.cpu_count() or 1
    return max(1, min(stem_count, cpu_count, 4))


def _fast_pitch_shift_channel(
    channel: np.ndarray,
    semitones: float,
) -> np.ndarray:
    samples = np.asarray(channel, dtype=np.float32).reshape(-1)
    if samples.size <= 1 or abs(float(semitones)) <= 1e-4:
        return np.array(samples, copy=True)

    pitch_ratio = float(2.0 ** (float(semitones) / 12.0))
    if not np.isfinite(pitch_ratio) or pitch_ratio <= 0.0:
        return np.array(samples, copy=True)

    shifted_length = max(int(round(samples.size / pitch_ratio)), 1)
    source_positions = np.arange(samples.size, dtype=np.float32)
    target_positions = np.linspace(
        0.0,
        float(samples.size - 1),
        num=shifted_length,
        dtype=np.float32,
    )
    return np.interp(target_positions, source_positions, samples).astype(
        np.float32,
        copy=False,
    )


def _load_pitch_shift_fn() -> (
    Callable[[np.ndarray, int, float], np.ndarray] | None
):
    return lambda channel, sample_rate, semitones: _fast_pitch_shift_channel(
        channel,
        semitones,
    )


def _apply_stem_tone_enrichment(
    signal: np.ndarray,
    sample_rate: int,
    stem_name: str,
    base_config: SmartMasteringConfig,
    *,
    pitch_shift_fn: Callable[[np.ndarray, int, float], np.ndarray]
    | None = None,
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
    dry_peak = (
        float(np.max(np.abs(stereo_signal))) if stereo_signal.size else 0.0
    )

    for semitones, mix in tone_layers:
        if mix <= 0.0:
            continue
        shifted_channels = []
        for channel in stereo_signal:
            shifted_channel = resolved_pitch_shift_fn(
                channel, sample_rate, semitones
            )
            shifted_channels.append(
                _match_signal_length(shifted_channel, stereo_signal.shape[-1])
            )
        shifted = np.vstack(shifted_channels).astype(np.float32, copy=False)
        shifted_peak = float(np.max(np.abs(shifted))) if shifted.size else 0.0
        if dry_peak > 1e-6 and shifted_peak > dry_peak:
            shifted = shifted * (dry_peak / shifted_peak)
        enriched = enriched + shifted * float(mix)

    return _constrain_stem_peak_growth(
        enriched,
        reference_signal=stereo_signal,
        peak_growth_limit=_STEM_TONE_ENRICHMENT_PEAK_GROWTH_LIMIT,
        absolute_peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
    )


def _apply_stem_glue_reverb(
    signal: np.ndarray,
    sample_rate: int,
    stem_name: str,
    base_config: SmartMasteringConfig,
) -> np.ndarray:
    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name not in {"vocals", "other"}:
        return _as_stereo(signal)

    stereo_signal = _as_stereo(signal)
    if stereo_signal.shape[-1] < 256 or sample_rate <= 0:
        return stereo_signal

    effects_macro = float(
        np.clip(getattr(base_config, "effects", 0.5), 0.0, 1.0)
    )
    glue_reverb_amount = float(
        np.clip(getattr(base_config, "stem_glue_reverb_amount", 1.0), 0.0, 1.5)
    )
    if glue_reverb_amount <= 1e-4:
        return stereo_signal
    enrichment_mix = float(
        np.clip(
            getattr(base_config, "stem_tone_enrichment_mix", 0.14),
            0.0,
            0.3,
        )
    )
    source = np.mean(stereo_signal, axis=0, dtype=np.float32)
    source_abs = np.abs(source).astype(np.float32, copy=False)
    if not np.any(source_abs > 0.0):
        return stereo_signal

    activity_threshold = float(
        max(
            np.percentile(source_abs, 62.0),
            np.mean(source_abs, dtype=np.float32) * 0.82,
            1e-6,
        )
    )
    activity_ratio = float(
        np.mean(source_abs >= activity_threshold, dtype=np.float32)
    )
    sustain_window = max(
        int(sample_rate * (0.02 if normalized_name == "vocals" else 0.026)),
        1,
    )
    sustain_envelope = _moving_average(source_abs, sustain_window)
    sustain_peak = (
        float(np.max(sustain_envelope)) if sustain_envelope.size else 0.0
    )
    sustain_ratio = (
        0.0
        if sustain_peak <= 1e-6
        else float(
            np.clip(
                np.mean(sustain_envelope, dtype=np.float32)
                / max(sustain_peak, 1e-6),
                0.0,
                1.0,
            )
        )
    )
    sparse_fill = float(np.clip((0.24 - activity_ratio) / 0.24, 0.0, 1.0))
    density_push = float(
        np.clip(sustain_ratio * 0.55 + sparse_fill * 0.85, 0.0, 1.0)
    )

    role_mix_floor = 0.078 if normalized_name == "vocals" else 0.064
    role_mix_ceiling = 0.24 if normalized_name == "vocals" else 0.2
    wet_mix = float(
        np.clip(
            role_mix_floor
            + effects_macro * 0.055
            + enrichment_mix * 0.26
            + sustain_ratio * 0.028
            + sparse_fill * 0.05,
            role_mix_floor,
            role_mix_ceiling,
        )
    )
    wet_mix = float(
        np.clip(
            wet_mix
            * (0.92 + glue_reverb_amount * 0.58)
            * (0.96 + density_push * 0.26),
            0.0,
            0.34,
        )
    )
    wet_mix = float(
        np.clip(
            wet_mix
            * (1.0 + max(glue_reverb_amount - 0.75, 0.0) * 0.18)
            * (1.0 + max(glue_reverb_amount - 1.0, 0.0) * 0.24),
            0.0,
            0.34,
        )
    )
    if wet_mix <= 1e-4:
        return stereo_signal

    tail_extension = float(
        np.clip(
            1.0
            + glue_reverb_amount * 0.42
            + effects_macro * 0.18
            + sparse_fill * 0.6
            + sustain_ratio * 0.34,
            1.0,
            2.85,
        )
    )
    wet = np.zeros_like(stereo_signal, dtype=np.float32)
    if normalized_name == "vocals":
        early_tap_spec = (
            (19.0, 0.34, 0.84, 1.0),
            (35.0, 0.24, 1.0, 0.82),
            (58.0, 0.17, 0.78, 0.97),
            (84.0, 0.12, 0.92, 0.76),
        )
        late_tap_spec = (
            (112.0, 0.1, 0.76, 1.0, 0.18),
            (156.0, 0.075, 1.0, 0.74, 0.22),
            (212.0, 0.055, 0.82, 0.95, 0.24),
            (286.0, 0.04, 0.95, 0.78, 0.2),
        )
    else:
        early_tap_spec = (
            (23.0, 0.26, 0.8, 0.95),
            (43.0, 0.18, 0.95, 0.82),
            (67.0, 0.125, 0.76, 0.9),
            (95.0, 0.085, 0.9, 0.72),
        )
        late_tap_spec = (
            (126.0, 0.075, 0.78, 0.96, 0.16),
            (181.0, 0.055, 0.96, 0.78, 0.2),
            (245.0, 0.04, 0.82, 0.92, 0.18),
        )

    early_delay_scale = float(np.clip(0.95 + tail_extension * 0.08, 1.0, 1.14))
    for delay_ms, gain, left_weight, right_weight in early_tap_spec:
        delay_samples = int(
            max(
                round(
                    float(sample_rate) * delay_ms * early_delay_scale / 1000.0
                ),
                1,
            )
        )
        if delay_samples >= source.shape[-1]:
            continue
        delayed = np.pad(source[:-delay_samples], (delay_samples, 0))
        wet[0] += delayed * float(gain) * float(left_weight)
        wet[1] += delayed * float(gain) * float(right_weight)

    diffusion_window = max(
        int(sample_rate * (0.005 + density_push * 0.0015)),
        1,
    )
    tail_source = _moving_average(source, diffusion_window)
    tail_source = _moving_average(
        tail_source,
        max(int(sample_rate * (0.009 + sustain_ratio * 0.005)), 1),
    )
    previous_tail = None
    late_gain_scale = float(
        np.clip(
            0.94
            + density_push * 0.42
            + glue_reverb_amount * 0.24
            + max(glue_reverb_amount - 1.0, 0.0) * 0.3,
            0.98,
            1.82,
        )
    )
    for delay_ms, gain, left_weight, right_weight, feedback in late_tap_spec:
        delay_samples = int(
            max(
                round(float(sample_rate) * delay_ms * tail_extension / 1000.0),
                1,
            )
        )
        if delay_samples >= tail_source.shape[-1]:
            continue
        delayed = np.pad(tail_source[:-delay_samples], (delay_samples, 0))
        if previous_tail is not None:
            delayed = delayed + previous_tail * float(feedback)
        wet[0] += delayed * float(gain) * late_gain_scale * float(left_weight)
        wet[1] += delayed * float(gain) * late_gain_scale * float(right_weight)
        previous_tail = delayed

    smoothing_window = max(
        int(sample_rate * (0.0045 + density_push * 0.0015)),
        1,
    )
    body_window = max(
        int(sample_rate * (0.02 + density_push * 0.014)),
        smoothing_window + 1,
    )
    low_cut_window = max(
        int(sample_rate * (0.024 + sparse_fill * 0.012)),
        body_window + 1,
    )
    wet = _moving_average(wet, smoothing_window)
    wet = wet + _moving_average(wet, body_window) * float(
        np.clip(0.14 + density_push * 0.14, 0.08, 0.3)
    )
    wet = wet - _moving_average(wet, low_cut_window) * 0.78

    dry_peak = (
        float(np.max(np.abs(stereo_signal))) if stereo_signal.size else 0.0
    )
    wet_peak = float(np.max(np.abs(wet))) if wet.size else 0.0
    wet_peak_limit_ratio = float(
        np.clip(
            0.66
            + sparse_fill * 0.16
            + sustain_ratio * 0.1
            + glue_reverb_amount * 0.05
            + max(glue_reverb_amount - 1.0, 0.0) * 0.06,
            0.6,
            0.92,
        )
    )
    if dry_peak > 1e-6 and wet_peak > dry_peak * wet_peak_limit_ratio:
        wet = wet * ((dry_peak * wet_peak_limit_ratio) / wet_peak)

    glued = stereo_signal + wet * wet_mix
    return _constrain_stem_peak_growth(
        glued,
        reference_signal=stereo_signal,
        peak_growth_limit=1.08,
        absolute_peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
    )


def _apply_drum_edge_finish(
    signal: np.ndarray,
    sample_rate: int,
    base_config: SmartMasteringConfig,
) -> np.ndarray:
    stereo_signal = _as_stereo(signal)
    if stereo_signal.shape[-1] < 256 or sample_rate <= 0:
        return stereo_signal

    effects_macro = float(
        np.clip(getattr(base_config, "effects", 0.5), 0.0, 1.0)
    )
    micro_dynamics_strength = float(
        np.clip(
            getattr(base_config, "micro_dynamics_strength", 0.1),
            0.0,
            0.3,
        )
    )
    exciter_mix = float(
        np.clip(getattr(base_config, "exciter_mix", 0.5), 0.0, 1.0)
    )
    drum_edge_amount = float(
        np.clip(getattr(base_config, "stem_drum_edge_amount", 1.0), 0.0, 1.5)
    )
    if drum_edge_amount <= 1e-4:
        return stereo_signal
    edge_mix = float(
        np.clip(
            0.08 + effects_macro * 0.05 + micro_dynamics_strength * 0.3,
            0.08,
            0.2,
        )
    )
    edge_mix = float(np.clip(edge_mix * drum_edge_amount, 0.0, 0.26))
    drive = float(
        np.clip(
            1.35
            + exciter_mix * 0.55
            + effects_macro * 0.35
            + max(drum_edge_amount - 1.0, 0.0) * 0.3,
            1.1,
            2.35,
        )
    )
    fast_window = max(int(sample_rate * 0.0018), 1)
    slow_window = max(int(sample_rate * 0.016), fast_window + 1)

    mono_energy = np.mean(np.abs(stereo_signal), axis=0, dtype=np.float32)
    fast_env = _moving_average(mono_energy, fast_window)
    slow_env = _moving_average(mono_energy, slow_window)
    transient_mask = np.clip(
        (fast_env - slow_env) / np.maximum(fast_env, 1e-6),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    transient_mask = _moving_average(
        transient_mask,
        max(int(sample_rate * 0.0024), 1),
    )
    transient_mask = np.clip(transient_mask, 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )
    transient_mask = transient_mask[np.newaxis, :]
    sustain_mask = 1.0 - transient_mask

    transient_component = stereo_signal - _moving_average(
        stereo_signal,
        max(int(sample_rate * 0.010), 1),
    )
    expanded = transient_component * (1.0 + transient_mask * 0.55)
    compressed = expanded * (1.0 - sustain_mask * 0.16)
    distorted = np.tanh(compressed * drive) / float(np.tanh(drive))
    finished = stereo_signal + distorted * edge_mix
    return _constrain_stem_peak_growth(
        finished,
        reference_signal=stereo_signal,
        peak_growth_limit=1.08,
        absolute_peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
    )


def _apply_stem_stereo_width_finish(
    signal: np.ndarray,
    sample_rate: int,
    stem_name: str,
    base_config: SmartMasteringConfig,
    *,
    stem_overrides: Mapping[str, float | str | None] | None = None,
) -> np.ndarray:
    stereo_signal = _as_stereo(signal)
    if stereo_signal.shape[-1] < 128 or sample_rate <= 0:
        return stereo_signal

    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "bass":
        return stereo_signal

    requested_width = _resolve_stem_override_float(
        stem_overrides,
        "stereo_width",
        getattr(base_config, "stereo_width", 1.0),
        minimum=0.8,
        maximum=1.6,
    )
    effects_macro = float(
        np.clip(getattr(base_config, "effects", 0.5), 0.0, 1.0)
    )
    glue_amount = float(
        np.clip(getattr(base_config, "stem_glue_reverb_amount", 1.0), 0.0, 1.5)
    )

    role_boost = 0.0
    synthesized_side_mix = 0.0
    if normalized_name == "vocals":
        role_boost = 0.16
        synthesized_side_mix = 0.56
    elif normalized_name == "other":
        role_boost = 0.18
        synthesized_side_mix = 0.56
    elif normalized_name == "drums":
        role_boost = 0.08
        synthesized_side_mix = 0.22
    else:
        role_boost = 0.12
        synthesized_side_mix = 0.32

    width_scale = float(
        np.clip(
            1.0
            + max(requested_width - 1.0, 0.0) * 0.68
            + role_boost
            + effects_macro * 0.06
            + max(glue_amount - 1.0, 0.0) * 0.08,
            0.92,
            1.48 if normalized_name != "drums" else 1.18,
        )
    )
    if width_scale <= 1.0 and normalized_name == "drums":
        return stereo_signal

    mid = 0.5 * (stereo_signal[0] + stereo_signal[1])
    side = 0.5 * (stereo_signal[0] - stereo_signal[1])
    current_width_ratio = _measure_stem_stereo_width_ratio(stereo_signal)
    width_deficit = float(
        np.clip((0.24 - current_width_ratio) / 0.24, 0.0, 1.0)
    )

    if synthesized_side_mix > 0.0 and width_deficit > 0.0:
        mono = np.mean(stereo_signal, axis=0, dtype=np.float32)
        detail_window = max(
            int(
                sample_rate
                * (0.0042 if normalized_name == "vocals" else 0.0058)
            ),
            1,
        )
        detail = mono - _moving_average(mono, detail_window)
        delay_samples = max(
            int(
                round(
                    sample_rate
                    * (0.0014 + width_deficit * 0.0012 + effects_macro * 0.0007)
                )
            ),
            1,
        )
        delayed = np.pad(detail[:-delay_samples], (delay_samples, 0))
        advanced = np.pad(detail[delay_samples:], (0, delay_samples))
        synthesized_side = (delayed - advanced) * float(
            np.clip(
                synthesized_side_mix
                * (0.88 + width_deficit * 0.9)
                * (1.0 + max(width_scale - 1.0, 0.0) * 1.1),
                0.0,
                0.72,
            )
        )
        if normalized_name == "vocals" and current_width_ratio <= 1e-4:
            synthesized_side = synthesized_side * 1.12
        side = side + synthesized_side.astype(np.float32, copy=False)

    side = side * width_scale
    widened = np.stack([mid + side, mid - side], axis=0)
    return _constrain_stem_peak_growth(
        widened,
        reference_signal=stereo_signal,
        peak_growth_limit=1.1,
        absolute_peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
    )


def _resolve_stem_dynamics_profile(stem_name: str) -> dict[str, float]:
    normalized_name = str(stem_name).strip().lower() or "other"
    if normalized_name == "drums":
        return {
            "threshold_percentile": 72.0,
            "ratio": 5.4,
            "attack_ms": 3.0,
            "release_ms": 54.0,
            "gain_smoothing_ms": 8.0,
            "wet_mix": 0.84,
            "body_mix": 0.04,
            "body_window_ms": 3.2,
            "sustain_bias": 0.94,
            "threshold_mean_scale": 1.22,
            "min_gain": 0.36,
            "makeup_scale": 0.58,
            "limiter_mix_bonus": 0.24,
            "drive_bonus": 0.18,
        }
    if normalized_name == "bass":
        return {
            "threshold_percentile": 82.0,
            "ratio": 4.8,
            "attack_ms": 8.5,
            "release_ms": 88.0,
            "gain_smoothing_ms": 18.0,
            "wet_mix": 0.74,
            "body_mix": 0.18,
            "body_window_ms": 6.5,
            "sustain_bias": 1.04,
            "threshold_mean_scale": 1.08,
            "min_gain": 0.44,
            "makeup_scale": 0.68,
            "limiter_mix_bonus": 0.14,
            "drive_bonus": 0.08,
        }
    if normalized_name == "vocals":
        return {
            "threshold_percentile": 79.0,
            "ratio": 4.9,
            "attack_ms": 6.5,
            "release_ms": 82.0,
            "gain_smoothing_ms": 14.0,
            "wet_mix": 0.72,
            "body_mix": 0.16,
            "body_window_ms": 4.6,
            "sustain_bias": 1.0,
            "threshold_mean_scale": 1.12,
            "min_gain": 0.42,
            "makeup_scale": 0.62,
            "limiter_mix_bonus": 0.18,
            "drive_bonus": 0.12,
        }
    return {
        "threshold_percentile": 78.0,
        "ratio": 4.3,
        "attack_ms": 5.8,
        "release_ms": 72.0,
        "gain_smoothing_ms": 13.0,
        "wet_mix": 0.7,
        "body_mix": 0.1,
        "body_window_ms": 4.0,
        "sustain_bias": 0.98,
        "threshold_mean_scale": 1.14,
        "min_gain": 0.42,
        "makeup_scale": 0.56,
        "limiter_mix_bonus": 0.16,
        "drive_bonus": 0.1,
    }


def _apply_stem_dynamics(
    signal: np.ndarray,
    sample_rate: int,
    stem_name: str,
    base_config: SmartMasteringConfig,
    *,
    stem_overrides: Mapping[str, float | str | None] | None = None,
) -> np.ndarray:
    stereo_signal = _as_stereo(signal)
    if stereo_signal.shape[-1] < 128 or sample_rate <= 0:
        return stereo_signal

    linked_energy = np.max(np.abs(stereo_signal), axis=0)
    if not np.any(linked_energy > 0.0):
        return stereo_signal

    dynamics_profile = _resolve_stem_dynamics_profile(stem_name)
    attack_window = max(
        int(round(sample_rate * dynamics_profile["attack_ms"] / 1000.0)),
        1,
    )
    release_window = max(
        int(round(sample_rate * dynamics_profile["release_ms"] / 1000.0)),
        attack_window + 1,
    )
    gain_window = max(
        int(
            round(sample_rate * dynamics_profile["gain_smoothing_ms"] / 1000.0)
        ),
        1,
    )
    fast_env = _moving_average(linked_energy, attack_window)
    slow_env = _moving_average(linked_energy, release_window)
    control_env = np.maximum(
        fast_env,
        slow_env * dynamics_profile["sustain_bias"],
    )
    threshold = float(
        max(
            np.percentile(
                control_env, dynamics_profile["threshold_percentile"]
            ),
            np.mean(control_env, dtype=np.float32)
            * dynamics_profile["threshold_mean_scale"],
            1e-5,
        )
    )
    ratio = float(max(dynamics_profile["ratio"], 1.0))
    compression_curve = np.maximum(control_env / threshold, 1.0)
    gain = np.power(compression_curve, -(1.0 - 1.0 / ratio))
    gain = _moving_average(gain, gain_window)
    gain = np.clip(gain, dynamics_profile["min_gain"], 1.0).astype(
        np.float32,
        copy=False,
    )
    compressed = stereo_signal * gain[np.newaxis, :]

    average_gain_loss = float(1.0 - np.mean(gain, dtype=np.float32))
    makeup_gain = float(
        1.0 + average_gain_loss * dynamics_profile["makeup_scale"]
    )
    compressed = compressed * makeup_gain

    body_mix = float(np.clip(dynamics_profile["body_mix"], 0.0, 0.3))
    if body_mix > 0.0:
        body_window = max(
            int(
                round(sample_rate * dynamics_profile["body_window_ms"] / 1000.0)
            ),
            1,
        )
        body_component = _moving_average(stereo_signal, body_window)
        compressed = compressed * (1.0 - body_mix) + body_component * body_mix

    wet_mix = float(np.clip(dynamics_profile["wet_mix"], 0.0, 1.0))
    shaped = stereo_signal * (1.0 - wet_mix) + compressed * wet_mix

    drive_db = _resolve_stem_override_float(
        stem_overrides,
        "drive_db",
        getattr(base_config, "drive_db", 0.0),
        minimum=0.0,
        maximum=8.0,
    )
    soft_clip_ratio = _resolve_stem_override_float(
        stem_overrides,
        "limiter_soft_clip_ratio",
        getattr(base_config, "limiter_soft_clip_ratio", 0.0),
        minimum=0.0,
        maximum=0.7,
    )
    saturation_ratio = _resolve_stem_override_float(
        stem_overrides,
        "pre_limiter_saturation_ratio",
        getattr(base_config, "pre_limiter_saturation_ratio", 0.0),
        minimum=0.0,
        maximum=0.45,
    )
    saturation_mix = float(np.clip(0.18 + saturation_ratio * 0.95, 0.0, 0.48))
    if saturation_mix > 0.0:
        saturation_drive = float(
            np.clip(1.0 + saturation_ratio * 2.8 + drive_db * 0.06, 1.0, 2.5)
        )
        saturated = np.tanh(stereo_signal * saturation_drive) / float(
            np.tanh(saturation_drive)
        )
        shaped = shaped * (1.0 - saturation_mix) + saturated * saturation_mix

    limiter_drive = float(
        np.clip(
            1.0
            + drive_db * 0.16
            + soft_clip_ratio * 1.95
            + saturation_ratio * 1.45
            + dynamics_profile["drive_bonus"],
            1.0,
            2.85,
        )
    )
    limited = np.tanh(shaped * limiter_drive) / float(np.tanh(limiter_drive))
    limiter_mix = float(
        np.clip(
            0.26
            + soft_clip_ratio * 1.18
            + saturation_ratio * 0.82
            + dynamics_profile["limiter_mix_bonus"],
            0.24,
            0.92,
        )
    )
    limited = shaped * (1.0 - limiter_mix) + limited * limiter_mix
    input_peak = (
        float(np.max(np.abs(stereo_signal))) if stereo_signal.size else 0.0
    )
    limited_peak = float(np.max(np.abs(limited))) if limited.size else 0.0
    if input_peak > 1e-6 and limited_peak > input_peak * 0.985:
        limited = limited * ((input_peak * 0.985) / limited_peak)
    return _constrain_stem_peak_growth(
        limited,
        reference_signal=stereo_signal,
        peak_growth_limit=1.06,
        absolute_peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
    )


def _apply_stem_role_finish(
    signal: np.ndarray,
    sample_rate: int,
    stem_name: str,
    base_config: SmartMasteringConfig,
    *,
    pitch_shift_fn: Callable[[np.ndarray, int, float], np.ndarray]
    | None = None,
    stem_overrides: Mapping[str, float | str | None] | None = None,
) -> np.ndarray:
    finished = _apply_stem_tone_enrichment(
        signal,
        sample_rate,
        stem_name,
        base_config,
        pitch_shift_fn=pitch_shift_fn,
    )
    finished = _apply_stem_glue_reverb(
        finished,
        sample_rate,
        stem_name,
        base_config,
    )
    if str(stem_name).strip().lower() == "drums":
        finished = _apply_drum_edge_finish(
            finished,
            sample_rate,
            base_config,
        )
    finished = _apply_stem_stereo_width_finish(
        finished,
        sample_rate,
        stem_name,
        base_config,
        stem_overrides=stem_overrides,
    )
    finished = _apply_stem_dynamics(
        finished,
        sample_rate,
        stem_name,
        base_config,
        stem_overrides=stem_overrides,
    )
    return _sanitize_stem_signal(
        finished,
        peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
    )


def resolve_stem_mastering_plan(
    stem_name: str,
    base_config: SmartMasteringConfig,
) -> StemMasteringPlan:
    normalized_name = str(stem_name).strip().lower() or "other"
    base_cleanup_strength = float(
        np.clip(getattr(base_config, "stem_cleanup_strength", 1.0), 0.0, 1.5)
    )
    base_noise_gate_strength = float(
        np.clip(getattr(base_config, "stem_noise_gate_strength", 1.0), 0.0, 1.5)
    )
    shared_target_lufs = float(base_config.target_lufs - 2.8)
    shared_ceil_db = float(min(base_config.ceil_db, -1.2))
    shared_drive_db = float(max(base_config.drive_db * 0.92, 0.55))
    shared_soft_clip_ratio = float(
        np.clip(base_config.limiter_soft_clip_ratio * 0.96 + 0.04, 0.0, 0.55)
    )
    shared_saturation_ratio = float(
        np.clip(
            base_config.pre_limiter_saturation_ratio * 0.9 + 0.035,
            0.0,
            0.4,
        )
    )
    shared_max_final_boost_db = float(
        max(base_config.max_final_boost_db * 0.55, 1.4)
    )
    shared_width = float(np.clip(base_config.stereo_width + 0.08, 0.96, 1.26))
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
        shared_mix_gain_db = 1.68
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.58,
                "bass_boost_db_per_oct": base_config.bass_boost_db_per_oct
                + 0.22,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct
                + 0.18,
                "exciter_mix": float(
                    np.clip(base_config.exciter_mix + 0.08, 0.0, 1.0)
                ),
                "exciter_max_drive": base_config.exciter_max_drive + 0.45,
                "exciter_high_frequency_cutoff_hz": None
                if base_config.exciter_high_frequency_cutoff_hz is None
                else max(
                    base_config.exciter_high_frequency_cutoff_hz - 700.0, 3500.0
                ),
                "limiter_recovery_style": "tight",
                "low_end_mono_tightening": "firm",
                "low_end_mono_tightening_amount": 0.95,
                "stem_cleanup_strength": float(
                    np.clip(base_cleanup_strength * 0.48, 0.0, 1.5)
                ),
                "stem_noise_gate_strength": float(
                    np.clip(base_noise_gate_strength * 0.54, 0.0, 1.5)
                ),
                "micro_dynamics_strength": float(
                    np.clip(
                        base_config.micro_dynamics_strength * 0.7, 0.0, 0.18
                    )
                ),
                "stereo_width": float(np.clip(shared_width + 0.04, 1.02, 1.16)),
            }
        )
    elif normalized_name == "bass":
        shared_mix_gain_db = 1.08
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.42,
                "bass_boost_db_per_oct": base_config.bass_boost_db_per_oct
                + 0.78,
                "treble_boost_db_per_oct": max(
                    base_config.treble_boost_db_per_oct - 0.72, 0.0
                ),
                "exciter_mix": float(
                    np.clip(
                        base_config.exciter_mix * 0.7 + 0.02,
                        0.0,
                        1.0,
                    )
                ),
                "exciter_max_drive": max(
                    base_config.exciter_max_drive * 0.92, 0.6
                ),
                "stereo_width": 0.94,
                "mono_bass_hz": max(base_config.mono_bass_hz, 160.0),
                "stereo_tone_variation_db": 0.0,
                "stereo_motion_mid_amount": 0.0,
                "stereo_motion_high_amount": 0.0,
                "low_end_mono_tightening": "firm",
                "low_end_mono_tightening_amount": 1.0,
                "stem_cleanup_strength": float(
                    np.clip(base_cleanup_strength * 0.5, 0.0, 1.5)
                ),
                "stem_noise_gate_strength": float(
                    np.clip(base_noise_gate_strength * 0.48, 0.0, 1.5)
                ),
                "micro_dynamics_strength": float(
                    np.clip(
                        base_config.micro_dynamics_strength * 0.46, 0.0, 0.1
                    )
                ),
            }
        )
    elif normalized_name == "vocals":
        shared_mix_gain_db = -0.72
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.22,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct
                + 0.08,
                "bass_boost_db_per_oct": max(
                    base_config.bass_boost_db_per_oct - 0.02, 0.0
                ),
                "exciter_mix": float(
                    np.clip(base_config.exciter_mix + 0.05, 0.0, 1.0)
                ),
                "exciter_max_drive": base_config.exciter_max_drive + 0.2,
                "stereo_width": float(np.clip(shared_width + 0.18, 1.08, 1.42)),
                "stem_cleanup_strength": float(
                    np.clip(base_cleanup_strength * 0.62, 0.0, 1.5)
                ),
                "stem_noise_gate_strength": float(
                    np.clip(base_noise_gate_strength * 0.68, 0.0, 1.5)
                ),
                "micro_dynamics_strength": float(
                    np.clip(
                        base_config.micro_dynamics_strength + 0.03, 0.0, 0.26
                    )
                ),
                "start_treble_boost_hz": max(
                    base_config.start_treble_boost_hz - 150.0, 2200.0
                ),
                "low_end_mono_tightening": "gentle",
                "low_end_mono_tightening_amount": 0.3,
            }
        )
    elif normalized_name == "guitar":
        shared_mix_gain_db = -0.18
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.05,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct
                + 0.12,
                "exciter_mix": float(
                    np.clip(base_config.exciter_mix + 0.05, 0.0, 1.0)
                ),
                "stereo_width": float(np.clip(shared_width + 0.16, 1.02, 1.38)),
                "stem_cleanup_strength": float(
                    np.clip(base_cleanup_strength * 0.74, 0.0, 1.5)
                ),
                "stem_noise_gate_strength": float(
                    np.clip(base_noise_gate_strength * 0.76, 0.0, 1.5)
                ),
                "micro_dynamics_strength": float(
                    np.clip(
                        base_config.micro_dynamics_strength + 0.02, 0.0, 0.25
                    )
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
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct
                + 0.1,
                "exciter_mix": float(
                    np.clip(base_config.exciter_mix + 0.04, 0.0, 1.0)
                ),
                "stereo_width": float(np.clip(shared_width + 0.14, 1.02, 1.36)),
                "stem_cleanup_strength": float(
                    np.clip(base_cleanup_strength * 0.74, 0.0, 1.5)
                ),
                "stem_noise_gate_strength": float(
                    np.clip(base_noise_gate_strength * 0.78, 0.0, 1.5)
                ),
                "micro_dynamics_strength": float(
                    np.clip(
                        base_config.micro_dynamics_strength + 0.03, 0.0, 0.28
                    )
                ),
                "low_end_mono_tightening": "gentle",
                "low_end_mono_tightening_amount": 0.3,
            }
        )
    else:
        shared_mix_gain_db = -0.16
        shared_overrides.update(
            {
                "drive_db": shared_drive_db + 0.05,
                "bass_boost_db_per_oct": base_config.bass_boost_db_per_oct
                + 0.08,
                "treble_boost_db_per_oct": base_config.treble_boost_db_per_oct
                + 0.12,
                "exciter_mix": float(
                    np.clip(base_config.exciter_mix + 0.05, 0.0, 1.0)
                ),
                "stereo_width": float(np.clip(shared_width + 0.2, 1.08, 1.42)),
                "stem_cleanup_strength": float(
                    np.clip(base_cleanup_strength * 0.72, 0.0, 1.5)
                ),
                "stem_noise_gate_strength": float(
                    np.clip(base_noise_gate_strength * 0.76, 0.0, 1.5)
                ),
                "micro_dynamics_strength": float(
                    np.clip(
                        base_config.micro_dynamics_strength + 0.02, 0.0, 0.25
                    )
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
    process_stem_fn: Callable[
        [np.ndarray, int, dict[str, object]], tuple[int, np.ndarray]
    ],
    separate_stems_fn: Callable[..., tuple[dict[str, str], str]],
    read_audio_fn: Callable[[str], tuple[int, np.ndarray]],
    delete_fn: Callable[[str], object],
    model_name: str = "mastering",
    shifts: int = 2,
    quality_flags: Sequence[str] = (),
    mix_headroom_db: float = 6.0,
    resample_fn: Callable[[np.ndarray, int, int], np.ndarray] | None = None,
    pitch_shift_fn: Callable[[np.ndarray, int, float], np.ndarray]
    | None = None,
    save_mastered_stems: bool = True,
    mastered_stems_output_dir: str | None = None,
    save_audio_fn: Callable[..., str | None] | None = None,
    mastered_stems_format: str = "wav",
    mastered_stems_bit_depth: int = 32,
    mastered_stems_bitrate: int = 320,
    mastered_stems_compression_level: int = 9,
) -> tuple[int, np.ndarray]:
    separated_output_dir: str | None = None
    resolved_pitch_shift_fn = pitch_shift_fn or _load_pitch_shift_fn()

    bind_download_activity_scope = None
    activity_scope_id: str | None = None
    try:
        from definers.system.download_activity import (
            bind_download_activity_scope as _bind_download_activity_scope,
            current_download_activity_scope,
        )

        bind_download_activity_scope = _bind_download_activity_scope
        activity_scope_id = current_download_activity_scope()
    except Exception:
        bind_download_activity_scope = None
        activity_scope_id = None

    def _run_single_stem(
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
        processed_signal = _sanitize_stem_signal(
            processed_signal,
            peak_ceiling=_STEM_SAVED_PEAK_CEILING_LINEAR,
        )
        finished_signal = _apply_stem_role_finish(
            processed_signal,
            processed_sample_rate,
            plan.stem_name,
            base_config,
            pitch_shift_fn=resolved_pitch_shift_fn,
            stem_overrides=plan.overrides,
        )
        balanced_signal, _applied_mix_gain_db = _apply_stem_mix_balance(
            finished_signal,
            plan.stem_name,
            plan.mix_gain_db,
            vocal_pullback_db=float(
                np.clip(
                    getattr(base_config, "stem_vocal_pullback_db", 0.0),
                    0.0,
                    3.0,
                )
            ),
        )
        return plan.stem_name, (
            processed_sample_rate,
            balanced_signal,
        )

    def process_single_stem(
        stem_name: str,
        stem_path: str,
    ) -> tuple[str, tuple[int, np.ndarray]]:
        if (
            bind_download_activity_scope is not None
            and activity_scope_id is not None
        ):
            with bind_download_activity_scope(activity_scope_id):
                return _run_single_stem(stem_name, stem_path)
        return _run_single_stem(stem_name, stem_path)

    try:
        stem_paths, separated_output_dir = separate_stems_fn(
            audio_path,
            model_name=model_name,
            shifts=shifts,
            quality_flags=tuple(
                str(flag).strip() for flag in quality_flags if str(flag).strip()
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

        mixed_sample_rate, aligned_layers, mixed_signal = (
            _prepare_mixed_stem_layers(
                mastered_layers,
                mix_headroom_db=mix_headroom_db,
                resample_fn=resample_fn,
            )
        )
        if (
            save_mastered_stems
            and mastered_stems_output_dir is not None
            and save_audio_fn is not None
        ):
            _save_mastered_stem_layers(
                mastered_layers,
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
