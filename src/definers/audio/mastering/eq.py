from __future__ import annotations

from collections import deque
from collections.abc import Callable

import numpy as np
from scipy import signal


def audio_eq(
    audio_data: np.ndarray,
    anchors: list[list[float]] | np.ndarray,
    sample_rate: int = 44100,
    nperseg: int = 8192,
) -> np.ndarray:
    anchors = sorted(anchors, key=lambda x: x[0])
    anchor_freqs = np.array([a[0] for a in anchors])
    anchor_gains_db = np.array([a[1] for a in anchors])

    unique_freqs, indices = np.unique(anchor_freqs, return_index=True)
    anchor_freqs = unique_freqs
    anchor_gains_db = anchor_gains_db[indices]

    f_axis, _times, stft_frames = signal.stft(
        audio_data,
        fs=sample_rate,
        nperseg=nperseg,
    )

    log_f = np.log10(f_axis + 1e-5)
    log_anchor_freqs = np.log10(anchor_freqs)

    interp_gains_db = np.interp(
        log_f,
        log_anchor_freqs,
        anchor_gains_db,
        left=anchor_gains_db[0],
        right=anchor_gains_db[-1],
    )

    gain_multipliers = 10 ** (interp_gains_db / 20)
    modified_frames = stft_frames * gain_multipliers[:, None]

    _reconstructed_times, output_audio = signal.istft(
        modified_frames,
        fs=sample_rate,
        nperseg=nperseg,
    )

    orig_len = audio_data.shape[-1]
    if output_audio.shape[-1] > orig_len:
        output_audio = output_audio[..., :orig_len]
    elif output_audio.shape[-1] < orig_len:
        pad_width = orig_len - output_audio.shape[-1]
        output_audio = (
            np.pad(output_audio, ((0, 0), (0, pad_width)))
            if output_audio.ndim > 1
            else np.pad(output_audio, (0, pad_width))
        )

    if np.issubdtype(audio_data.dtype, np.integer):
        info = np.iinfo(audio_data.dtype)
        return np.clip(output_audio, info.min, info.max).astype(
            audio_data.dtype
        )

    return output_audio.astype(np.float32)


def smooth_curve(
    self,
    curve: np.ndarray,
    f_axis: np.ndarray,
    smoothing_fraction: float | None = None,
) -> np.ndarray:
    if smoothing_fraction is None:
        return curve

    smoothed = np.copy(curve)

    for index, frequency_hz in enumerate(f_axis):
        bandwidth = frequency_hz * (
            2**smoothing_fraction - 2 ** (-smoothing_fraction)
        )
        low_f = frequency_hz - bandwidth / 2
        high_f = frequency_hz + bandwidth / 2
        mask = (f_axis >= low_f) & (f_axis <= high_f)

        if np.any(mask):
            smoothed[index] = np.mean(curve[mask])

    return smoothed


def _normalize_stem_role(stem_role: str | None) -> str:
    normalized_role = (
        "other" if stem_role is None else str(stem_role).strip().lower()
    )
    return normalized_role or "other"


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    if samples.size == 0 or window_size <= 1:
        return samples.astype(np.float32, copy=True)

    window_size = max(1, min(int(window_size), samples.size))
    if window_size <= 1:
        return samples.astype(np.float32, copy=True)

    left_pad = window_size // 2
    right_pad = window_size - left_pad - 1
    padded = np.pad(samples, (left_pad, right_pad), mode="edge")
    kernel = np.full(window_size, 1.0 / window_size, dtype=np.float32)
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _rolling_max(values: np.ndarray, window_size: int) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float32).reshape(-1)
    if samples.size == 0 or window_size <= 1:
        return samples.astype(np.float32, copy=True)

    window_size = max(1, min(int(window_size), samples.size))
    if window_size <= 1:
        return samples.astype(np.float32, copy=True)

    rolling_max: deque[int] = deque()
    output = np.empty_like(samples)

    for index, value in enumerate(samples):
        while rolling_max and rolling_max[0] <= index - window_size:
            rolling_max.popleft()
        while rolling_max and samples[rolling_max[-1]] <= value:
            rolling_max.pop()
        rolling_max.append(index)
        output[index] = samples[rolling_max[0]]

    return output


def _restore_audio_dtype(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(values, info.min, info.max).astype(dtype)

    return values.astype(dtype, copy=False)


def _resolve_stem_cleanup_pressure(
    stem_role: str | None,
    cleanup_pressure: float,
) -> float:
    normalized_role = _normalize_stem_role(stem_role)
    pressure = float(np.clip(cleanup_pressure, 0.0, 1.0))

    if normalized_role == "drums":
        return float(np.clip(pressure * 0.45, 0.0, 1.0))
    if normalized_role == "vocals":
        return float(np.clip(pressure * 0.68, 0.0, 1.0))
    if normalized_role == "bass":
        return float(np.clip(pressure * 0.74, 0.0, 1.0))
    return float(np.clip(pressure * 0.7, 0.0, 1.0))


def _resolve_stem_residual_profile(
    stem_role: str | None,
    cleanup_pressure: float,
) -> dict[str, float]:
    normalized_role = _normalize_stem_role(stem_role)

    if normalized_role == "drums":
        intensity = float(np.clip(0.62 + cleanup_pressure * 0.18, 0.0, 1.0))
        return {
            "fast_ms": 1.8,
            "slow_ms": 14.0,
            "hold_ms": 68.0,
            "release_ms": 14.0,
            "noise_percentile": 38.0,
            "suppression_floor": float(0.07 + (1.0 - intensity) * 0.05),
            "activity_exponent": 0.68,
            "activity_floor_scale": 0.94,
            "transient_blend": 0.9,
            "expansion_drive": float(0.42 + intensity * 0.36),
            "expansion_mix": float(0.16 + intensity * 0.18),
        }

    if normalized_role == "vocals":
        intensity = float(np.clip(0.52 + cleanup_pressure * 0.22, 0.0, 1.0))
        return {
            "fast_ms": 5.5,
            "slow_ms": 52.0,
            "hold_ms": 94.0,
            "release_ms": 34.0,
            "noise_percentile": 28.0,
            "suppression_floor": float(0.115 + (1.0 - intensity) * 0.08),
            "activity_exponent": 0.84,
            "activity_floor_scale": 0.44,
            "transient_blend": 0.16,
            "expansion_drive": float(0.46 + intensity * 0.24),
            "expansion_mix": float(0.18 + intensity * 0.14),
        }

    if normalized_role == "bass":
        intensity = float(np.clip(0.44 + cleanup_pressure * 0.18, 0.0, 1.0))
        return {
            "fast_ms": 7.0,
            "slow_ms": 76.0,
            "hold_ms": 126.0,
            "release_ms": 46.0,
            "noise_percentile": 24.0,
            "suppression_floor": float(0.18 + (1.0 - intensity) * 0.08),
            "activity_exponent": 0.96,
            "activity_floor_scale": 0.56,
            "transient_blend": 0.08,
            "expansion_drive": float(0.24 + intensity * 0.16),
            "expansion_mix": float(0.08 + intensity * 0.08),
        }

    intensity = float(np.clip(0.58 + cleanup_pressure * 0.22, 0.0, 1.0))
    return {
        "fast_ms": 4.5,
        "slow_ms": 42.0,
        "hold_ms": 86.0,
        "release_ms": 30.0,
        "noise_percentile": 30.0,
        "suppression_floor": float(0.1 + (1.0 - intensity) * 0.07),
        "activity_exponent": 0.8,
        "activity_floor_scale": 0.6,
        "transient_blend": 0.18,
        "expansion_drive": float(0.58 + intensity * 0.24),
        "expansion_mix": float(0.2 + intensity * 0.14),
    }


def _resolve_stem_noise_gate_profile(
    stem_role: str | None,
    cleanup_pressure: float,
    gate_strength: float,
) -> dict[str, float]:
    normalized_role = _normalize_stem_role(stem_role)
    strength = float(np.clip(gate_strength, 0.0, 1.5))

    if normalized_role == "drums":
        intensity = float(
            np.clip((0.74 + cleanup_pressure * 0.14) * strength, 0.0, 1.0)
        )
        return {
            "fast_ms": 1.3,
            "slow_ms": 11.0,
            "hold_ms": 54.0,
            "release_ms": 18.0,
            "noise_percentile": 38.0,
            "threshold_ratio": float(
                np.clip(0.07 + intensity * 0.02, 0.045, 0.18)
            ),
            "full_open_ratio": 0.16,
            "floor": float(np.clip(0.055 - intensity * 0.02, 0.022, 0.06)),
            "transient_bias": 0.92,
            "open_exponent": 0.68,
            "active_floor_scale": 0.98,
            "mix": float(np.clip(0.48 + intensity * 0.1, 0.0, 0.72)),
        }

    if normalized_role == "vocals":
        intensity = float(
            np.clip((0.6 + cleanup_pressure * 0.18) * strength, 0.0, 1.0)
        )
        return {
            "fast_ms": 3.6,
            "slow_ms": 32.0,
            "hold_ms": 96.0,
            "release_ms": 42.0,
            "noise_percentile": 24.0,
            "threshold_ratio": float(
                np.clip(0.065 + intensity * 0.03, 0.045, 0.18)
            ),
            "full_open_ratio": 0.24,
            "floor": float(np.clip(0.06 - intensity * 0.024, 0.02, 0.068)),
            "transient_bias": 0.32,
            "open_exponent": 0.88,
            "active_floor_scale": 0.82,
            "mix": float(np.clip(0.46 + intensity * 0.14, 0.0, 0.74)),
        }

    if normalized_role == "bass":
        intensity = float(
            np.clip((0.5 + cleanup_pressure * 0.14) * strength, 0.0, 0.94)
        )
        return {
            "fast_ms": 6.0,
            "slow_ms": 56.0,
            "hold_ms": 132.0,
            "release_ms": 56.0,
            "noise_percentile": 18.0,
            "threshold_ratio": float(
                np.clip(0.11 + intensity * 0.025, 0.075, 0.22)
            ),
            "full_open_ratio": 0.28,
            "floor": float(np.clip(0.13 - intensity * 0.018, 0.08, 0.13)),
            "transient_bias": 0.12,
            "open_exponent": 0.98,
            "active_floor_scale": 0.8,
            "mix": float(np.clip(0.22 + intensity * 0.1, 0.0, 0.56)),
        }

    intensity = float(
        np.clip((0.58 + cleanup_pressure * 0.18) * strength, 0.0, 1.0)
    )
    return {
        "fast_ms": 3.4,
        "slow_ms": 28.0,
        "hold_ms": 86.0,
        "release_ms": 36.0,
        "noise_percentile": 24.0,
        "threshold_ratio": float(np.clip(0.075 + intensity * 0.03, 0.045, 0.2)),
        "full_open_ratio": 0.24,
        "floor": float(np.clip(0.058 - intensity * 0.014, 0.03, 0.065)),
        "transient_bias": 0.28,
        "open_exponent": 0.9,
        "active_floor_scale": 0.82,
        "mix": float(np.clip(0.32 + intensity * 0.12, 0.0, 0.62)),
    }


def _build_stem_activity_mask(
    mono_energy: np.ndarray,
    *,
    sample_rate: int,
    cleanup_profile: dict[str, float],
) -> np.ndarray:
    if mono_energy.size == 0:
        return np.ones(0, dtype=np.float32)

    fast_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * cleanup_profile["fast_ms"] / 1000.0)),
        ),
    )
    slow_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * cleanup_profile["slow_ms"] / 1000.0)),
        ),
    )
    hold_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * cleanup_profile["hold_ms"] / 1000.0)),
        ),
    )
    release_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * cleanup_profile["release_ms"] / 1000.0)),
        ),
    )

    fast_env = _moving_average(mono_energy, fast_samples)
    slow_env = _moving_average(mono_energy, slow_samples)

    noise_floor = float(
        np.percentile(slow_env, cleanup_profile["noise_percentile"])
    )
    peak_env = float(np.percentile(slow_env, 99.6))
    if peak_env <= noise_floor + 1e-6:
        return np.ones_like(mono_energy, dtype=np.float32)

    activity = np.clip(
        (slow_env - noise_floor) / (peak_env - noise_floor + 1e-6),
        0.0,
        1.0,
    )
    activity = np.power(activity, cleanup_profile["activity_exponent"])
    transient = np.clip(
        (fast_env / np.maximum(slow_env, 1e-6) - 1.0) / 0.85,
        0.0,
        1.0,
    )
    activity = np.clip(
        activity
        + transient * cleanup_profile["transient_blend"] * (1.0 - activity),
        0.0,
        1.0,
    )

    held_activity = _rolling_max(activity, hold_samples)
    smoothed_activity = _moving_average(held_activity, release_samples)
    smoothed_activity = np.maximum(
        smoothed_activity,
        held_activity * cleanup_profile["activity_floor_scale"],
    )
    suppression_floor = cleanup_profile["suppression_floor"]
    return suppression_floor + (1.0 - suppression_floor) * np.clip(
        smoothed_activity,
        0.0,
        1.0,
    )


def _build_stem_noise_gate_mask(
    mono_energy: np.ndarray,
    *,
    sample_rate: int,
    gate_profile: dict[str, float],
) -> np.ndarray:
    if mono_energy.size == 0:
        return np.ones(0, dtype=np.float32)

    fast_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * gate_profile["fast_ms"] / 1000.0)),
        ),
    )
    slow_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * gate_profile["slow_ms"] / 1000.0)),
        ),
    )
    hold_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * gate_profile["hold_ms"] / 1000.0)),
        ),
    )
    release_samples = max(
        1,
        min(
            mono_energy.size,
            int(round(sample_rate * gate_profile["release_ms"] / 1000.0)),
        ),
    )

    fast_env = _moving_average(mono_energy, fast_samples)
    slow_env = _moving_average(mono_energy, slow_samples)
    transient = np.clip(
        fast_env / np.maximum(slow_env, 1e-6) - 1.0,
        0.0,
        1.0,
    )
    control_env = np.maximum(
        slow_env,
        fast_env * (1.0 + transient * gate_profile["transient_bias"]),
    )

    noise_floor = float(
        np.percentile(slow_env, gate_profile["noise_percentile"])
    )
    peak_env = float(np.percentile(control_env, 99.8))
    if peak_env <= noise_floor + 1e-6:
        return np.ones_like(mono_energy, dtype=np.float32)

    open_threshold = (
        noise_floor + (peak_env - noise_floor) * gate_profile["threshold_ratio"]
    )
    full_open_level = (
        open_threshold
        + (peak_env - open_threshold) * gate_profile["full_open_ratio"]
    )
    gate_curve = np.clip(
        (control_env - open_threshold)
        / (full_open_level - open_threshold + 1e-6),
        0.0,
        1.0,
    )
    gate_curve = np.power(gate_curve, gate_profile["open_exponent"])
    held_gate = _rolling_max(gate_curve, hold_samples)
    smoothed_gate = _moving_average(held_gate, release_samples)
    sustained_activity = np.clip(
        (slow_env - noise_floor) / (peak_env - noise_floor + 1e-6),
        0.0,
        1.0,
    )
    smoothed_gate = np.maximum(
        smoothed_gate,
        sustained_activity * gate_profile["active_floor_scale"],
    )
    floor = gate_profile["floor"]
    gate_mask = floor + (1.0 - floor) * np.clip(smoothed_gate, 0.0, 1.0)
    mix = gate_profile["mix"]
    return 1.0 - mix * (1.0 - gate_mask)


def _apply_stem_residual_suppression(
    self,
    y: np.ndarray,
    *,
    stem_role: str | None,
    cleanup_pressure: float,
) -> np.ndarray:
    if y.size == 0:
        return y

    cleanup_profile = _resolve_stem_residual_profile(
        stem_role, cleanup_pressure
    )
    channels = y if y.ndim > 1 else y[np.newaxis, :]
    working_channels = np.asarray(channels, dtype=np.float32)
    mono_energy = np.mean(np.abs(working_channels), axis=0)
    if not np.any(mono_energy > 0.0):
        return y

    activity_mask = _build_stem_activity_mask(
        mono_energy,
        sample_rate=self.resampling_target,
        cleanup_profile=cleanup_profile,
    )
    expansion_drive = float(cleanup_profile["expansion_drive"])
    expansion_mix = float(cleanup_profile["expansion_mix"])
    cleaned_channels: list[np.ndarray] = []

    for channel in working_channels:
        cleaned_channel = channel * activity_mask
        peak = float(np.max(np.abs(cleaned_channel)))
        if peak > 1e-6 and expansion_mix > 0.0:
            normalized = np.clip(np.abs(cleaned_channel) / peak, 0.0, 1.0)
            expanded_channel = (
                np.sign(cleaned_channel)
                * peak
                * np.power(
                    normalized,
                    1.0 + expansion_drive,
                )
            )
            cleaned_channel = (
                cleaned_channel * (1.0 - expansion_mix)
                + expanded_channel * expansion_mix
            )
        cleaned_channels.append(cleaned_channel)

    cleaned = np.vstack(cleaned_channels) if y.ndim > 1 else cleaned_channels[0]
    return _restore_audio_dtype(cleaned, y.dtype)


def _apply_stem_noise_gate(
    self,
    y: np.ndarray,
    *,
    stem_role: str | None,
    cleanup_pressure: float,
) -> np.ndarray:
    if y.size == 0:
        return y

    config = getattr(self, "config", None)
    if not bool(getattr(config, "stem_noise_gate_enabled", True)):
        return y

    gate_strength = float(
        np.clip(getattr(config, "stem_noise_gate_strength", 1.0), 0.0, 1.5)
    )
    if gate_strength <= 0.0:
        return y

    channels = y if y.ndim > 1 else y[np.newaxis, :]
    working_channels = np.asarray(channels, dtype=np.float32)
    mono_energy = np.mean(np.abs(working_channels), axis=0)
    if not np.any(mono_energy > 0.0):
        return y

    gate_profile = _resolve_stem_noise_gate_profile(
        stem_role,
        cleanup_pressure,
        gate_strength,
    )
    residual_profile = _resolve_stem_residual_profile(
        stem_role, cleanup_pressure
    )
    gate_mask = _build_stem_noise_gate_mask(
        mono_energy,
        sample_rate=self.resampling_target,
        gate_profile=gate_profile,
    )
    activity_mask = _build_stem_activity_mask(
        mono_energy,
        sample_rate=self.resampling_target,
        cleanup_profile=residual_profile,
    )
    gate_mask = np.maximum(gate_mask, activity_mask)
    gated = working_channels * gate_mask
    output = np.vstack(gated) if y.ndim > 1 else gated[0]
    return _restore_audio_dtype(output, y.dtype)


def _resolve_stem_cleanup_anchors(
    self,
    *,
    stem_role: str | None,
    restoration_factor: float,
    air_restoration_factor: float,
    body_restoration_factor: float,
    closure_repair_factor: float,
) -> list[list[float]]:
    normalized_role = _normalize_stem_role(stem_role)
    mud_low_hz = self._fit_frequency(max(self.low_cut * 2.4, 180.0))
    mud_high_hz = self._fit_frequency(max(mud_low_hz * 1.8, 420.0))
    focus_hz = self._fit_frequency(max(self.bass_transition_hz * 1.8, 1150.0))
    presence_lift_hz = self._fit_frequency(
        max(self.treble_transition_hz * 0.62, 1700.0)
    )
    presence_cut_hz = self._fit_frequency(
        max(self.treble_transition_hz * 0.82, 2900.0)
    )
    air_low_hz = self._fit_frequency(
        max(self.treble_transition_hz * 1.28, 5200.0)
    )
    air_high_hz = self._fit_frequency(
        max(self.treble_transition_hz * 1.95, 9000.0)
    )
    harmonic_low_hz = self._fit_frequency(
        max(self.bass_transition_hz * 2.0, 850.0)
    )
    harmonic_high_hz = self._fit_frequency(
        max(self.bass_transition_hz * 3.1, 1800.0)
    )

    if normalized_role == "bass":
        mud_cut_db = float(
            np.clip(0.18 + body_restoration_factor * 0.75, 0.0, 1.45)
        )
        harmonic_boost_db = float(
            np.clip(
                0.08
                + closure_repair_factor * 0.28
                + air_restoration_factor * 0.14,
                0.0,
                0.95,
            )
        )
        return [
            [self.low_cut, 0.0],
            [mud_low_hz, -mud_cut_db],
            [mud_high_hz, -mud_cut_db * 0.72],
            [harmonic_low_hz, harmonic_boost_db * 0.42],
            [harmonic_high_hz, harmonic_boost_db],
            [self.high_cut, 0.0],
        ]

    if normalized_role == "vocals":
        mud_cut_db = float(
            np.clip(
                0.22
                + body_restoration_factor * 0.92
                + closure_repair_factor * 0.14,
                0.0,
                1.7,
            )
        )
        presence_lift_db = float(
            np.clip(0.12 + closure_repair_factor * 0.3, 0.0, 0.9)
        )
        presence_cut_db = float(
            np.clip(0.2 + closure_repair_factor * 0.98, 0.0, 1.9)
        )
        air_low_boost_db = float(
            np.clip(
                0.08
                + air_restoration_factor * 0.45
                + closure_repair_factor * 0.18,
                0.0,
                0.95,
            )
        )
        air_high_boost_db = float(
            np.clip(
                0.16
                + air_restoration_factor * 1.08
                + closure_repair_factor * 0.36,
                0.0,
                2.25,
            )
        )
        return [
            [self.low_cut, 0.0],
            [mud_low_hz, -mud_cut_db * 0.65],
            [mud_high_hz, -mud_cut_db],
            [presence_lift_hz, presence_lift_db],
            [presence_cut_hz, -presence_cut_db],
            [air_low_hz, air_low_boost_db],
            [air_high_hz, air_high_boost_db],
            [self.high_cut, 0.0],
        ]

    if normalized_role == "drums":
        mud_cut_db = float(
            np.clip(0.12 + body_restoration_factor * 0.42, 0.0, 0.85)
        )
        presence_cut_db = float(
            np.clip(0.08 + closure_repair_factor * 0.35, 0.0, 0.75)
        )
        air_high_boost_db = float(
            np.clip(0.1 + air_restoration_factor * 0.76, 0.0, 1.0)
        )
        return [
            [self.low_cut, 0.0],
            [mud_low_hz, -mud_cut_db],
            [presence_cut_hz, -presence_cut_db],
            [air_low_hz, air_high_boost_db * 0.45],
            [air_high_hz, air_high_boost_db],
            [self.high_cut, 0.0],
        ]

    mud_cut_db = float(
        np.clip(
            0.16 + body_restoration_factor * 0.58 + restoration_factor * 0.12,
            0.0,
            1.2,
        )
    )
    presence_lift_db = float(
        np.clip(0.08 + closure_repair_factor * 0.2, 0.0, 0.55)
    )
    presence_cut_db = float(
        np.clip(0.1 + closure_repair_factor * 0.52, 0.0, 1.0)
    )
    air_high_boost_db = float(
        np.clip(0.1 + air_restoration_factor * 0.82, 0.0, 1.3)
    )
    focus_boost_db = float(
        np.clip(0.06 + closure_repair_factor * 0.18, 0.0, 0.4)
    )
    return [
        [self.low_cut, 0.0],
        [mud_low_hz, -mud_cut_db * 0.72],
        [mud_high_hz, -mud_cut_db],
        [focus_hz, focus_boost_db],
        [presence_lift_hz, presence_lift_db],
        [presence_cut_hz, -presence_cut_db],
        [air_low_hz, air_high_boost_db * 0.5],
        [air_high_hz, air_high_boost_db],
        [self.high_cut, 0.0],
    ]


def apply_stem_cleanup(
    self,
    y: np.ndarray,
    *,
    stem_role: str | None,
    audio_eq_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    profile = getattr(self, "spectral_balance_profile", None)
    restoration_factor = float(
        np.clip(getattr(profile, "restoration_factor", 0.0), 0.0, 1.0)
    )
    air_restoration_factor = float(
        np.clip(getattr(profile, "air_restoration_factor", 0.0), 0.0, 1.0)
    )
    body_restoration_factor = float(
        np.clip(getattr(profile, "body_restoration_factor", 0.0), 0.0, 1.0)
    )
    closure_repair_factor = float(
        np.clip(
            getattr(
                profile,
                "closure_repair_factor",
                max(
                    air_restoration_factor * 0.85,
                    restoration_factor * 0.55,
                ),
            ),
            0.0,
            1.0,
        )
    )
    cleanup_pressure = float(
        np.clip(
            max(
                restoration_factor * 0.35,
                air_restoration_factor * 0.45,
                body_restoration_factor * 0.3,
                closure_repair_factor * 0.6,
            ),
            0.0,
            1.0,
        )
    )
    stem_cleanup_strength = float(
        np.clip(
            getattr(
                self,
                "stem_cleanup_strength",
                getattr(self.config, "stem_cleanup_strength", 1.0),
            ),
            0.0,
            1.5,
        )
    )
    cleanup_pressure = float(
        np.clip(cleanup_pressure * stem_cleanup_strength, 0.0, 1.0)
    )
    cleanup_pressure = _resolve_stem_cleanup_pressure(
        stem_role,
        cleanup_pressure,
    )
    cleaned = y
    if cleanup_pressure > 0.04:
        cleanup_anchors = _resolve_stem_cleanup_anchors(
            self,
            stem_role=stem_role,
            restoration_factor=restoration_factor,
            air_restoration_factor=air_restoration_factor,
            body_restoration_factor=body_restoration_factor,
            closure_repair_factor=closure_repair_factor,
        )

        def eq_channel(channel: np.ndarray) -> np.ndarray:
            return audio_eq_fn(
                audio_data=channel,
                anchors=cleanup_anchors,
                sample_rate=self.resampling_target,
                nperseg=self.analysis_nperseg,
            )

        cleaned = (
            np.vstack([eq_channel(channel) for channel in y])
            if y.ndim > 1
            else eq_channel(y)
        )

    cleaned = _apply_stem_residual_suppression(
        self,
        cleaned,
        stem_role=stem_role,
        cleanup_pressure=cleanup_pressure,
    )
    return _apply_stem_noise_gate(
        self,
        cleaned,
        stem_role=stem_role,
        cleanup_pressure=cleanup_pressure,
    )


def apply_eq(
    self,
    y: np.ndarray,
    audio_eq_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

    input_db, f_axis = self.measure_spectrum(y_mono)
    input_db = self.smooth_curve(input_db, f_axis, self.smoothing_fraction)

    target_db = self.build_target_curve(f_axis)

    correction_db = target_db - input_db
    correction_db = np.nan_to_num(
        correction_db,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    self.spectral_balance_profile = self.build_spectral_balance_profile(
        correction_db,
        f_axis,
    )

    eq_stride = max(1, len(correction_db) // 192)

    correction_db = np.append(correction_db[:-1:eq_stride], correction_db[-1])
    f_axis = np.append(f_axis[:-1:eq_stride], f_axis[-1])

    reference_hz = float(
        np.sqrt(self.bass_transition_hz * self.treble_transition_hz)
    )
    reference_index = int(np.argmin(np.abs(f_axis - reference_hz)))
    restoration_factor = float(
        np.clip(
            getattr(self.spectral_balance_profile, "restoration_factor", 0.0),
            0.0,
            1.0,
        )
    )
    edge_baseline_db = float(np.average([correction_db[0], correction_db[-1]]))
    reference_baseline_db = float(correction_db[reference_index])
    baseline_db = float(
        edge_baseline_db * (1.0 - restoration_factor)
        + reference_baseline_db * restoration_factor
    )

    body_restoration_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "body_restoration_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    air_restoration_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "air_restoration_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    mud_cleanup_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "mud_cleanup_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    harshness_restraint_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "harshness_restraint_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    low_end_restraint_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "low_end_restraint_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    legacy_tonal_rebalance_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "legacy_tonal_rebalance_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    closed_top_end_repair_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "closed_top_end_repair_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    closure_repair_factor = float(
        np.clip(
            getattr(
                self.spectral_balance_profile,
                "closure_repair_factor",
                0.0,
            ),
            0.0,
            1.0,
        )
    )
    presence_start_hz = float(
        min(
            self.treble_transition_hz,
            max(reference_hz * 1.35, self.treble_transition_hz * 0.55),
        )
    )
    positive_correction_db = np.maximum(correction_db, 0.0)
    presence_mask = (f_axis >= presence_start_hz) & (
        f_axis < max(self.treble_transition_hz, presence_start_hz + 1.0)
    )
    high_band_mask = f_axis >= presence_start_hz
    negative_correction_db = np.maximum(-correction_db, 0.0)
    presence_deficit_db = (
        float(np.mean(positive_correction_db[presence_mask], dtype=np.float32))
        if np.any(presence_mask)
        else 0.0
    )
    high_band_deficit_db = (
        float(np.mean(positive_correction_db[high_band_mask], dtype=np.float32))
        if np.any(high_band_mask)
        else 0.0
    )
    if closure_repair_factor <= 0.0:
        closure_repair_factor = float(
            np.clip(
                max(presence_deficit_db * 0.9, high_band_deficit_db) / 9.0,
                0.0,
                1.0,
            )
        )
        closure_repair_factor = float(
            np.clip(
                closure_repair_factor
                * (
                    0.35
                    + restoration_factor * 0.35
                    + air_restoration_factor * 0.6
                ),
                0.0,
                1.0,
            )
        )
    mud_low_hz = float(
        min(self.high_cut, max(self.bass_transition_hz * 1.15, 160.0))
    )
    mud_high_hz = float(
        min(
            self.high_cut,
            max(
                mud_low_hz + 1.0,
                min(self.treble_transition_hz * 0.38, 550.0),
            ),
        )
    )
    mud_mask = (f_axis >= mud_low_hz) & (f_axis <= mud_high_hz)
    if mud_cleanup_factor <= 0.0 and np.any(mud_mask):
        mud_excess_db = float(
            np.mean(negative_correction_db[mud_mask], dtype=np.float32)
        )
        mud_peak_excess_db = float(
            np.percentile(negative_correction_db[mud_mask], 76.0)
        )
        mud_cleanup_factor = float(
            np.clip(
                (max(mud_excess_db, mud_peak_excess_db * 0.9) - 1.15) / 4.35,
                0.0,
                1.0,
            )
        )
        mud_cleanup_factor = float(
            np.clip(
                mud_cleanup_factor
                * (
                    0.45
                    + restoration_factor * 0.28
                    + air_restoration_factor * 0.18
                    + closure_repair_factor * 0.24
                ),
                0.0,
                1.0,
            )
        )
    low_end_focus_low_hz = float(
        min(
            self.high_cut,
            max(self.low_cut * 3.2, self.bass_transition_hz * 0.72, 75.0),
        )
    )
    low_end_focus_high_hz = float(
        min(
            self.high_cut,
            max(
                low_end_focus_low_hz + 1.0,
                min(self.bass_transition_hz * 1.9, 280.0),
            ),
        )
    )
    low_end_focus_mask = (f_axis >= low_end_focus_low_hz) & (
        f_axis <= low_end_focus_high_hz
    )
    if low_end_restraint_factor <= 0.0:
        bass_excess_db = (
            float(
                np.mean(
                    negative_correction_db[f_axis <= self.bass_transition_hz],
                    dtype=np.float32,
                )
            )
            if np.any(f_axis <= self.bass_transition_hz)
            else 0.0
        )
        bass_peak_excess_db = (
            float(
                np.percentile(
                    negative_correction_db[f_axis <= self.bass_transition_hz],
                    78.0,
                )
            )
            if np.any(f_axis <= self.bass_transition_hz)
            else 0.0
        )
        low_end_focus_excess_db = (
            float(
                np.mean(
                    negative_correction_db[low_end_focus_mask], dtype=np.float32
                )
            )
            if np.any(low_end_focus_mask)
            else 0.0
        )
        low_end_focus_peak_excess_db = (
            float(
                np.percentile(negative_correction_db[low_end_focus_mask], 80.0)
            )
            if np.any(low_end_focus_mask)
            else 0.0
        )
        low_end_restraint_factor = float(
            np.clip(
                (
                    max(
                        bass_excess_db * 0.82,
                        bass_peak_excess_db * 0.88,
                        low_end_focus_excess_db,
                        low_end_focus_peak_excess_db * 0.92,
                        mud_cleanup_factor * 3.2,
                    )
                    - 0.95
                )
                / 3.8,
                0.0,
                1.0,
            )
        )
        low_end_restraint_factor = float(
            np.clip(
                low_end_restraint_factor
                * (
                    0.16
                    + closure_repair_factor * 0.58
                    + mud_cleanup_factor * 0.34
                    + air_restoration_factor * 0.12
                ),
                0.0,
                1.0,
            )
        )
    harsh_low_hz = float(
        min(self.high_cut, max(presence_start_hz * 1.08, 2600.0))
    )
    harsh_high_hz = float(
        min(
            self.high_cut,
            max(
                harsh_low_hz + 1.0,
                min(max(self.treble_transition_hz * 1.18, 5200.0), 6200.0),
            ),
        )
    )
    harsh_mask = (f_axis >= harsh_low_hz) & (f_axis < harsh_high_hz)
    sibilance_low_hz = float(min(self.high_cut, max(harsh_high_hz, 5600.0)))
    sibilance_high_hz = float(
        min(
            self.high_cut,
            max(sibilance_low_hz + 1.0, min(self.high_cut, 9200.0)),
        )
    )
    sibilance_mask = (f_axis >= sibilance_low_hz) & (
        f_axis <= sibilance_high_hz
    )
    if harshness_restraint_factor <= 0.0:
        harsh_excess_db = (
            float(np.mean(negative_correction_db[harsh_mask], dtype=np.float32))
            if np.any(harsh_mask)
            else 0.0
        )
        harsh_peak_excess_db = (
            float(np.percentile(negative_correction_db[harsh_mask], 82.0))
            if np.any(harsh_mask)
            else 0.0
        )
        sibilance_peak_excess_db = (
            float(np.percentile(negative_correction_db[sibilance_mask], 78.0))
            if np.any(sibilance_mask)
            else 0.0
        )
        harshness_restraint_factor = float(
            np.clip(
                (
                    max(
                        harsh_excess_db,
                        harsh_peak_excess_db * 0.95,
                        sibilance_peak_excess_db * 0.86,
                    )
                    - 0.7
                )
                / 3.6,
                0.0,
                1.0,
            )
        )
        harshness_restraint_factor = float(
            np.clip(
                harshness_restraint_factor
                * (
                    0.12
                    + closure_repair_factor * 0.55
                    + air_restoration_factor * 0.35
                    + restoration_factor * 0.18
                ),
                0.0,
                1.0,
            )
        )
    baseline_db -= (
        max(reference_baseline_db, 0.0) * closure_repair_factor * 0.42
    )
    baseline_db -= max(reference_baseline_db, 0.0) * mud_cleanup_factor * 0.12
    correction_db -= baseline_db

    low_span_denominator = max(
        float(np.log2(self.bass_transition_hz / self.low_cut)),
        1e-6,
    )
    low_shape = np.clip(
        np.log2(self.bass_transition_hz / np.maximum(f_axis, self.low_cut))
        / low_span_denominator,
        0.0,
        1.0,
    )
    high_span_denominator = max(
        float(np.log2(self.high_cut / self.treble_transition_hz)),
        1e-6,
    )
    high_shape = np.clip(
        np.log2(
            np.maximum(f_axis, self.treble_transition_hz)
            / self.treble_transition_hz
        )
        / high_span_denominator,
        0.0,
        1.0,
    )
    presence_span_denominator = max(
        float(
            np.log2(
                max(self.treble_transition_hz, reference_hz + 1.0)
                / reference_hz
            )
        ),
        1e-6,
    )
    presence_shape = np.clip(
        np.log2(np.maximum(f_axis, reference_hz) / reference_hz)
        / presence_span_denominator,
        0.0,
        1.0,
    )
    presence_shape = np.clip(presence_shape - high_shape * 0.65, 0.0, 1.0)
    mud_center_hz = float(
        min(self.high_cut, max(self.bass_transition_hz * 1.9, 280.0))
    )
    mud_high_shape_hz = float(
        min(self.high_cut, max(mud_center_hz * 1.85, 620.0))
    )
    mud_rise_denominator = max(
        float(np.log2(max(mud_center_hz, mud_low_hz + 1.0) / mud_low_hz)),
        1e-6,
    )
    mud_fall_denominator = max(
        float(
            np.log2(max(mud_high_shape_hz, mud_center_hz + 1.0) / mud_center_hz)
        ),
        1e-6,
    )
    mud_rise_shape = np.clip(
        np.log2(np.maximum(f_axis, mud_low_hz) / mud_low_hz)
        / mud_rise_denominator,
        0.0,
        1.0,
    )
    mud_fall_shape = np.clip(
        np.log2(mud_high_shape_hz / np.maximum(f_axis, mud_center_hz))
        / mud_fall_denominator,
        0.0,
        1.0,
    )
    mud_shape = np.clip(
        np.minimum(mud_rise_shape, mud_fall_shape) * (1.0 - high_shape * 0.92),
        0.0,
        1.0,
    )
    low_end_focus_center_hz = float(
        min(
            self.high_cut,
            max(
                low_end_focus_low_hz * 1.28,
                self.bass_transition_hz * 1.05,
                135.0,
            ),
        )
    )
    low_end_focus_rise_denominator = max(
        float(
            np.log2(
                max(low_end_focus_center_hz, low_end_focus_low_hz + 1.0)
                / low_end_focus_low_hz
            )
        ),
        1e-6,
    )
    low_end_focus_fall_denominator = max(
        float(
            np.log2(
                max(low_end_focus_high_hz, low_end_focus_center_hz + 1.0)
                / low_end_focus_center_hz
            )
        ),
        1e-6,
    )
    low_end_focus_rise_shape = np.clip(
        np.log2(np.maximum(f_axis, low_end_focus_low_hz) / low_end_focus_low_hz)
        / low_end_focus_rise_denominator,
        0.0,
        1.0,
    )
    low_end_focus_fall_shape = np.clip(
        np.log2(
            low_end_focus_high_hz / np.maximum(f_axis, low_end_focus_center_hz)
        )
        / low_end_focus_fall_denominator,
        0.0,
        1.0,
    )
    low_end_focus_shape = np.clip(
        np.minimum(low_end_focus_rise_shape, low_end_focus_fall_shape)
        * (1.0 - presence_shape * 0.94)
        * (1.0 - high_shape * 0.96),
        0.0,
        1.0,
    )
    low_end_restraint_shape = np.clip(
        np.maximum(low_shape * 0.42, low_end_focus_shape + mud_shape * 0.02),
        0.0,
        1.0,
    )
    harsh_center_hz = float(
        min(
            self.high_cut,
            max(harsh_low_hz * 1.28, self.treble_transition_hz * 0.98, 3400.0),
        )
    )
    harsh_rise_denominator = max(
        float(np.log2(max(harsh_center_hz, harsh_low_hz + 1.0) / harsh_low_hz)),
        1e-6,
    )
    harsh_fall_denominator = max(
        float(
            np.log2(max(harsh_high_hz, harsh_center_hz + 1.0) / harsh_center_hz)
        ),
        1e-6,
    )
    harsh_rise_shape = np.clip(
        np.log2(np.maximum(f_axis, harsh_low_hz) / harsh_low_hz)
        / harsh_rise_denominator,
        0.0,
        1.0,
    )
    harsh_fall_shape = np.clip(
        np.log2(harsh_high_hz / np.maximum(f_axis, harsh_center_hz))
        / harsh_fall_denominator,
        0.0,
        1.0,
    )
    harshness_shape = np.clip(
        np.minimum(harsh_rise_shape, harsh_fall_shape)
        * (0.72 + presence_shape * 0.28)
        * (1.0 - high_shape * 0.38),
        0.0,
        1.0,
    )
    sibilance_shape = np.clip(
        np.log2(np.maximum(f_axis, sibilance_low_hz) / sibilance_low_hz)
        / max(
            float(
                np.log2(
                    max(sibilance_high_hz, sibilance_low_hz + 1.0)
                    / sibilance_low_hz
                )
            ),
            1e-6,
        ),
        0.0,
        1.0,
    )
    sibilance_shape = np.clip(
        sibilance_shape
        * (
            1.0
            - np.clip(
                (f_axis - sibilance_high_hz) / max(sibilance_high_hz, 1.0),
                0.0,
                1.0,
            )
        ),
        0.0,
        1.0,
    )
    closed_top_start_hz = float(
        min(
            self.high_cut,
            max(self.treble_transition_hz * 1.14, 4000.0),
        )
    )
    closed_top_span_denominator = max(
        float(
            np.log2(
                max(self.high_cut, closed_top_start_hz + 1.0)
                / closed_top_start_hz
            )
        ),
        1e-6,
    )
    closed_top_ramp = np.clip(
        np.log2(np.maximum(f_axis, closed_top_start_hz) / closed_top_start_hz)
        / closed_top_span_denominator,
        0.0,
        1.0,
    )
    closed_top_shape = np.where(
        f_axis >= closed_top_start_hz,
        0.3 + closed_top_ramp * 0.7,
        0.0,
    ).astype(np.float32, copy=False)
    treble_repair_factor = float(
        np.clip(
            max(
                closure_repair_factor,
                air_restoration_factor * 0.94,
                high_band_deficit_db / 8.5,
            ),
            0.0,
            1.0,
        )
    )
    air_shelf_shape = np.clip(high_shape + presence_shape * 0.42, 0.0, 1.0)
    correction_db += (
        low_shape
        * body_restoration_factor
        * (0.6 * (1.0 - low_end_restraint_factor * 0.9))
    )
    correction_db -= (
        low_end_restraint_shape
        * low_end_restraint_factor
        * (1.02 + mud_cleanup_factor * 0.5 + closure_repair_factor * 0.22)
    )
    correction_db -= (
        low_end_restraint_shape * legacy_tonal_rebalance_factor * 0.18
    )
    correction_db -= (
        mud_shape
        * mud_cleanup_factor
        * (1.55 + body_restoration_factor * 0.35 + closure_repair_factor * 0.3)
    )
    correction_db += presence_shape * closure_repair_factor * 0.95
    correction_db += high_shape * air_restoration_factor * 1.1
    correction_db += high_shape * closure_repair_factor * 0.55
    correction_db += air_shelf_shape * treble_repair_factor * 0.42
    correction_db += high_shape * legacy_tonal_rebalance_factor * 0.24
    correction_db += air_shelf_shape * legacy_tonal_rebalance_factor * 0.46
    correction_db += (
        closed_top_shape
        * closed_top_end_repair_factor
        * (0.88 + legacy_tonal_rebalance_factor * 0.16)
    )
    correction_db += air_shelf_shape * closed_top_end_repair_factor * 0.24
    correction_db -= (
        harshness_shape
        * harshness_restraint_factor
        * (0.9 + treble_repair_factor * 0.62 + closure_repair_factor * 0.22)
    )
    correction_db -= (
        sibilance_shape
        * harshness_restraint_factor
        * (0.18 + air_restoration_factor * 0.16)
    )

    correction_db[0], correction_db[-1] = 0.0, 0.0

    correction_db *= self.spectral_balance_profile.correction_strength
    correction_db = np.clip(
        correction_db,
        -self.spectral_balance_profile.max_cut_db,
        self.spectral_balance_profile.max_boost_db,
    )

    flat_anchors = np.column_stack((f_axis, correction_db))

    def eq_channel(channel: np.ndarray) -> np.ndarray:
        channel = audio_eq_fn(
            audio_data=channel,
            anchors=flat_anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )
        return audio_eq_fn(
            audio_data=channel,
            anchors=self.anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )

    if y.ndim > 1:
        return np.vstack([eq_channel(channel) for channel in y])

    return eq_channel(y)
