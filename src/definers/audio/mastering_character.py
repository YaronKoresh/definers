from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class LimiterRecoverySettings:
    style: str
    attack_ms: float
    release_ms_min: float
    release_ms_max: float
    window_ms: float

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _moving_average(values: np.ndarray, window_size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.array(array, copy=True)

    safe_window = min(max(int(window_size), 1), int(array.size))
    if safe_window <= 1:
        return np.array(array, copy=True)

    kernel = np.ones(safe_window, dtype=np.float32) / float(safe_window)
    return np.convolve(array, kernel, mode="same").astype(np.float32)


def resolve_limiter_recovery_settings(
    self,
    *,
    attack_ms: float,
    release_ms_min: float,
    release_ms_max: float,
    window_ms: float,
) -> LimiterRecoverySettings:
    style = (
        str(getattr(self, "limiter_recovery_style", "balanced")).strip().lower()
    )
    style_map = {
        "tight": (0.85, 0.7, 0.75, 0.85),
        "balanced": (1.0, 1.0, 1.0, 1.0),
        "glue": (1.15, 1.3, 1.45, 1.2),
    }
    attack_scale, release_min_scale, release_max_scale, window_scale = (
        style_map.get(
            style,
            style_map["balanced"],
        )
    )
    return LimiterRecoverySettings(
        style=style,
        attack_ms=max(float(attack_ms) * attack_scale, 0.1),
        release_ms_min=max(float(release_ms_min) * release_min_scale, 1.0),
        release_ms_max=max(
            float(release_ms_max) * release_max_scale,
            float(release_ms_min) * release_min_scale,
        ),
        window_ms=max(float(window_ms) * window_scale, 1.0),
    )


def resolve_low_end_mono_tightening_amount(self) -> float:
    style = (
        str(getattr(self, "low_end_mono_tightening", "balanced"))
        .strip()
        .lower()
    )
    style_map = {
        "off": 0.0,
        "gentle": 0.45,
        "balanced": 0.75,
        "firm": 1.0,
    }
    style_amount = style_map.get(style, style_map["balanced"])
    configured_amount = float(
        np.clip(getattr(self, "low_end_mono_tightening_amount", 1.0), 0.0, 1.0)
    )
    return float(np.clip(style_amount * configured_amount, 0.0, 1.0))


def apply_low_end_mono_tightening(
    self,
    y: np.ndarray,
    *,
    sample_rate: int,
    cutoff_hz: float | None = None,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal.ndim < 2 or signal.shape[0] < 2 or signal.shape[-1] == 0:
        return signal

    tightening_amount = resolve_low_end_mono_tightening_amount(self)
    if tightening_amount <= 0.0:
        return signal

    low_cutoff_hz = float(
        max(
            20.0,
            cutoff_hz
            if cutoff_hz is not None
            else getattr(self, "contract_low_end_mono_cutoff_hz", 160.0),
        )
    )
    if sample_rate <= 0:
        return signal

    window_size = max(int(round(sample_rate / low_cutoff_hz)), 1)
    if window_size % 2 == 0:
        window_size += 1

    mid = 0.5 * (signal[0] + signal[1])
    side = 0.5 * (signal[0] - signal[1])
    if signal.shape[-1] <= window_size:
        tightened_side = side * (1.0 - tightening_amount)
    else:
        side_low = _moving_average(side, window_size)
        side_high = side - side_low
        tightened_side = side_high + side_low * (1.0 - tightening_amount)
    output = np.stack([mid + tightened_side, mid - tightened_side], axis=0)

    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def apply_micro_dynamics_finish(
    self,
    y: np.ndarray,
    *,
    sample_rate: int,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    strength = float(
        np.clip(getattr(self, "micro_dynamics_strength", 0.0), 0.0, 1.0)
    )
    if strength <= 0.0 or signal.size == 0 or sample_rate <= 0:
        return signal

    fast_window_ms = float(
        max(getattr(self, "micro_dynamics_fast_window_ms", 8.0), 1.0)
    )
    slow_window_ms = float(
        max(
            getattr(self, "micro_dynamics_slow_window_ms", 45.0),
            fast_window_ms + 1.0,
        )
    )
    transient_bias = float(
        np.clip(getattr(self, "micro_dynamics_transient_bias", 0.75), 0.0, 1.0)
    )

    linked = (
        np.max(np.abs(signal), axis=0) if signal.ndim > 1 else np.abs(signal)
    )
    fast_window = max(int(round(sample_rate * fast_window_ms / 1000.0)), 1)
    slow_window = max(
        int(round(sample_rate * slow_window_ms / 1000.0)), fast_window + 1
    )
    fast_env = _moving_average(linked, fast_window)
    slow_env = _moving_average(linked, slow_window)
    transient_mask = np.clip(
        (fast_env - slow_env) / np.maximum(fast_env, 1e-6),
        0.0,
        1.0,
    )
    sustain_mask = 1.0 - transient_mask * transient_bias
    drive = 1.0 + sustain_mask * strength * 2.0
    normalized = np.tanh(signal * drive) / np.tanh(1.0 + strength * 2.0)
    blend = np.clip(strength * sustain_mask, 0.0, 0.6)

    if signal.ndim > 1:
        blend = blend[np.newaxis, :]
        transient_mask = transient_mask[np.newaxis, :]

    output = signal * (1.0 - blend) + normalized * blend
    transient_preserve = np.clip(transient_mask * strength * 0.18, 0.0, 0.2)
    output = output * (1.0 - transient_preserve) + signal * transient_preserve

    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = [
    "LimiterRecoverySettings",
    "apply_low_end_mono_tightening",
    "apply_micro_dynamics_finish",
    "resolve_limiter_recovery_settings",
    "resolve_low_end_mono_tightening_amount",
]
