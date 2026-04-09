from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def _moving_average_last_axis(
    values: np.ndarray, window_size: int
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.array(array, copy=True)

    safe_window = min(max(int(window_size), 1), int(array.shape[-1]))
    if safe_window <= 1:
        return np.array(array, copy=True)

    kernel = np.ones(safe_window, dtype=np.float32) / float(safe_window)
    return np.apply_along_axis(
        lambda channel: np.convolve(channel, kernel, mode="same").astype(
            np.float32
        ),
        -1,
        array,
    )


def _signal_rms(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(array), dtype=np.float32)))


def _band_correlation(
    left_band: np.ndarray,
    right_band: np.ndarray,
    window_size: int,
) -> np.ndarray:
    left_energy = _moving_average_last_axis(np.square(left_band), window_size)
    right_energy = _moving_average_last_axis(np.square(right_band), window_size)
    cross_energy = _moving_average_last_axis(
        left_band * right_band, window_size
    )
    denominator = np.maximum(np.sqrt(left_energy * right_energy), 1e-6)
    return np.clip(cross_energy / denominator, -1.0, 1.0)


def _repair_narrow_stereo_image(
    self,
    mid: np.ndarray,
    side: np.ndarray,
    *,
    signal_module: Any,
) -> np.ndarray:
    profile = getattr(self, "spectral_balance_profile", None)
    restoration_factor = float(
        np.clip(getattr(profile, "restoration_factor", 0.0), 0.0, 1.0)
    )
    air_restoration_factor = float(
        np.clip(
            getattr(profile, "air_restoration_factor", 0.0),
            0.0,
            1.0,
        )
    )
    if restoration_factor <= 0.0 or mid.shape[-1] < 64:
        return side

    mid_signal = np.asarray(mid, dtype=np.float32)
    side_signal = np.asarray(side, dtype=np.float32)
    mid_rms = _signal_rms(mid_signal)
    side_rms = _signal_rms(side_signal)
    current_width_ratio = 0.0
    if mid_rms + side_rms > 1e-6:
        current_width_ratio = float(
            np.clip(side_rms / (mid_rms + side_rms), 0.0, 1.0)
        )
    width_deficit = float(
        np.clip((0.14 - current_width_ratio) / 0.14, 0.0, 1.0)
    )
    if width_deficit <= 0.0:
        return side_signal

    nyquist = self.resampling_target / 2.0 - 1.0
    cutoff_hz = float(
        np.clip(
            max(
                getattr(self, "mono_bass_hz", 140.0) * 2.4,
                getattr(self, "stereo_tone_variation_cutoff_hz", 1400.0) * 0.9,
                700.0,
            ),
            350.0,
            max(nyquist, 351.0),
        )
    )
    if cutoff_hz >= nyquist:
        return side_signal

    high_sos = signal_module.butter(
        2,
        cutoff_hz / (self.resampling_target / 2.0),
        btype="high",
        output="sos",
    )
    upper_band = signal_module.sosfiltfilt(high_sos, mid_signal)
    shift_samples = max(
        int(
            round(
                self.resampling_target
                * (0.00045 + 0.00115 * restoration_factor * width_deficit)
            )
        ),
        1,
    )
    delayed = np.roll(upper_band, shift_samples)
    delayed[:shift_samples] = 0.0
    advanced = np.roll(upper_band, -shift_samples)
    advanced[-shift_samples:] = 0.0
    transient = np.diff(upper_band, prepend=upper_band[:1])
    synthesized_side = (delayed - advanced) * 0.5 + transient * (
        0.22 + air_restoration_factor * 0.18
    )

    synthesized_rms = _signal_rms(synthesized_side)
    upper_band_rms = _signal_rms(upper_band)
    target_width_ratio = float(
        np.clip(
            current_width_ratio
            + width_deficit
            * (
                0.05 + restoration_factor * 0.12 + air_restoration_factor * 0.05
            ),
            0.0,
            0.22,
        )
    )
    target_side_rms = (
        upper_band_rms
        * target_width_ratio
        / max(
            1.0 - target_width_ratio,
            1e-6,
        )
    )
    added_side_rms = max(target_side_rms - side_rms, 0.0)
    if synthesized_rms <= 1e-6 or added_side_rms <= 1e-6:
        return side_signal

    synthesized_side = synthesized_side * (added_side_rms / synthesized_rms)
    return np.nan_to_num(
        side_signal + synthesized_side.astype(np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def _apply_stereo_tonal_variation(
    self,
    y: np.ndarray,
    *,
    signal_module: Any,
) -> np.ndarray:
    setattr(self, "last_stereo_motion_activity", 0.0)
    setattr(self, "last_stereo_motion_correlation_guard", 1.0)
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    variation_db = float(
        np.clip(getattr(self, "stereo_tone_variation_db", 0.0), 0.0, 2.5)
    )
    if (
        variation_db <= 0.0
        or signal.ndim < 2
        or signal.shape[0] < 2
        or signal.shape[-1] < 32
    ):
        return signal

    nyquist = self.resampling_target / 2.0 - 1.0
    cutoff_hz = float(
        np.clip(
            getattr(self, "stereo_tone_variation_cutoff_hz", 1800.0),
            max(getattr(self, "mono_bass_hz", 140.0) * 2.0, 350.0),
            max(nyquist, 351.0),
        )
    )
    if cutoff_hz >= nyquist:
        return signal

    smoothing_ms = float(
        max(getattr(self, "stereo_tone_variation_smoothing_ms", 180.0), 10.0)
    )
    smoothing_samples = max(
        int(round(self.resampling_target * smoothing_ms / 1000.0)), 1
    )
    if smoothing_samples % 2 == 0:
        smoothing_samples += 1

    left = signal[0]
    right = signal[1]
    energy_left = _moving_average_last_axis(np.square(left), smoothing_samples)
    energy_right = _moving_average_last_axis(
        np.square(right), smoothing_samples
    )
    balance = (energy_left - energy_right) / np.maximum(
        energy_left + energy_right, 1e-6
    )
    balance = np.nan_to_num(
        balance - float(np.mean(balance, dtype=np.float32)), nan=0.0
    )
    balance = np.clip(balance * 0.9, -1.0, 1.0)
    balance_velocity = np.diff(balance, prepend=balance[:1])
    balance_velocity = _moving_average_last_axis(
        balance_velocity,
        max(smoothing_samples // 5, 3),
    )
    motion = np.clip(balance * 0.72 + balance_velocity * 10.0, -1.0, 1.0)

    high_sos = signal_module.butter(
        2,
        cutoff_hz / (self.resampling_target / 2.0),
        btype="high",
        output="sos",
    )
    mid_low_cutoff_hz = float(
        min(
            max(getattr(self, "mono_bass_hz", 140.0) * 1.4, 260.0),
            cutoff_hz * 0.72,
        )
    )
    if mid_low_cutoff_hz >= cutoff_hz - 120.0:
        mid_low_cutoff_hz = float(max(250.0, cutoff_hz * 0.58))
    low_sos = signal_module.butter(
        2,
        mid_low_cutoff_hz / (self.resampling_target / 2.0),
        btype="low",
        output="sos",
    )
    left_low = signal_module.sosfiltfilt(low_sos, left)
    right_low = signal_module.sosfiltfilt(low_sos, right)
    left_high = signal_module.sosfiltfilt(high_sos, left)
    right_high = signal_module.sosfiltfilt(high_sos, right)
    left_mid = left - left_low - left_high
    right_mid = right - right_low - right_high

    mid_amount = float(
        np.clip(getattr(self, "stereo_motion_mid_amount", 0.55), 0.0, 2.5)
    )
    high_amount = float(
        np.clip(getattr(self, "stereo_motion_high_amount", 1.0), 0.0, 2.5)
    )
    correlation_guard_strength = float(
        np.clip(getattr(self, "stereo_motion_correlation_guard", 1.0), 0.0, 2.5)
    )
    max_side_boost = float(
        np.clip(getattr(self, "stereo_motion_max_side_boost", 0.16), 0.0, 2.5)
    )

    mid_corr = _band_correlation(
        left_mid, right_mid, max(smoothing_samples // 3, 3)
    )
    high_corr = _band_correlation(
        left_high, right_high, max(smoothing_samples // 4, 3)
    )
    mid_guard = np.clip(
        1.0 - np.maximum(-mid_corr, 0.0) * 0.55 * correlation_guard_strength,
        0.35,
        1.0,
    )
    high_guard = np.clip(
        1.0 - np.maximum(-high_corr, 0.0) * 0.75 * correlation_guard_strength,
        0.25,
        1.0,
    )

    mid_gain_db = motion * variation_db * 0.38 * mid_amount
    high_gain_db = motion * variation_db * 0.92 * high_amount
    left_mid_gain = np.asarray(
        np.power(10.0, mid_gain_db / 20.0), dtype=np.float32
    )
    right_mid_gain = np.asarray(
        np.power(10.0, -mid_gain_db / 20.0), dtype=np.float32
    )
    left_high_gain = np.asarray(
        np.power(10.0, high_gain_db / 20.0), dtype=np.float32
    )
    right_high_gain = np.asarray(
        np.power(10.0, -high_gain_db / 20.0), dtype=np.float32
    )
    left_mid = left_mid * left_mid_gain
    right_mid = right_mid * right_mid_gain
    left_high = left_high * left_high_gain
    right_high = right_high * right_high_gain

    mid_width_motion = 1.0 + np.clip(
        np.abs(motion) * variation_db * 0.08 * mid_amount * mid_guard,
        0.0,
        max_side_boost * 0.6,
    )
    high_width_motion = 1.0 + np.clip(
        np.abs(motion) * variation_db * 0.14 * high_amount * high_guard,
        0.0,
        max_side_boost,
    )
    mid_mid = 0.5 * (left_mid + right_mid)
    mid_side = 0.5 * (left_mid - right_mid) * mid_width_motion
    high_mid = 0.5 * (left_high + right_high)
    high_side = 0.5 * (left_high - right_high) * high_width_motion
    setattr(
        self,
        "last_stereo_motion_activity",
        float(np.mean(np.abs(motion), dtype=np.float32)),
    )
    setattr(
        self,
        "last_stereo_motion_correlation_guard",
        float(np.mean(0.5 * (mid_guard + high_guard), dtype=np.float32)),
    )
    output = np.stack(
        [
            left_low + mid_mid + mid_side + high_mid + high_side,
            right_low + mid_mid - mid_side + high_mid - high_side,
        ],
        axis=0,
    )
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def apply_pre_limiter_saturation(
    self,
    y: np.ndarray,
    *,
    dynamic_drive_db: float = 0.0,
) -> np.ndarray:
    profile = getattr(self, "spectral_balance_profile", None)
    restoration_factor = float(
        np.clip(getattr(profile, "restoration_factor", 0.0), 0.0, 1.0)
    )
    body_restoration_factor = float(
        np.clip(
            getattr(profile, "body_restoration_factor", 0.0),
            0.0,
            1.0,
        )
    )
    saturation_ratio = float(
        np.clip(
            getattr(self, "pre_limiter_saturation_ratio", 0.0)
            + restoration_factor * 0.04
            + body_restoration_factor * 0.03,
            0.0,
            0.45,
        )
    )
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if saturation_ratio <= 0.0 or signal.size == 0:
        return signal

    drive_scale_db = max(float(dynamic_drive_db), 0.0) * saturation_ratio * 0.2
    drive_lin = float(10.0 ** (drive_scale_db / 20.0))
    threshold = float(np.clip(1.0 - saturation_ratio * 0.35, 0.65, 0.999))
    saturated = np.tanh((signal * drive_lin) / threshold) * threshold
    output = signal * (1.0 - saturation_ratio) + saturated * saturation_ratio

    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def apply_soft_clip_stage(
    y: np.ndarray,
    *,
    ceil_db: float = -0.1,
    soft_clip_ratio: float = 0.2,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_ratio = float(np.clip(soft_clip_ratio, 0.0, 0.95))
    if safe_ratio <= 0.0 or signal.size == 0:
        return signal

    limit_lin = float(10.0 ** (float(ceil_db) / 20.0))
    threshold = limit_lin * (1.0 - safe_ratio)
    if threshold <= 0.0:
        return signal

    output = np.array(signal, copy=True)
    mask = output > threshold
    output[mask] = threshold + (limit_lin - threshold) * np.tanh(
        (output[mask] - threshold) / (limit_lin - threshold)
    )
    mask_neg = output < -threshold
    output[mask_neg] = -threshold - (limit_lin - threshold) * np.tanh(
        (-output[mask_neg] - threshold) / (limit_lin - threshold)
    )
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def apply_safety_clamp(
    y: np.ndarray,
    *,
    ceil_db: float = -0.1,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    limit_lin = float(10.0 ** (float(ceil_db) / 20.0))
    peak = float(np.max(np.abs(signal))) if signal.size else 0.0
    if peak > limit_lin and peak > 0.0:
        signal = signal * (limit_lin / peak)
    return np.clip(signal, -limit_lin, limit_lin)


def apply_true_peak_limiter(
    self,
    y: np.ndarray,
    drive_db: float = 0.0,
    ceil_db: float = -0.1,
    os_factor: int = 4,
    lookahead_ms: float = 2.0,
    attack_ms: float = 1.0,
    release_ms_min: float = 30.0,
    release_ms_max: float = 130.0,
    window_ms: float = 4.0,
    *,
    signal_module: Any,
    maximum_filter1d_fn: Callable[..., np.ndarray],
    uniform_filter1d_fn: Callable[..., np.ndarray],
    limiter_smooth_env_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    input_dtype = y.dtype if np.issubdtype(y.dtype, np.floating) else np.float32
    y_in = np.asarray(y, dtype=np.float32)
    orig_len = y_in.shape[-1]

    drive_lin = 10.0 ** (drive_db / 20.0)
    limit_lin = 10.0 ** (ceil_db / 20.0)

    sr_os = self.resampling_target * os_factor
    y_os = signal_module.resample_poly(y_in, os_factor, 1, axis=-1)
    y_driven = y_os * drive_lin

    lookahead_samp = max(0, int(round(lookahead_ms * sr_os / 1000.0)))
    lookahead_span = lookahead_samp + 1
    abs_driven = np.abs(y_driven)
    linked_env = np.max(abs_driven, axis=0) if y_driven.ndim > 1 else abs_driven

    peak_env = maximum_filter1d_fn(
        linked_env[::-1],
        size=lookahead_span,
        mode="constant",
    )[::-1]

    rms_win = max(1, int(round(window_ms / 1000 * sr_os)))
    rms_env = np.sqrt(
        uniform_filter1d_fn(
            (linked_env**2)[::-1],
            size=rms_win,
            mode="constant",
        )[::-1]
    )

    crest = peak_env / (rms_env + 1e-12)
    release_ms = np.clip(
        release_ms_max / (crest + 1e-6),
        release_ms_min,
        release_ms_max,
    )

    atk_c = np.exp(-1.0 / (sr_os * attack_ms / 1000.0))
    rel_c = np.exp(-1.0 / (sr_os * release_ms / 1000.0))

    control_env = 0.9 * peak_env + 0.1 * rms_env
    control_smooth = limiter_smooth_env_fn(control_env, atk_c, rel_c)

    gain = np.ones_like(control_smooth)
    mask = control_smooth > limit_lin
    gain[mask] = limit_lin / control_smooth[mask]

    y_limited = np.clip(y_driven * gain, -limit_lin, limit_lin)
    y_down = signal_module.resample_poly(y_limited, 1, os_factor, axis=-1)
    output = y_down[..., :orig_len]
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    return output.astype(input_dtype)


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
    *,
    signal_module: Any,
    maximum_filter1d_fn: Callable[..., np.ndarray],
    uniform_filter1d_fn: Callable[..., np.ndarray],
    limiter_smooth_env_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    limited = apply_true_peak_limiter(
        self,
        y,
        drive_db=drive_db,
        ceil_db=ceil_db,
        os_factor=os_factor,
        lookahead_ms=lookahead_ms,
        attack_ms=attack_ms,
        release_ms_min=release_ms_min,
        release_ms_max=release_ms_max,
        window_ms=window_ms,
        signal_module=signal_module,
        maximum_filter1d_fn=maximum_filter1d_fn,
        uniform_filter1d_fn=uniform_filter1d_fn,
        limiter_smooth_env_fn=limiter_smooth_env_fn,
    )
    clipped = apply_soft_clip_stage(
        limited,
        ceil_db=ceil_db,
        soft_clip_ratio=soft_clip_ratio,
    )
    return apply_safety_clamp(clipped, ceil_db=ceil_db)


def apply_spatial_enhancement(
    self,
    y: np.ndarray,
    *,
    signal_module: Any,
) -> np.ndarray:
    if y.ndim < 2 or y.shape[0] != 2:
        return y

    width = float(max(self.stereo_width, 0.0))
    mid = 0.5 * (y[0] + y[1])
    side = 0.5 * (y[0] - y[1])

    cutoff = float(np.clip(self.mono_bass_hz, self.low_cut, self.high_cut))
    if side.shape[-1] > 32 and cutoff < self.resampling_target / 2.0 - 1.0:
        sos = signal_module.butter(
            4,
            cutoff / (self.resampling_target / 2.0),
            btype="high",
            output="sos",
        )
        side = signal_module.sosfiltfilt(sos, side)

    side = _repair_narrow_stereo_image(
        self,
        mid,
        side,
        signal_module=signal_module,
    )

    side *= width
    coef = np.sqrt(2.0 / (1.0 + width**2))
    widened = np.stack(
        [(mid + side) * coef, (mid - side) * coef],
        axis=0,
    )

    widened = _apply_stereo_tonal_variation(
        self,
        widened,
        signal_module=signal_module,
    )

    return np.nan_to_num(widened, nan=0.0, posinf=0.0, neginf=0.0)


def multiband_compress(
    self,
    y: np.ndarray,
    *,
    signal_module: Any,
    decoupled_envelope_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    def lr4(x: np.ndarray, fc: float) -> tuple[np.ndarray, np.ndarray]:
        if x.shape[-1] <= 9:
            return x, x

        sr2 = self.resampling_target / 2.0
        sos_l = signal_module.butter(2, fc / sr2, btype="low", output="sos")
        sos_h = signal_module.butter(2, fc / sr2, btype="high", output="sos")

        lp = signal_module.sosfiltfilt(sos_l, x, axis=-1)
        hp = signal_module.sosfiltfilt(sos_h, x, axis=-1)

        return lp, hp

    def compress(
        x: np.ndarray,
        control: np.ndarray | None,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
        makeup_db: float,
        knee_db: float = 3.0,
    ) -> np.ndarray:
        ratio = max(float(ratio), 1.0)
        knee_db = max(float(knee_db), 1e-3)
        ac = np.exp(-1.0 / (self.resampling_target * attack_ms / 1000.0 + 1e-9))
        rc = np.exp(
            -1.0 / (self.resampling_target * release_ms / 1000.0 + 1e-9)
        )
        mk = 10.0 ** (makeup_db / 20.0)
        env_source = np.abs(control) if control is not None else np.abs(x)
        env = decoupled_envelope_fn(
            20.0 * np.log10(env_source + 1e-12),
            ac,
            rc,
        )
        hk = knee_db / 2.0
        gain = np.zeros_like(env)
        above = env > threshold_db + hk
        in_kn = (env > threshold_db - hk) & ~above
        gain[above] = -(env[above] - threshold_db) * (1.0 - 1.0 / ratio)
        ki = env[in_kn] - threshold_db + hk
        gain[in_kn] = -(ki**2) / (2.0 * knee_db) * (1.0 - 1.0 / ratio)

        gain_lin = 10.0 ** (gain / 20.0) * mk
        if x.ndim > 1:
            return x * gain_lin[np.newaxis, :]

        return x * gain_lin

    def split_bands(ch: np.ndarray, fcs: list[float]) -> list[np.ndarray]:
        bands: list[np.ndarray] = []
        current = ch
        for fc in fcs[:-1]:
            lo, current = lr4(current, fc)
            bands.append(lo)
        bands.append(current)
        return bands

    sorted_bands = [
        band
        for band in sorted(self.bands, key=lambda band: band["fc"])
        if band["fc"] > 0.0
    ]
    if not sorted_bands:
        return y

    fcs_sorted = [band["fc"] for band in sorted_bands]

    if y.ndim > 1:
        channel_parts = [split_bands(channel, fcs_sorted) for channel in y]
        if any(len(parts) != len(sorted_bands) for parts in channel_parts):
            raise ValueError("Band split mismatch")

        comps: list[np.ndarray] = []
        for band_index, band_cfg in enumerate(sorted_bands):
            band_stack = np.stack(
                [parts[band_index] for parts in channel_parts],
                axis=0,
            )
            thr = band_cfg["base_threshold"] + band_cfg["makeup_db"]
            control = np.max(np.abs(band_stack), axis=0)
            comps.append(
                compress(
                    band_stack,
                    control,
                    thr,
                    band_cfg["ratio"],
                    band_cfg["attack_ms"],
                    band_cfg["release_ms"],
                    band_cfg["makeup_db"],
                    knee_db=band_cfg["knee_db"],
                )
            )

        return np.sum(comps, axis=0)

    signal_parts = split_bands(y, fcs_sorted)
    if len(signal_parts) != len(sorted_bands):
        raise ValueError("Band split mismatch")

    comps: list[np.ndarray] = []
    for band_cfg, sig in zip(sorted_bands, signal_parts, strict=True):
        thr = band_cfg["base_threshold"] + band_cfg["makeup_db"]
        comps.append(
            compress(
                sig,
                None,
                thr,
                band_cfg["ratio"],
                band_cfg["attack_ms"],
                band_cfg["release_ms"],
                band_cfg["makeup_db"],
                knee_db=band_cfg["knee_db"],
            )
        )

    return np.sum(comps, axis=0)
