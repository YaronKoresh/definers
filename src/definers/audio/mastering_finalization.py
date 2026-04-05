from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .mastering_contract import MasteringContract
from .mastering_dynamics import (
    apply_pre_limiter_saturation,
    apply_safety_clamp,
)
from .mastering_loudness import MasteringLoudnessMetrics
from .mastering_reference import measure_transient_density


@dataclass(frozen=True, slots=True)
class FinalizationAction:
    gain_db: float
    soft_clip_ratio: float
    stereo_width_scale: float
    integrated_gap_db: float
    reasons: tuple[str, ...]
    should_apply: bool

    def to_dict(self) -> dict[str, float | bool | tuple[str, ...]]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CharacterStageDecision:
    applied: bool
    reverted: bool
    reasons: tuple[str, ...]
    input_integrated_lufs: float
    output_integrated_lufs: float
    input_true_peak_dbfs: float
    output_true_peak_dbfs: float

    def to_dict(self) -> dict[str, float | bool | tuple[str, ...]]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PeakCatchEvent:
    attempt_index: int
    drive_db: float
    ceil_db: float
    soft_clip_ratio: float
    peak_over_db: float
    before_integrated_lufs: float
    after_integrated_lufs: float
    before_true_peak_dbfs: float
    after_true_peak_dbfs: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _measure_signal_crest_factor_db(y: np.ndarray) -> float:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal.size == 0:
        return 0.0

    peak = float(np.max(np.abs(signal)))
    if peak <= 0.0:
        return 0.0

    rms = float(np.sqrt(np.mean(np.square(signal), dtype=np.float32)))
    if rms <= 0.0:
        return 0.0

    return float(max(20.0 * np.log10(peak / max(rms, 1e-12)), 0.0))


def _candidate_respects_contract(
    candidate: np.ndarray,
    contract: MasteringContract | None,
) -> bool:
    if (
        contract is None
        or getattr(contract, "min_crest_factor_db", None) is None
    ):
        return True
    return bool(
        _measure_signal_crest_factor_db(candidate)
        >= float(getattr(contract, "min_crest_factor_db")) - 0.05
    )


def _resolve_headroom_recovery_profile(
    self,
    signal: np.ndarray,
    *,
    sample_rate: int,
) -> dict[str, float | str]:
    metrics = getattr(self, "last_post_clamp_metrics", None)
    integrated_gap_db = 0.0
    crest_factor_db = _measure_signal_crest_factor_db(signal)
    if metrics is not None:
        integrated_lufs = getattr(metrics, "integrated_lufs", None)
        if integrated_lufs is not None and np.isfinite(integrated_lufs):
            integrated_gap_db = max(
                float(self.target_lufs) - float(integrated_lufs), 0.0
            )
        metric_crest_factor = getattr(metrics, "crest_factor_db", None)
        if metric_crest_factor is not None and np.isfinite(metric_crest_factor):
            crest_factor_db = float(metric_crest_factor)

    transient_density = float(measure_transient_density(signal, sample_rate))
    contract = getattr(self, "last_mastering_contract", None)
    crest_safety = 1.0
    min_crest_factor_db = None
    if contract is not None:
        min_crest_factor_db = getattr(contract, "min_crest_factor_db", None)
    if min_crest_factor_db is not None:
        crest_safety = float(
            np.clip(
                (crest_factor_db - float(min_crest_factor_db) + 0.5) / 3.5,
                0.0,
                1.0,
            )
        )

    loudness_need = float(
        np.clip(
            max(
                integrated_gap_db
                - float(getattr(self, "final_lufs_tolerance", 0.0)),
                0.0,
            )
            / max(float(getattr(self, "max_final_boost_db", 1.0)), 1.0),
            0.0,
            1.0,
        )
    )
    loudness_deficit_db = float(
        max(
            integrated_gap_db
            - float(getattr(self, "final_lufs_tolerance", 0.0)),
            0.0,
        )
    )
    crest_pressure = float(np.clip((crest_factor_db - 5.0) / 4.0, 0.0, 1.0))
    transient_pressure = float(
        np.clip((transient_density - 0.08) / 0.18, 0.0, 1.0)
    )
    clip_pressure = float(
        np.clip(
            max(
                loudness_need * 0.75,
                crest_pressure * 0.65,
                transient_pressure * 0.85,
            )
            * (0.45 + crest_safety * 0.55),
            0.0,
            1.0,
        )
    )

    if loudness_deficit_db <= 0.05:
        mode = "disabled"
    elif clip_pressure < 0.18:
        mode = "makeup_only"
    elif transient_pressure >= 0.45 or crest_pressure >= 0.55:
        mode = "guarded"
    else:
        mode = "guarded"

    if mode == "disabled":
        max_step_db = 0.0
    elif mode == "makeup_only":
        max_step_db = min(1.35, 0.75 + loudness_need * 0.6 + crest_safety * 0.2)
    else:
        max_step_db = min(1.0, 0.45 + loudness_need * 0.4 + crest_safety * 0.15)

    return {
        "mode": mode,
        "integrated_gap_db": float(integrated_gap_db),
        "loudness_deficit_db": loudness_deficit_db,
        "transient_density": transient_density,
        "clip_pressure": clip_pressure,
        "max_step_db": float(max_step_db),
    }


def compute_dynamic_drive(self, current_lufs: float) -> float:
    if not np.isfinite(current_lufs):
        current_lufs = self.target_lufs - self.drive_db

    profile = getattr(self, "spectral_balance_profile", None)
    rescue_factor = float(
        np.clip(getattr(profile, "rescue_factor", 0.0), 0.0, 1.0)
    )
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

    return float(
        np.clip(
            self.target_lufs
            - current_lufs
            + self.drive_db
            + rescue_factor * self.spectral_drive_bias_db
            + restoration_factor * 0.9
            + body_restoration_factor * 0.35,
            -12.0,
            18.0,
        )
    )


def compute_primary_soft_clip_ratio(self, dynamic_drive_db: float) -> float:
    profile = getattr(self, "spectral_balance_profile", None)
    rescue_factor = float(
        np.clip(getattr(profile, "rescue_factor", 0.0), 0.0, 1.0)
    )
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
    drive_soft_clip_push = max(float(dynamic_drive_db), 0.0) * 0.006
    return float(
        np.clip(
            self.limiter_soft_clip_ratio
            + rescue_factor * 0.04
            + restoration_factor * 0.03
            + body_restoration_factor * 0.02
            + max(float(dynamic_drive_db), 0.0) * 0.004,
            0.0,
            0.45,
        )
    )


def resolve_final_true_peak_target(self) -> float:
    target_dbfs = float(self.ceil_db)
    codec_margin_db = float(
        max(getattr(self, "codec_headroom_margin_db", 0.0), 0.0)
    )
    delivery_target_dbfs = getattr(
        self, "delivery_decoded_true_peak_dbfs", None
    )
    if delivery_target_dbfs is not None and np.isfinite(delivery_target_dbfs):
        target_dbfs = min(
            target_dbfs, float(delivery_target_dbfs) - codec_margin_db
        )
    elif codec_margin_db > 0.0 and "lossy" in str(
        getattr(self, "delivery_profile", "")
    ):
        target_dbfs = min(target_dbfs, float(self.ceil_db) - codec_margin_db)
    return target_dbfs


def apply_stereo_width_restraint(
    y: np.ndarray,
    *,
    stereo_width_scale: float = 1.0,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    width_scale = float(np.clip(stereo_width_scale, 0.0, 1.0))
    if width_scale >= 0.999 or signal.ndim < 2 or signal.shape[0] < 2:
        return signal

    mid = 0.5 * (signal[0] + signal[1])
    side = 0.5 * (signal[0] - signal[1]) * width_scale
    output = np.stack([mid + side, mid - side], axis=0)
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def plan_follow_up_action(
    self,
    metrics: MasteringLoudnessMetrics,
    contract: MasteringContract,
) -> FinalizationAction:
    integrated_gap_db = float(contract.target_lufs - metrics.integrated_lufs)
    short_term_over_db = max(
        0.0,
        0.0
        if contract.max_short_term_lufs is None
        else metrics.max_short_term_lufs - contract.max_short_term_lufs,
    )
    momentary_over_db = max(
        0.0,
        0.0
        if contract.max_momentary_lufs is None
        else metrics.max_momentary_lufs - contract.max_momentary_lufs,
    )
    crest_under_db = max(
        0.0,
        0.0
        if contract.min_crest_factor_db is None
        else contract.min_crest_factor_db - metrics.crest_factor_db,
    )
    crest_over_db = max(
        0.0,
        0.0
        if contract.max_crest_factor_db is None
        else metrics.crest_factor_db - contract.max_crest_factor_db,
    )
    stereo_over_ratio = max(
        0.0,
        0.0
        if contract.max_stereo_width_ratio is None
        else metrics.stereo_width_ratio - contract.max_stereo_width_ratio,
    )
    low_end_mono_under_ratio = max(
        0.0,
        0.0
        if contract.min_low_end_mono_ratio is None
        else contract.min_low_end_mono_ratio - metrics.low_end_mono_ratio,
    )

    safe_gain_db = max(integrated_gap_db - self.final_lufs_tolerance, 0.0)
    if safe_gain_db > 0.0:
        safe_gain_db = max(
            safe_gain_db
            - short_term_over_db * 0.4
            - momentary_over_db * 0.6
            - stereo_over_ratio * 1.2
            - low_end_mono_under_ratio * 1.6,
            0.0,
        )
    gain_db = min(safe_gain_db, self.max_final_boost_db)

    clipping_push = 0.0
    if gain_db > 0.0 and integrated_gap_db > gain_db:
        clipping_push += min(0.06, (integrated_gap_db - gain_db) * 0.02)
    if gain_db > 0.0 and crest_over_db > 0.0:
        clipping_push += min(0.04, crest_over_db * 0.01)
    if crest_under_db > 0.0:
        clipping_push = max(
            clipping_push - min(0.05, crest_under_db * 0.02), 0.0
        )

    stereo_width_scale = 1.0
    if stereo_over_ratio > 0.0 or low_end_mono_under_ratio > 0.0:
        stereo_width_scale = float(
            np.clip(
                1.0 - stereo_over_ratio * 2.2 - low_end_mono_under_ratio * 2.8,
                0.7,
                1.0,
            )
        )

    if gain_db > 0.0:
        soft_clip_ratio = float(
            np.clip(
                self.compute_primary_soft_clip_ratio(gain_db) + clipping_push,
                0.0,
                0.7,
            )
        )
    else:
        soft_clip_ratio = float(np.clip(self.limiter_soft_clip_ratio, 0.0, 0.7))

    reasons: list[str] = []
    if gain_db > 0.0:
        reasons.append("gain")
    if gain_db > 0.0 and soft_clip_ratio > self.limiter_soft_clip_ratio + 1e-6:
        reasons.append("clip_density")
    if stereo_width_scale < 0.999:
        reasons.append("stereo_restraint")

    return FinalizationAction(
        gain_db=float(gain_db),
        soft_clip_ratio=soft_clip_ratio,
        stereo_width_scale=stereo_width_scale,
        integrated_gap_db=integrated_gap_db,
        reasons=tuple(reasons),
        should_apply=bool(reasons),
    )


def apply_delivery_trim(
    self,
    y: np.ndarray,
    *,
    sample_rate: int,
    measure_true_peak_fn,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal.size == 0:
        return signal

    target_dbfs = resolve_final_true_peak_target(self)
    measured_true_peak_dbfs = measure_true_peak_fn(
        signal,
        sample_rate,
        oversample_factor=self.true_peak_oversample_factor,
    )
    self.last_delivery_trim_target_dbfs = float(target_dbfs)
    self.last_delivery_trim_input_true_peak_dbfs = (
        None
        if not np.isfinite(measured_true_peak_dbfs)
        else float(measured_true_peak_dbfs)
    )
    self.last_delivery_trim_attenuation_db = 0.0
    self.last_delivery_trim_output_true_peak_dbfs = (
        self.last_delivery_trim_input_true_peak_dbfs
    )
    if (
        not np.isfinite(measured_true_peak_dbfs)
        or measured_true_peak_dbfs <= target_dbfs
    ):
        return signal

    attenuation_db = float(measured_true_peak_dbfs - target_dbfs)
    self.last_delivery_trim_attenuation_db = attenuation_db
    attenuation_lin = float(10.0 ** (-attenuation_db / 20.0))
    output = signal * attenuation_lin
    output_true_peak_dbfs = measure_true_peak_fn(
        output,
        sample_rate,
        oversample_factor=self.true_peak_oversample_factor,
    )
    self.last_delivery_trim_output_true_peak_dbfs = (
        None
        if not np.isfinite(output_true_peak_dbfs)
        else float(output_true_peak_dbfs)
    )

    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


def apply_final_headroom_recovery(
    self,
    y: np.ndarray,
    *,
    sample_rate: int,
    measure_true_peak_fn,
) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    self.last_headroom_recovery_gain_db = 0.0
    self.last_headroom_recovery_input_true_peak_dbfs = None
    self.last_headroom_recovery_output_true_peak_dbfs = None
    self.last_headroom_recovery_failure_reasons = ()
    self.last_headroom_recovery_mode = None
    self.last_headroom_recovery_integrated_gap_db = None
    self.last_headroom_recovery_transient_density = None
    self.last_headroom_recovery_closed_margin_db = None
    self.last_headroom_recovery_unused_margin_db = None
    if signal.size == 0:
        return signal

    target_dbfs = resolve_final_true_peak_target(self)
    measured_true_peak_dbfs = measure_true_peak_fn(
        signal,
        sample_rate,
        oversample_factor=self.true_peak_oversample_factor,
    )
    if not np.isfinite(measured_true_peak_dbfs):
        self.last_headroom_recovery_failure_reasons = ("invalid_input_peak",)
        return signal

    self.last_headroom_recovery_input_true_peak_dbfs = float(
        measured_true_peak_dbfs
    )
    recovery_profile = _resolve_headroom_recovery_profile(
        self,
        signal,
        sample_rate=sample_rate,
    )
    self.last_headroom_recovery_mode = str(recovery_profile["mode"])
    self.last_headroom_recovery_integrated_gap_db = float(
        recovery_profile["integrated_gap_db"]
    )
    self.last_headroom_recovery_transient_density = float(
        recovery_profile["transient_density"]
    )
    output = np.array(signal, dtype=np.float32, copy=True)
    output_true_peak_dbfs = float(measured_true_peak_dbfs)
    failure_reasons: list[str] = []
    contract = getattr(self, "last_mastering_contract", None)
    initial_available_headroom_db = float(
        max(target_dbfs - measured_true_peak_dbfs, 0.0)
    )
    loudness_deficit_db = float(recovery_profile["loudness_deficit_db"])
    if loudness_deficit_db <= 0.05:
        failure_reasons.append("loudness_already_within_tolerance")
    elif initial_available_headroom_db <= 0.01:
        failure_reasons.append("no_margin")
    else:
        gain_budget_db = float(
            min(
                loudness_deficit_db,
                initial_available_headroom_db,
                float(recovery_profile["max_step_db"]),
            )
        )
        if gain_budget_db <= 0.01:
            failure_reasons.append("no_recovery_budget")
        else:
            best_candidate = output
            best_true_peak_dbfs = output_true_peak_dbfs
            search_low_db = 0.0
            search_high_db = gain_budget_db
            for _ in range(10):
                if search_high_db - search_low_db <= 0.002:
                    break
                trial_gain_db = 0.5 * (search_low_db + search_high_db)
                trial = output * float(10.0 ** (trial_gain_db / 20.0))
                trial_true_peak_dbfs = float(
                    measure_true_peak_fn(
                        trial,
                        sample_rate,
                        oversample_factor=self.true_peak_oversample_factor,
                    )
                )
                if not np.isfinite(trial_true_peak_dbfs):
                    search_high_db = trial_gain_db
                    continue
                if (
                    trial_true_peak_dbfs <= target_dbfs + 0.01
                    and _candidate_respects_contract(trial, contract)
                ):
                    best_candidate = trial
                    best_true_peak_dbfs = trial_true_peak_dbfs
                    search_low_db = trial_gain_db
                else:
                    search_high_db = trial_gain_db

            if best_true_peak_dbfs > output_true_peak_dbfs + 0.001:
                output = best_candidate
                output_true_peak_dbfs = best_true_peak_dbfs
            else:
                failure_reasons.append("linear_recovery_stalled")

    remaining_margin_db = float(max(target_dbfs - output_true_peak_dbfs, 0.0))

    self.last_headroom_recovery_gain_db = float(
        max(output_true_peak_dbfs - measured_true_peak_dbfs, 0.0)
    )
    self.last_headroom_recovery_output_true_peak_dbfs = (
        None
        if not np.isfinite(output_true_peak_dbfs)
        else float(output_true_peak_dbfs)
    )
    self.last_headroom_recovery_closed_margin_db = float(
        max(initial_available_headroom_db - remaining_margin_db, 0.0)
    )
    self.last_headroom_recovery_unused_margin_db = remaining_margin_db
    self.last_headroom_recovery_failure_reasons = tuple(
        dict.fromkeys(failure_reasons)
    )
    return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)


__all__ = [
    "CharacterStageDecision",
    "FinalizationAction",
    "PeakCatchEvent",
    "apply_final_headroom_recovery",
    "apply_delivery_trim",
    "apply_pre_limiter_saturation",
    "apply_stereo_width_restraint",
    "compute_dynamic_drive",
    "compute_primary_soft_clip_ratio",
    "plan_follow_up_action",
    "resolve_final_true_peak_target",
]
