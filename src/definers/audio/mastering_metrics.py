from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .mastering_contract import (
    MasteringContract,
    MasteringContractAssessment,
    assess_mastering_contract,
)
from .mastering_finalization import CharacterStageDecision, PeakCatchEvent
from .mastering_loudness import (
    MasteringLoudnessMetrics,
    measure_mastering_loudness,
)
from .mastering_reference import measure_stereo_motion


@dataclass(frozen=True, slots=True)
class MasteringReport:
    input_metrics: MasteringLoudnessMetrics
    post_eq_metrics: MasteringLoudnessMetrics | None
    post_spatial_metrics: MasteringLoudnessMetrics | None
    post_limiter_metrics: MasteringLoudnessMetrics | None
    post_character_metrics: MasteringLoudnessMetrics | None
    post_peak_catch_metrics: MasteringLoudnessMetrics | None
    post_delivery_trim_metrics: MasteringLoudnessMetrics | None
    post_clamp_metrics: MasteringLoudnessMetrics | None
    final_in_memory_metrics: MasteringLoudnessMetrics
    output_metrics: MasteringLoudnessMetrics
    decoded_metrics: MasteringLoudnessMetrics | None
    decoded_sample_rate: int | None
    target_lufs: float | None
    ceil_db: float | None
    preset_name: str | None
    delivery_profile_name: str | None
    delivery_issues: tuple[str, ...]
    contract: MasteringContract | None
    output_contract_assessment: MasteringContractAssessment | None
    decoded_contract_assessment: MasteringContractAssessment | None
    character_stage_decision: CharacterStageDecision | None
    peak_catch_events: tuple[PeakCatchEvent, ...]
    resolved_true_peak_target_dbfs: float | None
    stereo_motion_activity: float | None
    stereo_motion_correlation_guard: float | None
    post_spatial_stereo_motion: float | None
    output_stereo_motion: float | None
    delivery_trim_attenuation_db: float
    delivery_trim_input_true_peak_dbfs: float | None
    delivery_trim_target_dbfs: float | None
    delivery_trim_output_true_peak_dbfs: float | None
    post_clamp_true_peak_dbfs: float | None
    post_clamp_true_peak_delta_db: float | None
    headroom_recovery_gain_db: float
    headroom_recovery_input_true_peak_dbfs: float | None
    headroom_recovery_output_true_peak_dbfs: float | None
    headroom_recovery_failure_reasons: tuple[str, ...]
    headroom_recovery_mode: str | None
    headroom_recovery_integrated_gap_db: float | None
    headroom_recovery_transient_density: float | None
    headroom_recovery_closed_margin_db: float | None
    headroom_recovery_unused_margin_db: float | None
    integrated_lufs_delta: float
    sample_peak_delta_db: float
    true_peak_delta_db: float
    crest_factor_delta_db: float
    target_lufs_error_db: float | None
    true_peak_margin_db: float | None
    decoded_true_peak_margin_db: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def generate_mastering_report(
    input_signal: np.ndarray,
    output_signal: np.ndarray,
    sample_rate: int,
    *,
    post_eq_signal: np.ndarray | None = None,
    post_spatial_signal: np.ndarray | None = None,
    post_limiter_signal: np.ndarray | None = None,
    post_character_signal: np.ndarray | None = None,
    post_peak_catch_signal: np.ndarray | None = None,
    post_delivery_trim_signal: np.ndarray | None = None,
    post_clamp_signal: np.ndarray | None = None,
    decoded_signal: np.ndarray | None = None,
    decoded_sample_rate: int | None = None,
    target_lufs: float | None = None,
    ceil_db: float | None = None,
    preset_name: str | None = None,
    delivery_profile_name: str | None = None,
    delivery_issues: tuple[str, ...] = (),
    contract: MasteringContract | None = None,
    decoded_contract: MasteringContract | None = None,
    character_stage_decision: CharacterStageDecision | None = None,
    peak_catch_events: tuple[PeakCatchEvent, ...] = (),
    resolved_true_peak_target_dbfs: float | None = None,
    stereo_motion_activity: float | None = None,
    stereo_motion_correlation_guard: float | None = None,
    delivery_trim_attenuation_db: float = 0.0,
    delivery_trim_input_true_peak_dbfs: float | None = None,
    delivery_trim_target_dbfs: float | None = None,
    delivery_trim_output_true_peak_dbfs: float | None = None,
    post_clamp_true_peak_dbfs: float | None = None,
    post_clamp_true_peak_delta_db: float | None = None,
    headroom_recovery_gain_db: float = 0.0,
    headroom_recovery_input_true_peak_dbfs: float | None = None,
    headroom_recovery_output_true_peak_dbfs: float | None = None,
    headroom_recovery_failure_reasons: tuple[str, ...] = (),
    headroom_recovery_mode: str | None = None,
    headroom_recovery_integrated_gap_db: float | None = None,
    headroom_recovery_transient_density: float | None = None,
    headroom_recovery_closed_margin_db: float | None = None,
    headroom_recovery_unused_margin_db: float | None = None,
    true_peak_oversample_factor: int = 4,
    signal_module: Any = signal,
) -> MasteringReport:
    low_end_mono_cutoff_hz = (
        160.0 if contract is None else contract.low_end_mono_cutoff_hz
    )

    def measure_optional_metrics(
        y: np.ndarray | None, sr_value: int
    ) -> MasteringLoudnessMetrics | None:
        if y is None:
            return None
        return measure_mastering_loudness(
            y,
            sr_value,
            true_peak_oversample_factor=true_peak_oversample_factor,
            low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
            signal_module=signal_module,
        )

    input_metrics = measure_mastering_loudness(
        input_signal,
        sample_rate,
        true_peak_oversample_factor=true_peak_oversample_factor,
        low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
        signal_module=signal_module,
    )
    output_metrics = measure_mastering_loudness(
        output_signal,
        sample_rate,
        true_peak_oversample_factor=true_peak_oversample_factor,
        low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
        signal_module=signal_module,
    )
    post_eq_metrics = measure_optional_metrics(post_eq_signal, sample_rate)
    post_spatial_metrics = measure_optional_metrics(
        post_spatial_signal, sample_rate
    )
    post_limiter_metrics = measure_optional_metrics(
        post_limiter_signal, sample_rate
    )
    post_character_metrics = measure_optional_metrics(
        post_character_signal, sample_rate
    )
    post_peak_catch_metrics = measure_optional_metrics(
        post_peak_catch_signal, sample_rate
    )
    post_delivery_trim_metrics = measure_optional_metrics(
        post_delivery_trim_signal, sample_rate
    )
    post_clamp_metrics = measure_optional_metrics(
        post_clamp_signal, sample_rate
    )
    final_in_memory_metrics = output_metrics
    post_spatial_stereo_motion = (
        None
        if post_spatial_signal is None
        else measure_stereo_motion(post_spatial_signal, sample_rate)
    )
    output_stereo_motion = measure_stereo_motion(output_signal, sample_rate)
    decoded_metrics = measure_optional_metrics(
        decoded_signal,
        sample_rate if decoded_sample_rate is None else decoded_sample_rate,
    )

    target_lufs_error_db = (
        None
        if target_lufs is None
        else output_metrics.integrated_lufs - float(target_lufs)
    )
    peak_ceiling = None
    if resolved_true_peak_target_dbfs is not None and np.isfinite(
        resolved_true_peak_target_dbfs
    ):
        peak_ceiling = float(resolved_true_peak_target_dbfs)
    elif contract is not None and contract.max_true_peak_dbfs is not None:
        peak_ceiling = float(contract.max_true_peak_dbfs)
    elif ceil_db is not None:
        peak_ceiling = float(ceil_db)

    true_peak_margin_db = (
        None
        if peak_ceiling is None
        else peak_ceiling - output_metrics.true_peak_dbfs
    )
    decoded_true_peak_margin_db = None
    decoded_peak_ceiling = None
    if (
        decoded_contract is not None
        and decoded_contract.max_true_peak_dbfs is not None
    ):
        decoded_peak_ceiling = decoded_contract.max_true_peak_dbfs
    elif ceil_db is not None:
        decoded_peak_ceiling = float(ceil_db)
    if decoded_peak_ceiling is not None and decoded_metrics is not None:
        decoded_true_peak_margin_db = (
            float(decoded_peak_ceiling) - decoded_metrics.true_peak_dbfs
        )

    output_contract_assessment = (
        None
        if contract is None
        else assess_mastering_contract(output_metrics, contract)
    )
    decoded_contract_assessment = (
        None
        if decoded_contract is None or decoded_metrics is None
        else assess_mastering_contract(
            decoded_metrics,
            decoded_contract,
            target_lufs_tolerance_db=decoded_contract.target_lufs_tolerance_db,
        )
    )

    return MasteringReport(
        input_metrics=input_metrics,
        post_eq_metrics=post_eq_metrics,
        post_spatial_metrics=post_spatial_metrics,
        post_limiter_metrics=post_limiter_metrics,
        post_character_metrics=post_character_metrics,
        post_peak_catch_metrics=post_peak_catch_metrics,
        post_delivery_trim_metrics=post_delivery_trim_metrics,
        post_clamp_metrics=post_clamp_metrics,
        final_in_memory_metrics=final_in_memory_metrics,
        output_metrics=output_metrics,
        decoded_metrics=decoded_metrics,
        decoded_sample_rate=decoded_sample_rate,
        target_lufs=target_lufs,
        ceil_db=ceil_db,
        preset_name=preset_name,
        delivery_profile_name=delivery_profile_name,
        delivery_issues=delivery_issues,
        contract=contract,
        output_contract_assessment=output_contract_assessment,
        decoded_contract_assessment=decoded_contract_assessment,
        character_stage_decision=character_stage_decision,
        peak_catch_events=tuple(peak_catch_events),
        resolved_true_peak_target_dbfs=resolved_true_peak_target_dbfs,
        stereo_motion_activity=stereo_motion_activity,
        stereo_motion_correlation_guard=stereo_motion_correlation_guard,
        post_spatial_stereo_motion=post_spatial_stereo_motion,
        output_stereo_motion=output_stereo_motion,
        delivery_trim_attenuation_db=float(delivery_trim_attenuation_db),
        delivery_trim_input_true_peak_dbfs=delivery_trim_input_true_peak_dbfs,
        delivery_trim_target_dbfs=delivery_trim_target_dbfs,
        delivery_trim_output_true_peak_dbfs=delivery_trim_output_true_peak_dbfs,
        post_clamp_true_peak_dbfs=post_clamp_true_peak_dbfs,
        post_clamp_true_peak_delta_db=post_clamp_true_peak_delta_db,
        headroom_recovery_gain_db=float(headroom_recovery_gain_db),
        headroom_recovery_input_true_peak_dbfs=headroom_recovery_input_true_peak_dbfs,
        headroom_recovery_output_true_peak_dbfs=headroom_recovery_output_true_peak_dbfs,
        headroom_recovery_failure_reasons=tuple(
            headroom_recovery_failure_reasons
        ),
        headroom_recovery_mode=headroom_recovery_mode,
        headroom_recovery_integrated_gap_db=headroom_recovery_integrated_gap_db,
        headroom_recovery_transient_density=headroom_recovery_transient_density,
        headroom_recovery_closed_margin_db=headroom_recovery_closed_margin_db,
        headroom_recovery_unused_margin_db=headroom_recovery_unused_margin_db,
        integrated_lufs_delta=output_metrics.integrated_lufs
        - input_metrics.integrated_lufs,
        sample_peak_delta_db=output_metrics.sample_peak_dbfs
        - input_metrics.sample_peak_dbfs,
        true_peak_delta_db=output_metrics.true_peak_dbfs
        - input_metrics.true_peak_dbfs,
        crest_factor_delta_db=output_metrics.crest_factor_db
        - input_metrics.crest_factor_db,
        target_lufs_error_db=target_lufs_error_db,
        true_peak_margin_db=true_peak_margin_db,
        decoded_true_peak_margin_db=decoded_true_peak_margin_db,
    )


def write_mastering_report(
    report: MasteringReport,
    destination_path: str,
    *,
    indent: int = 2,
) -> str:
    destination = Path(destination_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report.to_dict(), indent=max(int(indent), 0), sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return str(destination)


__all__ = [
    "MasteringReport",
    "generate_mastering_report",
    "write_mastering_report",
]
