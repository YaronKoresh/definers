from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .contract import (
    MasteringContract,
    MasteringContractAssessment,
    assess_mastering_contract,
)
from .finalization import CharacterStageDecision, PeakCatchEvent
from .loudness import (
    MasteringLoudnessMetrics,
    measure_mastering_loudness,
)
from .reference import measure_stereo_motion


def _measure_metric_batch(
    jobs: dict[str, tuple[np.ndarray, int]],
    *,
    true_peak_oversample_factor: int,
    low_end_mono_cutoff_hz: float,
    signal_module: Any,
) -> dict[str, MasteringLoudnessMetrics]:
    if not jobs:
        return {}

    results_by_key: dict[tuple[int, int], MasteringLoudnessMetrics] = {}
    aliases: dict[str, tuple[int, int]] = {}
    unique_jobs: list[tuple[tuple[int, int], np.ndarray, int]] = []
    max_length = 0

    for name, (signal_value, sample_rate_value) in jobs.items():
        array = np.asarray(signal_value)
        key = (id(signal_value), int(sample_rate_value))
        aliases[name] = key
        if key in results_by_key or any(
            existing_key == key for existing_key, *_ in unique_jobs
        ):
            continue
        max_length = max(
            max_length,
            int(array.shape[-1]) if array.ndim > 0 else 0,
        )
        unique_jobs.append((key, signal_value, int(sample_rate_value)))

    use_parallel = len(unique_jobs) > 2 and max_length >= 32768
    if use_parallel:
        worker_count = min(len(unique_jobs), 4)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    measure_mastering_loudness,
                    signal_value,
                    sample_rate_value,
                    true_peak_oversample_factor=true_peak_oversample_factor,
                    low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
                    signal_module=signal_module,
                ): key
                for key, signal_value, sample_rate_value in unique_jobs
            }
            for future, key in future_map.items():
                results_by_key[key] = future.result()
    else:
        for key, signal_value, sample_rate_value in unique_jobs:
            results_by_key[key] = measure_mastering_loudness(
                signal_value,
                sample_rate_value,
                true_peak_oversample_factor=true_peak_oversample_factor,
                low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
                signal_module=signal_module,
            )

    return {name: results_by_key[key] for name, key in aliases.items()}


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
    export_gain_applied_db: float
    export_peak_alignment_mode: str | None
    export_peak_alignment_target_dbfs: float | None
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
    final_in_memory_signal: np.ndarray | None = None,
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
    export_gain_applied_db: float = 0.0,
    export_peak_alignment_mode: str | None = None,
    export_peak_alignment_target_dbfs: float | None = None,
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

    metric_jobs: dict[str, tuple[np.ndarray, int]] = {
        "input": (input_signal, sample_rate),
        "output": (output_signal, sample_rate),
    }
    if final_in_memory_signal is not None:
        metric_jobs["final_in_memory"] = (
            final_in_memory_signal,
            sample_rate,
        )
    if post_eq_signal is not None:
        metric_jobs["post_eq"] = (post_eq_signal, sample_rate)
    if post_spatial_signal is not None:
        metric_jobs["post_spatial"] = (post_spatial_signal, sample_rate)
    if post_limiter_signal is not None:
        metric_jobs["post_limiter"] = (post_limiter_signal, sample_rate)
    if post_character_signal is not None:
        metric_jobs["post_character"] = (post_character_signal, sample_rate)
    if post_peak_catch_signal is not None:
        metric_jobs["post_peak_catch"] = (post_peak_catch_signal, sample_rate)
    if post_delivery_trim_signal is not None:
        metric_jobs["post_delivery_trim"] = (
            post_delivery_trim_signal,
            sample_rate,
        )
    if post_clamp_signal is not None:
        metric_jobs["post_clamp"] = (post_clamp_signal, sample_rate)
    if decoded_signal is not None:
        metric_jobs["decoded"] = (
            decoded_signal,
            sample_rate if decoded_sample_rate is None else decoded_sample_rate,
        )

    measured_metrics = _measure_metric_batch(
        metric_jobs,
        true_peak_oversample_factor=true_peak_oversample_factor,
        low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
        signal_module=signal_module,
    )

    input_metrics = measured_metrics["input"]
    output_metrics = measured_metrics["output"]
    post_eq_metrics = measured_metrics.get("post_eq")
    post_spatial_metrics = measured_metrics.get("post_spatial")
    post_limiter_metrics = measured_metrics.get("post_limiter")
    post_character_metrics = measured_metrics.get("post_character")
    post_peak_catch_metrics = measured_metrics.get("post_peak_catch")
    post_delivery_trim_metrics = measured_metrics.get("post_delivery_trim")
    post_clamp_metrics = measured_metrics.get("post_clamp")
    final_in_memory_metrics = measured_metrics.get(
        "final_in_memory",
        output_metrics,
    )
    post_spatial_stereo_motion = (
        None
        if post_spatial_signal is None
        else measure_stereo_motion(post_spatial_signal, sample_rate)
    )
    output_stereo_motion = measure_stereo_motion(output_signal, sample_rate)
    decoded_metrics = measured_metrics.get("decoded")

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
        export_gain_applied_db=float(export_gain_applied_db),
        export_peak_alignment_mode=export_peak_alignment_mode,
        export_peak_alignment_target_dbfs=export_peak_alignment_target_dbfs,
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
