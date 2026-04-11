from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()
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
    stem_mastered_input: bool
    stem_glue_reverb_amount: float | None
    stem_drum_edge_amount: float | None
    stem_vocal_pullback_db: float | None
    integrated_lufs_delta: float
    sample_peak_delta_db: float
    true_peak_delta_db: float
    crest_factor_delta_db: float
    target_lufs_error_db: float | None
    true_peak_margin_db: float | None
    decoded_true_peak_margin_db: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_musician_dict(self) -> dict[str, Any]:
        return _build_musician_report_payload(self)

    def to_musician_markdown(self) -> str:
        return _render_musician_report_markdown(self)


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        resolved = float(value)
    except Exception:
        return None
    if not np.isfinite(resolved):
        return None
    return resolved


def _format_value(
    value: object,
    unit: str = "",
    *,
    decimals: int = 2,
) -> str:
    resolved = _safe_float(value)
    if resolved is None:
        return "n/a"
    return f"{resolved:.{max(int(decimals), 0)}f}{unit}"


def _metric_value(metrics: object | None, attribute_name: str) -> float | None:
    if metrics is None:
        return None
    return _safe_float(getattr(metrics, attribute_name, None))


def _metric_block(metrics: object | None) -> dict[str, float | None]:
    return {
        "loudness_lufs": _metric_value(metrics, "integrated_lufs"),
        "true_peak_dbfs": _metric_value(metrics, "true_peak_dbfs"),
        "crest_factor_db": _metric_value(metrics, "crest_factor_db"),
        "stereo_width_ratio": _metric_value(metrics, "stereo_width_ratio"),
        "low_end_mono_ratio": _metric_value(metrics, "low_end_mono_ratio"),
    }


def _final_master_assessment(
    report: MasteringReport,
) -> MasteringContractAssessment | None:
    if report.contract is None:
        return None
    return assess_mastering_contract(
        report.final_in_memory_metrics, report.contract
    )


def _export_alignment_note(report: MasteringReport) -> str | None:
    alignment_mode = (
        str(report.export_peak_alignment_mode or "").strip().lower()
    )
    export_gain_applied_db = _safe_float(report.export_gain_applied_db)
    peak_alignment_target_dbfs = _safe_float(
        report.export_peak_alignment_target_dbfs
    )
    if alignment_mode != "align_to_ceil" or export_gain_applied_db is None:
        return None
    if abs(export_gain_applied_db) <= 1e-6:
        return None
    target_text = (
        f"{peak_alignment_target_dbfs:.2f} dBFS ceiling"
        if peak_alignment_target_dbfs is not None
        else "delivery ceiling"
    )
    if export_gain_applied_db > 0.0:
        return f"Export was raised by {export_gain_applied_db:.2f} dB to use the available {target_text}."
    return f"Export was trimmed by {abs(export_gain_applied_db):.2f} dB to stay under the {target_text}."


def _processing_actions(report: MasteringReport) -> tuple[str, ...]:
    actions: list[str] = []
    alignment_note = _export_alignment_note(report)
    if alignment_note is not None:
        actions.append(alignment_note)
    if report.character_stage_decision is not None and bool(
        getattr(report.character_stage_decision, "applied", False)
    ):
        if bool(getattr(report.character_stage_decision, "reverted", False)):
            actions.append("Character finish was auditioned and rolled back.")
        else:
            actions.append("Character finish was kept in the final master.")
    if report.peak_catch_events:
        actions.append(
            f"Peak catch engaged {len(report.peak_catch_events)} time(s) after the character stage."
        )
    delivery_trim_attenuation_db = _safe_float(
        report.delivery_trim_attenuation_db
    )
    if (
        delivery_trim_attenuation_db is not None
        and delivery_trim_attenuation_db > 0.05
    ):
        actions.append(
            f"Delivery trim removed {delivery_trim_attenuation_db:.2f} dB before the final clamp."
        )
    headroom_recovery_gain_db = _safe_float(report.headroom_recovery_gain_db)
    if (
        headroom_recovery_gain_db is not None
        and headroom_recovery_gain_db > 0.05
    ):
        actions.append(
            f"Headroom recovery added {headroom_recovery_gain_db:.2f} dB after the clamp."
        )
    elif str(report.headroom_recovery_mode or "").strip().lower() == "disabled":
        actions.append("Headroom recovery was not needed.")
    return tuple(actions)


def _attention_lines(report: MasteringReport) -> tuple[str, ...]:
    lines: list[str] = []
    tolerance_db = (
        0.0
        if report.contract is None
        else float(getattr(report.contract, "target_lufs_tolerance_db", 0.0))
    )
    master_assessment = _final_master_assessment(report)
    decoded_assessment = report.decoded_contract_assessment
    if master_assessment is not None and abs(
        master_assessment.target_lufs_error_db
    ) > max(tolerance_db + 0.25, 0.35):
        direction = (
            "hotter"
            if master_assessment.target_lufs_error_db > 0.0
            else "quieter"
        )
        lines.append(
            f"Final master lands {abs(master_assessment.target_lufs_error_db):.2f} dB {direction} than the target."
        )
    if (
        master_assessment is not None
        and master_assessment.short_term_over_db > 0.35
    ):
        lines.append(
            f"Short-term loudness still rises {master_assessment.short_term_over_db:.2f} dB above the profile window."
        )
    if (
        master_assessment is not None
        and master_assessment.momentary_over_db > 0.35
    ):
        lines.append(
            f"Momentary loudness still rises {master_assessment.momentary_over_db:.2f} dB above the profile window."
        )
    if decoded_assessment is not None and abs(
        decoded_assessment.target_lufs_error_db
    ) > max(tolerance_db + 0.25, 0.35):
        direction = (
            "hotter"
            if decoded_assessment.target_lufs_error_db > 0.0
            else "quieter"
        )
        lines.append(
            f"Decoded playback lands {abs(decoded_assessment.target_lufs_error_db):.2f} dB {direction} than the target."
        )
    if (
        decoded_assessment is not None
        and decoded_assessment.true_peak_over_db > 0.05
    ):
        lines.append(
            f"Decoded playback exceeds the delivery true-peak limit by {decoded_assessment.true_peak_over_db:.2f} dB."
        )
    return tuple(dict.fromkeys(lines))


def _verdict_lines(report: MasteringReport) -> tuple[str, str]:
    master_assessment = _final_master_assessment(report)
    decoded_assessment = report.decoded_contract_assessment
    attention_lines = _attention_lines(report)
    if master_assessment is not None and master_assessment.passed:
        if decoded_assessment is not None and not decoded_assessment.passed:
            return (
                "Final master is on target.",
                "Decoded playback still moves outside the chosen delivery profile.",
            )
        return (
            "Final master is on target.",
            "Delivery translation looks stable for the chosen profile.",
        )
    if attention_lines:
        return (
            "Final master needs another pass.",
            attention_lines[0],
        )
    return (
        "Mastering finished.",
        "Review the final master and delivery numbers below.",
    )


def _build_musician_report_payload(report: MasteringReport) -> dict[str, Any]:
    headline, detail = _verdict_lines(report)
    return {
        "report_title": "Mastering Report",
        "preset_name": report.preset_name,
        "delivery_profile_name": report.delivery_profile_name,
        "verdict": {
            "headline": headline,
            "detail": detail,
        },
        "final_master": _metric_block(report.final_in_memory_metrics),
        "delivery_file": {
            **_metric_block(report.output_metrics),
            "export_gain_applied_db": _safe_float(
                report.export_gain_applied_db
            ),
            "peak_alignment_mode": report.export_peak_alignment_mode,
            "peak_alignment_target_dbfs": _safe_float(
                report.export_peak_alignment_target_dbfs
            ),
        },
        "decoded_playback": (
            None
            if report.decoded_metrics is None
            else {
                **_metric_block(report.decoded_metrics),
                "sample_rate": report.decoded_sample_rate,
            }
        ),
        "stem_final_pass": _stem_final_pass_block(report),
        "actions_taken": list(_processing_actions(report)),
        "attention": list(_attention_lines(report)),
    }


def _render_musician_report_markdown(report: MasteringReport) -> str:
    headline, detail = _verdict_lines(report)
    final_master = report.final_in_memory_metrics
    output_metrics = report.output_metrics
    decoded_metrics = report.decoded_metrics
    sections = [
        "# Mastering Report",
        "",
        "## Verdict",
        headline,
        detail,
        "",
        "## Final Master",
        f"- Preset: {report.preset_name or 'n/a'}",
        f"- Delivery profile: {report.delivery_profile_name or 'n/a'}",
        f"- Target loudness: {_format_value(report.target_lufs, ' LUFS')}",
        f"- Final loudness: {_format_value(_metric_value(final_master, 'integrated_lufs'), ' LUFS')}",
        f"- Final true peak: {_format_value(_metric_value(final_master, 'true_peak_dbfs'), ' dBFS')}",
        f"- Crest factor: {_format_value(_metric_value(final_master, 'crest_factor_db'), ' dB')}",
        f"- Stereo width: {_format_value(_metric_value(final_master, 'stereo_width_ratio'))}",
        f"- Low-end mono focus: {_format_value(_metric_value(final_master, 'low_end_mono_ratio'))}",
        "",
        "## Delivered File",
        f"- Export loudness: {_format_value(_metric_value(output_metrics, 'integrated_lufs'), ' LUFS')}",
        f"- Export true peak: {_format_value(_metric_value(output_metrics, 'true_peak_dbfs'), ' dBFS')}",
        f"- Export gain: {_format_value(report.export_gain_applied_db, ' dB')}",
    ]
    alignment_note = _export_alignment_note(report)
    if alignment_note is not None:
        sections.append(f"- Ceiling alignment: {alignment_note}")
    if decoded_metrics is not None:
        sections.extend(
            [
                "",
                "## Decoded Playback",
                f"- Decoded loudness: {_format_value(_metric_value(decoded_metrics, 'integrated_lufs'), ' LUFS')}",
                f"- Decoded true peak: {_format_value(_metric_value(decoded_metrics, 'true_peak_dbfs'), ' dBFS')}",
                f"- Decoded sample rate: {report.decoded_sample_rate if report.decoded_sample_rate is not None else 'n/a'}",
            ]
        )
    stem_final_pass = _stem_final_pass_block(report)
    if stem_final_pass is not None:
        sections.extend(
            [
                "",
                "## Stem Final Pass",
                f"- Vocal/other glue reverb: {_format_value(stem_final_pass['glue_reverb_amount'])}x",
                f"- Drum edge amount: {_format_value(stem_final_pass['drum_edge_amount'])}x",
                f"- Extra vocal pullback: {_format_value(stem_final_pass['vocal_pullback_db'], ' dB')}",
                f"- Headroom recovery: {_format_value(stem_final_pass['headroom_recovery_gain_db'], ' dB')} ({report.headroom_recovery_mode or 'n/a'})",
                f"- Closed ceiling margin: {_format_value(stem_final_pass['headroom_recovery_closed_margin_db'], ' dB')}",
            ]
        )
    sections.extend(["", "## Processing Notes"])
    actions = _processing_actions(report)
    if actions:
        sections.extend(f"- {line}" for line in actions)
    else:
        sections.append("- No special follow-up moves were needed.")
    sections.extend(["", "## Attention"])
    attention = _attention_lines(report)
    if attention:
        sections.extend(f"- {line}" for line in attention)
    else:
        sections.append("- No major delivery warnings.")
    return "\n".join(sections) + "\n"


def _stem_final_pass_block(report: MasteringReport) -> dict[str, Any] | None:
    if not bool(getattr(report, "stem_mastered_input", False)):
        return None
    return {
        "glue_reverb_amount": _safe_float(report.stem_glue_reverb_amount),
        "drum_edge_amount": _safe_float(report.stem_drum_edge_amount),
        "vocal_pullback_db": _safe_float(report.stem_vocal_pullback_db),
        "headroom_recovery_gain_db": _safe_float(
            report.headroom_recovery_gain_db
        ),
        "headroom_recovery_mode": report.headroom_recovery_mode,
        "headroom_recovery_closed_margin_db": _safe_float(
            report.headroom_recovery_closed_margin_db
        ),
    }


def _serialize_report_payload(report: object) -> dict[str, Any]:
    to_musician_dict = getattr(report, "to_musician_dict", None)
    if callable(to_musician_dict):
        payload = to_musician_dict()
        if isinstance(payload, dict):
            return dict(payload)
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if isinstance(payload, dict):
            return dict(payload)
    return {"report": str(report)}


def _render_report_text(report: object) -> str:
    to_musician_markdown = getattr(report, "to_musician_markdown", None)
    if callable(to_musician_markdown):
        return str(to_musician_markdown())
    payload = _serialize_report_payload(report)
    lines = ["# Mastering Report", ""]
    for key, value in payload.items():
        lines.append(f"- {str(key).replace('_', ' ').title()}: {value}")
    return "\n".join(lines) + "\n"


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
    stem_mastered_input: bool = False,
    stem_glue_reverb_amount: float | None = None,
    stem_drum_edge_amount: float | None = None,
    stem_vocal_pullback_db: float | None = None,
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
        stem_mastered_input=bool(stem_mastered_input),
        stem_glue_reverb_amount=stem_glue_reverb_amount,
        stem_drum_edge_amount=stem_drum_edge_amount,
        stem_vocal_pullback_db=stem_vocal_pullback_db,
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
    if destination.suffix.lower() == ".json":
        destination.write_text(
            json.dumps(
                _serialize_report_payload(report),
                indent=max(int(indent), 0),
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        destination.write_text(
            _render_report_text(report),
            encoding="utf-8",
        )
    return str(destination)


__all__ = [
    "MasteringReport",
    "generate_mastering_report",
    "write_mastering_report",
]
