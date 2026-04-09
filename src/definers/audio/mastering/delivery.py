from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .contract import MasteringContract
from .metrics import MasteringReport, generate_mastering_report

_LOSSY_EXTENSIONS = {"aac", "m4a", "mp3", "ogg", "opus", "wma"}


@dataclass(frozen=True, slots=True)
class DeliveryProfile:
    name: str
    bitrate: int
    decoded_true_peak_dbfs: float | None
    decoded_lufs_tolerance_db: float
    is_lossy: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DeliveryVerificationResult:
    path: str
    profile: DeliveryProfile
    report: MasteringReport
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "profile": self.profile.to_dict(),
            "report": self.report.to_dict(),
            "issues": self.issues,
        }


def _normalized_extension(value: str) -> str:
    normalized = str(value).strip().lower()
    if "." in normalized:
        normalized = Path(normalized).suffix.lower().lstrip(".")
    return normalized.lstrip(".")


def resolve_delivery_profile(
    delivery_profile_name: str | None,
    output_path_or_ext: str,
    *,
    bitrate: int = 320,
    decoded_true_peak_dbfs: float | None = None,
    decoded_lufs_tolerance_db: float | None = None,
) -> DeliveryProfile:
    output_ext = _normalized_extension(output_path_or_ext)
    is_lossy = output_ext in _LOSSY_EXTENSIONS

    normalized_profile = (
        "lossy"
        if is_lossy
        else (
            str(delivery_profile_name).strip().lower()
            if delivery_profile_name is not None
            else "lossless"
        )
    )
    preset_map = {
        "analysis": {
            "bitrate": bitrate,
            "decoded_true_peak_dbfs": None,
            "decoded_lufs_tolerance_db": 1.25,
        },
        "lossless": {
            "bitrate": bitrate,
            "decoded_true_peak_dbfs": -0.1,
            "decoded_lufs_tolerance_db": 0.35,
        },
        "lossy": {
            "bitrate": bitrate,
            "decoded_true_peak_dbfs": -0.6 if is_lossy else -0.25,
            "decoded_lufs_tolerance_db": 0.75,
        },
        "streaming_lossy": {
            "bitrate": 256 if bitrate == 320 else bitrate,
            "decoded_true_peak_dbfs": -1.0,
            "decoded_lufs_tolerance_db": 1.0,
        },
    }
    preset = preset_map.get(normalized_profile)
    if preset is None:
        raise ValueError(f"Unknown delivery profile: {delivery_profile_name}")

    return DeliveryProfile(
        name=normalized_profile,
        bitrate=int(max(preset["bitrate"], 32)),
        decoded_true_peak_dbfs=(
            preset["decoded_true_peak_dbfs"]
            if decoded_true_peak_dbfs is None
            else float(decoded_true_peak_dbfs)
        ),
        decoded_lufs_tolerance_db=float(
            preset["decoded_lufs_tolerance_db"]
            if decoded_lufs_tolerance_db is None
            else decoded_lufs_tolerance_db
        ),
        is_lossy=is_lossy,
    )


def verify_delivery_export(
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
    output_path: str,
    profile: DeliveryProfile,
    read_audio_fn,
    target_lufs: float | None = None,
    ceil_db: float | None = None,
    preset_name: str | None = None,
    contract: MasteringContract | None = None,
    character_stage_decision=None,
    peak_catch_events: tuple[object, ...] = (),
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
) -> DeliveryVerificationResult:
    decoded_signal = None
    decoded_sample_rate = None
    issues: list[str] = []

    try:
        decoded_sample_rate, decoded_signal = read_audio_fn(output_path)
    except Exception as error:
        issues.append(f"decode failed: {error}")

    decoded_contract = None
    if contract is not None:
        decoded_contract = replace(
            contract,
            name=f"{contract.name}:{profile.name}:decoded",
            target_lufs_tolerance_db=max(
                contract.target_lufs_tolerance_db,
                profile.decoded_lufs_tolerance_db,
            ),
            max_true_peak_dbfs=(
                profile.decoded_true_peak_dbfs
                if profile.decoded_true_peak_dbfs is not None
                else contract.max_true_peak_dbfs
            ),
        )

    report = generate_mastering_report(
        input_signal,
        output_signal,
        sample_rate,
        final_in_memory_signal=final_in_memory_signal,
        post_eq_signal=post_eq_signal,
        post_spatial_signal=post_spatial_signal,
        post_limiter_signal=post_limiter_signal,
        post_character_signal=post_character_signal,
        post_peak_catch_signal=post_peak_catch_signal,
        post_delivery_trim_signal=post_delivery_trim_signal,
        post_clamp_signal=post_clamp_signal,
        decoded_signal=decoded_signal,
        decoded_sample_rate=decoded_sample_rate,
        target_lufs=target_lufs,
        ceil_db=ceil_db,
        preset_name=preset_name,
        delivery_profile_name=profile.name,
        contract=contract,
        decoded_contract=decoded_contract,
        character_stage_decision=character_stage_decision,
        peak_catch_events=peak_catch_events,
        resolved_true_peak_target_dbfs=resolved_true_peak_target_dbfs,
        stereo_motion_activity=stereo_motion_activity,
        stereo_motion_correlation_guard=stereo_motion_correlation_guard,
        export_gain_applied_db=export_gain_applied_db,
        export_peak_alignment_mode=export_peak_alignment_mode,
        export_peak_alignment_target_dbfs=export_peak_alignment_target_dbfs,
        delivery_trim_attenuation_db=delivery_trim_attenuation_db,
        delivery_trim_input_true_peak_dbfs=delivery_trim_input_true_peak_dbfs,
        delivery_trim_target_dbfs=delivery_trim_target_dbfs,
        delivery_trim_output_true_peak_dbfs=delivery_trim_output_true_peak_dbfs,
        post_clamp_true_peak_dbfs=post_clamp_true_peak_dbfs,
        post_clamp_true_peak_delta_db=post_clamp_true_peak_delta_db,
        headroom_recovery_gain_db=headroom_recovery_gain_db,
        headroom_recovery_input_true_peak_dbfs=headroom_recovery_input_true_peak_dbfs,
        headroom_recovery_output_true_peak_dbfs=headroom_recovery_output_true_peak_dbfs,
        headroom_recovery_failure_reasons=headroom_recovery_failure_reasons,
        headroom_recovery_mode=headroom_recovery_mode,
        headroom_recovery_integrated_gap_db=headroom_recovery_integrated_gap_db,
        headroom_recovery_transient_density=headroom_recovery_transient_density,
        headroom_recovery_closed_margin_db=headroom_recovery_closed_margin_db,
        headroom_recovery_unused_margin_db=headroom_recovery_unused_margin_db,
        true_peak_oversample_factor=true_peak_oversample_factor,
    )

    if (
        report.decoded_metrics is not None
        and profile.decoded_true_peak_dbfs is not None
    ):
        decoded_peak_over_db = (
            report.decoded_metrics.true_peak_dbfs
            - profile.decoded_true_peak_dbfs
        )
        if decoded_peak_over_db > 1e-6:
            issues.append(
                "decoded true peak exceeds "
                f"{profile.decoded_true_peak_dbfs:.2f} dBFS by {decoded_peak_over_db:.2f} dB"
            )

        decoded_lufs_delta = abs(
            report.decoded_metrics.integrated_lufs
            - report.output_metrics.integrated_lufs
        )
        if decoded_lufs_delta > profile.decoded_lufs_tolerance_db:
            issues.append(
                "decoded loudness drift exceeds "
                f"{profile.decoded_lufs_tolerance_db:.2f} dB by {decoded_lufs_delta:.2f} dB"
            )

    if report.output_contract_assessment is not None:
        issues.extend(
            f"output contract: {issue}"
            for issue in report.output_contract_assessment.issues
            if f"output contract: {issue}" not in issues
        )

    if report.decoded_contract_assessment is not None:
        issues.extend(
            f"decoded contract: {issue}"
            for issue in report.decoded_contract_assessment.issues
            if f"decoded contract: {issue}" not in issues
        )

    report = replace(
        report,
        delivery_profile_name=profile.name,
        delivery_issues=tuple(issues),
    )

    return DeliveryVerificationResult(
        path=output_path,
        profile=profile,
        report=report,
        issues=tuple(issues),
    )


def _apply_linear_export_ceiling(
    signal: np.ndarray,
    ceil_db: float | None,
    *,
    allow_positive_gain: bool = True,
) -> np.ndarray:
    working_signal = np.nan_to_num(
        np.asarray(signal, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if ceil_db is None or not np.isfinite(ceil_db):
        return working_signal

    limit_linear = float(10.0 ** (float(ceil_db) / 20.0))
    peak = float(np.max(np.abs(working_signal)))
    if peak <= 0.0:
        return working_signal
    if not allow_positive_gain and peak < limit_linear:
        return working_signal
    return working_signal * (limit_linear / peak)


def _measure_export_gain_applied_db(
    source_signal: np.ndarray,
    output_signal: np.ndarray,
) -> float:
    source_peak = float(
        np.max(np.abs(np.asarray(source_signal, dtype=np.float32)))
    )
    output_peak = float(
        np.max(np.abs(np.asarray(output_signal, dtype=np.float32)))
    )
    if source_peak <= 0.0 or output_peak <= 0.0:
        return 0.0
    return float(20.0 * np.log10(output_peak / source_peak))


def _resolve_export_peak_alignment_mode(
    signal: np.ndarray,
    ceil_db: float | None,
) -> str | None:
    if ceil_db is None or not np.isfinite(ceil_db):
        return None
    peak = float(np.max(np.abs(np.asarray(signal, dtype=np.float32))))
    if peak <= 0.0:
        return None
    return "align_to_ceil"


def _decoded_peak_excess_db(
    verification_result: DeliveryVerificationResult,
) -> float | None:
    decoded_metrics = getattr(
        verification_result.report, "decoded_metrics", None
    )
    decoded_limit_dbfs = verification_result.profile.decoded_true_peak_dbfs
    if decoded_metrics is None or decoded_limit_dbfs is None:
        return None

    decoded_true_peak_dbfs = getattr(decoded_metrics, "true_peak_dbfs", None)
    if decoded_true_peak_dbfs is None or not np.isfinite(
        decoded_true_peak_dbfs
    ):
        return None

    return float(decoded_true_peak_dbfs - float(decoded_limit_dbfs))


def save_verified_audio(
    destination_path: str,
    audio_signal: np.ndarray,
    sample_rate: int,
    *,
    input_signal: np.ndarray,
    post_eq_signal: np.ndarray | None = None,
    post_spatial_signal: np.ndarray | None = None,
    post_limiter_signal: np.ndarray | None = None,
    post_character_signal: np.ndarray | None = None,
    post_peak_catch_signal: np.ndarray | None = None,
    post_delivery_trim_signal: np.ndarray | None = None,
    post_clamp_signal: np.ndarray | None = None,
    save_audio_fn,
    read_audio_fn,
    target_lufs: float | None = None,
    ceil_db: float | None = None,
    preset_name: str | None = None,
    contract: MasteringContract | None = None,
    delivery_profile_name: str | None = None,
    character_stage_decision=None,
    peak_catch_events: tuple[object, ...] = (),
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
    decoded_true_peak_dbfs: float | None = None,
    decoded_lufs_tolerance_db: float | None = None,
    true_peak_oversample_factor: int = 4,
    bit_depth: int = 32,
    bitrate: int = 320,
    compression_level: int = 9,
) -> tuple[str, np.ndarray, DeliveryVerificationResult]:
    profile = resolve_delivery_profile(
        delivery_profile_name,
        destination_path,
        bitrate=bitrate,
        decoded_true_peak_dbfs=decoded_true_peak_dbfs,
        decoded_lufs_tolerance_db=decoded_lufs_tolerance_db,
    )

    working_signal = np.nan_to_num(
        np.asarray(audio_signal, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    working_signal = _apply_linear_export_ceiling(working_signal, ceil_db)
    export_peak_alignment_mode = _resolve_export_peak_alignment_mode(
        audio_signal,
        ceil_db,
    )
    final_path = destination_path
    verification_result: DeliveryVerificationResult | None = None

    max_attempts = 3 if profile.is_lossy else 1
    for _attempt_index in range(max_attempts):
        final_path = save_audio_fn(
            destination_path=destination_path,
            audio_signal=working_signal,
            sample_rate=sample_rate,
            bit_depth=bit_depth,
            bitrate=profile.bitrate,
            compression_level=compression_level,
        )

        verification_result = verify_delivery_export(
            input_signal,
            working_signal,
            sample_rate,
            final_in_memory_signal=audio_signal,
            post_eq_signal=post_eq_signal,
            post_spatial_signal=post_spatial_signal,
            post_limiter_signal=post_limiter_signal,
            post_character_signal=post_character_signal,
            post_peak_catch_signal=post_peak_catch_signal,
            post_delivery_trim_signal=post_delivery_trim_signal,
            post_clamp_signal=post_clamp_signal,
            output_path=final_path,
            profile=profile,
            read_audio_fn=read_audio_fn,
            target_lufs=target_lufs,
            ceil_db=ceil_db,
            preset_name=preset_name,
            contract=contract,
            character_stage_decision=character_stage_decision,
            peak_catch_events=peak_catch_events,
            resolved_true_peak_target_dbfs=resolved_true_peak_target_dbfs,
            stereo_motion_activity=stereo_motion_activity,
            stereo_motion_correlation_guard=stereo_motion_correlation_guard,
            export_gain_applied_db=_measure_export_gain_applied_db(
                audio_signal,
                working_signal,
            ),
            export_peak_alignment_mode=export_peak_alignment_mode,
            export_peak_alignment_target_dbfs=(
                None
                if ceil_db is None or not np.isfinite(ceil_db)
                else float(ceil_db)
            ),
            delivery_trim_attenuation_db=delivery_trim_attenuation_db,
            delivery_trim_input_true_peak_dbfs=delivery_trim_input_true_peak_dbfs,
            delivery_trim_target_dbfs=delivery_trim_target_dbfs,
            delivery_trim_output_true_peak_dbfs=delivery_trim_output_true_peak_dbfs,
            post_clamp_true_peak_dbfs=post_clamp_true_peak_dbfs,
            post_clamp_true_peak_delta_db=post_clamp_true_peak_delta_db,
            headroom_recovery_gain_db=headroom_recovery_gain_db,
            headroom_recovery_input_true_peak_dbfs=headroom_recovery_input_true_peak_dbfs,
            headroom_recovery_output_true_peak_dbfs=headroom_recovery_output_true_peak_dbfs,
            headroom_recovery_failure_reasons=headroom_recovery_failure_reasons,
            headroom_recovery_mode=headroom_recovery_mode,
            headroom_recovery_integrated_gap_db=headroom_recovery_integrated_gap_db,
            headroom_recovery_transient_density=headroom_recovery_transient_density,
            headroom_recovery_closed_margin_db=headroom_recovery_closed_margin_db,
            headroom_recovery_unused_margin_db=headroom_recovery_unused_margin_db,
            true_peak_oversample_factor=true_peak_oversample_factor,
        )

        decoded_peak_excess_db = _decoded_peak_excess_db(verification_result)
        if decoded_peak_excess_db is None or decoded_peak_excess_db <= 1e-6:
            break

        attenuation_db = float(decoded_peak_excess_db + 0.1)
        attenuation_linear = float(10.0 ** (-attenuation_db / 20.0))
        attenuated_signal = working_signal * attenuation_linear
        attenuated_signal = _apply_linear_export_ceiling(
            attenuated_signal,
            ceil_db,
            allow_positive_gain=True,
        )
        if np.allclose(attenuated_signal, working_signal, atol=1e-8, rtol=0.0):
            break
        working_signal = attenuated_signal

    if verification_result is None:
        raise RuntimeError("Delivery verification did not run")

    return final_path, working_signal, verification_result


__all__ = [
    "DeliveryProfile",
    "DeliveryVerificationResult",
    "resolve_delivery_profile",
    "save_verified_audio",
    "verify_delivery_export",
]
