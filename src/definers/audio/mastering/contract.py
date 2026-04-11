from __future__ import annotations

from dataclasses import asdict, dataclass

from .loudness import MasteringLoudnessMetrics


@dataclass(frozen=True, slots=True)
class MasteringContract:
    name: str
    target_lufs: float
    target_lufs_tolerance_db: float
    max_short_term_lufs: float | None
    max_momentary_lufs: float | None
    max_true_peak_dbfs: float | None
    min_crest_factor_db: float | None
    max_crest_factor_db: float | None
    max_stereo_width_ratio: float | None
    min_low_end_mono_ratio: float | None
    low_end_mono_cutoff_hz: float

    def to_dict(self) -> dict[str, float | str | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MasteringContractAssessment:
    contract_name: str
    passed: bool
    issues: tuple[str, ...]
    target_lufs_error_db: float
    short_term_over_db: float
    momentary_over_db: float
    true_peak_over_db: float
    crest_factor_under_db: float
    crest_factor_over_db: float
    stereo_width_over_ratio: float
    low_end_mono_under_ratio: float

    def to_dict(self) -> dict[str, float | str | bool | tuple[str, ...]]:
        return asdict(self)


def resolve_mastering_contract(
    preset_name: str | None,
    *,
    target_lufs: float,
    ceil_db: float,
    target_lufs_tolerance_db: float = 0.5,
    max_short_term_lufs: float | None = None,
    max_momentary_lufs: float | None = None,
    min_crest_factor_db: float | None = 4.0,
    max_crest_factor_db: float | None = 12.0,
    max_stereo_width_ratio: float | None = 0.55,
    min_low_end_mono_ratio: float | None = 0.85,
    low_end_mono_cutoff_hz: float = 160.0,
) -> MasteringContract:
    resolved_short_term = (
        target_lufs + 1.5
        if max_short_term_lufs is None
        else float(max_short_term_lufs)
    )
    resolved_momentary = (
        resolved_short_term + 1.0
        if max_momentary_lufs is None
        else float(max_momentary_lufs)
    )

    return MasteringContract(
        name="default" if preset_name is None else str(preset_name),
        target_lufs=float(target_lufs),
        target_lufs_tolerance_db=max(float(target_lufs_tolerance_db), 0.0),
        max_short_term_lufs=resolved_short_term,
        max_momentary_lufs=resolved_momentary,
        max_true_peak_dbfs=float(ceil_db),
        min_crest_factor_db=(
            None if min_crest_factor_db is None else float(min_crest_factor_db)
        ),
        max_crest_factor_db=(
            None if max_crest_factor_db is None else float(max_crest_factor_db)
        ),
        max_stereo_width_ratio=(
            None
            if max_stereo_width_ratio is None
            else float(max_stereo_width_ratio)
        ),
        min_low_end_mono_ratio=(
            None
            if min_low_end_mono_ratio is None
            else float(min_low_end_mono_ratio)
        ),
        low_end_mono_cutoff_hz=max(float(low_end_mono_cutoff_hz), 20.0),
    )


def assess_mastering_contract(
    metrics: MasteringLoudnessMetrics,
    contract: MasteringContract,
    *,
    target_lufs_tolerance_db: float | None = None,
) -> MasteringContractAssessment:
    issues: list[str] = []
    tolerance_db = (
        contract.target_lufs_tolerance_db
        if target_lufs_tolerance_db is None
        else max(float(target_lufs_tolerance_db), 0.0)
    )

    target_lufs_error_db = float(metrics.integrated_lufs - contract.target_lufs)
    short_term_over_db = 0.0
    momentary_over_db = 0.0
    true_peak_over_db = 0.0
    crest_factor_under_db = 0.0
    crest_factor_over_db = 0.0
    stereo_width_over_ratio = 0.0
    low_end_mono_under_ratio = 0.0

    if abs(target_lufs_error_db) > tolerance_db:
        issues.append(
            "integrated loudness misses target "
            f"{contract.target_lufs:.2f} LUFS by {target_lufs_error_db:.2f} dB"
        )

    if contract.max_short_term_lufs is not None:
        short_term_over_db = max(
            float(metrics.max_short_term_lufs - contract.max_short_term_lufs),
            0.0,
        )
        if short_term_over_db > 1e-6:
            issues.append(
                "short-term loudness exceeds contract by "
                f"{short_term_over_db:.2f} dB"
            )

    if contract.max_momentary_lufs is not None:
        momentary_over_db = max(
            float(metrics.max_momentary_lufs - contract.max_momentary_lufs),
            0.0,
        )
        if momentary_over_db > 1e-6:
            issues.append(
                "momentary loudness exceeds contract by "
                f"{momentary_over_db:.2f} dB"
            )

    if contract.max_true_peak_dbfs is not None:
        true_peak_over_db = max(
            float(metrics.true_peak_dbfs - contract.max_true_peak_dbfs),
            0.0,
        )
        if true_peak_over_db > 1e-6:
            issues.append(
                f"true peak exceeds contract by {true_peak_over_db:.2f} dB"
            )

    if contract.min_crest_factor_db is not None:
        crest_factor_under_db = max(
            float(contract.min_crest_factor_db - metrics.crest_factor_db),
            0.0,
        )
        if crest_factor_under_db > 1e-6:
            issues.append(
                "crest factor is below contract by "
                f"{crest_factor_under_db:.2f} dB"
            )

    if contract.max_crest_factor_db is not None:
        crest_factor_over_db = max(
            float(metrics.crest_factor_db - contract.max_crest_factor_db),
            0.0,
        )
        if crest_factor_over_db > 1e-6:
            issues.append(
                "crest factor exceeds contract by "
                f"{crest_factor_over_db:.2f} dB"
            )

    if contract.max_stereo_width_ratio is not None:
        stereo_width_over_ratio = max(
            float(metrics.stereo_width_ratio - contract.max_stereo_width_ratio),
            0.0,
        )
        if stereo_width_over_ratio > 1e-6:
            issues.append(
                "stereo width exceeds contract by "
                f"{stereo_width_over_ratio:.3f}"
            )

    if contract.min_low_end_mono_ratio is not None:
        low_end_mono_under_ratio = max(
            float(contract.min_low_end_mono_ratio - metrics.low_end_mono_ratio),
            0.0,
        )
        if low_end_mono_under_ratio > 1e-6:
            issues.append(
                "low-end mono ratio is below contract by "
                f"{low_end_mono_under_ratio:.3f}"
            )

    return MasteringContractAssessment(
        contract_name=contract.name,
        passed=not issues,
        issues=tuple(issues),
        target_lufs_error_db=target_lufs_error_db,
        short_term_over_db=short_term_over_db,
        momentary_over_db=momentary_over_db,
        true_peak_over_db=true_peak_over_db,
        crest_factor_under_db=crest_factor_under_db,
        crest_factor_over_db=crest_factor_over_db,
        stereo_width_over_ratio=stereo_width_over_ratio,
        low_end_mono_under_ratio=low_end_mono_under_ratio,
    )


__all__ = [
    "MasteringContract",
    "MasteringContractAssessment",
    "assess_mastering_contract",
    "resolve_mastering_contract",
]
