from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .mastering_contract import MasteringContract
from .mastering_finalization import CharacterStageDecision, PeakCatchEvent
from .utils import apply_lufs


def process(
    self,
    y: np.ndarray,
    sr: int | None = None,
    *,
    resample_fn: Callable[..., np.ndarray],
    stereo_fn: Callable[[np.ndarray], np.ndarray],
    freq_cut_fn: Callable[..., np.ndarray],
    apply_exciter_fn: Callable[..., np.ndarray],
    get_lufs_fn: Callable[..., float],
    measure_mastering_loudness_fn: Callable[..., object],
    log_fn: Callable[..., object],
) -> tuple[int, np.ndarray]:
    stage_signals: dict[str, np.ndarray] = {}
    if sr is not None:
        self.sr = sr

    if self.sr != self.resampling_target:
        y = resample_fn(y, self.sr, self.resampling_target)

    y = stereo_fn(y)
    y = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    log_fn("Mastering", "Applying filtering...")
    y = freq_cut_fn(
        y,
        self.resampling_target,
        low_cut=self.filter_low_cut,
        high_cut=self.filter_high_cut,
    )

    log_fn("Mastering", "Applying equalizer...")
    y = self.apply_eq(y)
    stage_signals["post_eq"] = np.array(y, dtype=np.float32, copy=True)

    log_fn("Mastering", "Applying multiband compression...")
    self.update_bands(self.spectral_balance_profile.band_intensity)
    y = self.multiband_compress(y)

    log_fn("Mastering", "Applying exciter...")
    y = apply_exciter_fn(
        y,
        self.resampling_target,
        self.exciter_cutoff_hz,
        self.exciter_mix,
        self.exciter_max_drive,
        self.exciter_high_frequency_cutoff_hz,
    )
    y = stereo_fn(y)

    log_fn("Mastering", "Applying filtering...")
    y = freq_cut_fn(
        y,
        self.resampling_target,
        low_cut=self.filter_low_cut,
        high_cut=self.filter_high_cut,
    )

    log_fn("Mastering", "Applying stereo enhancement...")
    y = self.apply_spatial_enhancement(y)
    stage_signals["post_spatial"] = np.array(y, dtype=np.float32, copy=True)

    log_fn("Mastering", "Applying low-end mono tightening...")
    y = self.apply_low_end_mono_tightening(y)

    log_fn("Mastering", "Preparing finalization...")
    current_lufs = get_lufs_fn(y, self.resampling_target)
    dynamic_drive_db = self.compute_dynamic_drive(current_lufs)
    primary_soft_clip_ratio = self.compute_primary_soft_clip_ratio(
        dynamic_drive_db
    )

    log_fn("Mastering", "Applying pre-limiter saturation...")
    y = self.apply_pre_limiter_saturation(y, dynamic_drive_db=dynamic_drive_db)

    log_fn("Mastering", "Applying limiter...")

    y = self.apply_limiter(
        y,
        drive_db=dynamic_drive_db,
        ceil_db=self.ceil_db,
        os_factor=self.limiter_oversample_factor,
        soft_clip_ratio=primary_soft_clip_ratio,
    )

    contract: MasteringContract = self.resolve_mastering_contract()
    self.last_mastering_contract = contract
    finalization_actions: list[object] = []
    metrics = measure_mastering_loudness_fn(
        y,
        self.resampling_target,
        true_peak_oversample_factor=self.true_peak_oversample_factor,
        low_end_mono_cutoff_hz=contract.low_end_mono_cutoff_hz,
    )
    for follow_up_index in range(self.max_follow_up_passes):
        action = self.plan_follow_up_action(metrics, contract)
        if not action.should_apply:
            break

        finalization_actions.append(action)
        if action.stereo_width_scale < 0.999:
            y = self.apply_stereo_width_restraint(
                y,
                stereo_width_scale=action.stereo_width_scale,
            )

        follow_up_soft_clip_ratio = float(
            np.clip(
                action.soft_clip_ratio
                + self.follow_up_soft_clip_ratio_step * follow_up_index,
                0.0,
                0.7,
            )
        )

        should_relimit = bool(
            float(action.gain_db) > 1e-6
            or follow_up_soft_clip_ratio
            > float(self.limiter_soft_clip_ratio) + 1e-6
        )
        if should_relimit:
            y = self.apply_limiter(
                y,
                drive_db=action.gain_db,
                ceil_db=self.ceil_db,
                os_factor=self.limiter_oversample_factor,
                lookahead_ms=3.0,
                soft_clip_ratio=follow_up_soft_clip_ratio,
                window_ms=6.0 + follow_up_index,
            )
        metrics = measure_mastering_loudness_fn(
            y,
            self.resampling_target,
            true_peak_oversample_factor=self.true_peak_oversample_factor,
            low_end_mono_cutoff_hz=contract.low_end_mono_cutoff_hz,
        )

    stage_signals["post_limiter"] = np.array(y, dtype=np.float32, copy=True)

    final_ceiling_db = self.resolve_final_true_peak_target()
    self.last_resolved_final_true_peak_target_dbfs = float(final_ceiling_db)

    log_fn("Mastering", "Applying micro-dynamics finish...")
    character_source = np.array(y, dtype=np.float32, copy=True)
    character_reasons: list[str] = []
    character_applied = (
        float(getattr(self, "micro_dynamics_strength", 0.0)) > 0.0
    )
    if float(getattr(self, "micro_dynamics_strength", 0.0)) > 0.0:
        character_candidate = self.apply_micro_dynamics_finish(y)
        character_metrics = measure_mastering_loudness_fn(
            character_candidate,
            self.resampling_target,
            true_peak_oversample_factor=self.true_peak_oversample_factor,
            low_end_mono_cutoff_hz=contract.low_end_mono_cutoff_hz,
        )
        base_short_term = float(
            getattr(
                metrics,
                "max_short_term_lufs",
                contract.max_short_term_lufs or -120.0,
            )
        )
        base_momentary = float(
            getattr(
                metrics,
                "max_momentary_lufs",
                contract.max_momentary_lufs or -120.0,
            )
        )
        character_short_term = float(
            getattr(character_metrics, "max_short_term_lufs", base_short_term)
        )
        character_momentary = float(
            getattr(character_metrics, "max_momentary_lufs", base_momentary)
        )
        base_true_peak_dbfs = float(
            getattr(metrics, "true_peak_dbfs", final_ceiling_db)
        )
        character_true_peak_dbfs = float(
            getattr(character_metrics, "true_peak_dbfs", base_true_peak_dbfs)
        )
        base_integrated_lufs = float(
            getattr(metrics, "integrated_lufs", self.target_lufs)
        )
        character_integrated_lufs = float(
            getattr(character_metrics, "integrated_lufs", base_integrated_lufs)
        )
        short_term_over_db = max(
            0.0,
            0.0
            if contract.max_short_term_lufs is None
            else character_short_term - contract.max_short_term_lufs,
        )
        momentary_over_db = max(
            0.0,
            0.0
            if contract.max_momentary_lufs is None
            else character_momentary - contract.max_momentary_lufs,
        )
        true_peak_regression_db = float(
            character_true_peak_dbfs - base_true_peak_dbfs
        )
        integrated_gain_db = float(
            character_integrated_lufs - base_integrated_lufs
        )
        if short_term_over_db > 1e-6:
            character_reasons.append("short_term")
        if momentary_over_db > 1e-6:
            character_reasons.append("momentary")
        if (
            true_peak_regression_db > 0.25
            and integrated_gain_db < true_peak_regression_db
        ):
            character_reasons.append("true_peak_efficiency")
        should_revert_character = bool(
            short_term_over_db > 1e-6
            or momentary_over_db > 1e-6
            or (
                true_peak_regression_db > 0.25
                and integrated_gain_db < true_peak_regression_db
            )
        )
        if should_revert_character:
            y = character_source
            character_metrics = metrics
        else:
            y = character_candidate
    else:
        character_metrics = metrics
        should_revert_character = False
    self.last_character_stage_decision = CharacterStageDecision(
        applied=character_applied,
        reverted=bool(should_revert_character),
        reasons=tuple(character_reasons),
        input_integrated_lufs=float(
            getattr(metrics, "integrated_lufs", self.target_lufs)
        ),
        output_integrated_lufs=float(
            getattr(
                character_metrics,
                "integrated_lufs",
                getattr(metrics, "integrated_lufs", self.target_lufs),
            )
        ),
        input_true_peak_dbfs=float(
            getattr(metrics, "true_peak_dbfs", final_ceiling_db)
        ),
        output_true_peak_dbfs=float(
            getattr(
                character_metrics,
                "true_peak_dbfs",
                getattr(metrics, "true_peak_dbfs", final_ceiling_db),
            )
        ),
    )
    stage_signals["post_character"] = np.array(y, dtype=np.float32, copy=True)

    peak_catch_events: list[PeakCatchEvent] = []
    peak_catch_metrics = character_metrics
    for peak_catch_index in range(3):
        before_true_peak_dbfs = float(
            getattr(peak_catch_metrics, "true_peak_dbfs", final_ceiling_db)
        )
        peak_over_db = before_true_peak_dbfs - final_ceiling_db
        if peak_over_db <= 1e-6:
            break
        before_integrated_lufs = float(
            getattr(peak_catch_metrics, "integrated_lufs", self.target_lufs)
        )
        peak_catch_ceil_db = float(
            final_ceiling_db
            - np.clip(
                peak_over_db * 0.9 + peak_catch_index * 0.02,
                0.08,
                0.8,
            )
        )
        soft_clip_ratio = float(
            np.clip(
                max(self.limiter_soft_clip_ratio * 0.8, 0.16)
                + peak_over_db * 0.22
                + peak_catch_index * 0.05,
                0.16,
                0.6,
            )
        )
        log_fn("Mastering", "Applying final peak catch...")
        y = self.apply_limiter(
            y,
            drive_db=0.0,
            ceil_db=peak_catch_ceil_db,
            os_factor=self.limiter_oversample_factor,
            lookahead_ms=2.0,
            attack_ms=0.5,
            release_ms_min=8.0,
            release_ms_max=30.0,
            soft_clip_ratio=soft_clip_ratio,
            window_ms=2.5 + peak_catch_index,
        )
        peak_catch_metrics = measure_mastering_loudness_fn(
            y,
            self.resampling_target,
            true_peak_oversample_factor=self.true_peak_oversample_factor,
            low_end_mono_cutoff_hz=contract.low_end_mono_cutoff_hz,
        )
        peak_catch_events.append(
            PeakCatchEvent(
                attempt_index=peak_catch_index + 1,
                drive_db=0.0,
                ceil_db=peak_catch_ceil_db,
                soft_clip_ratio=soft_clip_ratio,
                peak_over_db=float(peak_over_db),
                before_integrated_lufs=before_integrated_lufs,
                after_integrated_lufs=float(
                    getattr(
                        peak_catch_metrics,
                        "integrated_lufs",
                        before_integrated_lufs,
                    )
                ),
                before_true_peak_dbfs=before_true_peak_dbfs,
                after_true_peak_dbfs=float(
                    getattr(
                        peak_catch_metrics,
                        "true_peak_dbfs",
                        before_true_peak_dbfs,
                    )
                ),
            )
        )
    stage_signals["post_peak_catch"] = np.array(y, dtype=np.float32, copy=True)
    self.last_peak_catch_events = tuple(peak_catch_events)

    log_fn("Mastering", "Applying delivery trim...")
    y = self.apply_delivery_trim(y)
    stage_signals["post_delivery_trim"] = np.array(
        y, dtype=np.float32, copy=True
    )

    log_fn("Mastering", "Applying hard clipping...")
    y = self.apply_safety_clamp(y, ceil_db=final_ceiling_db)
    stage_signals["post_clamp"] = np.array(y, dtype=np.float32, copy=True)
    self.last_post_clamp_metrics = measure_mastering_loudness_fn(
        y,
        self.resampling_target,
        true_peak_oversample_factor=self.true_peak_oversample_factor,
        low_end_mono_cutoff_hz=contract.low_end_mono_cutoff_hz,
    )

    log_fn("Mastering", "Recovering final headroom...")
    y = self.apply_final_headroom_recovery(y)
    self.last_post_clamp_true_peak_dbfs = getattr(
        self,
        "last_headroom_recovery_input_true_peak_dbfs",
        None,
    )
    delivery_trim_output_true_peak_dbfs = getattr(
        self,
        "last_delivery_trim_output_true_peak_dbfs",
        None,
    )
    if (
        self.last_post_clamp_true_peak_dbfs is not None
        and delivery_trim_output_true_peak_dbfs is not None
    ):
        self.last_post_clamp_true_peak_delta_db = float(
            self.last_post_clamp_true_peak_dbfs
            - delivery_trim_output_true_peak_dbfs
        )
    else:
        self.last_post_clamp_true_peak_delta_db = None

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    stage_signals["final_in_memory"] = np.array(y, dtype=np.float32, copy=True)
    self.last_stage_signals = stage_signals
    self.last_finalization_actions = tuple(finalization_actions)

    return self.resampling_target, y
