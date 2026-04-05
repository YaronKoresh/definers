from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .utils import generate_bands


@dataclass(frozen=True, slots=True)
class SpectralBalanceProfile:
    rescue_factor: float
    correction_strength: float
    max_boost_db: float
    max_cut_db: float
    band_intensity: float
    restoration_factor: float = 0.0
    air_restoration_factor: float = 0.0
    body_restoration_factor: float = 0.0
    closure_repair_factor: float = 0.0
    mud_cleanup_factor: float = 0.0
    harshness_restraint_factor: float = 0.0
    low_end_restraint_factor: float = 0.0
    legacy_tonal_rebalance_factor: float = 0.0
    closed_top_end_repair_factor: float = 0.0


def fit_frequency(self, frequency_hz: float) -> float:
    return float(np.clip(frequency_hz, self.low_cut, self.high_cut))


def build_target_curve(self, f_axis: np.ndarray) -> np.ndarray:
    freqs = np.clip(
        np.asarray(f_axis, dtype=np.float32),
        self.low_cut,
        self.high_cut,
    )

    bass_octaves = np.maximum(
        0.0,
        np.log2(self.bass_transition_hz / np.maximum(freqs, self.low_cut)),
    )
    target = bass_octaves * self._slope_db

    mid_span_octaves = max(
        float(np.log2(self.treble_transition_hz / self.bass_transition_hz)),
        1e-6,
    )
    mid_positions = np.clip(
        np.log2(
            np.maximum(freqs, self.bass_transition_hz) / self.bass_transition_hz
        )
        / mid_span_octaves,
        0.0,
        1.0,
    )
    mid_curve = self.mid_slope * mid_positions
    target = np.where(freqs > self.bass_transition_hz, mid_curve, target)

    treble_octaves = np.maximum(
        0.0,
        np.log2(
            np.maximum(freqs, self.treble_transition_hz)
            / self.treble_transition_hz
        ),
    )
    treble_curve = self.mid_slope + (
        treble_octaves * self.treble_boost_db_per_oct
    )
    target = np.where(freqs > self.treble_transition_hz, treble_curve, target)

    reference_hz = float(
        np.sqrt(self.bass_transition_hz * self.treble_transition_hz)
    )
    reference_index = int(np.argmin(np.abs(freqs - reference_hz)))
    target = target - float(target[reference_index])

    return np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)


def update_profile(self) -> None:
    bass_focus_hz = self._fit_frequency(
        max(self.low_cut * 1.5, self.bass_transition_hz / 2.0)
    )
    mid_focus_hz = self._fit_frequency(
        np.sqrt(self.bass_transition_hz * self.treble_transition_hz)
    )
    treble_focus_hz = self._fit_frequency(
        min(self.high_cut, self.treble_transition_hz * 2.0)
    )
    air_focus_hz = self._fit_frequency(
        min(
            self.high_cut,
            max(treble_focus_hz * 1.6, self.treble_transition_hz * 2.8),
        )
    )

    anchor_freqs = np.array(
        [
            self.low_cut,
            bass_focus_hz,
            self.bass_transition_hz,
            mid_focus_hz,
            self.treble_transition_hz,
            treble_focus_hz,
            air_focus_hz,
            self.high_cut,
        ],
        dtype=np.float32,
    )
    anchor_gains = (
        self.build_target_curve(anchor_freqs) * self.anchor_curve_strength
    )
    anchor_gains[0] = 0.0
    anchor_gains[-1] = 0.0

    self.anchors = np.column_stack((anchor_freqs, anchor_gains)).tolist()


def update_bands(self, intensity: float | None = None) -> None:
    count = self.num_bands
    fcs = generate_bands(self.low_cut, self.high_cut, count)
    band_config = (
        self.config
        if intensity is None
        else replace(self.config, intensity=float(intensity))
    )
    self.bands = band_config.build_bands_from_fcs(
        fcs, self.low_cut, self.high_cut
    )


def build_spectral_balance_profile(
    self, correction_db: np.ndarray, f_axis: np.ndarray
) -> SpectralBalanceProfile:
    if correction_db.size == 0 or f_axis.size == 0:
        return SpectralBalanceProfile(
            rescue_factor=0.0,
            correction_strength=self.correction_strength,
            max_boost_db=self.max_spectrum_boost_db,
            max_cut_db=self.max_spectrum_cut_db,
            band_intensity=float(np.clip(self.config.intensity, 0.25, 2.5)),
            restoration_factor=0.0,
            air_restoration_factor=0.0,
            body_restoration_factor=0.0,
            closure_repair_factor=0.0,
            mud_cleanup_factor=0.0,
            harshness_restraint_factor=0.0,
            low_end_restraint_factor=0.0,
            legacy_tonal_rebalance_factor=0.0,
            closed_top_end_repair_factor=0.0,
        )

    safe_correction = np.nan_to_num(
        np.asarray(correction_db, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    safe_freqs = np.nan_to_num(
        np.asarray(f_axis, dtype=np.float32),
        nan=self.low_cut,
        posinf=self.high_cut,
        neginf=self.low_cut,
    )

    overall_mismatch_db = float(
        np.mean(np.abs(safe_correction), dtype=np.float32)
    )
    positive_correction_db = np.maximum(safe_correction, 0.0)
    negative_correction_db = np.maximum(-safe_correction, 0.0)

    bass_mask = safe_freqs <= self.bass_transition_hz
    treble_mask = safe_freqs >= self.treble_transition_hz
    mud_low_hz = float(min(self.high_cut, max(self.bass_transition_hz * 1.15, 160.0)))
    mud_high_hz = float(
        min(
            self.high_cut,
            max(
                mud_low_hz + 1.0,
                min(self.treble_transition_hz * 0.38, 550.0),
            ),
        )
    )
    mud_mask = (safe_freqs >= mud_low_hz) & (safe_freqs <= mud_high_hz)
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
    low_end_focus_mask = (safe_freqs >= low_end_focus_low_hz) & (
        safe_freqs <= low_end_focus_high_hz
    )
    air_focus_hz = float(
        min(
            self.high_cut,
            max(self.treble_transition_hz * 1.65, 6500.0),
        )
    )
    presence_low_hz = float(
        min(
            air_focus_hz,
            max(self.treble_transition_hz * 0.55, 1800.0),
        )
    )
    presence_mask = (safe_freqs >= presence_low_hz) & (safe_freqs < air_focus_hz)
    harsh_low_hz = float(min(self.high_cut, max(presence_low_hz * 1.08, 2600.0)))
    harsh_high_hz = float(
        min(
            self.high_cut,
            max(
                harsh_low_hz + 1.0,
                min(air_focus_hz * 0.9, 6200.0),
            ),
        )
    )
    harsh_mask = (safe_freqs >= harsh_low_hz) & (safe_freqs < harsh_high_hz)
    sibilance_low_hz = float(min(self.high_cut, max(harsh_high_hz, 5600.0)))
    sibilance_high_hz = float(
        min(
            self.high_cut,
            max(sibilance_low_hz + 1.0, min(self.high_cut, 9200.0)),
        )
    )
    sibilance_mask = (safe_freqs >= sibilance_low_hz) & (safe_freqs <= sibilance_high_hz)
    air_mask = safe_freqs >= air_focus_hz
    closed_top_low_hz = float(
        min(
            self.high_cut,
            max(self.treble_transition_hz * 1.14, 4000.0),
        )
    )
    closed_top_mask = safe_freqs >= closed_top_low_hz

    bass_deficit_db = 0.0
    if np.any(bass_mask):
        bass_deficit_db = float(
            max(
                0.0,
                np.mean(positive_correction_db[bass_mask], dtype=np.float32),
            )
        )

    treble_deficit_db = 0.0
    if np.any(treble_mask):
        treble_deficit_db = float(
            max(
                0.0,
                np.mean(positive_correction_db[treble_mask], dtype=np.float32),
            )
        )

    air_deficit_db = 0.0
    if np.any(air_mask):
        air_deficit_db = float(
            max(
                0.0,
                np.mean(positive_correction_db[air_mask], dtype=np.float32),
            )
        )

    presence_deficit_db = 0.0
    if np.any(presence_mask):
        presence_deficit_db = float(
            max(
                0.0,
                np.mean(
                    positive_correction_db[presence_mask],
                    dtype=np.float32,
                ),
            )
        )

    treble_peak_deficit_db = 0.0
    if np.any(treble_mask):
        treble_peak_deficit_db = float(
            np.percentile(positive_correction_db[treble_mask], 78.0)
        )

    air_peak_deficit_db = 0.0
    if np.any(air_mask):
        air_peak_deficit_db = float(
            np.percentile(positive_correction_db[air_mask], 82.0)
        )

    closed_top_deficit_db = 0.0
    if np.any(closed_top_mask):
        closed_top_deficit_db = float(
            max(
                0.0,
                np.mean(positive_correction_db[closed_top_mask], dtype=np.float32),
            )
        )

    closed_top_peak_deficit_db = 0.0
    if np.any(closed_top_mask):
        closed_top_peak_deficit_db = float(
            np.percentile(positive_correction_db[closed_top_mask], 80.0)
        )

    presence_peak_deficit_db = 0.0
    if np.any(presence_mask):
        presence_peak_deficit_db = float(
            np.percentile(positive_correction_db[presence_mask], 76.0)
        )

    mud_excess_db = 0.0
    if np.any(mud_mask):
        mud_excess_db = float(
            np.mean(negative_correction_db[mud_mask], dtype=np.float32)
        )

    mud_peak_excess_db = 0.0
    if np.any(mud_mask):
        mud_peak_excess_db = float(
            np.percentile(negative_correction_db[mud_mask], 76.0)
        )

    bass_excess_db = 0.0
    if np.any(bass_mask):
        bass_excess_db = float(
            np.mean(negative_correction_db[bass_mask], dtype=np.float32)
        )

    bass_peak_excess_db = 0.0
    if np.any(bass_mask):
        bass_peak_excess_db = float(
            np.percentile(negative_correction_db[bass_mask], 78.0)
        )

    low_end_focus_excess_db = 0.0
    if np.any(low_end_focus_mask):
        low_end_focus_excess_db = float(
            np.mean(negative_correction_db[low_end_focus_mask], dtype=np.float32)
        )

    low_end_focus_peak_excess_db = 0.0
    if np.any(low_end_focus_mask):
        low_end_focus_peak_excess_db = float(
            np.percentile(negative_correction_db[low_end_focus_mask], 80.0)
        )

    harsh_excess_db = 0.0
    if np.any(harsh_mask):
        harsh_excess_db = float(
            np.mean(negative_correction_db[harsh_mask], dtype=np.float32)
        )

    harsh_peak_excess_db = 0.0
    if np.any(harsh_mask):
        harsh_peak_excess_db = float(
            np.percentile(negative_correction_db[harsh_mask], 82.0)
        )

    sibilance_peak_excess_db = 0.0
    if np.any(sibilance_mask):
        sibilance_peak_excess_db = float(
            np.percentile(negative_correction_db[sibilance_mask], 78.0)
        )

    club_voicing_factor = float(
        np.clip((bass_deficit_db + treble_deficit_db) / 10.0, 0.0, 1.0)
    )
    mismatch_factor = float(
        np.clip((overall_mismatch_db - 2.5) / 6.5, 0.0, 1.0)
    )
    dual_end_repair_factor = float(
        np.clip((min(bass_deficit_db, treble_deficit_db) - 2.0) / 5.5, 0.0, 1.0)
    )
    high_restore_pressure_db = float(
        max(
            presence_deficit_db * 0.9,
            treble_deficit_db,
            air_deficit_db,
            presence_peak_deficit_db * 0.94,
            treble_peak_deficit_db * 0.9,
            air_peak_deficit_db,
        )
    )
    closure_repair_factor = float(
        np.clip(
            (high_restore_pressure_db - 2.3) / 5.8,
            0.0,
            1.0,
        )
    )
    air_restoration_factor = float(
        np.clip(
            max(
                treble_deficit_db * 0.75,
                air_deficit_db,
                presence_deficit_db * 0.85,
                treble_peak_deficit_db * 0.82,
                air_peak_deficit_db,
                presence_peak_deficit_db * 0.88,
            )
            / 8.4,
            0.0,
            1.0,
        )
    )
    closed_top_end_repair_pressure = float(
        np.clip(
            (
                max(
                    closed_top_deficit_db * 0.9,
                    closed_top_peak_deficit_db,
                    high_restore_pressure_db * 0.78,
                )
                - 2.1
            )
            / 6.2,
            0.0,
            1.0,
        )
    )
    body_restoration_factor = float(
        np.clip((bass_deficit_db + dual_end_repair_factor * 3.0) / 10.0, 0.0, 1.0)
    )
    rescue_factor = float(
        np.clip(max(mismatch_factor, club_voicing_factor * 0.9), 0.0, 1.0)
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
                + mismatch_factor * 0.32
                + rescue_factor * 0.16
                + closure_repair_factor * 0.26
                + air_restoration_factor * 0.18
            ),
            0.0,
            1.0,
        )
    )
    low_end_restraint_factor = float(
        np.clip(
            (
                max(
                    bass_excess_db * 0.82,
                    bass_peak_excess_db * 0.88,
                    low_end_focus_excess_db,
                    low_end_focus_peak_excess_db * 0.92,
                    mud_excess_db * 0.78,
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
    body_restoration_factor = float(
        np.clip(
            body_restoration_factor * (1.0 - low_end_restraint_factor * 0.92),
            0.0,
            1.0,
        )
    )
    high_restore_blend_factor = float(
        max(
            closure_repair_factor,
            air_restoration_factor * 0.85,
            rescue_factor * 0.35,
        )
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
                + high_restore_blend_factor * 0.18
            ),
            0.0,
            1.0,
        )
    )
    closed_top_end_repair_factor = float(
        np.clip(
            closed_top_end_repair_pressure
            * (
                0.18
                + closure_repair_factor * 0.58
                + air_restoration_factor * 0.42
                + low_end_restraint_factor * 0.16
            )
            * (1.0 - harshness_restraint_factor * 0.35),
            0.0,
            1.0,
        )
    )
    restoration_factor = float(
        np.clip(
            max(
                rescue_factor,
                dual_end_repair_factor * 0.95,
                air_restoration_factor * 0.9,
                body_restoration_factor * 0.75,
                closure_repair_factor * 0.92,
                mud_cleanup_factor * 0.42,
                harshness_restraint_factor * 0.22,
                low_end_restraint_factor * 0.16,
            ),
            0.0,
            1.0,
        )
    )
    legacy_tonal_rebalance_factor = float(
        np.clip(
            min(
                low_end_restraint_factor,
                max(
                    closure_repair_factor,
                    air_restoration_factor * 0.92,
                    restoration_factor * 0.74,
                ),
            )
            * (
                0.56
                + mud_cleanup_factor * 0.22
                + restoration_factor * 0.12
            )
            * (1.0 - harshness_restraint_factor * 0.45),
            0.0,
            1.0,
        )
    )

    correction_strength = float(
        np.clip(
            self.correction_strength
            + rescue_factor * self.spectral_rescue_strength
            + club_voicing_factor * 0.05,
            0.25,
            1.35,
        )
    )
    correction_strength = float(
        np.clip(
            correction_strength
            + restoration_factor * 0.16
            + air_restoration_factor * 0.07
            + body_restoration_factor * 0.05,
            0.25,
            1.35,
        )
    )
    correction_strength = float(
        np.clip(
            correction_strength
            + closure_repair_factor * 0.12
            + mud_cleanup_factor * 0.08,
            0.25,
            1.45,
        )
    )
    correction_strength = float(
        np.clip(
            correction_strength + closed_top_end_repair_factor * 0.08,
            0.25,
            1.45,
        )
    )
    max_boost_db = float(
        self.max_spectrum_boost_db
        + rescue_factor * self.spectral_rescue_boost_db
        + club_voicing_factor * 0.5
        + restoration_factor * 2.2
        + air_restoration_factor * 1.4
        + body_restoration_factor * 0.85
        + closure_repair_factor * 1.75
        + mud_cleanup_factor * 0.35
        + closed_top_end_repair_factor * 0.95
    )
    max_cut_db = float(
        self.max_spectrum_cut_db
        + rescue_factor * self.spectral_rescue_cut_db
        + restoration_factor * 0.35
        + mud_cleanup_factor * 1.55
        + harshness_restraint_factor * 0.9
        + low_end_restraint_factor * 1.1
    )
    band_intensity = float(
        np.clip(
            self.config.intensity
            + rescue_factor * self.spectral_rescue_band_intensity
            + club_voicing_factor * 0.15,
            0.25,
            2.5,
        )
    )
    band_intensity = float(
        np.clip(
            band_intensity
            + restoration_factor * 0.28
            + body_restoration_factor * 0.1
            + closure_repair_factor * 0.1,
            0.25,
            2.5,
        )
    )
    band_intensity = float(
        np.clip(
            band_intensity
            - low_end_restraint_factor * 0.2
            - harshness_restraint_factor * 0.05
            - legacy_tonal_rebalance_factor * 0.38,
            0.25,
            2.5,
        )
    )

    return SpectralBalanceProfile(
        rescue_factor=rescue_factor,
        correction_strength=correction_strength,
        max_boost_db=max_boost_db,
        max_cut_db=max_cut_db,
        band_intensity=band_intensity,
        restoration_factor=restoration_factor,
        air_restoration_factor=air_restoration_factor,
        body_restoration_factor=body_restoration_factor,
        closure_repair_factor=closure_repair_factor,
        mud_cleanup_factor=mud_cleanup_factor,
        harshness_restraint_factor=harshness_restraint_factor,
        low_end_restraint_factor=low_end_restraint_factor,
        legacy_tonal_rebalance_factor=legacy_tonal_rebalance_factor,
        closed_top_end_repair_factor=closed_top_end_repair_factor,
    )


def plan_follow_up_drives(
    self, remaining_lufs: float, rescue_factor: float = 0.0
) -> list[float]:
    if (
        not np.isfinite(remaining_lufs)
        or remaining_lufs <= self.final_lufs_tolerance
    ):
        return []

    total_budget = float(
        min(
            max(remaining_lufs, 0.0),
            self.max_final_boost_db + rescue_factor * 1.5,
        )
    )
    if total_budget <= 0.0:
        return []

    planned_passes = min(
        self.max_follow_up_passes,
        1 + int(total_budget > 1.0) + int(total_budget > 2.5),
    )
    weights = np.linspace(planned_passes, 1, planned_passes, dtype=np.float32)
    weights /= np.sum(weights, dtype=np.float32)
    drives = [float(total_budget * weight) for weight in weights]
    return [drive for drive in drives if drive > 0.05]
