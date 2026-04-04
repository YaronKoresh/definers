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

    bass_mask = safe_freqs <= self.bass_transition_hz
    treble_mask = safe_freqs >= self.treble_transition_hz

    bass_deficit_db = 0.0
    if np.any(bass_mask):
        bass_deficit_db = float(
            max(0.0, np.mean(safe_correction[bass_mask], dtype=np.float32))
        )

    treble_deficit_db = 0.0
    if np.any(treble_mask):
        treble_deficit_db = float(
            max(
                0.0,
                np.mean(safe_correction[treble_mask], dtype=np.float32),
            )
        )

    club_voicing_factor = float(
        np.clip((bass_deficit_db + treble_deficit_db) / 10.0, 0.0, 1.0)
    )
    mismatch_factor = float(
        np.clip((overall_mismatch_db - 2.5) / 6.5, 0.0, 1.0)
    )
    rescue_factor = float(
        np.clip(max(mismatch_factor, club_voicing_factor * 0.9), 0.0, 1.0)
    )

    correction_strength = float(
        np.clip(
            self.correction_strength
            + rescue_factor * self.spectral_rescue_strength
            + club_voicing_factor * 0.05,
            0.25,
            1.2,
        )
    )
    max_boost_db = float(
        self.max_spectrum_boost_db
        + rescue_factor * self.spectral_rescue_boost_db
        + club_voicing_factor * 0.5
    )
    max_cut_db = float(
        self.max_spectrum_cut_db + rescue_factor * self.spectral_rescue_cut_db
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

    return SpectralBalanceProfile(
        rescue_factor=rescue_factor,
        correction_strength=correction_strength,
        max_boost_db=max_boost_db,
        max_cut_db=max_cut_db,
        band_intensity=band_intensity,
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
