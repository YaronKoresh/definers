from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmartMasteringConfig:
    num_bands: int = 6
    intensity: float = 1.0

    bass_knee_db: float = 10.0
    treb_knee_db: float = 4.0

    bass_ratio: float = 5.0
    bass_attack_ms: float = 20.0
    bass_release_ms: float = 250.0
    bass_threshold_db: float = -16.0

    treb_ratio: float = 3.5
    treb_attack_ms: float = 2.0
    treb_release_ms: float = 100.0
    treb_threshold_db: float = -26.0

    resampling_target: int = 44100

    target_lufs: float = -5.5

    stop_bass_boost_hz: float = 140.0
    start_treble_boost_hz: float = 4500.0

    bass_boost_db_per_oct: float = 2.0
    mid_slope: float = -3.0
    treble_boost_db_per_oct: float = 2.0

    smoothing_fraction: float | None = 0.2

    correction_strength: float = 1.0

    low_cut: float | None = 30.0
    high_cut: float | None = None

    drive_db: float = 3.0
    ceil_db: float = -0.3

    @classmethod
    def make_bands_from_fcs(
        cls, fcs: list[float], freq_min: float, freq_max: float
    ) -> list[dict]:
        if not fcs:
            return []

        fcs_arr = np.array(fcs, dtype=float)
        ref_min = np.log2(freq_min)
        ref_max = np.log2(freq_max)
        fcs_safe = np.clip(fcs_arr, freq_min, freq_max)
        fcs_log = np.log2(fcs_safe)

        positions = (fcs_log - ref_min) / (ref_max - ref_min)

        knees = (
            cls.bass_knee_db 
            + (cls.treb_knee_db - cls.bass_knee_db) * positions
        )

        base_thr = (
            cls.bass_threshold_db
            + (cls.treb_threshold_db - cls.bass_threshold_db) * positions
        )

        ratios = cls.bass_ratio + (cls.treb_ratio - cls.bass_ratio) * positions

        attacks = (
            cls.bass_attack_ms
            + (cls.treb_attack_ms - cls.bass_attack_ms) * positions
        )

        releases = (
            cls.bass_release_ms
            + (cls.treb_release_ms - cls.bass_release_ms) * positions
        )

        thr = np.mean(base_thr) + (base_thr - np.mean(base_thr)) * cls.intensity
        ratios = 1.0 + (ratios - 1.0) * cls.intensity
        attacks *= cls.intensity
        releases *= cls.intensity

        knees = 0.5 * knees + 0.5 * (knees / max(cls.intensity, 0.1))

        makeups = np.abs(base_thr) * (1 - 1/ratios) * 0.5 * cls.intensity

        return [
            {
                "fc": float(fcs_arr[i]),
                "base_threshold": float(thr[i]),
                "ratio": float(ratios[i]),
                "attack_ms": float(attacks[i]),
                "release_ms": float(releases[i]),
                "makeup_db": float(makeups[i]),
                "knee_db": float(knees[i]),
            }
            for i in range(len(fcs_arr))
        ]
