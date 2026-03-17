from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SmartMasteringConfig:
    num_bands: int = 6
    intensity: float = 1.0

    bass_ratio: float = 2.4
    bass_attack_ms: float = 10.0
    bass_release_ms: float = 160.0
    bass_threshold_db: float = -23.0

    treb_ratio: float = 1.9
    treb_attack_ms: float = 0.001
    treb_release_ms: float = 14.0
    treb_threshold_db: float = -17.0

    resampling_target: int = 44100

    target_lufs: float = -9.0

    slope_db: float = 3.0
    slope_hz: float = 320.0

    phase_type: str = "minimal"

    anchors: list[list[float]] | None = None

    smoothing_fraction: float = 1.0 / 4.0

    correction_strength: float = 1.0

    low_cut: float | None = None
    high_cut: float | None = None

    drive_db: float = 1.0
    ceil_db: float | None = -1.0

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

        makeups = np.zeros_like(fcs_arr, dtype=float)

        ratios = 1.0 + (ratios - 1.0) * cls.intensity
        attacks *= cls.intensity
        releases *= cls.intensity

        knees = np.full_like(fcs, 6.0, dtype=float)

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
