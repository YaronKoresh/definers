import math
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import librosa
import numpy as _np
from numba import njit
from scipy import signal
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, lfilter, resample_poly, sosfilt, welch

from definers.constants import (
    MADMOM_AVAILABLE,
    MODELS,
    PROCESSORS,
    language_codes,
)
from definers.cuda import device
from definers.data import cupy_to_numpy, numpy_to_cupy
from definers.image import get_max_resolution
from definers.system import (
    catch,
    cores,
    delete,
    exist,
    full_path,
    get_ext,
    log,
    run,
    tmp,
)
from definers.text import random_string
from definers.web import google_drive_download

try:
    import cupy as np
except Exception:
    import numpy as np


@dataclass
class SmartMasteringConfig:
    num_bands: int = 16
    intensity: float = 1.0

    bass_ratio: float = 2.0
    bass_attack_ms: float = 0.1
    bass_release_ms: float = 70.0
    bass_threshold_db: float = -20.0

    treb_ratio: float = 2.0
    treb_attack_ms: float = 0.1
    treb_release_ms: float = 30.0
    treb_threshold_db: float = -20.0

    sample_rate: int | None = None
    slope_db: float = 3.0
    slope_hz: float = 1000.0
    smoothing_fraction: float = 1.0 / 3.0
    target_lufs: float = -9.0
    correction_strength: float = 1.0
    low_cut: int | None = None
    high_cut: int | None = None
    phase_type: str = "minimal"
    drive_db: float = 0.0
    ceil_db: float = -0.1

    @classmethod
    def make_bands_from_fcs(
        cls, fcs: list[float], freq_min: int, freq_max: int
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
        attacks *= 1.0 - 0.5 * cls.intensity
        releases *= 1.0 + 0.5 * cls.intensity

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


def resample(y, sr, target_sr=44100):
    if np.issubdtype(y.dtype, np.integer):
        y = librosa.util.buf_to_float(y)
    else:
        y = y.astype(np.float64)

    y_out = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_out


@njit(cache=True)
def decoupled_envelope(env_db, attack_coef, release_coef):
    out = np.empty_like(env_db)
    prev = env_db[0]
    for i in range(len(env_db)):
        coef = attack_coef if env_db[i] > prev else release_coef
        prev = coef * prev + (1.0 - coef) * env_db[i]
        out[i] = prev
    return out


@njit(cache=True)
def limiter_smooth_env(env, attack_coef, release_coef):
    out = np.empty_like(env)
    prev = env[0]

    is_adaptive = release_coef.ndim > 0

    for i in range(len(env)):
        rc = release_coef[i] if is_adaptive else release_coef

        coef = attack_coef if env[i] > prev else rc

        prev = coef * prev + (1.0 - coef) * env[i]
        out[i] = prev

    return out


def calculate_dynamic_cutoff(
    y: np.ndarray, sr: int, rolloff_percent: float
) -> float:
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

    freqs, psd = welch(y_mono, fs=sr, nperseg=4096)

    cumulative_energy = np.cumsum(psd)
    total_energy = cumulative_energy[-1]

    rolloff_index = np.argmax(
        cumulative_energy >= (rolloff_percent * total_energy)
    )
    optimal_cutoff_hz = freqs[rolloff_index]

    optimal_cutoff_hz = np.clip(optimal_cutoff_hz, 4000.0, 12000.0)

    print(
        f"Calculated dynamic exciter cutoff: {optimal_cutoff_hz:.1f} Hz (rolloff at {rolloff_percent * 100:.1f}%)"
    )
    print(
        f"Total energy: {total_energy:.6e}, Rolloff energy: {cumulative_energy[rolloff_index]:.6e}"
    )
    print(
        f"Rolloff frequency: {freqs[rolloff_index]:.1f} Hz, PSD at rolloff: {psd[rolloff_index]:.6e}"
    )

    return float(optimal_cutoff_hz)


def apply_exciter(
    y: np.ndarray,
    sr: int,
    cutoff_hz: float = None,
    drive: float = None,
    oversample: int = 2,
    mix: float = 1.0,
) -> np.ndarray:
    nyquist = sr / 2.0

    if cutoff_hz is None:
        cutoff_hz = calculate_dynamic_cutoff(y, sr, rolloff_percent=0.7)

    sos = butter(
        4, cutoff_hz / nyquist * oversample, btype="high", output="sos"
    )
    high_band = sosfilt(sos, y)

    print(
        f"Extracted high band for exciter. Cutoff: {cutoff_hz:.1f} Hz, High band stats - min: {high_band.min():.6f}, max: {high_band.max():.6f}"
    )

    if drive is None:
        peak_highs = np.max(np.abs(high_band))
        drive = np.clip(4.0 / (peak_highs + 1e-6), 1.0, 50.0)

        print(f"Calculated drive for exciter: {drive:.2f}")

    y_up = resample_poly(high_band, up=oversample, down=1, axis=-1)

    print(
        f"Oversampled high band. Oversample factor: {oversample}, Shape: {y_up.shape}, dtype: {y_up.dtype}, max: {np.max(np.abs(y_up)):.6f}"
    )

    distorted_up = np.tanh(y_up * drive)
    distorted_up *= peak_highs / (np.max(np.abs(distorted_up)) + 1e-6)

    distorted_highs = resample_poly(
        distorted_up, up=1, down=oversample, axis=-1
    )

    if distorted_highs.shape[-1] > high_band.shape[-1]:
        print(
            f"Trimming distorted highs from {distorted_highs.shape[-1]} to {high_band.shape[-1]} samples"
        )

        distorted_highs = distorted_highs[..., : high_band.shape[-1]]

    pure_air = sosfilt(sos, distorted_highs)

    print(
        f"Generated pure air signal. Shape: {pure_air.shape}, dtype: {pure_air.dtype}, max: {np.max(np.abs(pure_air)):.6f}"
    )

    y, pure_air = pad_audio(y, pure_air)
    y_enhanced = y + (pure_air * mix)

    print(
        f"Exciter applied with cutoff {cutoff_hz:.1f} Hz, drive {drive:.2f}, mix {mix:.2f}"
    )
    print(f"Peak of pure air signal: {np.max(np.abs(pure_air)):.6f}")
    print(f"Peak of enhanced signal: {np.max(np.abs(y_enhanced)):.6f}")

    return y_enhanced


def stereo(*y_s, lr=True, sr=44100):
    stereo_result = None

    if lr and len(y_s) == 2:
        print("Processing stereo pair with LR mode")
        a, b = y_s
        a = stereo(a)
        b = stereo(b)
        _l, _r = pad_audio(a, b)
        y_s = [np.vstack([_l[0], _r[1]])]

    for y in y_s:
        if y.ndim == 1:
            print(
                f"Input is mono, converting to stereo. Original shape: {y.shape}, dtype: {y.dtype}"
            )
            y = y[np.newaxis, :]
            y = np.vstack([y, y])
        if y.ndim > 2:
            print(
                f"Input has more than 2 channels, taking first two. Original shape: {y.shape}, dtype: {y.dtype}"
            )
            y = np.vstack([y[0], y[1]])
        if y.shape[1] < y.shape[0]:
            print(
                f"Transposing input to have shape (channels, samples). Original shape: {y.shape}, dtype: {y.dtype}"
            )
            y = y.T

        y = np.asarray(y, dtype=np.float64)

        if stereo_result is None:
            print(
                f"Initializing stereo result with first input. Shape: {y.shape}, dtype: {y.dtype}"
            )
            stereo_result = y
        else:
            print(
                f"Mixing input into stereo result. Input shape: {y.shape}, dtype: {y.dtype}, Current stereo result shape: {stereo_result.shape}, dtype: {stereo_result.dtype}"
            )
            stereo_result = mix_audio(stereo_result, y, sr, weight=0.5)

    print(
        f"Final stereo result shape: {stereo_result.shape}, dtype: {stereo_result.dtype}, max: {np.max(np.abs(stereo_result)):.6f}"
    )
    return stereo_result


def pad_audio(y1: np.ndarray, y2: np.ndarray):
    y1 = stereo(y1)
    y2 = stereo(y2)

    max_len = max(y1.shape[1], y2.shape[1])

    y1_padded = np.zeros((y1.shape[0], max_len), dtype=np.float64)
    y2_padded = np.zeros((y2.shape[0], max_len), dtype=np.float64)
    y1_padded[:, : y1.shape[1]] = y1
    y2_padded[:, : y2.shape[1]] = y2

    print(
        f"Padded audio to length {max_len}. Y1 shape: {y1_padded.shape}, dtype: {y1_padded.dtype}, Y2 shape: {y2_padded.shape}, dtype: {y2_padded.dtype}"
    )
    print(
        f"Y1 stats after padding - min: {y1_padded.min():.6f}, max: {y1_padded.max():.6f}, mean: {y1_padded.mean():.6f}"
    )
    print(
        f"Y2 stats after padding - min: {y2_padded.min():.6f}, max: {y2_padded.max():.6f}, mean: {y2_padded.mean():.6f}"
    )

    return y1_padded, y2_padded


def mix_audio(
    y1: np.ndarray,
    y2: np.ndarray,
    sr: int,
    weight: float = 0.3,
    fade_duration: float = 2.0,
    duck_depth: float = 0.3,
    cutoff_hz: int | None = None,
):
    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)

    y1, y2 = stereo(y1), stereo(y2)
    y1_padded, y2_padded = pad_audio(y1, y2)

    y1_padded = y1_padded.T
    y2_padded = y2_padded.T

    if cutoff_hz is None:
        y1_filtered = y1_padded
    else:
        nyquist = 0.5 * sr
        normal_cutoff = cutoff_hz / nyquist
        b, a = butter(4, normal_cutoff, btype="low", analog=False)
        y1_filtered = lfilter(b, a, y1_padded, axis=0)

    print(
        f"Applied low-pass filter to Y1 with cutoff {cutoff_hz} Hz. Filtered Y1 stats - min: {y1_filtered.min():.6f}, max: {y1_filtered.max():.6f}, mean: {y1_filtered.mean():.6f}"
    )

    window_size = int(sr * 0.1)
    envelope = np.abs(y2_padded.mean(axis=1))
    envelope = np.convolve(
        envelope, np.ones(window_size) / window_size, mode="same"
    )

    if np.max(envelope) > 0:
        envelope = envelope / np.max(envelope)
    envelope = envelope[:, np.newaxis]

    print(
        f"Computed envelope for Y2. Envelope stats - min: {envelope.min():.6f}, max: {envelope.max():.6f}, mean: {envelope.mean():.6f}"
    )

    y1_processed = (y1_padded * (1 - envelope)) + (y1_filtered * envelope)

    duck_mask = 1.0 - (envelope * (1.0 - duck_depth))

    y_length = len(y1_padded)

    fade_samples = min(int(fade_duration * sr), y_length)
    t = np.linspace(0, np.pi / 2, fade_samples).reshape(-1, 1)
    fade_in, fade_out = np.sin(t), np.cos(t)

    print(
        f"Fade in stats - min: {fade_in.min():.6f}, max: {fade_in.max():.6f}, mean: {fade_in.mean():.6f}"
    )
    print(
        f"Fade out stats - min: {fade_out.min():.6f}, max: {fade_out.max():.6f}, mean: {fade_out.mean():.6f}"
    )

    mask1 = np.ones((y_length, 1)) * weight
    mask1[-fade_samples:] *= fade_out

    mask2 = np.zeros((y_length, 1))
    mask2[:fade_samples] = (1.0 - weight) * fade_in
    mask2[fade_samples:] = 1.0 - weight

    y1_weighted = y1_processed * mask1 * duck_mask
    y2_weighted = y2_padded * mask2

    print("Applied mixing masks")
    print(
        f"Y1 weighted stats - min: {y1_weighted.min():.6f}, max: {y1_weighted.max():.6f}, mean: {y1_weighted.mean():.6f}"
    )
    print(
        f"Y2 weighted stats - min: {y2_weighted.min():.6f}, max: {y2_weighted.max():.6f}, mean: {y2_weighted.mean():.6f}"
    )

    y_mixed = y1_weighted + y2_weighted

    max_val = np.max(np.abs(y_mixed))
    if max_val > 1.0:
        print(
            f"Normalizing mixed audio. Max value before normalization: {max_val:.6f}"
        )
        y_mixed /= max_val

    if y_mixed.shape[0] > y_mixed.shape[1]:
        print(
            f"Transposing mixed audio to have shape (channels, samples). Original shape: {y_mixed.shape}, dtype: {y_mixed.dtype}"
        )
        y_mixed = y_mixed.T

    print(
        f"Final mixed audio stats - min: {y_mixed.min():.6f}, max: {y_mixed.max():.6f}, mean: {y_mixed.mean():.6f}"
    )
    print(
        f"Mixing parameters - weight: {weight:.2f}, fade_duration: {fade_duration:.2f} s, duck_depth: {duck_depth:.2f}, cutoff_hz: {cutoff_hz}"
    )

    return y_mixed


def freq_cut(y: np.ndarray, sr: int, low_cut: int | None, high_cut: int | None):
    y = np.asarray(y, dtype=np.float64)
    original_length = y.shape[-1]

    if y.ndim > 1:
        for i in range(y.shape[0]):
            y[i] -= np.mean(y[i])
    else:
        y -= np.mean(y)

    original_length = y.shape[-1]

    if high_cut:
        target_high_sr = high_cut * 2

        y_high_cut_resampled = resample(y, sr, target_high_sr)
        y_main = resample(y_high_cut_resampled, target_high_sr, sr)

        if y_main.shape[-1] > original_length:
            y = y_main[..., :original_length]
        elif y_main.shape[-1] < original_length:
            pad_width = original_length - y_main.shape[-1]
            y = np.pad(y_main, (*((0, 0),) * (y_main.ndim - 1), (0, pad_width)))

    if low_cut:
        target_low_sr = low_cut * 2
        y_low_only_resampled = resample(y, sr, target_low_sr)
        y_low_reconstructed = resample(y_low_only_resampled, target_low_sr, sr)

        if y_low_reconstructed.shape[-1] > original_length:
            y_low_reconstructed = y_low_reconstructed[..., :original_length]
        elif y_low_reconstructed.shape[-1] < original_length:
            pad_width = original_length - y_low_reconstructed.shape[-1]
            y_low_reconstructed = np.pad(
                y_low_reconstructed,
                (*((0, 0),) * (y_low_reconstructed.ndim - 1), (0, pad_width)),
            )

        y -= y_low_reconstructed

    print(
        f"Applied frequency cuts. Low cut: {low_cut} Hz, High cut: {high_cut} Hz"
    )
    print(
        f"Output shape after freq cut: {y.shape}, dtype: {y.dtype}, max: {np.max(np.abs(y)):.6f}"
    )

    return y


class SmartMastering:
    @property
    def slope_db(self) -> float:
        return self._slope_db

    @slope_db.setter
    def slope_db(self, value: float) -> None:
        self._slope_db = value
        self.update_bands()

    def __init__(
        self,
        config: SmartMasteringConfig | dict | None = None,
    ) -> None:
        cfg = config or SmartMasteringConfig()

        self.config = cfg

        self.resampling_target = 44100

        self.drive_db = (
            cfg.drive_db
            if cfg.drive_db is not None
            else SmartMasteringConfig().drive_db
        )

        self.ceil_db = (
            cfg.ceil_db
            if cfg.ceil_db is not None
            else SmartMasteringConfig().ceil_db
        )

        self.sr = (
            cfg.sample_rate
            if cfg.sample_rate is not None
            else self.resampling_target
        )

        self.num_bands = (
            cfg.num_bands
            if cfg.num_bands is not None
            else SmartMasteringConfig().num_bands
        )

        self._slope_db = (
            cfg.slope_db
            if cfg.slope_db is not None
            else SmartMasteringConfig().slope_db
        )

        self.slope_hz = (
            cfg.slope_hz
            if cfg.slope_hz is not None
            else SmartMasteringConfig().slope_hz
        )
        self.smoothing_fraction = (
            cfg.smoothing_fraction
            if cfg.smoothing_fraction is not None
            else SmartMasteringConfig().smoothing_fraction
        )
        self.target_lufs = (
            cfg.target_lufs
            if cfg.target_lufs is not None
            else SmartMasteringConfig().target_lufs
        )
        self.low_cut = (
            cfg.low_cut
            if cfg.low_cut is not None
            else SmartMasteringConfig().low_cut
        )
        self.high_cut = (
            cfg.high_cut
            if cfg.high_cut is not None
            else SmartMasteringConfig().high_cut
        )
        self.correction_strength = (
            cfg.correction_strength
            if cfg.correction_strength is not None
            else SmartMasteringConfig().correction_strength
        )
        self.phase_type = (
            cfg.phase_type
            if cfg.phase_type is not None
            else SmartMasteringConfig().phase_type
        )

        self.update_bands()

        self.nperseg = int(2 ** np.ceil(np.log2(self.sr / 5)))

        self.target_freqs_hz = np.array(
            generate_bands(0.1, self.sr / 2.0, self.nperseg),
            dtype=float,
        )

    def update_bands(self) -> None:
        count = self.num_bands
        fcs = generate_bands(
            self.low_cut or 0.1,
            self.high_cut or (self.resampling_target / 2 - 0.1),
            count,
        )
        self.bands = SmartMasteringConfig.make_bands_from_fcs(
            fcs,
            self.low_cut or 0.1,
            self.high_cut or (self.resampling_target / 2 - 0.1),
        )

    def measure_spectrum(self, y_mono: np.ndarray) -> np.ndarray:
        NPERSEG = self.nperseg
        FLOOR_DB = -120.0
        CEILING_DB = 20.0

        if len(y_mono) < NPERSEG:
            y_mono_padded = np.pad(y_mono, (0, NPERSEG - len(y_mono)))
        else:
            y_mono_padded = y_mono

        f_axis, psd = signal.welch(
            y_mono_padded,
            fs=self.sr,
            nperseg=NPERSEG,
            noverlap=int(NPERSEG * 0.75),
            window=("kaiser", 18),
            scaling="density",
            average="median",
            detrend="constant",
        )

        print(
            f"Welch method completed. Frequency bins: {len(f_axis)}, PSD bins: {len(psd)}"
        )

        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
        psd_db = np.clip(psd_db, FLOOR_DB, CEILING_DB)

        print(
            f"Measured spectrum. Frequency axis shape: {f_axis.shape}, PSD shape: {psd_db.shape}"
        )
        print(
            f"Frequency axis stats - min: {f_axis.min():.1f} Hz, max: {f_axis.max():.1f} Hz, mean: {f_axis.mean():.1f} Hz"
        )
        print(
            f"PSD stats - min: {psd_db.min():.2f} dB, max: {psd_db.max():.2f} dB, mean: {psd_db.mean():.2f} dB"
        )

        return psd_db, f_axis

    def compute_spectrum(self, f_axis: np.ndarray) -> np.ndarray:
        min_freq = self.low_cut if self.low_cut else 0.1
        max_freq = self.high_cut if self.high_cut else f_axis[-1]

        f_axis_safe = np.maximum(f_axis, min_freq)
        f_axis_safe = np.minimum(f_axis_safe, max_freq)

        octaves = np.log2(f_axis_safe / self.slope_hz)

        target_db = -self.slope_db * octaves

        print(
            f"Built spectrum. Shape: {target_db.shape}, dtype: {target_db.dtype}, min: {target_db.min():.2f} dB, max: {target_db.max():.2f} dB, mean: {target_db.mean():.2f} dB"
        )
        print(
            f"Spectrum frequency axis stats - min: {f_axis.min():.1f} Hz, max: {f_axis.max():.1f} Hz, mean: {f_axis.mean():.1f} Hz"
        )
        print(
            f"Spectrum parameters - slope_db: {self.slope_db:.2f} dB/octave, slope_hz: {self.slope_hz:.1f} Hz, low_cut: {self.low_cut}, high_cut: {self.high_cut}"
        )

        return target_db

    def smooth_curve(
        self,
        curve: np.ndarray,
        f_axis: np.ndarray,
        smoothing_fraction: float | None = None,
    ) -> np.ndarray:
        if smoothing_fraction is None:
            smoothing_fraction = self.smoothing_fraction

        smoothed = np.copy(curve)

        for i, f in enumerate(f_axis):
            bandwidth = f * (2**smoothing_fraction - 2 ** (-smoothing_fraction))

            low_f = f - bandwidth / 2
            high_f = f + bandwidth / 2

            mask = (f_axis >= low_f) & (f_axis <= high_f)

            if np.any(mask):
                smoothed[i] = np.mean(curve[mask])

        print(
            f"Smoothed spectrum. Shape: {smoothed.shape}, dtype: {smoothed.dtype}"
        )
        print(
            f"Smoothed spectrum stats - min: {smoothed.min():.2f} dB, max: {smoothed.max():.2f} dB, mean: {smoothed.mean():.2f} dB"
        )

        return smoothed

    def apply_phase_correction(self, y: np.ndarray, tp: str) -> np.ndarray:
        start = __import__("time").perf_counter()

        orig_len = y.shape[-1]
        y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

        print(
            f"Measuring spectrum for phase correction. Input shape: {y_mono.shape}, dtype: {y_mono.dtype}, sample rate: {self.sr} Hz"
        )

        input_db, f_axis = self.measure_spectrum(y_mono)
        target_db = self.compute_spectrum(f_axis)

        print(
            f"Measured input spectrum. Shape: {input_db.shape}, dtype: {input_db.dtype}"
        )
        print(
            f"Built target spectrum. Shape: {target_db.shape}, dtype: {target_db.dtype}"
        )

        if not np.all(np.isfinite(target_db)):
            target_db = np.nan_to_num(
                target_db, nan=0.0, posinf=0.0, neginf=0.0
            )

        target_db = self.smooth_curve(
            target_db, f_axis, self.smoothing_fraction
        )

        print(
            f"Smoothed target spectrum. Shape: {target_db.shape}, dtype: {target_db.dtype}"
        )
        print(
            f"Target spectrum stats - min: {target_db.min():.2f} dB, max: {target_db.max():.2f} dB, mean: {target_db.mean():.2f} dB"
        )

        min_len = min(len(target_db), len(input_db))

        target_db = target_db[:min_len]
        input_db = input_db[:min_len]

        print(
            f"Padded spectra. Target shape: {target_db.shape}, Input shape: {input_db.shape}"
        )

        mean_target = np.mean(target_db)
        mean_input = np.mean(input_db)
        target_db = target_db - mean_target + mean_input

        correction_db = target_db - input_db
        correction_db = (target_db - input_db) * self.correction_strength

        print(
            f"Calculated correction curve. Shape: {correction_db.shape}, dtype: {correction_db.dtype}, min: {correction_db.min():.2f} dB, max: {correction_db.max():.2f} dB, mean: {correction_db.mean():.2f} dB"
        )

        correction_db_norm = correction_db - np.max(correction_db)
        H_lin = 10.0 ** (correction_db_norm / 20.0)

        print(
            f"Converted correction to linear scale. Shape: {H_lin.shape}, dtype: {H_lin.dtype}, min: {H_lin.min():.6f}, max: {H_lin.max():.6f}"
        )

        out = None

        if tp == "minimal":
            log_mag = np.log(np.maximum(H_lin, 1e-12))

            cepstrum = np.fft.irfft(log_mag)
            n = len(cepstrum)

            lifter = np.zeros(n)
            lifter[0] = 1.0
            lifter[1 : n // 2] = 2.0
            if n % 2 == 0:
                lifter[n // 2] = 1.0

            print(
                f"Applying minimal phase correction. Cepstrum length: {n}, lifter shape: {lifter.shape}"
            )

            causal_cepstrum = cepstrum * lifter

            min_phase_spectrum = np.exp(np.fft.rfft(causal_cepstrum))

            h_ir = np.fft.irfft(min_phase_spectrum)

            FIR_LEN = min(len(h_ir), 8192)
            h_ir_cut = h_ir[:FIR_LEN]

            window = np.ones(FIR_LEN)
            fade_size = FIR_LEN // 2
            if fade_size > 0:
                window[-fade_size:] = np.hamming(fade_size * 2)[fade_size:]

            print(
                f"Applied windowing to impulse response. Window shape: {window.shape}"
            )

            h_ir_final = h_ir_cut * window

            h_ir_final = h_ir_final[np.newaxis, :]

            out = signal.oaconvolve(y, h_ir_final, mode="full")

            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "minimal_phase output contains non-finite values"
            )
            end = __import__("time").perf_counter()
            self._profile("minimal_phase", end - start)

            print(
                f"Applied minimal phase correction. Output shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
            )
            print(
                f"Correction curve stats - min: {correction_db.min():.2f} dB, max: {correction_db.max():.2f} dB, mean: {correction_db.mean():.2f} dB"
            )
            print(
                f"Impulse response stats - min: {h_ir_final.min():.6f}, max: {h_ir_final.max():.6f}, mean: {h_ir_final.mean():.6f}"
            )

        elif tp == "linear":
            H_full = np.concatenate([H_lin, H_lin[-2:0:-1]])

            h = np.real(np.fft.ifft(H_full))

            print(
                f"Calculated linear phase impulse response. Length: {len(h)}, dtype: {h.dtype}, max: {np.max(np.abs(h)):.6f}"
            )

            h_centered = np.fft.fftshift(h)
            FIR_LEN = min(len(h), 8192)

            print(
                f"Centered impulse response. Length: {len(h_centered)}, dtype: {h_centered.dtype}, max: {np.max(np.abs(h_centered)):.6f}"
            )

            center = len(h_centered) // 2
            half_len = FIR_LEN // 2
            h_final = h_centered[center - half_len : center + half_len]

            print(
                f"Extracted central portion of impulse response. Length: {len(h_final)}, dtype: {h_final.dtype}, max: {np.max(np.abs(h_final)):.6f}"
            )

            h_final *= signal.windows.hamming(len(h_final))
            h_final = h_final[np.newaxis, :]

            out = signal.oaconvolve(y, h_final, mode="same")
            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "linear_phase output contains non-finite values"
            )
            end = __import__("time").perf_counter()
            self._profile("linear_phase", end - start)

            print(
                f"Applied linear phase correction. Output shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
            )
            print(
                f"Correction curve stats - min: {correction_db.min():.2f} dB, max: {correction_db.max():.2f} dB, mean: {correction_db.mean():.2f} dB"
            )
            print(
                f"Impulse response stats - min: {h_final.min():.6f}, max: {h_final.max():.6f}, mean: {h_final.mean():.6f}"
            )

        else:
            raise ValueError(f"Unknown phase type: {tp}")

        if out is None:
            raise RuntimeError("Phase correction failed to produce output")

        print(
            f"Output before final length adjustment. Shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
        )

        out = (
            out[..., :orig_len]
            if out.shape[-1] > orig_len
            else np.pad(
                out,
                (*((0, 0),) * (out.ndim - 1), (0, orig_len - out.shape[-1])),
            )
        )

        print(
            f"Final output after phase correction. Shape: {out.shape}, dtype: {out.dtype}, max: {np.max(np.abs(out)):.6f}"
        )

        return out

    def multiband_compress(self, y: np.ndarray) -> np.ndarray:

        if hasattr(self, "_profile"):
            start = __import__("time").perf_counter()

        def lr4(x, fc):

            if x.shape[-1] <= 9:
                return x, x

            sr2 = self.sr / 2.0
            sos_l = signal.butter(2, fc / sr2, btype="low", output="sos")
            sos_h = signal.butter(2, fc / sr2, btype="high", output="sos")
            lp = signal.sosfiltfilt(sos_l, x)
            hp = signal.sosfiltfilt(sos_h, x)

            print(
                f"Applied LR4 crossover at {fc:.1f} Hz. LP max: {np.max(np.abs(lp)):.6f}, HP max: {np.max(np.abs(hp)):.6f}"
            )
            print(
                f"LP stats - min: {lp.min():.6f}, max: {lp.max():.6f}, mean: {lp.mean():.6f}"
            )
            print(
                f"HP stats - min: {hp.min():.6f}, max: {hp.max():.6f}, mean: {hp.mean():.6f}"
            )

            return lp, hp

        def compress(
            x,
            threshold_db,
            ratio,
            attack_ms,
            release_ms,
            makeup_db,
            knee_db=3.0,
        ):
            ac = np.exp(-1.0 / (self.sr * attack_ms / 1000.0 + 1e-9))
            rc = np.exp(-1.0 / (self.sr * release_ms / 1000.0 + 1e-9))
            mk = 10.0 ** (makeup_db / 20.0)
            env = decoupled_envelope(20.0 * np.log10(np.abs(x) + 1e-12), ac, rc)
            hk = knee_db / 2.0
            gain = np.zeros_like(env)
            above = env > threshold_db + hk
            in_kn = (env > threshold_db - hk) & ~above
            gain[above] = -(env[above] - threshold_db) * (1.0 - 1.0 / ratio)
            ki = env[in_kn] - threshold_db + hk
            gain[in_kn] = -(ki**2) / (2.0 * knee_db) * (1.0 - 1.0 / ratio)

            print(
                f"Compressor applied. Threshold: {threshold_db:.2f} dB, Ratio: {ratio:.2f}, Attack: {attack_ms:.2f} ms, Release: {release_ms:.2f} ms, Makeup: {makeup_db:.2f} dB, Knee: {knee_db:.2f} dB"
            )
            print(
                f"Envelope stats - min: {env.min():.2f} dB, max: {env.max():.2f} dB, mean: {env.mean():.2f} dB"
            )
            print(
                f"Gain stats - min: {gain.min():.2f} dB, max: {gain.max():.2f} dB, mean: {gain.mean():.2f} dB"
            )
            print(
                f"Input signal stats - min: {x.min():.6f}, max: {x.max():.6f}, mean: {x.mean():.6f}"
            )
            print(
                f"Output signal stats - min: {(x * 10.0 ** (gain / 20.0) * mk).min():.6f}, max: {(x * 10.0 ** (gain / 20.0) * mk).max():.6f}, mean: {(x * 10.0 ** (gain / 20.0) * mk).mean():.6f}"
            )

            return x * 10.0 ** (gain / 20.0) * mk

        def split_bands(ch, fcs):
            bands = []
            current = ch
            for fc in fcs[:-1]:
                lo, current = lr4(current, fc)
                bands.append(lo)
            bands.append(current)
            return bands

        def process_ch(ch):

            fcs = [b.fc for b in self.bands if b.fc > 0]
            if not fcs:
                return ch

            sorted_bands = sorted(self.bands, key=lambda b: b.fc)
            fcs_sorted = [b.fc for b in sorted_bands if b.fc > 0]
            signal_parts = split_bands(ch, fcs_sorted)

            comps: list[np.ndarray] = []
            for band_cfg, sig in zip(sorted_bands, signal_parts, strict=False):
                thr = band_cfg.base_threshold + band_cfg.makeup_db
                comps.append(
                    compress(
                        sig,
                        thr,
                        band_cfg.ratio,
                        band_cfg.attack_ms,
                        band_cfg.release_ms,
                        band_cfg.makeup_db,
                        knee_db=band_cfg.knee_db,
                    )
                )
            print(
                f"Processed multiband compression for one channel. Number of bands: {len(comps)}"
            )
            print(
                f"Band thresholds: {[b.base_threshold for b in sorted_bands]}"
            )
            print(f"Band ratios: {[b.ratio for b in sorted_bands]}")
            print(
                f"Band attack times: {[b.attack_ms for b in sorted_bands]} ms"
            )
            print(
                f"Band release times: {[b.release_ms for b in sorted_bands]} ms"
            )
            print(
                f"Band makeup gains: {[b.makeup_db for b in sorted_bands]} dB"
            )
            print(f"Band knee values: {[b.knee_db for b in sorted_bands]} dB")
            print(
                f"First band stats - min: {comps[0].min():.6f}, max: {comps[0].max():.6f}, mean: {comps[0].mean():.6f}"
            )
            print(
                f"Last band stats - min: {comps[-1].min():.6f}, max: {comps[-1].max():.6f}, mean: {comps[-1].mean():.6f}"
            )
            print(
                f"Sum of bands stats - min: {np.sum(comps, axis=0).min():.6f}, max: {np.sum(comps, axis=0).max():.6f}, mean: {np.sum(comps, axis=0).mean():.6f}"
            )
            print(
                f"Original channel stats - min: {ch.min():.6f}, max: {ch.max():.6f}, mean: {ch.mean():.6f}"
            )

            return sum(comps)

        if y.ndim > 1:
            out = np.stack([process_ch(ch) for ch in y])
        else:
            out = process_ch(y)
        if hasattr(self, "_profile"):
            end = __import__("time").perf_counter()
            self._profile("multiband_compress", end - start)
        return out

    def apply_stereo_widening(
        self, y: np.ndarray, width: float = 1.5
    ) -> np.ndarray:
        if y.ndim < 2 or y.shape[0] != 2:
            return y
        coef = np.sqrt(2.0 / (1.0 + width**2))
        mid = (y[0] + y[1]) * 0.5
        side = (y[0] - y[1]) * 0.5 * width

        print(
            f"Applying stereo widening. Width: {width:.2f}, Coefficient: {coef:.4f}"
        )
        print(
            f"Mid signal stats - min: {mid.min():.6f}, max: {mid.max():.6f}, mean: {mid.mean():.6f}"
        )
        print(
            f"Side signal stats - min: {side.min():.6f}, max: {side.max():.6f}, mean: {side.mean():.6f}"
        )
        print(
            f"Output left channel stats - min: {(mid + side).min():.6f}, max: {(mid + side).max():.6f}, mean: {(mid + side).mean():.6f}"
        )
        print(
            f"Output right channel stats - min: {(mid - side).min():.6f}, max: {(mid - side).max():.6f}, mean: {(mid - side).mean():.6f}"
        )
        print(
            f"Original left channel stats - min: {y[0].min():.6f}, max: {y[0].max():.6f}, mean: {y[0].mean():.6f}"
        )
        print(
            f"Original right channel stats - min: {y[1].min():.6f}, max: {y[1].max():.6f}, mean: {y[1].mean():.6f}"
        )
        print(f"Width factor applied to side signal: {width:.2f}")
        print(f"Energy of mid signal before widening: {np.sum(mid**2):.6f}")
        print(f"Energy of side signal before widening: {np.sum(side**2):.6f}")

        return np.stack([(mid + side) * coef, (mid - side) * coef])

    def apply_limiter(
        self,
        y: np.ndarray,
        drive_db: float,
        ceil_db: float,
        os_factor: int = 2,
        lookahead_ms: float = 4.0,
        attack_ms: float = 0.1,
        release_ms_min: float = 30.0,
        release_ms_max: float = 70.0,
        soft_clip_ratio: float = 0.5,
        up_beta: float = 12.0,
        down_beta: float = 18.0,
    ) -> np.ndarray:

        if hasattr(self, "_profile"):
            start = __import__("time").perf_counter()

        orig_len = y.shape[-1]
        limit_lin = 10.0 ** (ceil_db / 20.0)
        drive_lin = 10.0 ** (drive_db / 20.0)
        sr_os = self.sr * os_factor

        y_os = signal.resample_poly(
            y, os_factor, 1, axis=-1, window=("kaiser", up_beta)
        )
        y_driven = y_os * drive_lin

        y_env = (
            np.max(np.abs(y_driven), axis=0)
            if y_driven.ndim > 1
            else np.abs(y_driven)
        )
        lookahead_samp = int(lookahead_ms * sr_os / 1000.0)
        peak_env = maximum_filter1d(y_env, size=lookahead_samp)

        rms_win = int(sr_os * 0.05)
        rms_env = np.sqrt(maximum_filter1d(y_env**2, size=rms_win))
        crest = peak_env / (rms_env + 1e-12)

        rel_ms = np.clip(
            release_ms_max / (crest + 1e-12), release_ms_min, release_ms_max
        )
        ac = np.exp(-1.0 / (sr_os * attack_ms / 1000.0))
        rc = np.exp(-1.0 / (sr_os * rel_ms / 1000.0))

        peak_smooth = limiter_smooth_env(peak_env, ac, rc)

        gain = np.where(
            peak_smooth > limit_lin, limit_lin / (peak_smooth + 1e-12), 1.0
        )

        y_delayed = np.roll(y_driven, lookahead_samp, axis=-1)
        if y_delayed.ndim > 1:
            y_delayed[:, :lookahead_samp] = 0
        else:
            y_delayed[:lookahead_samp] = 0

        y_lim = y_delayed * gain
        y_sat = (
            y_lim * (1.0 - soft_clip_ratio)
            + np.tanh(y_lim / limit_lin) * limit_lin * soft_clip_ratio
        )

        y_down = signal.resample_poly(
            y_sat, 1, os_factor, axis=-1, window=("kaiser", down_beta)
        )

        delay_samp = int(lookahead_ms * self.sr / 1000.0)
        out = y_down[..., delay_samp : delay_samp + orig_len]

        out -= (
            np.mean(out, axis=-1, keepdims=True)
            if out.ndim > 1
            else np.mean(out)
        )

        out = np.clip(out, -limit_lin, limit_lin)

        if hasattr(self, "_profile"):
            self._profile("limiter", __import__("time").perf_counter() - start)

        print(
            f"Applied limiter. Drive: {drive_db} dB, Ceiling: {ceil_db} dB, Lookahead: {lookahead_ms} ms, Attack: {attack_ms} ms, Release: {release_ms_min}-{release_ms_max} ms, Soft clip ratio: {soft_clip_ratio}, Up beta: {up_beta}, Down beta: {down_beta}"
        )
        print(
            f"Limiter input stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )
        print(
            f"Limiter output stats - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}"
        )
        print(
            f"Limiter gain stats - min: {10.0 * np.log10(gain.min() + 1e-12):.2f} dB, max: {10.0 * np.log10(gain.max() + 1e-12):.2f} dB, mean: {10.0 * np.log10(gain.mean() + 1e-12):.2f} dB"
        )
        print(
            f"Limiter peak envelope stats - min: {peak_env.min():.6f}, max: {peak_env.max():.6f}, mean: {peak_env.mean():.6f}"
        )
        print(
            f"Limiter RMS envelope stats - min: {rms_env.min():.6f}, max: {rms_env.max():.6f}, mean: {rms_env.mean():.6f}"
        )
        print(
            f"Limiter crest stats - min: {crest.min():.6f}, max: {crest.max():.6f}, mean: {crest.mean():.6f}"
        )
        print(
            f"Limiter smoothed peak envelope stats - min: {peak_smooth.min():.6f}, max: {peak_smooth.max():.6f}, mean: {peak_smooth.mean():.6f}"
        )
        print(
            f"Limiter delayed signal stats - min: {y_delayed.min():.6f}, max: {y_delayed.max():.6f}, mean: {y_delayed.mean():.6f}"
        )
        print(
            f"Limiter driven signal stats - min: {y_driven.min():.6f}, max: {y_driven.max():.6f}, mean: {y_driven.mean():.6f}"
        )

        return out

    def apply_rms(self, y: np.ndarray, rms: float):
        y *= rms / (
            (
                np.max(np.sqrt(np.mean(y**2, axis=1)))
                if y.ndim > 1
                else np.sqrt(np.mean(y**2))
            )
            + 1e-12
        )

        print(
            f"Applied RMS normalization. Target RMS: {rms:.6f}, Output stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )
        print(
            f"Original signal stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )

        return y

    def apply_lufs(self, y: np.ndarray, target_lufs: float) -> np.ndarray:
        if hasattr(self, "_profile"):
            start = __import__("time").perf_counter()

        b1, a1 = (
            [1.5309096471, -2.6511690392, 1.1691662955],
            [1.0, -1.6906592931, 0.7324805897],
        )
        b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.9900474541, 0.9900722501]

        y_filt = signal.lfilter(b1, a1, y, axis=-1)
        y_filt = signal.lfilter(b2, a2, y_filt, axis=-1)

        win_size = int(self.sr * 0.4)
        hop_size = int(self.sr * 0.1)

        if y_filt.ndim > 1:
            y_sq = np.sum(y_filt**2, axis=0)
            n_samples = y_filt.shape[1]
        else:
            y_sq = y_filt**2
            n_samples = len(y_sq)

        shape = ((n_samples - win_size) // hop_size + 1, win_size)
        strides = (y_sq.strides[0] * hop_size, y_sq.strides[0])
        y_strided = np.lib.stride_tricks.as_strided(
            y_sq, shape=shape, strides=strides
        )
        ms = np.mean(y_strided, axis=-1)

        abs_threshold = 10.0 ** (-70.0 / 10.0)
        ms_gated = ms[ms > abs_threshold]

        if len(ms_gated) == 0:
            return y

        gamma_rel = 0.1 * np.mean(ms_gated)
        ms_final = ms_gated[ms_gated > gamma_rel]

        loudness_ms = (
            np.mean(ms_final) if len(ms_final) > 0 else np.mean(ms_gated)
        )

        current_lufs = -0.691 + 10.0 * np.log10(np.maximum(loudness_ms, 1e-12))

        gain_lin = 10.0 ** ((target_lufs - current_lufs) / 20.0)
        out = y * gain_lin

        if hasattr(self, "_profile"):
            self._profile("lufs", __import__("time").perf_counter() - start)

        print(
            f"Applied LUFS normalization. Target LUFS: {target_lufs:.2f}, Current LUFS: {current_lufs:.2f}, Gain applied: {gain_lin:.4f} ({20.0 * np.log10(gain_lin):.2f} dB)"
        )
        print(
            f"Input signal stats - min: {y.min():.6f}, max: {y.max():.6f}, mean: {y.mean():.6f}"
        )
        print(
            f"Output signal stats - min: {out.min():.6f}, max: {out.max():.6f}, mean: {out.mean():.6f}"
        )
        print(
            f"Measured loudness stats - min: {loudness_ms:.6f}, max: {loudness_ms:.6f}, mean: {loudness_ms:.6f}"
        )
        print(
            f"Number of frames used for loudness measurement: {len(ms_final)}"
        )
        print(
            f"Absolute threshold for gating: {abs_threshold:.6e}, Relative threshold for gating: {gamma_rel:.6e}"
        )
        print(
            f"LUFS normalization applied with biquad filters. Filter 1 - b: {b1}, a: {a1}, Filter 2 - b: {b2}, a: {a2}"
        )

        return out

    def process(self, y: np.ndarray, sr: int = None) -> np.ndarray:
        orig_len = y.shape[-1]

        print(
            f"Starting processing with input shape: {y.shape}, dtype: {y.dtype}, sample rate: {sr if sr is not None else self.sr} Hz"
        )

        if sr is not None:
            self.sr = sr

        y_resamp = y

        if self.sr != self.resampling_target:
            print(
                f"Resampling to {self.resampling_target} Hz for processing..."
            )

            y_resamp = resample(y_resamp, self.sr, self.resampling_target)
            self.sr = self.resampling_target

        print(
            f"Resampled shape: {y_resamp.shape}, dtype: {y_resamp.dtype}, sample rate: {self.sr} Hz"
        )

        y_stereo = stereo(y_resamp)

        y_exciter = apply_exciter(y_stereo, self.sr)

        print(
            f"Applied exciter. Shape: {y_exciter.shape}, dtype: {y_exciter.dtype}, sample rate: {self.sr} Hz"
        )

        y_mbc = self.multiband_compress(y_exciter)

        print(
            f"Applied multiband compression. Shape: {y_mbc.shape}, dtype: {y_mbc.dtype}, sample rate: {self.sr} Hz"
        )

        self.update_bands()

        print(
            f"Updated multiband compressor bands based on current slope settings. Bands: {[(b.fc, b.base_threshold, b.ratio) for b in self.bands]}"
        )

        p = self.apply_phase_correction(y_mbc, self.phase_type)

        y_stereo2 = stereo(p)

        y_widen = self.apply_stereo_widening(y_stereo2)

        print(
            f"Applied stereo widening. Shape: {y_widen.shape}, dtype: {y_widen.dtype}, sample rate: {self.sr} Hz"
        )

        y_cut = freq_cut(
            y_widen, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        y_lufs = self.apply_lufs(y_cut, self.target_lufs)

        print(
            f"Applied LUFS normalization. Shape: {y_lufs.shape}, dtype: {y_lufs.dtype}, sample rate: {self.sr} Hz"
        )

        y_limiter = self.apply_limiter(
            y_lufs, drive_db=self.drive_db, ceil_db=self.ceil_db
        )

        print(
            f"Applied limiter with drive {self.drive_db} dB and ceiling {self.ceil_db} dB. Shape: {y_limiter.shape}, dtype: {y_limiter.dtype}, sample rate: {self.sr} Hz"
        )

        y_lufs2 = self.apply_lufs(y_limiter, self.target_lufs)

        y_cut2 = freq_cut(
            y_lufs2, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        lin_amp = 10 ** (self.ceil_db / 20.0)

        print(
            f"Applied final LUFS normalization and frequency cut. Shape: {y_cut2.shape}, dtype: {y_cut2.dtype}, sample rate: {self.sr} Hz"
        )

        y_clip = np.clip(y_cut2, -lin_amp, lin_amp)

        print(
            f"Applied final clipping to ceiling of {self.ceil_db} dB. Shape: {y_clip.shape}, dtype: {y_clip.dtype}, sample rate: {self.sr} Hz"
        )

        if y_clip.shape[-1] != orig_len:
            if y_clip.shape[-1] > orig_len:
                print(
                    f"Output is longer than original by {y_clip.shape[-1] - orig_len} samples, trimming to original length."
                )
                y_clip = y_clip[..., :orig_len]
            else:
                print(
                    f"Output is shorter than original by {orig_len - y_clip.shape[-1]} samples, padding with zeros to original length."
                )
                y_clip = np.pad(
                    y_clip,
                    (
                        *((0, 0),) * (y_clip.ndim - 1),
                        (0, orig_len - y_clip.shape[-1]),
                    ),
                )

        print(f"Final output peak: {np.max(np.abs(y_clip)):.6f}")
        print(
            f"Final output spectrum (dB): {10.0 * np.log10(np.maximum(np.mean(y_clip**2, axis=-1), 1e-12))}"
        )
        print(f"Final output shape: {y_clip.shape}, dtype: {y_clip.dtype}")
        print(
            f"Limiter settings - Drive: {self.drive_db} dB, Ceiling: {self.ceil_db} dB"
        )
        print(f"Sample rate: {self.sr} Hz, Original length: {orig_len} samples")
        print(
            f"Multiband compression settings: {[(b.fc, b.base_threshold, b.ratio, b.attack_ms, b.release_ms, b.makeup_db) for b in self.bands]}"
        )
        print(
            f"Phase correction settings: Slope dB/octave: {self.slope_db}, Slope reference frequency: {self.slope_hz} Hz, Smoothing fraction: {self.smoothing_fraction}"
        )
        print(
            f"Frequency cut settings: Low cut: {self.low_cut} Hz, High cut: {self.high_cut} Hz"
        )
        print(f"Target LUFS: {self.target_lufs} LUFS")
        print(
            f"Output stats - Mean: {np.mean(y_clip):.6e}, Std: {np.std(y_clip):.6e}, Max: {np.max(y_clip):.6f}, Min: {np.min(y_clip):.6f}"
        )
        print(
            f"Output clipping - Samples at ceiling: {(y_clip >= lin_amp).sum()}, Samples at floor: {(y_clip <= -lin_amp).sum()}"
        )
        print(
            f"Output headroom: {10.0 * np.log10(lin_amp / (np.max(np.abs(y_clip)) + 1e-12)):.2f} dB"
        )
        print(
            f"Output spectrum after final processing (dB): {10.0 * np.log10(np.maximum(np.mean(y_clip**2, axis=-1), 1e-12))}"
        )
        print(f"Output shape: {y_clip.shape}, dtype: {y_clip.dtype}")
        print("Processing steps completed successfully without errors.")

        return y_clip, self.sr


def generate_bands(
    start_freq: float, end_freq: float, num_bands: int
) -> list[float]:
    if num_bands < 2:
        return [start_freq]

    bands = []
    factor = (end_freq / start_freq) ** (1 / (num_bands - 1))

    for i in range(num_bands):
        freq = start_freq * (factor**i)
        bands.append(freq)

    return bands


def get_audio_duration(file_path: str) -> float | None:
    try:
        sr, audio = read_audio(file_path)
        return len(audio) / sr
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None


def audio_preview(file_path: str, max_duration: float = 30) -> str | None:
    file_path = full_path(file_path)
    if not exist(file_path):
        catch(f"Error: Audio file not found at {file_path}")
        return None
    if max_duration <= 0:
        catch("Error: max_duration must be positive.")
        return None
    try:
        total_duration = get_audio_duration(file_path)
        if total_duration is None:
            catch(f"Error: Could not get duration for {file_path}")
            return None
        log("Total audio duration", f"{total_duration:.2f} seconds")
        if total_duration <= max_duration:
            log("Audio duration <= max_duration", "Returning copy of original.")
            preview_paths = split_audio(
                file_path, chunk_duration=total_duration, chunks_limit=1, skip_time=0
            )
            return preview_paths[0] if preview_paths else None
        start_time = 0.0
        timeline = get_active_audio_timeline(
            file_path, threshold_db=-25, min_silence_len=0.5
        )
        if timeline:
            longest_segment_duration = 0.0
            longest_segment_center = 0.0
            for start, end in timeline:
                duration = end - start
                if duration > longest_segment_duration:
                    longest_segment_duration = duration
                    longest_segment_center = start + duration / 2.0
            log(
                "Longest active segment",
                f"Duration: {longest_segment_duration:.2f}s, Center: {longest_segment_center:.2f}s",
            )
            ideal_start = longest_segment_center - max_duration / 2.0
            start_time = max(0.0, ideal_start)
            start_time = min(start_time, total_duration - max_duration)
            log("Calculated preview start time", f"{start_time:.2f} seconds")
        else:
            start_time = min(
                total_duration * 0.1, total_duration - max_duration
            )
            start_time = max(0.0, start_time)
            log(
                "No significant active segments found",
                f"Defaulting preview start time to {start_time:.2f} seconds",
            )
        log(
            "Extracting preview chunk",
            f"Start: {start_time:.2f}s, Duration: {max_duration:.2f}s",
        )
        preview_paths = split_audio(
            file_path, chunk_duration=max_duration, chunks_limit=1, skip_time=start_time
        )
        if preview_paths:
            log("Preview extraction successful", preview_paths[0])
            return preview_paths[0]
        else:
            catch(
                "Error: split_audio did not return any paths for the preview."
            )
            return None
    except Exception as e:
        catch(f"An unexpected error occurred in audio_preview: {e}")
        return None


def extract_audio_features(file_path, n_mfcc=20):
    import librosa

    try:
        (y, sr) = librosa.load(file_path, sr=None)
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None
    try:
        mfccs = librosa.feature.mfcc(
            y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, n_mels=80
        ).flatten()
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=sr
        ).flatten()
        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y, sr=sr
        ).flatten()
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=sr
        ).flatten()
        spectral_features = _np.concatenate(
            (spectral_centroid, spectral_bandwidth, spectral_rolloff)
        )
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).flatten()
        chroma = librosa.feature.chroma_stft(y=y, sr=sr).flatten()
        all_features = _np.concatenate(
            (mfccs, spectral_features, zero_crossing_rate, chroma)
        ).astype(_np.float32)
        return all_features
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None
    except Exception as e:
        catch(e)
        return None


def features_to_audio(
    predicted_features,
    sr=32000,
    n_mfcc=20,
    n_mels=80,
    n_fft=2048,
    hop_length=512,
):
    import librosa

    expected_freq_bins = n_fft // 2 + 1
    try:
        predicted_features = _np.asarray(predicted_features)
        remainder = predicted_features.size % n_mfcc
        if remainder != 0:
            padding_needed = n_mfcc - remainder
            print(
                f"Padding with {padding_needed} zeros to make the predicted features ({predicted_features.size}) a multiple of n_mfcc ({n_mfcc})."
            )
            predicted_features = _np.pad(
                predicted_features,
                (0, padding_needed),
                mode="constant",
                constant_values=0,
            )
        mfccs = predicted_features.reshape((n_mfcc, -1))
        if mfccs.shape[1] == 0:
            print(
                "Error: Reshaped MFCCs have zero frames. Cannot proceed with audio reconstruction."
            )
            return None
        mel_spectrogram_db = librosa.feature.inverse.mfcc_to_mel(
            mfccs, n_mels=n_mels
        )
        mel_spectrogram = librosa.db_to_amplitude(mel_spectrogram_db)
        mel_spectrogram = _np.nan_to_num(
            mel_spectrogram,
            nan=0.0,
            posinf=_np.finfo(_np.float16).max,
            neginf=_np.finfo(_np.float16).min,
        )
        mel_spectrogram = _np.maximum(0, mel_spectrogram)
        magnitude_spectrogram = librosa.feature.inverse.mel_to_stft(
            M=mel_spectrogram, sr=sr, n_fft=n_fft
        )
        magnitude_spectrogram = _np.nan_to_num(
            magnitude_spectrogram,
            nan=0.0,
            posinf=_np.finfo(_np.float16).max,
            neginf=_np.finfo(_np.float16).min,
        )
        magnitude_spectrogram = _np.maximum(0, magnitude_spectrogram)
        magnitude_spectrogram = _np.nan_to_num(
            magnitude_spectrogram,
            nan=0.0,
            posinf=_np.finfo(_np.float16).max,
            neginf=_np.finfo(_np.float16).min,
        )
        if magnitude_spectrogram.shape[0] != expected_freq_bins:
            print(
                f"Error: Magnitude spectrogram has incorrect frequency bin count ({magnitude_spectrogram.shape[0]}) for n_fft ({n_fft}).\nExpected {expected_freq_bins}.\nCannot perform Griffin-Lim."
            )
            return None
        if magnitude_spectrogram.shape[1] == 0:
            print(
                "Error: Magnitude spectrogram has zero frames. Skipping Griffin-Lim."
            )
            return None
        griffin_lim_iterations = [12, 32]
        for n_iter in griffin_lim_iterations:
            try:
                audio_waveform = librosa.griffinlim(
                    magnitude_spectrogram,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    n_iter=n_iter,
                )
                if audio_waveform.size > 0:
                    print(f"Griffin-Lim finished {n_iter} iterations")
                    audio_waveform = _np.nan_to_num(
                        audio_waveform,
                        nan=0.0,
                        posinf=_np.finfo(_np.float16).max,
                        neginf=_np.finfo(_np.float16).min,
                    )
                    audio_waveform = _np.clip(audio_waveform, -1.0, 1.0)
                    if not _np.all(_np.isfinite(audio_waveform)):
                        print(
                            "Warning: Audio waveform contains non-finite values after clipping.\nThis is unexpected.\nReturning None."
                        )
                        return None
                    return audio_waveform
                else:
                    print(
                        f"Griffin-Lim with n_iter={n_iter} produced an empty output."
                    )
            except Exception as e:
                print(f"Griffin-Lim with n_iter={n_iter} failed!")
                catch(e)
                if n_iter == griffin_lim_iterations[-1]:
                    print("Griffin-Lim failed. Returning None.")
                    return None
                else:
                    print("Trying again with more iterations...")
        return None
    except Exception as e:
        catch(e)
        return None


def predict_audio(model, audio_file):
    import os

    import librosa
    import soundfile as sf

    import definers as _d

    audio_file = full_path(audio_file)

    if not os.path.exists(audio_file):
        return None

    try:
        (audio_data, sr) = librosa.load(audio_file, sr=32000, mono=True)
        timeline = _d.get_active_audio_timeline(audio_file)
        log("Audio shape", audio_data.shape)
        log("Active audio timeline", timeline)
        predicted_audio = _np.zeros_like(audio_data)
        if not timeline:
            log("Silent timeline", "No active audio segments found.")
        for i, (start_time, end_time) in enumerate(timeline):
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            start_sample = max(0, start_sample)
            end_sample = min(len(audio_data), end_sample)
            active_audio_part_np = audio_data[start_sample:end_sample]
            if active_audio_part_np.size == 0:
                log(
                    "Segment skipped",
                    f"Skipping empty audio segment from {start_time:.2f}s to {end_time:.2f}s",
                )
                continue
            active_audio_part_model_input = numpy_to_cupy(active_audio_part_np)
            log(
                "Predicting segment",
                f"Predicting audio segment {i + 1}/{len(timeline)} with shape {active_audio_part_model_input.shape}",
            )
            prediction = model.predict(active_audio_part_model_input)
            if _d.is_clusters_model(model):
                log(
                    "Getting prediction cluster content",
                    f"Predicted cluster for segment {i + 1}: {int(prediction[0])}",
                )
                part_feat = cupy_to_numpy(
                    _d.get_cluster_content(model, int(prediction[0]))
                )
            else:
                part_feat = cupy_to_numpy(prediction)
            log(
                "Prediction shape",
                f"Predicted features shape for segment {i + 1}: {part_feat.shape}",
            )
            part_aud = _d.features_to_audio(part_feat)
            if part_aud is None:
                log(
                    "Segment failure",
                    f"Failed to convert features to audio for segment {i + 1}. Skipping this segment.",
                )
                continue
            part_length = end_sample - start_sample
            min_len = min(part_aud.shape[0], part_length)
            predicted_audio[start_sample : start_sample + min_len] = part_aud[
                :min_len
            ]
        output_file = _d.tmp("wav")
        sf.write(output_file, predicted_audio, sr)
        log("Audio output", f"Predicted audio saved to: {output_file}")
        return output_file
    except Exception as e:
        catch(e)
        return None


def save_audio(
    destination_path: str,
    audio_signal: np.ndarray,
    sample_rate: int = 44100,
    output_format: str = "mp3",
    *,
    audio_bit_depth: int = 32,
    audio_bitrate: int = 320,
) -> None:

    import lameenc
    import soundfile as sf

    ceiling_db = -0.1
    lin_amp = 10 ** (ceiling_db / 20.0)

    audio_signal = np.clip(audio_signal * lin_amp, -lin_amp, lin_amp)

    base_path = os.path.splitext(destination_path)[0]

    if output_format.lower() == "mp3":
        final_path = f"{base_path}.mp3"

        INT16_MAX = np.iinfo(np.int16).max
        y_scaled = (audio_signal * INT16_MAX).astype(np.int16)

        if y_scaled.ndim == 1:
            channels = 1
            y_interleaved = y_scaled
        else:
            y_interleaved = np.ascontiguousarray(
                y_scaled.T
                if y_scaled.shape[0] < y_scaled.shape[1]
                else y_scaled
            )
            channels = y_interleaved.shape[1]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(audio_bitrate)
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(channels)
        encoder.set_quality(2)

        mp3_data = encoder.encode(y_interleaved.tobytes())
        mp3_data += encoder.flush()

        with open(final_path, "wb") as f:
            f.write(mp3_data)
        return

    final_path = f"{base_path}.wav"
    if audio_bit_depth == 16:
        INT16_MIN = np.iinfo(np.int16).min
        INT16_MAX = np.iinfo(np.int16).max
        y_scaled = audio_signal * float(INT16_MAX)

        tpdf = (
            np.random.uniform(-1.0, 1.0, y_scaled.shape)
            + np.random.uniform(-1.0, 1.0, y_scaled.shape)
        ) * 0.5

        y_dithered = y_scaled + tpdf
        audio_signal = np.clip(np.rint(y_dithered), INT16_MIN, INT16_MAX).astype(np.int16)

    elif audio_bit_depth == 32:
        audio_signal = audio_signal.astype(np.float32)

    elif audio_bit_depth == 64:
        audio_signal = audio_signal.astype(np.float64)

    else:
        raise ValueError(f"Unsupported bit depth: {audio_bit_depth}. Use 16, 32, or 64.")

    if audio_signal.ndim == 2 and audio_signal.shape[0] < audio_signal.shape[1]:
        audio_signal = audio_signal.T

    subtype_map = {16: "PCM_16", 32: "FLOAT", 64: "DOUBLE"}
    sf.write(destination_path, audio_signal, sample_rate, subtype=subtype_map[audio_bit_depth], format="RF64")


def master(
    audio_file_path: str,
    audio_format: str = "mp3"
) -> str | None:
    try:
        sr, y = read_audio(audio_file_path)
        y_mastered = SmartMastering().process(y, sr)
        output_path = tmp(audio_format, keep=False)
        save_audio(destination_path=output_path, audio_signal=y_mastered, sample_rate=sr, output_format=audio_format)
        return output_path
    except Exception as e:
        catch(e)
        return None


def split_audio(
    audio_file_path: str,
    chunk_duration: float = 5.0,
    audio_format: str = "mp3",
    *,
    chunks_limit: int | None = None,
    skip_time: float = 0.0,
    target_sample_rate: int | None = None,
    output_folder: str | None = None,
) -> list[str]:
    audio_file_path = full_path(audio_file_path)
    if not exist(audio_file_path):
        return []

    try:
        sr, audio = read_audio(audio_file_path)
    except Exception:
        catch(f"Failed to load audio file: {audio_file_path}")
        return []

    total_ms = len(audio)
    skip_ms = int(skip_time * 1000.0)
    if skip_ms >= total_ms:
        return []

    duration_ms = int(chunk_duration * 1000.0)
    if duration_ms <= 0:
        return []

    remaining_ms = total_ms - skip_ms
    max_chunks = math.ceil(remaining_ms / duration_ms)
    num_chunks = min(chunks_limit, max_chunks) if chunks_limit is not None else max_chunks

    if output_folder:
        output_folder = full_path(output_folder)
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = tmp(dir=True)

    res_paths: list[str] = []
    for i in range(num_chunks):
        start_ms = skip_ms + i * duration_ms
        if start_ms >= total_ms:
            break
        end_ms = min(start_ms + duration_ms, total_ms)
        chunk = audio[start_ms:end_ms]
        if target_sample_rate:
            chunk = chunk.set_frame_rate(target_sample_rate)
        chunk_path = full_path(output_folder, f"chunk_{i:04d}.{audio_format}")
        export_kwargs = {}
        if audio_format.lower() == "mp3":
            export_kwargs["bitrate"] = "192k"
        chunk.export(chunk_path, format=audio_format, **export_kwargs)
        res_paths.append(chunk_path)

    return res_paths


def remove_silence(input_file: str, output_file: str):
    import definers as _d

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_file,
                "-ac",
                "2",
                "-af",
                "silenceremove=stop_duration=0.1:stop_threshold=-32dB",
                output_file,
            ],
            check=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        _d.catch(e)


def compact_audio(input_file: str, output_file: str):
    import definers as _d

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                input_file,
                "-ar",
                "32000",
                "-ab",
                "96k",
                "-ac",
                "1",
                output_file,
            ],
            check=True,
        )
        return output_file
    except subprocess.CalledProcessError as e:
        _d.catch(e)


def read_audio(audio_file, normalized=False):
    import pydub

    audio_segment = pydub.AudioSegment.from_file(audio_file)
    samples = np.array(audio_segment.get_array_of_samples())
    if audio_segment.channels == 2:
        audio_data = samples.reshape((-1, 2)).T
    else:
        audio_data = samples.reshape((1, -1))
    if normalized:
        return (audio_segment.frame_rate, np.float32(audio_data) / 32768.0)
    else:
        return (audio_segment.frame_rate, audio_data)


def write_mp3(file_path, sr, audio_data):
    import pydub

    if audio_data.ndim == 1:
        channels = 1
    else:
        channels = audio_data.shape[0]
    y = np.int8(
        audio_data * 128.0 / 2 + 128.0 + (audio_data * 128.0 / 2 - 128.0)
    )
    interleaved_data = np.ascontiguousarray(y.T)
    song = pydub.AudioSegment(
        interleaved_data.tobytes(),
        frame_rate=sr,
        sample_width=1,
        channels=channels,
    )
    song.export(file_path, format="mp3", bitrate="320k")


def export_to_pkl(model, pkl_path):
    import pickle

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)


import numpy as np


def process_audio_chunks(fn, data, chunk_size, overlap=0):
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")
    data = data.astype(np.float32)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    (num_channels, audio_length) = data.shape
    step = chunk_size - overlap
    final_result = np.zeros_like(data, dtype=np.float32)
    window_sum = np.zeros_like(data, dtype=np.float32)
    window = np.hanning(chunk_size)
    window = window[np.newaxis, :]
    start = 0
    while start < audio_length:
        end = min(start + chunk_size, audio_length)
        current_chunk_size = end - start
        chunk = data[:, start:end]
        if current_chunk_size < chunk_size:
            padding_size = chunk_size - current_chunk_size
            chunk = np.pad(chunk, ((0, 0), (0, padding_size)), "constant")
        processed_chunk = fn(chunk)
        if processed_chunk.ndim == 1:
            processed_chunk = processed_chunk[np.newaxis, :]
        final_result[:, start:end] += (
            processed_chunk[:, :current_chunk_size]
            * window[:, :current_chunk_size]
        )
        window_sum[:, start:end] += window[:, :current_chunk_size] ** 2
        if end == audio_length:
            break
        start += step
    window_sum[window_sum == 0] = 1.0
    final_result /= window_sum
    return final_result


def get_active_audio_timeline(
    audio_file, threshold_db=-16, min_silence_len=0.1
):
    import librosa

    (audio_data, sample_rate) = librosa.load(audio_file, sr=32000)
    silence_mask = detect_silence_mask(
        audio_data, sample_rate, threshold_db, min_silence_len
    )
    active_regions = librosa.effects.split(
        _np.logical_not(silence_mask).astype(float),
        frame_length=1,
        hop_length=1,
    )
    timeline = [
        (start.item() / int(sample_rate), end.item() / int(sample_rate))
        for (start, end) in active_regions
    ]
    return timeline


def detect_silence_mask(
    audio_data, sample_rate, threshold_db=-16, min_silence_len=0.1
):
    import librosa

    threshold_amplitude = librosa.db_to_amplitude(threshold_db)
    frame_length = int(0.02 * sample_rate)
    hop_length = frame_length // 4
    rms = librosa.feature.rms(
        y=audio_data, frame_length=frame_length, hop_length=hop_length
    )[0]
    silence_mask_rms = rms < threshold_amplitude
    silence_mask = np.repeat(silence_mask_rms, hop_length)
    if len(silence_mask) > len(audio_data):
        silence_mask = silence_mask[: len(audio_data)]
    elif len(silence_mask) < len(audio_data):
        padding = np.ones(len(audio_data) - len(silence_mask), dtype=bool)
        silence_mask = np.concatenate((silence_mask, padding))
    min_silence_samples = int(min_silence_len * sample_rate)
    silence_mask_filtered = silence_mask.copy()
    silence_regions = librosa.effects.split(
        silence_mask.astype(float), top_db=0.5
    )
    for start, end in silence_regions:
        if end - start < min_silence_samples:
            silence_mask_filtered[start:end] = False
    return silence_mask_filtered


def create_share_links(hf_username, space_name, file_path, text_description):
    file_url = f"https://{hf_username}-{space_name}.hf.space/gradio_api/file={file_path}"
    encoded_text = quote(text_description)
    encoded_url = quote(file_url)
    twitter_link = f"https://twitter.com/intent/tweet?text={encoded_text}&url={encoded_url}"
    facebook_link = (
        f"https://www.facebook.com/sharer/sharer.php?u={encoded_url}"
    )
    reddit_link = (
        f"https://www.reddit.com/submit?url={encoded_url}&title={encoded_text}"
    )
    whatsapp_link = (
        f"https://api.whatsapp.com/send?text={encoded_text}%20{encoded_url}"
    )
    return f"<div style='text-align:center; padding-top: 10px;'><p style='font-weight: bold;'>Share your creation!</p><a href='{twitter_link}' target='_blank' style='margin: 0 5px;'>X/Twitter</a> | <a href='{facebook_link}' target='_blank' style='margin: 0 5px;'>Facebook</a> | <a href='{reddit_link}' target='_blank' style='margin: 0 5px;'>Reddit</a> | <a href='{whatsapp_link}' target='_blank' style='margin: 0 5px;'>WhatsApp</a></div>"


def humanize_vocals(audio_path, amount=0.5):
    import librosa
    import soundfile as sf

    if not exist(audio_path):
        catch(f"Error: Input file not found at {audio_path}")
        return None
    temp_dir = tmp(dir=True)
    try:
        print("--- Loading Audio ---")
        (y, sr) = librosa.load(audio_path, sr=None, mono=True)
        print("--- Analyzing Vocal Pitch ---")
        (n_fft, hop_length) = (2048, 1024)
        (f0, voiced_flag, _) = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0)
        print("--- Generating Humanized Pitch Map ---")
        target_f0 = np.copy(f0)
        if amount > 0:
            deviation_scale = amount * 5.0
            max_deviation_cents = 20.0
            for i in range(len(f0)):
                if voiced_flag[i] and f0[i] > 0:
                    cents_dev = np.random.normal(0, scale=deviation_scale)
                    cents_dev = np.clip(
                        cents_dev, -max_deviation_cents, max_deviation_cents
                    )
                    target_f0[i] = f0[i] * 2 ** (cents_dev / 1200)
        freq_map_path = os.path.join(temp_dir, "freqmap.txt")
        with open(freq_map_path, "w") as f:
            for i in range(len(f0)):
                if voiced_flag[i] and f0[i] > 0 and (target_f0[i] > 0):
                    sample_num = i * hop_length
                    ratio = target_f0[i] / f0[i]
                    f.write(f"{sample_num} {ratio:.6f}\n")
        if os.path.getsize(freq_map_path) > 0:
            print("--- Applying Pitch Variations with Rubberband ---")
            temp_input_wav = os.path.join(temp_dir, "input.wav")
            temp_output_wav = os.path.join(temp_dir, "output.wav")
            sf.write(temp_input_wav, y, sr)
            run(
                f"rubberband --formant --freqmap {freq_map_path} {temp_input_wav} {temp_output_wav}"
            )
            (y_humanized, _) = librosa.load(temp_output_wav, sr=sr)
            print("Humanization complete.")
        else:
            print("Warning: No voiced frames detected. Skipping processing.")
            y_humanized = np.copy(y)
        input_p = Path(audio_path)
        output_path = input_p.parent / f"{input_p.stem}_humanized.wav"
        sf.write(str(output_path), y_humanized, sr)
        print("\n--- Processing Complete ---")
        print(f"Humanized audio saved to: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"An error occurred during processing: {e}", file=sys.stderr)
        return None
    finally:
        print("Cleaning up temporary files.")
        delete(temp_dir)


def value_to_keys(dictionary, target_value):
    return [key for (key, value) in dictionary.items() if value == target_value]


def transcribe_audio(audio_path, language):
    from definers.ml import init_pretrained_model

    if MODELS["speech-recognition"] is None:
        init_pretrained_model("speech-recognition")
    audio_path = normalize_audio_to_peak(audio_path)
    (vocal, _) = separate_stems(audio_path)
    lang_code = value_to_keys(language_codes, language)[0]
    lang_code = lang_code.replace("iw", "he")
    return MODELS["speech-recognition"](
        vocal, generate_kwargs={"language": lang_code}, return_timestamps=True
    )["text"]


def generate_voice(text, reference_audio, format_choice):
    import pydub
    import soundfile as sf

    from definers.ml import init_pretrained_model

    if not MODELS["tts"]:
        init_pretrained_model("tts")
    try:
        temp_wav_path = tmp("wav", False)
        wav = MODELS["tts"].generate(
            text=text, audio_prompt_path=reference_audio
        )
        wav = normalize_audio_to_peak(wav)
        sf.write(temp_wav_path, wav, 24000)
        sound = pydub.AudioSegment.from_file(temp_wav_path)
        output_stem = tmp(keep=False).replace(".data", "")
        final_output_path = save_audio(
            destination_path=output_stem, audio_signal=sound, sample_rate=24000, output_format=format_choice
        )
        return final_output_path
    except Exception as e:
        catch(f"Generation failed: {e}")


def generate_music(prompt, duration_s, format_choice):
    import pydub
    from scipy.io.wavfile import write as write_wav

    inputs = PROCESSORS["music"](
        text=[prompt], padding=True, return_tensors="pt"
    ).to(device())
    max_new_tokens = int(duration_s * 50)
    audio_values = MODELS["music"].generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=max_new_tokens,
    )
    sampling_rate = MODELS["music"].config.audio_encoder.sampling_rate
    wav_output = audio_values[0, 0].cpu().numpy()
    temp_wav_path = tmp("wav", keep=False)
    write_wav(temp_wav_path, rate=sampling_rate, data=wav_output)
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"generated_{random_string()}")
    output_path = save_audio(
        destination_path=output_stem, audio_signal=sound, sample_rate=32000, output_format=format_choice
    )
    delete(temp_wav_path)
    return output_path


def dj_mix(
    files, mix_type=None, target_bpm=None, transition_sec=5, format_choice="mp3"
):
    import madmom
    import pydub

    if not files or len(files) < 2:
        catch("Please upload at least two audio files.")
        return None
    transition_ms = int(transition_sec * 1000)
    processed_tracks = []
    if target_bpm is None or target_bpm == 0:
        all_bpms = []
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        beat_processor = madmom.features.beats.RNNBeatProcessor()
        print("Analyzing BPM for all tracks to determine the average...")
        for file in files:
            try:
                act = beat_processor(str(file))
                bpm = np.median(60 / np.diff(proc(act)))
                if bpm > 0:
                    all_bpms.append(bpm)
            except Exception as e:
                print(
                    f"Could not analyze BPM for {Path(str(file)).name}, skipping this track for BPM calculation. Error: {e}"
                )
                continue
        if all_bpms:
            target_bpm = np.mean(all_bpms)
            print(f"Average target BPM calculated as: {target_bpm:.2f}")
        else:
            catch(
                "Could not determine BPM for any track. Beatmatching will be skipped."
            )
            target_bpm = 0
    for file in files:
        try:
            temp_stretched_path = None
            current_path = str(file)
            if (
                mix_type is not None
                and "beatmatched" in mix_type.lower()
                and (target_bpm > 0)
            ):
                proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
                act = madmom.features.beats.RNNBeatProcessor()(current_path)
                original_bpm = np.median(60 / np.diff(proc(act)))
                if original_bpm > 0 and target_bpm > 0:
                    speed_factor = target_bpm / original_bpm
                    temp_stretched_path = tmp(Path(current_path).suffix)
                    stretch_audio(
                        current_path, temp_stretched_path, speed_factor
                    )
                    current_path = temp_stretched_path
            track_segment = pydub.AudioSegment.from_file(current_path)
            processed_tracks.append(track_segment)
            if temp_stretched_path:
                delete(temp_stretched_path)
        except Exception as e:
            print(
                f"Could not process track {Path(file.name).name}, skipping. Error: {e}"
            )
            continue
    if not processed_tracks:
        catch("No tracks could be processed.")
        return None
    final_mix = processed_tracks[0]
    for i in range(1, len(processed_tracks)):
        final_mix = final_mix.append(
            processed_tracks[i], crossfade=transition_ms
        )
    output_stem = tmp("dj_mix", keep=False)
    final_output_path = save_audio(
        destination_path=output_stem, audio_signal=final_mix, output_format=format_choice
    )
    return final_output_path


def beat_visualizer(
    image_path, audio_path, image_effect, animation_style, scale_intensity
):
    import librosa
    from moviepy import AudioFileClip, ColorClip, CompositeVideoClip, ImageClip
    from PIL import Image, ImageFilter

    img = Image.open(image_path)
    (w, h) = get_max_resolution(*img.size)
    img = img.resize((w, h), Image.Resampling.LANCZOS)
    (W, H) = img.size
    effect_map = {
        "Blur": ImageFilter.BLUR,
        "Sharpen": ImageFilter.SHARPEN,
        "Contour": ImageFilter.CONTOUR,
        "Emboss": ImageFilter.EMBOSS,
    }
    if image_effect in effect_map:
        img = img.filter(effect_map[image_effect])
    output_path = tmp(".mp4")
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    (y, sr) = librosa.load(audio_path, sr=None)
    hop_length = 512
    effect_strength = scale_intensity - 1.0
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-07)
    rms_scales = 1.0 + rms_normalized * effect_strength * 0.5
    (tempo, beat_frames) = librosa.beat.beat_track(
        y=y, sr=sr, hop_length=hop_length
    )
    beat_impulses = np.zeros_like(rms_normalized)
    decay_rate = 0.75
    for beat_frame in beat_frames:
        frame = beat_frame
        impulse = 1.0
        while frame < len(beat_impulses) and impulse > 0.01:
            beat_impulses[frame] = max(beat_impulses[frame], impulse)
            impulse *= decay_rate
            frame += 1
    beat_scales = 1.0 + beat_impulses * effect_strength

    def base_animation_func(t):
        if animation_style == "Zoom In":
            return 1 + 0.1 * (t / duration)
        elif animation_style == "Zoom Out":
            return 1.1 - 0.1 * (t / duration)
        return 1.0

    def final_scale_func(t):
        frame_index = int(t * sr / hop_length)
        frame_index = min(frame_index, len(rms_scales) - 1)
        return (
            base_animation_func(t)
            * rms_scales[frame_index]
            * beat_scales[frame_index]
        )

    image_clip = ImageClip(np.array(img), duration=duration)
    animated_image = image_clip.with_position(("center", "center")).resized(
        final_scale_func
    )
    background = ColorClip(size=(W, H), color=(0, 0, 0), duration=duration)
    final_clip = CompositeVideoClip([background, animated_image])
    final_clip = final_clip.with_audio(audio_clip)
    final_clip.write_videofile(
        output_path,
        fps=20,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=cores(),
    )
    return output_path


def analyze_audio(audio_path, hop_length=1024, duration=None, offset=0.0):
    import librosa

    (y, sr) = librosa.load(
        audio_path, sr=None, duration=duration, offset=offset
    )
    actual_duration = librosa.get_duration(y=y, sr=sr)
    stft = librosa.stft(y, hop_length=hop_length)
    (mag, _) = librosa.magphase(stft)
    stft_db = librosa.amplitude_to_db(mag, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low_mask = (freqs >= 20) & (freqs < 250)
    mid_mask = (freqs >= 250) & (freqs < 4000)
    high_mask = freqs >= 4000
    rms_all = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_low = np.mean(mag[low_mask, :], axis=0) if np.any(low_mask) else rms_all
    rms_mid = np.mean(mag[mid_mask, :], axis=0) if np.any(mid_mask) else rms_all
    rms_high = (
        np.mean(mag[high_mask, :], axis=0) if np.any(high_mask) else rms_all
    )
    spectral_centroid = librosa.feature.spectral_centroid(
        y=y, sr=sr, hop_length=hop_length
    )[0]

    def normalize(v):
        (v_min, v_max) = (np.min(v), np.max(v))
        if v_max - v_min == 0:
            return np.zeros_like(v)
        return (v - v_min) / (v_max - v_min)

    beat_frames = []
    bpm = 120
    if MADMOM_AVAILABLE and (duration is None or duration > 10):
        try:
            import madmom

            proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
            act = madmom.features.beats.RNNBeatProcessor()(audio_path)
            beat_times = proc(act)
            beat_frames = librosa.time_to_frames(
                beat_times, sr=sr, hop_length=hop_length
            )
            if len(beat_times) > 1:
                bpm = int(round(float(60.0 / np.mean(np.diff(beat_times)))))
        except:
            (tempo, beat_frames) = librosa.beat.beat_track(
                y=y, sr=sr, hop_length=hop_length
            )
            bpm = tempo
    else:
        (tempo, beat_frames) = librosa.beat.beat_track(
            y=y, sr=sr, hop_length=hop_length
        )
        bpm = tempo
    return {
        "y": y,
        "sr": sr,
        "duration": actual_duration,
        "bpm": bpm,
        "rms": normalize(rms_all),
        "rms_low": normalize(rms_low),
        "rms_mid": normalize(rms_mid),
        "rms_high": normalize(rms_high),
        "centroid": normalize(spectral_centroid),
        "stft": (stft_db - np.min(stft_db))
        / (np.max(stft_db) - np.min(stft_db) + 1e-06),
        "beat_frames": beat_frames,
        "hop_length": hop_length,
    }


def get_color_palette(name):
    palettes = {
        "Cyberpunk": [(0, 255, 255), (255, 0, 128), (128, 0, 255)],
        "Sunset": [(255, 94, 77), (255, 195, 0), (199, 0, 57)],
        "Ocean": [(0, 105, 148), (0, 168, 107), (72, 209, 204)],
        "Toxic": [(57, 255, 20), (170, 255, 0), (20, 20, 20)],
        "Gold": [(255, 215, 0), (218, 165, 32), (50, 50, 50)],
        "Israel": [(0, 56, 184), (255, 255, 255), (200, 200, 255)],
        "Matrix": [(0, 255, 0), (0, 128, 0), (0, 50, 0)],
        "Neon Red": [(255, 0, 0), (100, 0, 0), (20, 0, 0)],
        "Deep Space": [(10, 10, 30), (138, 43, 226), (75, 0, 130)],
    }
    return palettes.get(name, palettes["Cyberpunk"])


def get_audio_feedback(audio_path):
    import librosa
    from scipy.stats import pearsonr

    if not audio_path:
        catch("Please upload an audio file for feedback.")
        return None
    try:
        (y_stereo, sr) = librosa.load(audio_path, sr=None, mono=False)
        y_mono = librosa.to_mono(y_stereo) if y_stereo.ndim > 1 else y_stereo
        rms = librosa.feature.rms(y=y_mono)[0]
        librosa.feature.spectral_contrast(y=y_mono, sr=sr)
        stft = librosa.stft(y_mono)
        freqs = librosa.fft_frequencies(sr=sr)
        bass_energy = np.mean(np.abs(stft[(freqs >= 20) & (freqs < 250)]))
        high_energy = np.mean(np.abs(stft[(freqs >= 5000) & (freqs < 20000)]))
        peak_amp = np.max(np.abs(y_mono))
        mean_rms = np.mean(rms)
        crest_factor = 20 * np.log10(peak_amp / mean_rms) if mean_rms > 0 else 0
        stereo_width = 0
        if y_stereo.ndim > 1 and y_stereo.shape[0] == 2:
            (corr, _) = pearsonr(y_stereo[0], y_stereo[1])
            stereo_width = (1 - corr) * 100
        feedback = "### AI Track Feedback\n\n"
        feedback += "#### Technical Analysis\n"
        feedback += f"- **Loudness & Dynamics:** The track has a crest factor of **{crest_factor:.2f} dB**. "
        if crest_factor > 14:
            feedback += "This suggests the track is very dynamic and punchy.\n"
        elif crest_factor > 8:
            feedback += "This is a good balance between punch and loudness, typical for many genres.\n"
        else:
            feedback += "This suggests the track is heavily compressed or limited, prioritizing loudness over dynamic range.\n"
        feedback += f"- **Stereo Image:** The stereo width is estimated at **{stereo_width:.1f}%**. "
        if stereo_width > 60:
            feedback += "The mix feels wide and immersive.\n"
        elif stereo_width > 20:
            feedback += "The mix has a balanced stereo field.\n"
        else:
            feedback += "The mix is narrow or mostly mono.\n"
        feedback += f"- **Frequency Balance:** Bass energy is at **{bass_energy:.2f}** and high-frequency energy is at **{high_energy:.2f}**. "
        if bass_energy > high_energy * 2:
            feedback += "The track is bass-heavy.\n"
        elif high_energy > bass_energy * 2:
            feedback += "The track is bright or treble-heavy.\n"
        else:
            feedback += (
                "The track has a relatively balanced frequency spectrum.\n"
            )
        feedback += "\n#### Advice\n"
        if crest_factor < 8:
            feedback += "- **Compression:** The track might be over-compressed. Consider reducing the amount of compression to bring back some life and punch to the transients.\n"
        if stereo_width < 20 and y_stereo.ndim > 1:
            feedback += "- **Stereo Width:** To make the mix sound bigger, try using stereo widening tools or panning instruments differently to create more space.\n"
        if bass_energy > high_energy * 2.5:
            feedback += "- **Bass Management:** The low-end might be overpowering. Ensure it's not masking other instruments. A high-pass filter on non-bass elements can clean up muddiness.\n"
        if high_energy > bass_energy * 2.5:
            feedback += "- **Tame the Highs:** The track is very bright, which can be fatiguing. Check for harshness in cymbals or vocals, and consider using a de-esser or a gentle high-shelf cut.\n"
        if mean_rms < 0.05:
            feedback += "- **Mastering:** The overall volume is low. The track would benefit from mastering to increase its loudness and competitiveness with commercial tracks.\n"
        else:
            feedback += "- **General Mix:** The track has a solid technical foundation. Focus on creative choices, arrangement, and ensuring all elements have their own space in the mix.\n"
        return feedback
    except Exception as e:
        raise catch(f"Analysis failed: {e}")


def analyze_audio_features(audio_path, txt=True):
    import librosa
    import madmom
    from scipy.stats import pearsonr

    try:
        proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
        act = madmom.features.beats.RNNBeatProcessor()(audio_path)
        beat_times = proc(act)
        bpm = np.median(60 / np.diff(beat_times)) if len(beat_times) > 1 else 0
        (y, sr) = librosa.load(audio_path)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_profile = np.mean(chroma, axis=1)
        major_template = np.array(
            [
                6.35,
                2.23,
                3.48,
                2.33,
                4.38,
                4.09,
                2.52,
                5.19,
                2.39,
                3.66,
                2.29,
                2.88,
            ]
        )
        minor_template = np.array(
            [
                6.33,
                2.68,
                3.52,
                5.38,
                2.6,
                3.53,
                2.54,
                4.75,
                3.98,
                2.69,
                3.34,
                3.17,
            ]
        )
        notes = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]
        best_correlation = -1
        detected_key = None
        detected_mode = None
        for i in range(12):
            (major_corr, _) = pearsonr(
                chroma_profile, np.roll(major_template, i)
            )
            if major_corr > best_correlation:
                best_correlation = major_corr
                detected_key = notes[i]
                detected_mode = "major"
            (minor_corr, _) = pearsonr(
                chroma_profile, np.roll(minor_template, i)
            )
            if minor_corr > best_correlation:
                best_correlation = minor_corr
                detected_key = notes[i]
                detected_mode = "minor"
        if txt:
            return f"{detected_key} {detected_mode}, {bpm:.2f} BPM"
        return (detected_key, detected_mode, bpm)
    except Exception as e:
        print(f"Analysis failed: {e}")
        return (None, None, None)


def change_audio_speed(audio_path, speed_factor, preserve_pitch, format_choice):
    import pydub

    sound_out = None
    if preserve_pitch:
        audio_path_out = tmp(Path(audio_path).suffix)
        if stretch_audio(audio_path, audio_path_out, speed_factor):
            sound_out = pydub.AudioSegment.from_file(audio_path_out)
            delete(audio_path_out)
        else:
            catch("Failed to stretch audio while preserving pitch.")
            return None
    else:
        sound = pydub.AudioSegment.from_file(audio_path)
        new_frame_rate = int(sound.frame_rate * speed_factor)
        sound_out = sound._spawn(
            sound.raw_data, overrides={"frame_rate": new_frame_rate}
        ).set_frame_rate(sound.frame_rate)
    if sound_out:
        output_stem = str(
            Path(audio_path).with_name(
                f"{Path(audio_path).stem}_speed_{speed_factor}x"
            )
        )
        return save_audio(
            destination_path=output_stem, audio_signal=sound_out, sample_rate=24000, output_format=format_choice
        )
    else:
        catch("Could not process audio speed change.")
        return None


def separate_stems(audio_path, separation_type=None, format_choice="wav"):
    output_dir = tmp(dir=True)
    run(
        f'"{sys.executable}" -m demucs.separate -n hdemucs_mmi --shifts=2 --two-stems=vocals -o "{output_dir}" "{audio_path}"'
    )
    separated_dir = Path(output_dir) / "hdemucs_mmi" / Path(audio_path).stem
    vocals_path = separated_dir / "vocals.wav"
    accompaniment_path = separated_dir / "no_vocals.wav"
    if not vocals_path.exists() or not accompaniment_path.exists():
        delete(output_dir)
        catch("Stem separation failed.")
        return None

    def _export_stem(chosen_stem_path, suffix):
        sr, sound = read_audio(chosen_stem_path)
        output_stem = str(
            Path(audio_path).with_name(Path(audio_path).stem + suffix)
        )
        final_output_path = save_audio(
            audio_signal=sound, destination_path=output_stem, sample_rate=sr, output_format=format_choice
        )
        return final_output_path

    if "acapella" == separation_type:
        voice = _export_stem(vocals_path, "_acapella")
        delete(output_dir)
        return normalize_audio_to_peak(voice)
    if "karaoke" == separation_type:
        music = _export_stem(accompaniment_path, "_karaoke")
        delete(output_dir)
        return normalize_audio_to_peak(music)
    (voice, music) = (
        normalize_audio_to_peak(_export_stem(vocals_path, "_acapella")),
        normalize_audio_to_peak(_export_stem(accompaniment_path, "_karaoke")),
    )
    delete(output_dir)
    return (voice, music)


def pitch_shift_vocals(
    audio_path, pitch_shift, format_choice="wav", seperated=False
):
    import librosa
    import pydub
    import soundfile as sf

    if seperated:
        (y_vocals, sr) = librosa.load(str(audio_path), sr=None)
        y_shifted = librosa.effects.pitch_shift(
            y=y_vocals, sr=sr, n_steps=float(pitch_shift)
        )
        output_path = tmp(format_choice, keep=False)
        sf.write(output_path, y_shifted, sr)
        return output_path
    (vocals_path, instrumental_path) = separate_stems(audio_path)
    if not vocals_path.exists() or not instrumental_path.exists():
        delete(str(Path(vocals_path).parent))
        catch("Vocal separation failed.")
        return None
    (y_vocals, sr) = librosa.load(str(vocals_path), sr=None)
    y_shifted = librosa.effects.pitch_shift(
        y=y_vocals, sr=sr, n_steps=float(pitch_shift)
    )
    shifted_vocals_path = tmp("wav", keep=False)
    sf.write(shifted_vocals_path, y_shifted, sr)
    instrumental = pydub.AudioSegment.from_file(instrumental_path)
    shifted_vocals = pydub.AudioSegment.from_file(shifted_vocals_path)
    combined = instrumental.overlay(shifted_vocals)
    output_stem = str(
        Path(audio_path).with_name(
            f"{Path(audio_path).stem}_vocal_pitch_shifted"
        )
    )
    final_output_path = save_audio(
        audio_signal=combined, destination_path=output_stem, output_format=format_choice
    )
    delete(str(Path(vocals_path).parent))
    delete(shifted_vocals_path)
    return final_output_path


def create_spectrum_visualization(audio_path):
    import librosa
    import matplotlib.pyplot as plt

    try:
        (y, sr) = librosa.load(audio_path, sr=None)
        n_fft = 8192
        hop_length = 512
        stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft_result)
        avg_magnitude = np.mean(magnitude, axis=1)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        magnitude_db = librosa.amplitude_to_db(avg_magnitude, ref=np.max)
        (fig, ax) = plt.subplots(figsize=(8, 5), facecolor="#f0f0f0")
        ax.set_facecolor("white")
        ax.fill_between(
            freqs,
            magnitude_db,
            y2=np.min(magnitude_db) - 1,
            color="#7c3aed",
            alpha=0.8,
            zorder=2,
        )
        ax.plot(freqs, magnitude_db, color="#4c2a8c", linewidth=1, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(20, sr / 2)
        ax.set_ylim(np.min(magnitude_db) - 1, np.max(magnitude_db) + 5)
        xticks = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        xtick_labels = [
            "50",
            "100",
            "200",
            "500",
            "1k",
            "2k",
            "5k",
            "10k",
            "20k",
        ]
        ax.set_xticks([x for x in xticks if x < sr / 2])
        ax.set_xticklabels(
            [label for (x, label) in zip(xticks, xtick_labels) if x < sr / 2]
        )
        ax.grid(True, which="both", ls="--", color="gray", alpha=0.6, zorder=1)
        ax.set_title("Frequency Analysis", color="black")
        ax.set_xlabel("Frequency (Hz)", color="black")
        ax.set_ylabel("Amplitude (dB)", color="black")
        ax.tick_params(colors="black", which="both")
        audible_mask = freqs > 20
        if np.any(audible_mask):
            peak_idx = np.argmax(magnitude_db[audible_mask])
            peak_freq = freqs[audible_mask][peak_idx]
            peak_db = magnitude_db[audible_mask][peak_idx]
            peak_text = f"Peak: {peak_freq:.0f} Hz at {peak_db:.1f} dB"
            ax.text(
                0.98,
                0.95,
                peak_text,
                transform=ax.transAxes,
                color="black",
                ha="right",
                va="top",
            )
        fig.tight_layout()
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
        fig.savefig(temp_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        return temp_path
    except Exception as e:
        print(f"Error creating spectrum: {e}")
        return None


def stem_mixer(files, format_choice):
    import librosa
    import madmom
    import pydub
    import soundfile as sf
    from scipy.io.wavfile import write as write_wav

    if not files or len(files) < 2:
        catch("Please upload at least two stem files.")
        return None
    processed_stems = []
    target_sr = None
    max_length = 0
    print("--- Processing Stems for Simple Mixing ---")
    for i, _file in enumerate(files):
        file_obj = Path(_file)
        print(f"Processing file {i + 1}/{len(files)}: {file_obj.name}")
        try:
            (y, sr) = librosa.load(_file, sr=None)
        except Exception as e:
            catch(f"Could not load file: {file_obj.name}. Error: {e}")
            continue
        if target_sr is None:
            target_sr = sr
        if sr != target_sr:
            print(f"Resampling {file_obj.name} from {sr}Hz to {target_sr}Hz.")
            y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        processed_stems.append(y)
        if len(y) > max_length:
            max_length = len(y)
    if not processed_stems:
        catch("No audio files were successfully processed.")
        return None
    print("\n--- Mixing Stems ---")
    mixed_y = np.zeros(max_length, dtype=np.float32)
    for stem_audio in processed_stems:
        mixed_y[: len(stem_audio)] += stem_audio
    print("Mixing complete. Normalizing volume...")
    peak_amplitude = np.max(np.abs(mixed_y))
    if peak_amplitude > 0:
        mixed_y = mixed_y / peak_amplitude * 0.99
    print("Exporting final mix...")
    temp_wav_path = tmp(".wav", keep=False)
    write_wav(temp_wav_path, target_sr, (mixed_y * 32767).astype(np.int16))
    sound = pydub.AudioSegment.from_file(temp_wav_path)
    output_stem = Path(temp_wav_path).with_name(f"stem_mix_{random_string()}")
    output_path = save_audio(
        audio_signal=sound, destination_path=output_stem, output_format=format_choice
    )
    delete(temp_wav_path)
    print(f"--- Success! Mix saved to: {output_path} ---")
    return output_path


def identify_instruments(audio_path):
    if MODELS["audio-classification"] is None:
        catch("Audio identification model is not available.")
        return None
    predictions = MODELS["audio-classification"](audio_path, top_k=10)
    instrument_list = [
        "guitar",
        "piano",
        "violin",
        "drum",
        "bass",
        "saxophone",
        "trumpet",
        "flute",
        "cello",
        "clarinet",
        "synthesizer",
        "organ",
        "accordion",
        "banjo",
        "harp",
        "voice",
        "speech",
    ]
    detected_instruments = "### Detected Instruments\n\n"
    found = False
    for p in predictions:
        label = p["label"].lower()
        if any(instrument in label for instrument in instrument_list):
            detected_instruments += (
                f"- **{p['label'].title()}** (Score: {p['score']:.2f})\n"
            )
            found = True
    if not found:
        detected_instruments += "Could not identify specific instruments with high confidence. Top sound events:\n"
        for p in predictions[:3]:
            detected_instruments += (
                f"- {p['label'].title()} (Score: {p['score']:.2f})\n"
            )
    return detected_instruments


def extend_audio(audio_path, extend_duration_s, format_choice):
    import librosa
    import pydub
    import soundfile as sf

    if MODELS["music"] is None or PROCESSORS["music"] is None:
        catch("MusicGen model is not available for audio extension.")
        return None
    (y, sr) = librosa.load(audio_path, sr=None, mono=True)
    prompt_duration_s = min(15.0, len(y) / sr)
    prompt_wav = y[-int(prompt_duration_s * sr) :]
    inputs = PROCESSORS["music"](
        audio=prompt_wav, sampling_rate=sr, return_tensors="pt"
    ).to(device())
    total_duration_s = prompt_duration_s + extend_duration_s
    max_new_tokens = int(total_duration_s * 50)
    generated_audio_values = MODELS["music"].generate(
        **inputs,
        do_sample=True,
        guidance_scale=3,
        max_new_tokens=max_new_tokens,
    )
    generated_wav = generated_audio_values[0, 0].cpu().numpy()
    extension_start_sample = int(
        prompt_duration_s * MODELS["music"].config.audio_encoder.sampling_rate
    )
    extension_wav = generated_wav[extension_start_sample:]
    temp_extension_path = tmp(".wav")
    sf.write(
        temp_extension_path,
        extension_wav,
        MODELS["music"].config.audio_encoder.sampling_rate,
    )
    original_sr, original_sound = read_audio(audio_path)
    extension_sr, extension_sound = read_audio(temp_extension_path)
    if original_sound.channels != extension_sound.channels:
        extension_sound = extension_sound.set_channels(original_sound.channels)
    final_sound = original_sound + extension_sound
    output_stem = str(
        Path(audio_path).with_name(f"{Path(audio_path).stem}_extended")
    )
    final_output_path = save_audio(
        audio_signal=final_sound, destination_path=output_stem, output_format=format_choice
    )
    delete(temp_extension_path)
    return final_output_path


def audio_to_midi(audio_path):
    import madmom
    from basic_pitch.inference import predict

    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(audio_path)
    bpm = np.median(60 / np.diff(proc(act)))
    (model_output, midi_data, note_events) = predict(
        audio_path,
        midi_tempo=bpm,
        onset_threshold=0.95,
        frame_threshold=0.25,
        minimum_note_length=80,
        minimum_frequency=60,
        maximum_frequency=4200,
    )
    name = random_string() + ".mid"
    midi_data.write(f"./{name}")
    return name


def midi_to_audio(midi_path, format_choice):
    import pydub
    from midi2audio import FluidSynth

    soundfont_paths = [
        os.path.join(
            os.path.expanduser("~"),
            "app_dependencies",
            "soundfonts",
            "VintageDreamsWaves-v2.sf3",
        ),
        "/usr/share/sounds/sf2/VintageDreamsWaves-v2.sf3",
        "C:/Windows/System32/drivers/gm.dls",
    ]
    soundfont_file = None
    for path in soundfont_paths:
        if os.path.exists(path):
            soundfont_file = path
            break
    if soundfont_file is None:
        catch(
            "SoundFont file not found. MIDI to Audio conversion cannot proceed. Please re-run the dependency installer."
        )
        return None
    fs = FluidSynth(sound_font=soundfont_file)
    temp_wav_path = tmp(".wav")
    fs.midi_to_audio(midi_path, temp_wav_path)
    sr, sound = read_audio(temp_wav_path)
    output_stem = str(
        Path(midi_path).with_name(f"{Path(midi_path).stem}_render")
    )
    final_output_path = save_audio(
        audio_signal=sound, destination_path=output_stem, sample_rate=sr, output_format=format_choice
    )
    delete(temp_wav_path)
    return final_output_path


def subdivide_beats(beat_times, subdivision):
    if subdivision <= 1 or len(beat_times) < 2:
        return np.array(beat_times)
    new_beats = []
    for i in range(len(beat_times) - 1):
        start_beat = beat_times[i]
        end_beat = beat_times[i + 1]
        interval = (end_beat - start_beat) / subdivision
        for j in range(subdivision):
            new_beats.append(start_beat + j * interval)
    new_beats.append(beat_times[-1])
    return np.array(sorted(list(set(new_beats))))


def calculate_active_rms(y, sr):
    import librosa

    non_silent_intervals = librosa.effects.split(
        y, top_db=40, frame_length=1024, hop_length=256
    )
    active_audio = np.concatenate(
        [y[start:end] for (start, end) in non_silent_intervals]
    )
    if len(active_audio) == 0:
        return 1e-06
    return np.sqrt(np.mean(active_audio**2))


def normalize_audio_to_peak(
    input_path: str, target_level: float = 0.9, format: str = None
):
    from pydub import AudioSegment

    if not 0.0 < target_level <= 1.0:
        catch("target_level must be between 0.0 and 1.0")
        return None
    if format is None:
        format = get_ext(input_path) or "wav"
    output_path = tmp(format)
    try:
        sr, audio = read_audio(input_path)
    except FileNotFoundError:
        catch(f"Input file not found at {input_path}")
        return None
    if target_level == 0.0 or audio.max_dBFS == -float("inf"):
        silent_audio = AudioSegment.silent(duration=len(audio))
        silent_audio.export(output_path, format=format)
        log(
            "Exported silent file"
            f"Silent audio detected or target level is 0. Saved silent file to '{output_path}'"
        )
        return output_path
    target_dbfs = 20 * math.log10(target_level)
    gain_to_apply = target_dbfs - audio.max_dBFS
    normalized_audio = audio.apply_gain(gain_to_apply)
    normalized_audio.export(output_path, format=format)
    log(
        f"Successfully normalized '{input_path}' to a peak of {target_dbfs:.2f} dBFS."
    )
    print(f"Saved result to '{output_path}'")
    return output_path


def stretch_audio(input_path, output_path=None, speed_factor=0.85):
    if not exist(input_path):
        return None
    if output_path is None:
        output_path = tmp("wav")
    command = [
        "rubberband",
        "--formant",
        "--tempo",
        str(speed_factor),
        "-q",
        input_path,
        output_path,
    ]
    try:
        run(command)
        return normalize_audio_to_peak(output_path)
    except Exception as e:
        catch(f"Error during audio stretching with rubberband: {e}")
        return None


def get_scale_notes(key="C", scale="major", start_octave=1, end_octave=9):
    NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
    start_note_midi = (start_octave - 1) * 12 + NOTES.index(key.upper())
    scale_intervals = SCALES.get(scale.lower(), SCALES["major"])
    scale_notes = []
    for i in range((end_octave - start_octave) * 12):
        if i % 12 in scale_intervals:
            scale_notes.append(start_note_midi + i)
    return np.array(scale_notes)


def autotune_song(
    audio_path,
    output_path=None,
    strength=0.7,
    correct_timing=True,
    quantize_grid_strength=16,
    tolerance_cents=15,
    attack_smoothing_ms=0.1,
):
    import librosa
    import madmom
    import pydub
    import soundfile as sf
    from scipy.signal import medfilt

    if output_path is None:
        output_path = tmp("wav", keep=False)
    audio_path = normalize_audio_to_peak(audio_path)
    if not exist(audio_path):
        catch("Input audio file not found.")
        return None
    temp_files = []
    try:
        (detected_key, detected_mode, detected_bpm) = analyze_audio_features(
            audio_path, False
        )
        if not detected_key:
            catch("Could not determine song key. Aborting.")
            return None
        (vocals_path, instrumental_path) = separate_stems(audio_path)
        if not vocals_path or not instrumental_path:
            catch("Vocal separation failed.")
            return None
        temp_files.extend([vocals_path, instrumental_path])
        (y_vocals, sr) = librosa.load(vocals_path, sr=None, mono=True)
        n_fft = 16384
        hop_length = 8192
        processed_vocals_path = vocals_path
        if correct_timing:
            beat_proc = madmom.features.beats.RNNBeatProcessor()
            beat_act = beat_proc(instrumental_path)
            beat_times = madmom.features.beats.BeatTrackingProcessor(fps=100)(
                beat_act
            )
            if quantize_grid_strength > 1 and len(beat_times) > 1:
                beat_interval = np.mean(np.diff(beat_times))
                beat_interval / quantize_grid_strength
                quantized_beat_times = []
                for i in range(len(beat_times) - 1):
                    quantized_beat_times.extend(
                        np.linspace(
                            beat_times[i],
                            beat_times[i + 1],
                            quantize_grid_strength,
                            endpoint=False,
                        )
                    )
                quantized_beat_times.append(beat_times[-1])
                beat_times = np.array(sorted(quantized_beat_times))
            onsets = librosa.onset.onset_detect(
                y=y_vocals, sr=sr, hop_length=hop_length, units="time"
            )
            if len(onsets) > 1 and len(beat_times) > 0:
                time_map_data = []
                for onset_time in onsets:
                    closest_beat_index = np.argmin(
                        np.abs(beat_times - onset_time)
                    )
                    target_time = beat_times[closest_beat_index]
                    time_map_data.append(f"{onset_time:.6f} {target_time:.6f}")
                time_map_path = tmp(".txt")
                temp_files.append(time_map_path)
                with open(time_map_path, "w") as f:
                    f.write("\n".join(time_map_data))
                quantized_vocals_path = tmp(".wav")
                command = [
                    "rubberband",
                    "--time",
                    "1",
                    "--timemap",
                    time_map_path,
                    vocals_path,
                    quantized_vocals_path,
                ]
                run(command)
                if exist(quantized_vocals_path):
                    (y_vocals, sr) = librosa.load(quantized_vocals_path, sr=sr)
                    processed_vocals_path = quantized_vocals_path
                    temp_files.append(quantized_vocals_path)
        allowed_notes_midi = get_scale_notes(
            key=detected_key, scale=detected_mode
        )
        (f0, voiced_flag, _) = librosa.pyin(
            y_vocals,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
            sr=sr,
            frame_length=n_fft,
            hop_length=hop_length,
        )
        f0 = np.nan_to_num(f0, nan=0.0)
        target_f0 = np.copy(f0)
        voiced_mask = voiced_flag & (f0 > 0)
        if np.any(voiced_mask):
            voiced_f0 = f0[voiced_mask]
            voiced_midi = librosa.hz_to_midi(voiced_f0)
            note_diffs = np.abs(allowed_notes_midi.reshape(-1, 1) - voiced_midi)
            closest_note_indices = np.argmin(note_diffs, axis=0)
            target_midi = allowed_notes_midi[closest_note_indices]
            cents_deviation = np.abs(voiced_midi - target_midi) * 100
            correction_mask = cents_deviation > tolerance_cents
            ideal_f0 = librosa.midi_to_hz(target_midi[correction_mask])
            original_f0_to_correct = voiced_f0[correction_mask]
            corrected_f0_subset = (
                original_f0_to_correct
                + (ideal_f0 - original_f0_to_correct) * strength
            )
            temp_voiced_f0 = np.copy(voiced_f0)
            temp_voiced_f0[correction_mask] = corrected_f0_subset
            if attack_smoothing_ms > 0:
                smoothing_window_size = int(
                    sr / hop_length * (attack_smoothing_ms / 1000.0)
                )
                if smoothing_window_size % 2 == 0:
                    smoothing_window_size += 1
                if smoothing_window_size > 1:
                    temp_voiced_f0 = medfilt(
                        temp_voiced_f0, kernel_size=smoothing_window_size
                    )
            target_f0[voiced_mask] = temp_voiced_f0
        freq_map_path = tmp(".txt")
        temp_files.append(freq_map_path)
        with open(freq_map_path, "w") as f:
            ratios = target_f0 / f0
            ratios[~voiced_flag | (f0 == 0)] = 1.0
            for i in range(len(ratios)):
                sample_num = i * hop_length
                f.write(f"{sample_num} {ratios[i]:.6f}\n")
        tuned_vocals_path = tmp(".wav")
        command = [
            "rubberband",
            "--formant",
            "--freqmap",
            freq_map_path,
            processed_vocals_path,
            tuned_vocals_path,
        ]
        run(command)
        if not exist(tuned_vocals_path):
            catch("Pitch correction with rubberband failed.")
            return None
        temp_files.append(tuned_vocals_path)
        instrumental_audio = pydub.AudioSegment.from_file(instrumental_path)
        tuned_vocals_audio = pydub.AudioSegment.from_file(tuned_vocals_path)
        tuned_vocals_audio = tuned_vocals_audio.set_frame_rate(
            instrumental_audio.frame_rate
        )
        combined = instrumental_audio.overlay(tuned_vocals_audio)
        output_format = get_ext(output_path)
        combined.export(output_path, format=output_format)
        normalize_audio_to_peak(output_path)
        return output_path
    finally:
        for path in temp_files:
            delete(path)


def compute_gain_envelope(
    sidechain, sample_rate, threshold, attack_ms, release_ms
):
    gain = 1.0
    envelope = np.zeros_like(sidechain)
    attack_samples = attack_ms / 1000.0 * sample_rate
    release_samples = release_ms / 1000.0 * sample_rate
    attack_coeff = (
        np.exp(-np.log(9) / attack_samples) if attack_samples > 0 else 0.0
    )
    release_coeff = (
        np.exp(-np.log(9) / release_samples) if release_samples > 0 else 0.0
    )
    for i in range(len(sidechain)):
        current_sample_abs = np.abs(sidechain[i])
        if current_sample_abs > threshold:
            target_gain = threshold / current_sample_abs
        else:
            target_gain = 1.0
        if target_gain < gain:
            gain = attack_coeff * gain + (1 - attack_coeff) * target_gain
        else:
            gain = release_coeff * gain + (1 - release_coeff) * target_gain
        envelope[i] = gain
    return envelope


def loudness_maximizer(
    input_filename,
    output_filename=None,
    comp_threshold_db=-12.0,
    comp_ratio=10.0,
    comp_attack_ms=0.1,
    comp_release_ms=300.0,
    db_boost=12.0,
    db_limit=-0.2,
    limit_attack_ms=0.1,
    limit_release_ms=400.0,
    lookahead_ms=1.5,
    oversampling=2,
):
    from scipy import signal as scipy_signal
    from scipy.io import wavfile

    if output_filename is None:
        output_filename = tmp("mp3", keep=False)
    try:
        (sample_rate, audio) = wavfile.read(input_filename)
        print(f"Reading '{input_filename}' at {sample_rate} Hz.")
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None
    original_dtype = audio.dtype
    if original_dtype == np.int16:
        max_val = 32767.0
    elif original_dtype == np.int32:
        max_val = 2147483647.0
    elif original_dtype == np.uint8:
        audio = audio.astype(np.float32) - 128.0
        max_val = 127.0
    else:
        max_val = 1.0
    audio_float = audio.astype(np.float32) / max_val
    original_length = audio.shape[0]
    effective_rate = sample_rate
    if oversampling > 1 and isinstance(oversampling, int):
        print(f"Oversampling audio by a factor of {oversampling}x...")
        effective_rate = sample_rate * oversampling
        audio_float = scipy_signal.resample_poly(
            audio_float, oversampling, 1, axis=0
        )
    compressed_audio = apply_compressor(
        audio_float,
        effective_rate,
        threshold_db=comp_threshold_db,
        ratio=comp_ratio,
        attack_ms=comp_attack_ms,
        release_ms=comp_release_ms,
    )
    initial_rms_db = 20 * np.log10(np.sqrt(np.mean(compressed_audio**2)))
    print(f"Post-Compressor RMS: {initial_rms_db:.2f} dB")
    linear_boost = 10 ** (db_boost / 20.0)
    boosted_audio = compressed_audio * linear_boost
    print(f"Applied {db_boost} dB of makeup gain.")
    if boosted_audio.ndim > 1:
        sidechain = np.max(np.abs(boosted_audio), axis=1)
    else:
        sidechain = np.abs(boosted_audio)
    lookahead_samples = int(lookahead_ms * effective_rate / 1000.0)
    if lookahead_samples > 0:
        pad_width_audio = [(lookahead_samples, 0)] + [(0, 0)] * (
            boosted_audio.ndim - 1
        )
        delayed_audio = np.pad(boosted_audio, pad_width_audio, "constant")
    else:
        delayed_audio = boosted_audio
    threshold = 10 ** (db_limit / 20.0)
    print("Computing gain envelope...")
    if lookahead_samples > 0:
        sidechain_padded = np.pad(sidechain, (0, lookahead_samples), "constant")
    else:
        sidechain_padded = sidechain
    gain = compute_gain_envelope(
        sidechain_padded,
        effective_rate,
        threshold,
        limit_attack_ms,
        limit_release_ms,
    )
    if boosted_audio.ndim > 1:
        gain = np.tile(gain[:, np.newaxis], (1, boosted_audio.shape[1]))
    limited_audio = delayed_audio * gain
    print(f"Applied limiting with a ceiling of {db_limit} dB.")
    if oversampling > 1 and isinstance(oversampling, int):
        print(f"Downsampling back to {sample_rate} Hz...")
        limited_audio = scipy_signal.resample_poly(
            limited_audio, 1, oversampling, axis=0
        )
    current_length = int(limited_audio.shape[0])
    lookahead_len = int(lookahead_ms / 1000 * sample_rate)
    max_len = lookahead_len + original_length
    if current_length > original_length:
        final_processed_audio = limited_audio[lookahead_len:max_len]
    elif current_length < original_length:
        pad_amount = original_length - current_length
        pad_width = [(0, pad_amount)] + [(0, 0)] * (limited_audio.ndim - 1)
        final_processed_audio = np.pad(limited_audio, pad_width, "constant")
    else:
        final_processed_audio = limited_audio
    print(f"Applying final brickwall clip at {db_limit} dB.")
    np.clip(
        final_processed_audio, -threshold, threshold, out=final_processed_audio
    )
    final_rms_db = 20 * np.log10(np.sqrt(np.mean(final_processed_audio**2)))
    print(f"Final RMS: {final_rms_db:.2f} dB.")
    processed_audio_int = final_processed_audio * max_val
    final_audio = processed_audio_int.astype(original_dtype)
    try:
        wavfile.write(output_filename, sample_rate, final_audio)
        print(f"✅ Successfully saved processed audio to '{output_filename}'.")
        return output_filename
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None


def apply_compressor(
    audio,
    sample_rate,
    threshold_db=-20.0,
    ratio=4.0,
    attack_ms=5.0,
    release_ms=150.0,
    knee_db=5.0,
):
    print(f"Applying compressor: Threshold={threshold_db}dB, Ratio={ratio}:1")
    10.0 ** (threshold_db / 20.0)
    attack_samples = sample_rate / 1000.0 * attack_ms
    release_samples = sample_rate / 1000.0 * release_ms
    alpha_attack = np.exp(-1.0 / attack_samples)
    alpha_release = np.exp(-1.0 / release_samples)
    if audio.ndim > 1:
        sidechain = np.max(np.abs(audio), axis=1)
    else:
        sidechain = np.abs(audio)
    sidechain = np.maximum(sidechain, 1e-08)
    sidechain_db = 20.0 * np.log10(sidechain)
    gain_reduction_db = np.zeros_like(sidechain_db)
    half_knee = knee_db / 2.0
    knee_start = threshold_db - half_knee
    knee_end = threshold_db + half_knee
    above_knee_start_indices = np.where(sidechain_db > knee_start)[0]
    for i in above_knee_start_indices:
        level = sidechain_db[i]
        if level <= knee_end:
            x = (level - knee_start) / knee_db
            reduction = x * x * (threshold_db - level) * (1.0 - 1.0 / ratio)
            gain_reduction_db[i] = reduction
        else:
            gain_reduction_db[i] = (threshold_db - level) * (1.0 - 1.0 / ratio)
    smoothed_gain_db = np.zeros_like(gain_reduction_db)
    for i in range(1, len(gain_reduction_db)):
        if gain_reduction_db[i] < smoothed_gain_db[i - 1]:
            smoothed_gain_db[i] = (
                alpha_attack * smoothed_gain_db[i - 1]
                + (1 - alpha_attack) * gain_reduction_db[i]
            )
        else:
            smoothed_gain_db[i] = (
                alpha_release * smoothed_gain_db[i - 1]
                + (1 - alpha_release) * gain_reduction_db[i]
            )
    final_gain = 10.0 ** (smoothed_gain_db / 20.0)
    if audio.ndim > 1:
        final_gain = np.tile(final_gain[:, np.newaxis], (1, audio.shape[1]))
    return audio * final_gain


def create_sample_audio(
    filename="sample_audio.wav", duration=5, sample_rate=44100
):
    from scipy.io import wavfile

    print("Creating a sample audio file for testing...")
    t = np.linspace(0.0, duration, int(sample_rate * duration))
    amplitude_ramp = np.linspace(0.1, 1.0, int(sample_rate * duration))
    audio_data = amplitude_ramp * np.sin(2.0 * np.pi * 440.0 * t)
    audio_data_int = np.int16(audio_data * 32767)
    wavfile.write(filename, sample_rate, audio_data_int)
    print(f"Sample audio saved as '{filename}'.")


def riaa_filter(input_filename, bass_factor=1.0):
    import librosa
    from scipy.io import wavfile
    from scipy.signal import bilinear, freqs, lfilter

    output_filename = tmp("wav")
    try:
        (audio_data, sample_rate) = librosa.load(
            input_filename, sr=None, mono=False
        )
        if audio_data.dtype != np.float32 and audio_data.dtype != np.float64:
            info = np.iinfo(audio_data.dtype)
            audio_data = audio_data.astype(np.float32) / (info.max + 1)
        print(f"Read '{input_filename}' with sample rate {sample_rate} Hz.")
    except FileNotFoundError:
        print(
            f"File '{input_filename}' not found. Generating white noise for demonstration."
        )
        sample_rate = 44100
        duration = 5
        audio_data = np.random.randn(sample_rate * duration)
        audio_data /= np.max(np.abs(audio_data))
    t1_original = 0.00318
    t2 = 0.000318
    t3 = 7.5e-05
    print(
        f"Applying a custom RIAA de-emphasis with a bass factor of {bass_factor}..."
    )
    t1_modified = (1 - bass_factor) * t2 + bass_factor * t1_original
    num_s = [t2, 1]
    den_s = [t1_modified * t3, t1_modified + t3, 1]
    w1k = 2 * np.pi * 1000
    (_, h) = freqs(num_s, den_s, worN=[w1k])
    gain_at_1k = np.abs(h[0])
    num_s_normalized = [c / gain_at_1k for c in num_s]
    (b_riaa, a_riaa) = bilinear(num_s_normalized, den_s, fs=sample_rate)
    if audio_data.ndim > 1:
        processed_audio = np.array(
            [lfilter(b_riaa, a_riaa, channel) for channel in audio_data]
        )
    else:
        processed_audio = lfilter(b_riaa, a_riaa, audio_data)
    max_abs = np.max(np.abs(processed_audio))
    if max_abs > 0:
        processed_audio /= max_abs
    processed_audio_int16 = np.int16((processed_audio * 32767).T)
    wavfile.write(output_filename, sample_rate, processed_audio_int16)
    print(
        f"Successfully applied custom RIAA EQ and saved to '{output_filename}'."
    )
    return output_filename
