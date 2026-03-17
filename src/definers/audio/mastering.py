from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import RLock

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d
from scipy.signal import fftconvolve, firwin2

from ..file_ops import log
from .config import SmartMasteringConfig
from .dsp import (
    decoupled_envelope,
    limiter_smooth_env,
    remove_spectral_spikes,
    resample,
)
from .effects import apply_exciter, stereo
from .filters import freq_cut
from .utils import apply_lufs, generate_bands, stereo_widen

_AUDIO_EQUALIZER_CACHE_LOCK = RLock()
_AUDIO_EQUALIZER_KERNEL_CACHE: OrderedDict[
    tuple[int, int, float, bytes], np.ndarray
] = OrderedDict()
_AUDIO_EQUALIZER_CACHE_LIMIT = 32


class AudioEqualizer:
    def __init__(
        self,
        sr: int,
        anchors: list[list[float]] | np.ndarray,
        *,
        taps: int = 131071,
        correction_strength: float = 1.0,
    ) -> None:
        sample_rate = int(sr)
        if sample_rate <= 0:
            raise ValueError("sr must be a positive integer")

        num_taps = int(taps)
        if num_taps < 3:
            raise ValueError("taps must be at least 3")
        if num_taps % 2 == 0:
            num_taps += 1

        strength = float(correction_strength)
        if not np.isfinite(strength):
            raise ValueError("correction_strength must be finite")

        self.sr = sample_rate
        self.correction_strength = strength
        self.num_taps = num_taps
        self.anchors = self.normalize_anchors(anchors, sample_rate)
        self.build_filter()

    def normalize_anchors(
        self,
        anchors: list[list[float]] | np.ndarray,
        sr: int,
    ) -> np.ndarray:
        anchor_array = np.asarray(anchors, dtype=np.float64)

        if (
            anchor_array.ndim != 2
            or anchor_array.shape[1] != 2
            or anchor_array.shape[0] < 2
        ):
            raise ValueError(
                "anchors must be a 2D array of [frequency_hz, gain_db] pairs with at least two rows"
            )

        if not np.all(np.isfinite(anchor_array)):
            raise ValueError("anchors must contain only finite values")

        nyquist = float(sr) / 2.0
        if nyquist <= 0.1:
            raise ValueError("sample rate is too low for equalizer design")

        freqs = np.clip(anchor_array[:, 0], 0.1, nyquist)
        gains = anchor_array[:, 1]

        order = np.argsort(freqs, kind="mergesort")
        freqs = freqs[order]
        gains = gains[order]

        unique_freqs, inverse = np.unique(freqs, return_inverse=True)
        unique_gains = np.zeros_like(unique_freqs)
        counts = np.zeros_like(unique_freqs)

        np.add.at(unique_gains, inverse, gains)
        np.add.at(counts, inverse, 1.0)
        unique_gains /= np.maximum(counts, 1.0)

        if unique_freqs[0] > 0.1:
            unique_freqs = np.insert(unique_freqs, 0, 0.1)
            unique_gains = np.insert(unique_gains, 0, unique_gains[0])

        if unique_freqs[-1] < nyquist:
            unique_freqs = np.append(unique_freqs, nyquist)
            unique_gains = np.append(unique_gains, unique_gains[-1])

        normalized = np.column_stack((unique_freqs, unique_gains)).astype(
            np.float64, copy=False
        )

        if normalized.shape[0] < 2:
            raise ValueError(
                "normalized anchors must contain at least two unique frequency points"
            )

        return normalized

    def cache_key(
        self,
        sr: int,
        taps: int,
        correction_strength: float,
        anchors: np.ndarray,
    ) -> tuple[int, int, float, bytes]:
        digest = hashlib.blake2b(
            np.ascontiguousarray(anchors, dtype=np.float64).tobytes(),
            digest_size=16,
        ).digest()
        return sr, taps, round(float(correction_strength), 12), digest

    def cache_get(
        self,
        key: tuple[int, int, float, bytes],
    ) -> np.ndarray | None:
        with _AUDIO_EQUALIZER_CACHE_LOCK:
            kernel = _AUDIO_EQUALIZER_KERNEL_CACHE.get(key)
            if kernel is None:
                return None
            _AUDIO_EQUALIZER_KERNEL_CACHE.move_to_end(key)
            return kernel

    def cache_put(
        self,
        key: tuple[int, int, float, bytes],
        kernel: np.ndarray,
    ) -> None:
        with _AUDIO_EQUALIZER_CACHE_LOCK:
            _AUDIO_EQUALIZER_KERNEL_CACHE[key] = kernel
            _AUDIO_EQUALIZER_KERNEL_CACHE.move_to_end(key)
            while (
                len(_AUDIO_EQUALIZER_KERNEL_CACHE)
                > _AUDIO_EQUALIZER_CACHE_LIMIT
            ):
                _AUDIO_EQUALIZER_KERNEL_CACHE.popitem(last=False)

    def clear_kernel_cache(self) -> None:
        with _AUDIO_EQUALIZER_CACHE_LOCK:
            _AUDIO_EQUALIZER_KERNEL_CACHE.clear()

    def build_filter(self) -> None:
        self.anchors = self.normalize_anchors(self.anchors, self.sr)
        cache_key = self.cache_key(
            self.sr,
            self.num_taps,
            self.correction_strength,
            self.anchors,
        )
        cached_kernel = self.cache_get(cache_key)

        if cached_kernel is not None:
            self.fir_kernel = cached_kernel
            return

        anchor_freqs = self.anchors[:, 0]
        anchor_dbs = self.anchors[:, 1] * self.correction_strength
        nyquist = self.sr / 2.0

        dense_count = int(
            min(
                262144,
                max(
                    32768,
                    1
                    << int(
                        np.ceil(
                            np.log2(
                                max(self.num_taps, self.anchors.shape[0] * 4096)
                            )
                        )
                    ),
                ),
            )
        )

        dense_freqs_log = np.geomspace(
            anchor_freqs[0], nyquist, num=dense_count
        )
        dense_dbs = np.interp(
            np.log10(dense_freqs_log),
            np.log10(anchor_freqs),
            anchor_dbs,
        )

        dense_freqs = np.insert(dense_freqs_log, 0, 0.0)
        dense_dbs = np.insert(dense_dbs, 0, dense_dbs[0])

        norm_freqs = np.clip(dense_freqs / nyquist, 0.0, 1.0)
        norm_freqs[-1] = 1.0

        linear_gains = np.asarray(10.0 ** (dense_dbs / 20.0), dtype=np.float64)
        if not np.all(np.isfinite(linear_gains)):
            raise ValueError("equalizer gain curve contains non-finite values")

        kernel = firwin2(
            numtaps=self.num_taps,
            freq=norm_freqs,
            gain=linear_gains,
            window=("kaiser", 8.6),
        ).astype(np.float64, copy=False)

        if not np.all(np.isfinite(kernel)):
            raise RuntimeError("equalizer kernel contains non-finite values")

        kernel.setflags(write=False)
        self.fir_kernel = kernel
        self.cache_put(cache_key, kernel)

    def select_convolver(self, signal_length: int, kernel_length: int):
        if signal_length <= 0:
            return signal.oaconvolve
        if kernel_length >= 16384 or signal_length >= kernel_length * 4:
            return signal.oaconvolve
        return fftconvolve

    def apply_correction(self, y: np.ndarray) -> np.ndarray:
        input_array = np.asarray(y)
        input_dtype = (
            input_array.dtype
            if np.issubdtype(input_array.dtype, np.floating)
            else np.float64
        )
        sanitized = np.nan_to_num(
            input_array, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )
        signal_array = np.asarray(sanitized, dtype=np.float64, order="C")

        convolver = self.select_convolver(
            signal_array.shape[-1], self.fir_kernel.shape[0]
        )

        if signal_array.ndim == 1:
            corrected = convolver(signal_array, self.fir_kernel, mode="same")
        elif signal_array.ndim == 2:
            corrected = convolver(
                signal_array,
                self.fir_kernel[np.newaxis, :],
                mode="same",
                axes=-1,
            )
        else:
            raise ValueError("audio signal must be 1D or 2D")

        corrected = np.nan_to_num(
            corrected, copy=False, nan=0.0, posinf=0.0, neginf=0.0
        )
        return np.asarray(corrected, dtype=input_dtype)

    def frequency_response(
        self,
        worN: int = 8192,
    ) -> tuple[np.ndarray, np.ndarray]:
        freqs, response = signal.freqz(
            self.fir_kernel,
            worN=int(worN),
            fs=float(self.sr),
        )
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-12))
        return freqs.astype(np.float64, copy=False), magnitude_db.astype(
            np.float64, copy=False
        )

    def reconfigure(
        self,
        *,
        sr: int | None = None,
        anchors: list[list[float]] | np.ndarray | None = None,
        taps: int | None = None,
        correction_strength: float | None = None,
    ) -> AudioEqualizer:
        changed = False

        if sr is not None:
            sample_rate = int(sr)
            if sample_rate <= 0:
                raise ValueError("sr must be a positive integer")
            if sample_rate != self.sr:
                self.sr = sample_rate
                changed = True

        if taps is not None:
            num_taps = int(taps)
            if num_taps < 3:
                raise ValueError("taps must be at least 3")
            if num_taps % 2 == 0:
                num_taps += 1
            if num_taps != self.num_taps:
                self.num_taps = num_taps
                changed = True

        if correction_strength is not None:
            strength = float(correction_strength)
            if not np.isfinite(strength):
                raise ValueError("correction_strength must be finite")
            if strength != self.correction_strength:
                self.correction_strength = strength
                changed = True

        if anchors is not None:
            normalized_anchors = self.normalize_anchors(anchors, self.sr)
            if not np.array_equal(normalized_anchors, self.anchors):
                self.anchors = normalized_anchors
                changed = True
        elif changed:
            self.anchors = self.normalize_anchors(self.anchors, self.sr)

        if changed:
            self.build_filter()

        return self


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
        sr: int,
        **config: dict,
    ) -> None:
        cfg = SmartMasteringConfig()

        if isinstance(config, dict):
            for key, val in config.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)

        self.resampling_target = cfg.resampling_target

        default_anchors = [
            [9000.0, 0.0],
            [self.resampling_target / 2.0, 3.0],
        ]

        self.anchors = (
            cfg.anchors if cfg.anchors is not None else default_anchors
        )

        self.sr = sr
        self.drive_db = cfg.drive_db
        self.ceil_db = cfg.ceil_db
        self.num_bands = cfg.num_bands
        self._slope_db = cfg.slope_db
        self.slope_hz = cfg.slope_hz
        self.smoothing_fraction = cfg.smoothing_fraction
        self.target_lufs = cfg.target_lufs
        self.low_cut = cfg.low_cut
        self.high_cut = cfg.high_cut
        self.correction_strength = cfg.correction_strength
        self.phase_type = cfg.phase_type

        self.update_bands()

        self.nperseg = int(2 ** np.ceil(np.log2(self.resampling_target / 5)))

        self.target_freqs_hz = np.array(
            generate_bands(0.1, self.resampling_target / 2.0, self.num_bands),
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
            fs=self.resampling_target,
            nperseg=NPERSEG,
            noverlap=int(NPERSEG * 0.75),
            window=("kaiser", 18),
            scaling="density",
            average="median",
            detrend="constant",
        )

        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
        psd_db = np.clip(psd_db, FLOOR_DB, CEILING_DB)

        return psd_db, f_axis

    def compute_spectrum(self, f_axis: np.ndarray) -> np.ndarray:
        min_freq = self.low_cut if self.low_cut else 0.1
        max_freq = self.high_cut if self.high_cut else f_axis[-1]

        f_axis_safe = np.maximum(f_axis, min_freq)
        f_axis_safe = np.minimum(f_axis_safe, max_freq)
        bounded_octaves = np.clip(
            np.log2(f_axis_safe / np.clip(self.slope_hz, min_freq, max_freq)),
            -2.0,
            2.0,
        )
        target_db = -self.slope_db * bounded_octaves

        return np.nan_to_num(target_db, nan=0.0, posinf=0.0, neginf=0.0)

    def apply_anchor_correction(self, y: np.ndarray) -> np.ndarray:
        return AudioEqualizer(
            sr=self.resampling_target,
            anchors=self.anchors,
            correction_strength=self.correction_strength,
        ).apply_correction(y)

    def apply_limiter(
        self,
        y: np.ndarray,
        drive_db: float,
        ceil_db: float | None,
        os_factor: int = 2,
        lookahead_ms: float = 3.0,
        attack_ms: float = 0.03,
        release_ms_min: float = 24.0,
        release_ms_max: float = 110.0,
        soft_clip_ratio: float = 0.82,
        up_beta: float = 16.0,
        down_beta: float = 20.0,
    ) -> np.ndarray:
        input_dtype = (
            y.dtype if np.issubdtype(y.dtype, np.floating) else np.float64
        )
        y_in = np.asarray(y, dtype=np.float64)

        orig_len = y_in.shape[-1]
        drive_lin = 10.0 ** (drive_db / 20.0)
        sr_os = self.resampling_target * os_factor

        y_os = signal.resample_poly(
            y_in, os_factor, 1, axis=-1, window=("kaiser", up_beta)
        )
        y_driven = y_os * drive_lin

        if ceil_db is None:
            out = signal.resample_poly(
                y_driven, 1, os_factor, axis=-1, window=("kaiser", down_beta)
            )
        else:
            limit_lin = 10.0 ** (ceil_db / 20.0)
            lookahead_samp = max(1, int(round(lookahead_ms * sr_os / 1000.0)))
            y_driven_work = np.pad(
                y_driven,
                (*((0, 0),) * (y_driven.ndim - 1), (0, lookahead_samp)),
                mode="edge" if y_driven.shape[-1] > 0 else "constant",
            )

            abs_driven = np.abs(y_driven_work)
            linked_env = (
                np.max(abs_driven, axis=0) if y_driven.ndim > 1 else abs_driven
            )

            peak_env = maximum_filter1d(
                linked_env, size=lookahead_samp, mode="reflect"
            )

            rms_win = max(1, int(round(sr_os * 0.035)))
            rms_env = np.sqrt(
                maximum_filter1d(
                    linked_env * linked_env, size=rms_win, mode="reflect"
                )
            )
            crest = peak_env / (rms_env + 1e-12)

            release_ms = np.clip(
                release_ms_max / np.sqrt(np.maximum(crest, 1.0)),
                release_ms_min,
                release_ms_max,
            )
            attack_coeff = np.exp(-1.0 / max(sr_os * attack_ms / 1000.0, 1e-9))
            release_coeff = np.exp(
                -1.0 / np.maximum(sr_os * release_ms / 1000.0, 1e-9)
            )

            control_env = 0.88 * peak_env + 0.12 * rms_env
            control_smooth = limiter_smooth_env(
                control_env, attack_coeff, release_coeff
            )
            gain = np.minimum(
                1.0, limit_lin / np.maximum(control_smooth, 1e-12)
            )

            y_delayed = np.empty_like(y_driven_work)
            y_delayed[..., :lookahead_samp] = y_driven_work[..., :1]
            y_delayed[..., lookahead_samp:] = y_driven_work[
                ..., :-lookahead_samp
            ]

            y_limited = y_delayed * gain

            density = np.clip((2.2 - crest) / 1.4, 0.0, 1.0)
            clip_mix = float(
                np.clip(soft_clip_ratio + 0.06 * np.median(density), 0.0, 0.95)
            )
            y_clipped = np.tanh(y_limited / max(limit_lin, 1e-12)) * limit_lin
            y_saturated = y_limited * (1.0 - clip_mix) + y_clipped * clip_mix

            y_down = signal.resample_poly(
                y_saturated, 1, os_factor, axis=-1, window=("kaiser", down_beta)
            )

            delay_samp = max(
                0, int(round(lookahead_ms * self.resampling_target / 1000.0))
            )
            start = min(delay_samp, y_down.shape[-1])
            out = y_down[..., start : start + orig_len]

        if out.shape[-1] < orig_len:
            out = np.pad(
                out,
                (*((0, 0),) * (out.ndim - 1), (0, orig_len - out.shape[-1])),
                mode="edge" if out.shape[-1] > 0 else "constant",
            )

        out = out - (
            np.mean(out, axis=-1, keepdims=True)
            if out.ndim > 1
            else np.mean(out)
        )

        if ceil_db is not None:
            out = np.clip(out, -limit_lin, limit_lin)

        return out.astype(input_dtype, copy=False)

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

        return smoothed

    def apply_phase_correction(self, y: np.ndarray, tp: str) -> np.ndarray:
        orig_len = y.shape[-1]
        y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

        input_db, f_axis = self.measure_spectrum(y_mono)
        target_db = self.compute_spectrum(f_axis)

        if not np.all(np.isfinite(target_db)):
            target_db = np.nan_to_num(
                target_db, nan=0.0, posinf=0.0, neginf=0.0
            )

        target_db = self.smooth_curve(
            target_db, f_axis, self.smoothing_fraction
        )

        min_len = min(len(target_db), len(input_db))

        target_db = target_db[:min_len]
        input_db = input_db[:min_len]

        mean_target = np.mean(target_db)
        mean_input = np.mean(input_db)
        target_db = target_db - mean_target + mean_input

        correction_db = (target_db - input_db) * self.correction_strength
        correction_db = np.nan_to_num(
            correction_db, nan=0.0, posinf=0.0, neginf=0.0
        )

        correction_center = float(np.median(correction_db))
        centered_correction = correction_db - correction_center
        phase_limit_db = 12.0
        max_abs_correction = float(
            np.quantile(np.abs(centered_correction), 0.98)
        )
        correction_scale = (
            phase_limit_db / max_abs_correction
            if np.isfinite(max_abs_correction)
            and max_abs_correction > phase_limit_db
            else 1.0
        )
        correction_db_norm = np.clip(
            centered_correction * correction_scale,
            -phase_limit_db,
            phase_limit_db,
        )
        H_lin = np.nan_to_num(
            10.0 ** (correction_db_norm / 20.0),
            nan=1.0,
            posinf=10.0 ** (phase_limit_db / 20.0),
            neginf=10.0 ** (-phase_limit_db / 20.0),
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

            causal_cepstrum = cepstrum * lifter

            min_phase_spectrum = np.exp(np.fft.rfft(causal_cepstrum))

            h_ir = np.fft.irfft(min_phase_spectrum)

            FIR_LEN = min(len(h_ir), 8192)
            h_ir_cut = h_ir[:FIR_LEN]

            window = np.ones(FIR_LEN)
            fade_size = FIR_LEN // 2
            if fade_size > 0:
                window[-fade_size:] = np.hamming(fade_size * 2)[fade_size:]

            h_ir_final = h_ir_cut * window

            h_ir_final = h_ir_final[np.newaxis, :]

            phase_pad = min(max(FIR_LEN // 2, 1), max(orig_len - 1, 0))
            if phase_pad > 0:
                pad_mode = "reflect" if phase_pad <= orig_len - 1 else "edge"
                y_work = np.pad(
                    y,
                    (*((0, 0),) * (y.ndim - 1), (phase_pad, phase_pad)),
                    mode=pad_mode,
                )
            else:
                y_work = y

            out = signal.oaconvolve(y_work, h_ir_final, mode="full")
            out = out[..., : y_work.shape[-1]]
            if phase_pad > 0:
                out = out[..., phase_pad : phase_pad + orig_len]

            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "minimal_phase output contains non-finite values"
            )

        elif tp == "linear":
            H_full = np.concatenate([H_lin, H_lin[-2:0:-1]])

            h = np.real(np.fft.ifft(H_full))

            h_centered = np.fft.fftshift(h)
            FIR_LEN = min(len(h), 8192)

            center = len(h_centered) // 2
            half_len = FIR_LEN // 2
            h_final = h_centered[center - half_len : center + half_len]

            h_final *= signal.windows.hamming(len(h_final))
            h_final = h_final[np.newaxis, :]

            phase_pad = min(max(len(h_final) // 2, 1), max(orig_len - 1, 0))
            if phase_pad > 0:
                pad_mode = "reflect" if phase_pad <= orig_len - 1 else "edge"
                y_work = np.pad(
                    y,
                    (*((0, 0),) * (y.ndim - 1), (phase_pad, phase_pad)),
                    mode=pad_mode,
                )
            else:
                y_work = y

            out = signal.oaconvolve(y_work, h_final, mode="same")
            if phase_pad > 0:
                out = out[..., phase_pad : phase_pad + orig_len]
            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "linear_phase output contains non-finite values"
            )
        else:
            raise ValueError(f"Unknown phase type: {tp}")

        if out is None:
            raise RuntimeError("Phase correction failed to produce output")

        out = (
            out[..., :orig_len]
            if out.shape[-1] > orig_len
            else np.pad(
                out,
                (*((0, 0),) * (out.ndim - 1), (0, orig_len - out.shape[-1])),
                mode="edge" if out.shape[-1] > 0 else "constant",
            )
        )

        return out

    def multiband_compress(self, y: np.ndarray) -> np.ndarray:
        def lr4(x, fc):
            if x.shape[-1] <= 9:
                return x, x

            sr2 = self.resampling_target / 2.0
            sos_l = signal.butter(2, fc / sr2, btype="low", output="sos")
            sos_h = signal.butter(2, fc / sr2, btype="high", output="sos")
            lp = signal.sosfiltfilt(sos_l, x)
            hp = signal.sosfiltfilt(sos_h, x)

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
            ac = np.exp(
                -1.0 / (self.resampling_target * attack_ms / 1000.0 + 1e-9)
            )
            rc = np.exp(
                -1.0 / (self.resampling_target * release_ms / 1000.0 + 1e-9)
            )
            mk = 10.0 ** (makeup_db / 20.0)
            env = decoupled_envelope(20.0 * np.log10(np.abs(x) + 1e-12), ac, rc)
            hk = knee_db / 2.0
            gain = np.zeros_like(env)
            above = env > threshold_db + hk
            in_kn = (env > threshold_db - hk) & ~above
            gain[above] = -(env[above] - threshold_db) * (1.0 - 1.0 / ratio)
            ki = env[in_kn] - threshold_db + hk
            gain[in_kn] = -(ki**2) / (2.0 * knee_db) * (1.0 - 1.0 / ratio)

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
            fcs = [b["fc"] for b in self.bands if b["fc"] > 0]
            if not fcs:
                return ch

            sorted_bands = sorted(self.bands, key=lambda b: b["fc"])
            fcs_sorted = [b["fc"] for b in sorted_bands if b["fc"] > 0]
            signal_parts = split_bands(ch, fcs_sorted)

            comps: list[np.ndarray] = []
            for band_cfg, sig in zip(sorted_bands, signal_parts, strict=False):
                thr = band_cfg["base_threshold"] + band_cfg["makeup_db"]
                comps.append(
                    compress(
                        sig,
                        thr,
                        band_cfg["ratio"],
                        band_cfg["attack_ms"],
                        band_cfg["release_ms"],
                        band_cfg["makeup_db"],
                        knee_db=band_cfg["knee_db"],
                    )
                )

            return np.sum(comps, axis=0)

        if y.ndim > 1:
            return np.vstack([process_ch(y_ch) for y_ch in y])
        else:
            return process_ch(y)

    def process(self, y: np.ndarray, sr: int | None = None) -> np.ndarray:
        if sr is not None:
            self.sr = sr

        if self.sr != self.resampling_target:
            y = resample(y, self.sr, self.resampling_target)

        y = stereo(y)

        log("Mastering", "Applying pre-filtering...")

        y = freq_cut(
            y,
            self.resampling_target,
            low_cut=self.low_cut,
            high_cut=self.high_cut,
        )

        log("Mastering", "Applying multiband compression...")

        self.update_bands()

        y = self.multiband_compress(y)

        log("Mastering", "Applying exciter...")

        y = apply_exciter(y, self.resampling_target)

        log("Mastering", "Removing audio spikes...")

        y = remove_spectral_spikes(y)

        log("Mastering", "Applying phase correction...")

        y = self.apply_phase_correction(y, self.phase_type)

        log("Mastering", "Applying anchor correction...")

        y = self.apply_anchor_correction(y)

        y = stereo(y)

        log("Mastering", "Applying stereo widening...")

        y = stereo_widen(y)

        log("Mastering", "Applying LUFS normalization...")

        y = apply_lufs(
            y,
            self.resampling_target,
            self.target_lufs,
        )

        log("Mastering", "Applying limiter...")

        y = self.apply_limiter(
            y,
            self.drive_db,
            self.ceil_db,
        )

        log("Mastering", "Applying final filtering...")

        y = freq_cut(
            y,
            self.resampling_target,
            low_cut=self.low_cut,
            high_cut=self.high_cut,
        )

        log("Mastering", "Applying final loudness targeting...")

        y = apply_lufs(
            y,
            self.resampling_target,
            self.target_lufs,
        )

        if self.ceil_db is not None:
            log("Mastering", "Applying final ceiling...")

            lin_amp = 10 ** (self.ceil_db / 20.0)
            y = np.clip(y, -lin_amp, lin_amp)

        y = freq_cut(
            y,
            self.resampling_target,
            low_cut=self.low_cut,
            high_cut=self.high_cut,
        )

        log("Mastering", "Mastering complete.")

        return self.resampling_target, y


def master(
    input_path: str,
    output_path: str = None,
    target_format: str = "mp3",
    **kwargs: dict,
) -> str | None:

    from definers.system import tmp

    from .io import save_audio

    try:
        from .io import read_audio

        sr, y = read_audio(input_path)

        sr_mastered, y_mastered = SmartMastering(sr, **kwargs).process(y, sr)

        output_path = output_path or tmp(target_format, keep=False)
        save_audio(
            destination_path=output_path,
            audio_signal=y_mastered,
            sample_rate=sr_mastered,
            output_format=target_format,
        )
        return output_path

    except Exception as e:
        from definers.system import catch

        catch(e)
        return None
