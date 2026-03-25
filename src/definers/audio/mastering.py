from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

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


def audio_eq(
    audio_data: np.ndarray,
    anchors: list[list[float]],
    sample_rate: int = 44100,
    nperseg: int = 8192,
) -> np.ndarray:
    anchors = sorted(anchors, key=lambda x: x[0])
    anchor_freqs = np.array([a[0] for a in anchors])
    anchor_gains_db = np.array([a[1] for a in anchors])

    f, _t, Zxx = signal.stft(audio_data, fs=sample_rate, nperseg=nperseg)

    log_f = np.log10(f + 1e-5)
    log_anchor_freqs = np.log10(anchor_freqs)

    interp_gains_db = np.interp(
        log_f,
        log_anchor_freqs,
        anchor_gains_db,
        left=anchor_gains_db[0],
        right=anchor_gains_db[-1],
    )

    gain_multipliers = 10 ** (interp_gains_db / 20)

    Zxx_modified = Zxx * gain_multipliers[:, None]

    _, output_audio = signal.istft(
        Zxx_modified, fs=sample_rate, nperseg=nperseg
    )

    if len(output_audio) > len(audio_data):
        output_audio = output_audio[: len(audio_data)]
    elif len(output_audio) < len(audio_data):
        output_audio = np.pad(
            output_audio, (0, len(audio_data) - len(output_audio))
        )

    if np.issubdtype(audio_data.dtype, np.integer):
        info = np.iinfo(audio_data.dtype)
        return np.clip(output_audio, info.min, info.max).astype(
            audio_data.dtype
        )

    return output_audio.astype(np.float32)


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

        nyquist_hz = self.resampling_target / 2.0 - 1.0

        self.low_cut = cfg.low_cut or 0.1
        self.high_cut = cfg.high_cut or nyquist_hz

        self.slope_hz = float(
            np.clip(cfg.stop_bass_boost_hz, self.low_cut, self.high_cut)
        )
        self._slope_db = cfg.bass_boost_db_per_oct

        bass_octaves = float(np.log2(self.slope_hz / self.low_cut))
        float(np.log2(self.high_cut / cfg.start_treble_boost_hz))

        bass_gain_db = bass_octaves * self._slope_db
        treble_gain_db = bass_gain_db * cfg.treble_boost_ratio

        self.anchors = [
            [self.low_cut, 0.0],
            [self.low_cut + 1.0, bass_gain_db],
            [self.slope_hz, 0.0],
            [cfg.start_treble_boost_hz, 0.0],
            [self.high_cut - 1.0, treble_gain_db],
            [self.high_cut, 0.0],
        ]

        self.sr = sr
        self.drive_db = cfg.drive_db
        self.ceil_db = cfg.ceil_db
        self.num_bands = cfg.num_bands
        self.smoothing_fraction = cfg.smoothing_fraction
        self.target_lufs = cfg.target_lufs
        self.correction_strength = cfg.correction_strength

        self.update_bands()

        self.analysis_nperseg = (
            int(2 ** np.ceil(np.log2(self.resampling_target / 5))) * 8
        )
        self.fft_n = self.analysis_nperseg * 8

        self.target_freqs_hz = np.array(
            generate_bands(0.1, self.resampling_target / 2.0, self.num_bands),
            dtype=np.float32,
        )

    def update_bands(self) -> None:
        count = self.num_bands
        fcs = generate_bands(self.low_cut, self.high_cut, count)
        self.bands = SmartMasteringConfig.make_bands_from_fcs(
            fcs, self.low_cut, self.high_cut
        )

    def measure_spectrum(self, y_mono: np.ndarray) -> np.ndarray:
        FLOOR_DB = -120.0
        CEILING_DB = 20.0

        if len(y_mono) < self.analysis_nperseg:
            y_mono_padded = np.pad(
                y_mono,
                (0, self.analysis_nperseg - len(y_mono)),
            )
        else:
            y_mono_padded = y_mono

        f_axis, psd = signal.welch(
            y_mono_padded,
            fs=self.resampling_target,
            nperseg=self.analysis_nperseg,
            nfft=self.fft_n,
            noverlap=int(self.analysis_nperseg * 0.875),
            window=("kaiser", 18.0),
            scaling="density",
            average="median",
            detrend="constant",
        )

        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
        psd_db = np.clip(psd_db, FLOOR_DB, CEILING_DB)

        f_axis = np.clip(f_axis, self.low_cut, self.high_cut)

        return psd_db, f_axis

    def apply_limiter(
        self,
        y: np.ndarray,
        drive_db: float = 0.0,
        ceil_db: float | None = -0.1,
        os_factor: int = 2,
        lookahead_ms: float = 2.5,
        attack_ms: float = 25.0,
        release_ms_min: float = 40.0,
        release_ms_max: float = 160.0,
        soft_clip_ratio: float = 0.8,
        up_beta: float = 14.0,
        down_beta: float = 18.0,
    ) -> np.ndarray:
        input_dtype = (
            y.dtype if np.issubdtype(y.dtype, np.floating) else np.float32
        )
        y_in = np.asarray(y, dtype=np.float32)

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
            return curve

        smoothed = np.copy(curve)

        for i, f in enumerate(f_axis):
            bandwidth = f * (2**smoothing_fraction - 2 ** (-smoothing_fraction))

            low_f = f - bandwidth / 2
            high_f = f + bandwidth / 2

            mask = (f_axis >= low_f) & (f_axis <= high_f)

            if np.any(mask):
                smoothed[i] = np.mean(curve[mask])

        return smoothed

    def apply_eq(self, y: np.ndarray) -> np.ndarray:
        y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

        input_db, f_axis = self.measure_spectrum(y_mono)

        input_db = self.smooth_curve(input_db, f_axis, self.smoothing_fraction)

        correction_db = -input_db
        correction_db = np.nan_to_num(
            correction_db, nan=0.0, posinf=0.0, neginf=0.0
        )

        eq_flat = max(1, len(correction_db) // self.analysis_nperseg)

        correction_db = np.append(correction_db[:-1:eq_flat], correction_db[-1])
        f_axis = np.append(f_axis[:-1:eq_flat], f_axis[-1])

        dec_eq = np.average([correction_db[0], correction_db[-1]])
        correction_db -= dec_eq
        correction_db[0], correction_db[-1] = 0.0, 0.0

        flat_anchors = np.column_stack((f_axis, correction_db))

        equalized = audio_eq(
            audio_data=y_mono,
            anchors=flat_anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )

        return audio_eq(
            audio_data=equalized,
            anchors=self.anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )

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

        log("Mastering", "Applying equalizer...")

        y = self.apply_eq(y)

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
