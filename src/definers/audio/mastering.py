from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from ..file_ops import log
from .config import SmartMasteringConfig
from .dsp import (
    decoupled_envelope,
    limiter_smooth_env,
    resample,
)
from .effects import apply_exciter, stereo
from .filters import freq_cut
from .utils import generate_bands, get_lufs, stereo_widen

from definers.system import get_ext

def audio_eq(
    audio_data: np.ndarray,
    anchors: list[list[float]],
    sample_rate: int = 44100,
    nperseg: int = 8192,
) -> np.ndarray:
    anchors = sorted(anchors, key=lambda x: x[0])
    anchor_freqs = np.array([a[0] for a in anchors])
    anchor_gains_db = np.array([a[1] for a in anchors])

    unique_freqs, indices = np.unique(anchor_freqs, return_index=True)
    anchor_freqs = unique_freqs
    anchor_gains_db = anchor_gains_db[indices]

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

    orig_len = audio_data.shape[-1]
    if output_audio.shape[-1] > orig_len:
        output_audio = output_audio[..., :orig_len]
    elif output_audio.shape[-1] < orig_len:
        pad_width = orig_len - output_audio.shape[-1]
        output_audio = np.pad(output_audio, ((0, 0), (0, pad_width))) if output_audio.ndim > 1 else np.pad(output_audio, (0, pad_width))

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

        self.low_cut = max(cfg.low_cut or 0.1, 0.1)
        self.high_cut = min(cfg.high_cut or nyquist_hz, nyquist_hz)

        self.slope_hz = float(
            np.clip(cfg.stop_bass_boost_hz, self.low_cut, self.high_cut)
        )
        self._slope_db = cfg.bass_boost_db_per_oct

        bass_octaves = float(np.log2(self.slope_hz / self.low_cut))
        treble_octaves = float(np.log2(self.high_cut / cfg.start_treble_boost_hz))

        bass_gain_db = bass_octaves * cfg.bass_boost_db_per_oct
        treble_gain_db = treble_octaves * cfg.treble_boost_db_per_oct

        min_boost = min(bass_gain_db, treble_gain_db)
        base = 0.0
        if min_boost < 0.0:
            bass_gain_db -= min_boost
            treble_gain_db -= min_boost
            base -= min_boost

        self.anchors = [
            [self.low_cut, 0.0],
            [self.low_cut + 5.0, base],
            [self.low_cut + 10.0, bass_gain_db],
            [self.slope_hz, base],
            [cfg.start_treble_boost_hz, base],
            [self.high_cut - 400.0, treble_gain_db],
            [self.high_cut - 200.0, base],
            [self.high_cut, 0.0],
        ]

        self.sr = sr
        self.drive_db = cfg.drive_db
        self.ceil_db = cfg.ceil_db
        self.num_bands = cfg.num_bands
        self.smoothing_fraction = cfg.smoothing_fraction
        self.target_lufs = cfg.target_lufs
        self.correction_strength = cfg.correction_strength
        self.mid_slope = cfg.mid_slope

        self.update_bands()

        self.analysis_nperseg = int(2 ** np.ceil(np.log2(self.resampling_target / 5))) * 2
        self.fft_n = self.analysis_nperseg * 2

        self.target_freqs_hz = np.array(
            generate_bands(0.1, self.resampling_target / 2.0 - 1.0, self.num_bands),
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
        ceil_db: float = -0.1,
        os_factor: int = 2,
        lookahead_ms: float = 2.5,
        attack_ms: float = 2.0,
        release_ms_min: float = 40.0,
        release_ms_max: float = 200.0,
        soft_clip_ratio: float = 0.2,
        window_ms: float = 4.0
    ) -> np.ndarray:
        input_dtype = y.dtype if np.issubdtype(y.dtype, np.floating) else np.float32
        y_in = np.asarray(y, dtype=np.float32)
        orig_len = y_in.shape[-1]

        drive_lin = 10.0 ** (drive_db / 20.0)
        limit_lin = 10.0 ** (ceil_db / 20.0)

        sr_os = self.resampling_target * os_factor
        y_os = signal.resample_poly(y_in, os_factor, 1, axis=-1)
        y_driven = y_os * drive_lin

        lookahead_samp = max(0, int(round(lookahead_ms * sr_os / 1000.0)))
        lookahead_span = lookahead_samp + 1
        abs_driven = np.abs(y_driven)
        linked_env = np.max(abs_driven, axis=0) if y_driven.ndim > 1 else abs_driven

        peak_env = maximum_filter1d(
            linked_env[::-1],
            size=lookahead_span,
            mode="constant",
        )[::-1]

        rms_win = max(1, int(round(window_ms / 1000 * sr_os)))
        rms_env = np.sqrt(
            uniform_filter1d(
                (linked_env**2)[::-1],
                size=rms_win,
                mode="constant",
            )[::-1]
        )

        crest = peak_env / (rms_env + 1e-12)
        release_ms = np.clip(release_ms_max / (crest + 1e-6), release_ms_min, release_ms_max)

        atk_c = np.exp(-1.0 / (sr_os * attack_ms / 1000.0))
        rel_c = np.exp(-1.0 / (sr_os * release_ms / 1000.0))

        control_env = 0.9 * peak_env + 0.1 * rms_env

        control_smooth = limiter_smooth_env(control_env, atk_c, rel_c)

        gain = np.ones_like(control_smooth)
        mask = control_smooth > limit_lin
        gain[mask] = limit_lin / control_smooth[mask]

        y_limited = y_driven * gain

        if soft_clip_ratio > 0:
            threshold = limit_lin * (1.0 - soft_clip_ratio)
            mask = y_limited > threshold
            y_limited[mask] = threshold + (limit_lin - threshold) * np.tanh((y_limited[mask] - threshold) / (limit_lin - threshold))
            mask_neg = y_limited < -threshold
            y_limited[mask_neg] = -threshold - (limit_lin - threshold) * np.tanh((-y_limited[mask_neg] - threshold) / (limit_lin - threshold))

        y_down = signal.resample_poly(y_limited, 1, os_factor, axis=-1)
        
        out = y_down[..., :orig_len]

        return out.astype(input_dtype)

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

        octaves_from_low_hz = np.log2(np.maximum(f_axis, self.low_cut) / self.low_cut)
        target_db = self.mid_slope * octaves_from_low_hz

        correction_db = target_db - input_db
        correction_db = np.nan_to_num(
            correction_db, nan=0.0, posinf=0.0, neginf=0.0
        )

        eq_flat = max(1, len(correction_db) // self.analysis_nperseg)

        correction_db = np.append(correction_db[:-1:eq_flat], correction_db[-1])
        f_axis = np.append(f_axis[:-1:eq_flat], f_axis[-1])

        dec_eq = np.average([correction_db[0], correction_db[-1]])
        correction_db -= dec_eq
        correction_db[0], correction_db[-1] = 0.0, 0.0

        correction_db *= self.correction_strength

        flat_anchors = np.column_stack((f_axis, correction_db))

        def eq_channel(channel: np.ndarray) -> np.ndarray:
            channel = audio_eq(
                audio_data=channel,
                anchors=flat_anchors,
                sample_rate=self.resampling_target,
                nperseg=self.analysis_nperseg,
            )
            return audio_eq(
                audio_data=channel,
                anchors=self.anchors,
                sample_rate=self.resampling_target,
                nperseg=self.analysis_nperseg,
            )
        
        if y.ndim > 1:
            return np.vstack([eq_channel(channel) for channel in y])

        return eq_channel(y)

    def multiband_compress(self, y: np.ndarray) -> np.ndarray:
        def lr4(x, fc):
            if x.shape[-1] <= 9:
                return x, x

            sr2 = self.resampling_target / 2.0
            sos_l = signal.butter(2, fc / sr2, btype="low", output="sos")
            sos_h = signal.butter(2, fc / sr2, btype="high", output="sos")

            lp = signal.sosfilt(sos_l, x)
            lp = signal.sosfilt(sos_l, lp)
            hp = signal.sosfilt(sos_h, x)
            hp = signal.sosfilt(sos_h, hp)

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

        log("Mastering", "Applying exciter...")

        y = apply_exciter(y, self.resampling_target)

        log("Mastering", "Applying multiband compression...")

        self.update_bands()

        y = self.multiband_compress(y)

        log("Mastering", "Applying equalizer...")

        y = self.apply_eq(y)

        y = stereo(y)

        log("Mastering", "Applying filtering...")

        y = freq_cut(
            y,
            self.resampling_target,
            low_cut=self.low_cut,
            high_cut=self.high_cut,
        )

        log("Mastering", "Applying stereo widening...")

        y = stereo_widen(y)

        log("Mastering", "Applying limiter...")

        current_lufs = get_lufs(y, self.resampling_target)
        lufs_diff = self.target_lufs - current_lufs
        dynamic_drive_db = max(0.0, lufs_diff)

        y = self.apply_limiter(
            y,
            drive_db=dynamic_drive_db,
            ceil_db=self.ceil_db,
        )

        log("Mastering", "Applying hard clipping...")

        lin_amp = 10 ** (self.ceil_db / 20.0)
        y = np.clip(y, -lin_amp, lin_amp)

        log("Mastering", "Mastering complete.")

        return self.resampling_target, y


def master(
    input_path: str,
    output_path: str = None,
    **kwargs: dict,
) -> str | None:

    from definers.system import tmp

    from .io import save_audio

    try:
        from .io import read_audio

        sr, y = read_audio(input_path)

        sr_mastered, y_mastered = SmartMastering(sr, **kwargs).process(y, sr)

        output_path = output_path or tmp(get_ext(input_path), keep=False)

        save_audio(
            destination_path=output_path,
            audio_signal=y_mastered,
            sample_rate=sr_mastered,
        )

        return output_path

    except Exception as e:
        from definers.system import catch

        catch(e)
        return None
