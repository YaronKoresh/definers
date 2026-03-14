
from __future__ import annotations

import numpy as np
from scipy import signal

from .config import SmartMasteringConfig
from .dsp import decoupled_envelope, limiter_smooth_env, resample
from .effects import apply_exciter, mix_audio, pad_audio, stereo
from .filters import freq_cut


def generate_bands(start_freq: float, end_freq: float, num_bands: int) -> list[float]:
    if num_bands < 2:
        return [start_freq]

    bands = []
    factor = (end_freq / start_freq) ** (1 / (num_bands - 1))

    for i in range(num_bands):
        freq = start_freq * (factor**i)
        bands.append(freq)

    return bands


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

        psd_db = 10.0 * np.log10(np.maximum(psd, 1e-24))
        psd_db = np.clip(psd_db, FLOOR_DB, CEILING_DB)

        return psd_db, f_axis

    def compute_spectrum(self, f_axis: np.ndarray) -> np.ndarray:
        min_freq = self.low_cut if self.low_cut else 0.1
        max_freq = self.high_cut if self.high_cut else f_axis[-1]

        f_axis_safe = np.maximum(f_axis, min_freq)
        f_axis_safe = np.minimum(f_axis_safe, max_freq)

        octaves = np.log2(f_axis_safe / self.slope_hz)

        target_db = -self.slope_db * octaves

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

        return smoothed

    def apply_phase_correction(self, y: np.ndarray, tp: str) -> np.ndarray:
        start = __import__("time").perf_counter()

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

        correction_db = target_db - input_db
        correction_db = (target_db - input_db) * self.correction_strength

        correction_db_norm = correction_db - np.max(correction_db)
        H_lin = 10.0 ** (correction_db_norm / 20.0)

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

            out = signal.oaconvolve(y, h_ir_final, mode="full")

            if not np.all(np.isfinite(out)):
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            assert np.all(np.isfinite(out)), (
                "minimal_phase output contains non-finite values"
            )
            end = __import__("time").perf_counter()
            self._profile("minimal_phase", end - start)

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

            out = signal.oaconvolve(y, h_final, mode="same")
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
            )
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

            return np.sum(comps, axis=0)

        if y.ndim > 1:
            return np.vstack([process_ch(y_ch) for y_ch in y])
        else:
            return process_ch(y)

    def process(self, y: np.ndarray, sr: int | None = None) -> np.ndarray:
        orig_len = y.shape[-1]

        if sr is not None:
            self.sr = sr

        y_resamp = y

        if self.sr != self.resampling_target:
            y_resamp = resample(y_resamp, self.sr, self.resampling_target)
            self.sr = self.resampling_target

        y_stereo = stereo(y_resamp)

        y_exciter = apply_exciter(y_stereo, self.sr)

        y_mbc = self.multiband_compress(y_exciter)

        self.update_bands()

        p = self.apply_phase_correction(y_mbc, self.phase_type)

        y_stereo2 = stereo(p)

        y_widen = self.apply_stereo_widening(y_stereo2)

        y_cut = freq_cut(
            y_widen, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        y_lufs = self.apply_lufs(y_cut, self.target_lufs)

        y_limiter = self.apply_limiter(
            y_lufs, drive_db=self.drive_db, ceil_db=self.ceil_db
        )

        y_lufs2 = self.apply_lufs(y_limiter, self.target_lufs)

        y_cut2 = freq_cut(
            y_lufs2, self.sr, low_cut=self.low_cut, high_cut=self.high_cut
        )

        lin_amp = 10 ** (self.ceil_db / 20.0)

        y_clip = np.clip(y_cut2, -lin_amp, lin_amp)

        if y_clip.shape[-1] != orig_len:
            if y_clip.shape[-1] > orig_len:
                y_clip = y_clip[..., :orig_len]
            else:
                y_clip = np.pad(
                    y_clip,
                    (
                        *((0, 0),) * (y_clip.ndim - 1),
                        (0, orig_len - y_clip.shape[-1]),
                    ),
                )

        return y_clip, self.sr


def master(audio_file_path: str, audio_format: str = "mp3") -> str | None:

    from .io import save_audio
    from . import tmp

    try:
        from .io import read_audio

        sr, y = read_audio(audio_file_path)
        y_mastered, mastered_sr = SmartMastering().process(y, sr)
        output_path = tmp(audio_format, keep=False)
        save_audio(
            destination_path=output_path,
            audio_signal=y_mastered,
            sample_rate=mastered_sr,
            output_format=audio_format,
        )
        return output_path
    except Exception as e:
        from definers.system import catch

        catch(e)
        return None
