from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal


def audio_eq(
    audio_data: np.ndarray,
    anchors: list[list[float]] | np.ndarray,
    sample_rate: int = 44100,
    nperseg: int = 8192,
) -> np.ndarray:
    anchors = sorted(anchors, key=lambda x: x[0])
    anchor_freqs = np.array([a[0] for a in anchors])
    anchor_gains_db = np.array([a[1] for a in anchors])

    unique_freqs, indices = np.unique(anchor_freqs, return_index=True)
    anchor_freqs = unique_freqs
    anchor_gains_db = anchor_gains_db[indices]

    f_axis, _times, stft_frames = signal.stft(
        audio_data,
        fs=sample_rate,
        nperseg=nperseg,
    )

    log_f = np.log10(f_axis + 1e-5)
    log_anchor_freqs = np.log10(anchor_freqs)

    interp_gains_db = np.interp(
        log_f,
        log_anchor_freqs,
        anchor_gains_db,
        left=anchor_gains_db[0],
        right=anchor_gains_db[-1],
    )

    gain_multipliers = 10 ** (interp_gains_db / 20)
    modified_frames = stft_frames * gain_multipliers[:, None]

    _reconstructed_times, output_audio = signal.istft(
        modified_frames,
        fs=sample_rate,
        nperseg=nperseg,
    )

    orig_len = audio_data.shape[-1]
    if output_audio.shape[-1] > orig_len:
        output_audio = output_audio[..., :orig_len]
    elif output_audio.shape[-1] < orig_len:
        pad_width = orig_len - output_audio.shape[-1]
        output_audio = (
            np.pad(output_audio, ((0, 0), (0, pad_width)))
            if output_audio.ndim > 1
            else np.pad(output_audio, (0, pad_width))
        )

    if np.issubdtype(audio_data.dtype, np.integer):
        info = np.iinfo(audio_data.dtype)
        return np.clip(output_audio, info.min, info.max).astype(
            audio_data.dtype
        )

    return output_audio.astype(np.float32)


def smooth_curve(
    self,
    curve: np.ndarray,
    f_axis: np.ndarray,
    smoothing_fraction: float | None = None,
) -> np.ndarray:
    if smoothing_fraction is None:
        return curve

    smoothed = np.copy(curve)

    for index, frequency_hz in enumerate(f_axis):
        bandwidth = frequency_hz * (
            2**smoothing_fraction - 2 ** (-smoothing_fraction)
        )
        low_f = frequency_hz - bandwidth / 2
        high_f = frequency_hz + bandwidth / 2
        mask = (f_axis >= low_f) & (f_axis <= high_f)

        if np.any(mask):
            smoothed[index] = np.mean(curve[mask])

    return smoothed


def apply_eq(
    self,
    y: np.ndarray,
    audio_eq_fn: Callable[..., np.ndarray],
) -> np.ndarray:
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y

    input_db, f_axis = self.measure_spectrum(y_mono)
    input_db = self.smooth_curve(input_db, f_axis, self.smoothing_fraction)

    target_db = self.build_target_curve(f_axis)

    correction_db = target_db - input_db
    correction_db = np.nan_to_num(
        correction_db,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    self.spectral_balance_profile = self.build_spectral_balance_profile(
        correction_db,
        f_axis,
    )

    eq_stride = max(1, len(correction_db) // 192)

    correction_db = np.append(correction_db[:-1:eq_stride], correction_db[-1])
    f_axis = np.append(f_axis[:-1:eq_stride], f_axis[-1])

    dec_eq = np.average([correction_db[0], correction_db[-1]])
    correction_db -= dec_eq
    correction_db[0], correction_db[-1] = 0.0, 0.0

    correction_db *= self.spectral_balance_profile.correction_strength
    correction_db = np.clip(
        correction_db,
        -self.spectral_balance_profile.max_cut_db,
        self.spectral_balance_profile.max_boost_db,
    )

    flat_anchors = np.column_stack((f_axis, correction_db))

    def eq_channel(channel: np.ndarray) -> np.ndarray:
        channel = audio_eq_fn(
            audio_data=channel,
            anchors=flat_anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )
        return audio_eq_fn(
            audio_data=channel,
            anchors=self.anchors,
            sample_rate=self.resampling_target,
            nperseg=self.analysis_nperseg,
        )

    if y.ndim > 1:
        return np.vstack([eq_channel(channel) for channel in y])

    return eq_channel(y)
