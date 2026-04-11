from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()

from ..config import SmartMasteringConfig
from .loudness import (
    MasteringLoudnessMetrics,
    measure_mastering_loudness,
)


@dataclass(frozen=True, slots=True)
class ReferenceAnalysis:
    reference_metrics: MasteringLoudnessMetrics
    candidate_metrics: MasteringLoudnessMetrics
    integrated_lufs_delta_db: float
    short_term_lufs_delta_db: float
    momentary_lufs_delta_db: float
    true_peak_delta_db: float
    crest_factor_delta_db: float
    stereo_width_delta: float
    low_end_mono_ratio_delta: float
    spectral_tilt_delta_db_per_oct: float
    transient_density_delta: float
    stereo_motion_delta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReferenceMatchAssist:
    analysis: ReferenceAnalysis
    match_amount: float
    suggested_overrides: dict[str, float | str]
    remaining_delta_estimate: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sanitize_mono(y: np.ndarray) -> np.ndarray:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if signal.ndim > 1:
        signal = np.mean(signal, axis=0)
    return np.asarray(signal, dtype=np.float32)


def measure_spectral_tilt(y: np.ndarray, sample_rate: int) -> float:
    mono = _sanitize_mono(y)
    if mono.size == 0 or sample_rate <= 0:
        return 0.0

    fft_size = max(1024, int(2 ** np.ceil(np.log2(max(mono.size, 16)))))
    padded = mono
    if mono.size < fft_size:
        padded = np.pad(mono, (0, fft_size - mono.size))
    else:
        padded = mono[:fft_size]

    window = np.hanning(padded.size).astype(np.float32)
    spectrum = np.fft.rfft(padded * window)
    power = np.square(np.abs(spectrum))
    freqs = np.fft.rfftfreq(padded.size, d=1.0 / float(sample_rate))
    mask = (freqs >= 40.0) & (freqs <= min(sample_rate / 2.0 - 1.0, 16000.0))
    if np.count_nonzero(mask) < 2:
        return 0.0

    x_axis = np.log2(freqs[mask])
    y_axis = 10.0 * np.log10(np.maximum(power[mask], 1e-24))
    slope, _intercept = np.polyfit(x_axis, y_axis, 1)
    return float(slope)


def measure_transient_density(y: np.ndarray, sample_rate: int) -> float:
    mono = _sanitize_mono(y)
    if mono.size < 2 or sample_rate <= 0:
        return 0.0

    delta = np.abs(np.diff(mono, prepend=mono[0]))
    window_size = max(int(round(sample_rate * 0.01)), 1)
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    smoothed = np.convolve(delta, kernel, mode="same")
    threshold = float(np.mean(smoothed) + np.std(smoothed))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return 0.0
    return float(np.mean(smoothed > threshold))


def measure_stereo_motion(y: np.ndarray, sample_rate: int) -> float:
    signal = np.nan_to_num(
        np.asarray(y, dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if (
        signal.ndim < 2
        or signal.shape[0] < 2
        or signal.shape[-1] < 4
        or sample_rate <= 0
    ):
        return 0.0

    left = signal[0]
    right = signal[1]
    emphasized_left = np.diff(left, prepend=left[0])
    emphasized_right = np.diff(right, prepend=right[0])
    window_size = max(int(round(sample_rate * 0.05)), 3)
    if window_size % 2 == 0:
        window_size += 1
    kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
    left_energy = np.convolve(
        emphasized_left * emphasized_left, kernel, mode="same"
    )
    right_energy = np.convolve(
        emphasized_right * emphasized_right, kernel, mode="same"
    )
    balance = (left_energy - right_energy) / np.maximum(
        left_energy + right_energy, 1e-6
    )
    balance = np.nan_to_num(
        balance - float(np.mean(balance, dtype=np.float32)),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    balance = np.clip(balance, -1.0, 1.0)

    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    mid_energy = np.convolve(mid * mid, kernel, mode="same")
    side_energy = np.convolve(side * side, kernel, mode="same")
    width_series = side_energy / np.maximum(mid_energy + side_energy, 1e-6)
    balance_delta = np.abs(np.diff(balance, prepend=balance[0]))
    width_delta = np.abs(np.diff(width_series, prepend=width_series[0]))

    return float(
        np.clip(
            np.mean(balance_delta, dtype=np.float32) * 8.0
            + np.mean(width_delta, dtype=np.float32) * 3.0,
            0.0,
            1.0,
        )
    )


def analyze_reference(
    reference_signal: np.ndarray,
    candidate_signal: np.ndarray,
    sample_rate: int,
    *,
    low_end_mono_cutoff_hz: float = 160.0,
    true_peak_oversample_factor: int = 4,
) -> ReferenceAnalysis:
    reference_length = int(np.asarray(reference_signal).shape[-1])
    candidate_length = int(np.asarray(candidate_signal).shape[-1])
    if max(reference_length, candidate_length) >= 32768:
        with ThreadPoolExecutor(max_workers=2) as executor:
            reference_future = executor.submit(
                measure_mastering_loudness,
                reference_signal,
                sample_rate,
                true_peak_oversample_factor=true_peak_oversample_factor,
                low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
            )
            candidate_future = executor.submit(
                measure_mastering_loudness,
                candidate_signal,
                sample_rate,
                true_peak_oversample_factor=true_peak_oversample_factor,
                low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
            )
            reference_metrics = reference_future.result()
            candidate_metrics = candidate_future.result()
    else:
        reference_metrics = measure_mastering_loudness(
            reference_signal,
            sample_rate,
            true_peak_oversample_factor=true_peak_oversample_factor,
            low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
        )
        candidate_metrics = measure_mastering_loudness(
            candidate_signal,
            sample_rate,
            true_peak_oversample_factor=true_peak_oversample_factor,
            low_end_mono_cutoff_hz=low_end_mono_cutoff_hz,
        )
    reference_tilt = measure_spectral_tilt(reference_signal, sample_rate)
    candidate_tilt = measure_spectral_tilt(candidate_signal, sample_rate)
    reference_transients = measure_transient_density(
        reference_signal, sample_rate
    )
    candidate_transients = measure_transient_density(
        candidate_signal, sample_rate
    )
    reference_stereo_motion = measure_stereo_motion(
        reference_signal, sample_rate
    )
    candidate_stereo_motion = measure_stereo_motion(
        candidate_signal, sample_rate
    )

    return ReferenceAnalysis(
        reference_metrics=reference_metrics,
        candidate_metrics=candidate_metrics,
        integrated_lufs_delta_db=(
            candidate_metrics.integrated_lufs
            - reference_metrics.integrated_lufs
        ),
        short_term_lufs_delta_db=(
            candidate_metrics.max_short_term_lufs
            - reference_metrics.max_short_term_lufs
        ),
        momentary_lufs_delta_db=(
            candidate_metrics.max_momentary_lufs
            - reference_metrics.max_momentary_lufs
        ),
        true_peak_delta_db=(
            candidate_metrics.true_peak_dbfs - reference_metrics.true_peak_dbfs
        ),
        crest_factor_delta_db=(
            candidate_metrics.crest_factor_db
            - reference_metrics.crest_factor_db
        ),
        stereo_width_delta=(
            candidate_metrics.stereo_width_ratio
            - reference_metrics.stereo_width_ratio
        ),
        low_end_mono_ratio_delta=(
            candidate_metrics.low_end_mono_ratio
            - reference_metrics.low_end_mono_ratio
        ),
        spectral_tilt_delta_db_per_oct=candidate_tilt - reference_tilt,
        transient_density_delta=candidate_transients - reference_transients,
        stereo_motion_delta=candidate_stereo_motion - reference_stereo_motion,
    )


def reference_match_assist(
    reference_signal: np.ndarray,
    candidate_signal: np.ndarray,
    sample_rate: int,
    *,
    current_config: SmartMasteringConfig | None = None,
    match_amount: float | None = None,
    low_end_mono_cutoff_hz: float | None = None,
    true_peak_oversample_factor: int = 4,
) -> ReferenceMatchAssist:
    config = (
        SmartMasteringConfig() if current_config is None else current_config
    )
    amount = float(
        np.clip(
            config.reference_match_amount
            if match_amount is None
            else match_amount,
            0.0,
            1.0,
        )
    )
    cutoff_hz = float(
        config.contract_low_end_mono_cutoff_hz
        if low_end_mono_cutoff_hz is None
        else low_end_mono_cutoff_hz
    )
    analysis = analyze_reference(
        reference_signal,
        candidate_signal,
        sample_rate,
        low_end_mono_cutoff_hz=cutoff_hz,
        true_peak_oversample_factor=true_peak_oversample_factor,
    )

    suggested_low_end_amount = float(
        np.clip(
            config.low_end_mono_tightening_amount
            - analysis.low_end_mono_ratio_delta * 1.5 * amount,
            0.0,
            1.0,
        )
    )
    if suggested_low_end_amount >= 0.85:
        low_end_policy = "firm"
    elif suggested_low_end_amount >= 0.6:
        low_end_policy = "balanced"
    elif suggested_low_end_amount > 0.0:
        low_end_policy = "gentle"
    else:
        low_end_policy = "off"

    limiter_recovery_style = config.limiter_recovery_style
    if analysis.transient_density_delta < -0.02:
        limiter_recovery_style = "tight"
    elif analysis.transient_density_delta > 0.02:
        limiter_recovery_style = "glue"

    suggested_overrides: dict[str, float | str] = {
        "target_lufs": float(
            config.target_lufs - analysis.integrated_lufs_delta_db * amount
        ),
        "stereo_width": float(
            np.clip(
                config.stereo_width
                - analysis.stereo_width_delta * 1.25 * amount,
                0.8,
                1.5,
            )
        ),
        "bass_boost_db_per_oct": float(
            np.clip(
                config.bass_boost_db_per_oct
                - analysis.spectral_tilt_delta_db_per_oct * 0.22 * amount,
                0.0,
                2.5,
            )
        ),
        "treble_boost_db_per_oct": float(
            np.clip(
                config.treble_boost_db_per_oct
                - analysis.spectral_tilt_delta_db_per_oct * 0.15 * amount,
                0.0,
                2.5,
            )
        ),
        "pre_limiter_saturation_ratio": float(
            np.clip(
                config.pre_limiter_saturation_ratio
                + analysis.crest_factor_delta_db * 0.03 * amount,
                0.0,
                0.6,
            )
        ),
        "micro_dynamics_strength": float(
            np.clip(
                config.micro_dynamics_strength
                - analysis.transient_density_delta * 0.9 * amount,
                0.0,
                0.5,
            )
        ),
        "stereo_tone_variation_db": float(
            np.clip(
                config.stereo_tone_variation_db
                - analysis.stereo_motion_delta * 0.9 * amount,
                0.0,
                1.5,
            )
        ),
        "stereo_motion_mid_amount": float(
            np.clip(
                config.stereo_motion_mid_amount
                - analysis.stereo_motion_delta * 0.75 * amount,
                0.0,
                1.5,
            )
        ),
        "low_end_mono_tightening": low_end_policy,
        "low_end_mono_tightening_amount": suggested_low_end_amount,
        "limiter_recovery_style": limiter_recovery_style,
        "codec_headroom_margin_db": float(
            np.clip(
                config.codec_headroom_margin_db
                + max(analysis.true_peak_delta_db, 0.0) * 0.2 * amount,
                0.0,
                1.0,
            )
        ),
    }

    remaining_delta_estimate = {
        "integrated_lufs_delta_db": float(
            analysis.integrated_lufs_delta_db * (1.0 - amount * 0.85)
        ),
        "short_term_lufs_delta_db": float(
            analysis.short_term_lufs_delta_db * (1.0 - amount * 0.7)
        ),
        "momentary_lufs_delta_db": float(
            analysis.momentary_lufs_delta_db * (1.0 - amount * 0.7)
        ),
        "true_peak_delta_db": float(
            analysis.true_peak_delta_db * (1.0 - amount * 0.45)
        ),
        "crest_factor_delta_db": float(
            analysis.crest_factor_delta_db * (1.0 - amount * 0.65)
        ),
        "stereo_width_delta": float(
            analysis.stereo_width_delta * (1.0 - amount * 0.8)
        ),
        "low_end_mono_ratio_delta": float(
            analysis.low_end_mono_ratio_delta * (1.0 - amount * 0.8)
        ),
        "spectral_tilt_delta_db_per_oct": float(
            analysis.spectral_tilt_delta_db_per_oct * (1.0 - amount * 0.7)
        ),
        "transient_density_delta": float(
            analysis.transient_density_delta * (1.0 - amount * 0.6)
        ),
        "stereo_motion_delta": float(
            analysis.stereo_motion_delta * (1.0 - amount * 0.75)
        ),
    }

    return ReferenceMatchAssist(
        analysis=analysis,
        match_amount=amount,
        suggested_overrides=suggested_overrides,
        remaining_delta_estimate=remaining_delta_estimate,
    )


__all__ = [
    "ReferenceAnalysis",
    "ReferenceMatchAssist",
    "analyze_reference",
    "measure_spectral_tilt",
    "measure_stereo_motion",
    "measure_transient_density",
    "reference_match_assist",
]
