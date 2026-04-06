import numpy as np
import pytest

from definers.audio.mastering_loudness import (
    get_lufs,
    measure_mastering_loudness,
)

_REFERENCE_METRICS_BY_SAMPLE_RATE = {
    32000: {
        "integrated": -9.911759460558093,
        "momentary": -9.911687244059383,
        "short": -9.911687244062952,
        "lra": 0.9949244189635547,
    },
    44100: {
        "integrated": -9.92954061436167,
        "momentary": -9.929468532844016,
        "short": -9.929468532847588,
        "lra": 0.994932174406852,
    },
    48000: {
        "integrated": -9.933363490017145,
        "momentary": -9.933291437740133,
        "short": -9.93329143774369,
        "lra": 0.9949338285939238,
    },
    88200: {
        "integrated": -9.953077992378073,
        "momentary": -9.95300609203843,
        "short": -9.953006092041976,
        "lra": 0.9949422846096763,
    },
    96000: {
        "integrated": -9.954991687349517,
        "momentary": -9.954919801856795,
        "short": -9.954919801860324,
        "lra": 0.994943098726262,
    },
    176400: {
        "integrated": -9.964857596499176,
        "momentary": -9.96478578779999,
        "short": -9.964785787803548,
        "lra": 0.9949472770041528,
    },
    192000: {
        "integrated": -9.965815015084956,
        "momentary": -9.965743213861371,
        "short": -9.965743213864855,
        "lra": 0.9949476807751001,
    },
}

_BLOCK_LOUDNESS_ABSOLUTE_TOLERANCE = 5e-5


def _tone(frequency_hz: float, duration_seconds: float, sr: int) -> np.ndarray:
    time_axis = np.linspace(
        0.0, duration_seconds, int(sr * duration_seconds), endpoint=False
    )
    return (0.5 * np.sin(2.0 * np.pi * frequency_hz * time_axis)).astype(
        np.float32
    )


@pytest.mark.parametrize(
    "sample_rate", sorted(_REFERENCE_METRICS_BY_SAMPLE_RATE)
)
def test_get_lufs_matches_bs1770_reference_across_sample_rates(
    sample_rate: int,
) -> None:
    tone = _tone(220.0, 10.0, sample_rate)

    assert get_lufs(tone, sample_rate) == pytest.approx(
        _REFERENCE_METRICS_BY_SAMPLE_RATE[sample_rate]["integrated"],
        abs=5e-5,
    )


@pytest.mark.parametrize(
    "sample_rate", sorted(_REFERENCE_METRICS_BY_SAMPLE_RATE)
)
def test_measure_mastering_loudness_matches_bs1770_reference_metrics_across_sample_rates(
    sample_rate: int,
) -> None:
    tone = _tone(220.0, 10.0, sample_rate)
    expected = _REFERENCE_METRICS_BY_SAMPLE_RATE[sample_rate]

    metrics = measure_mastering_loudness(tone, sample_rate)

    assert metrics.integrated_lufs == pytest.approx(
        expected["integrated"], abs=5e-5
    )
    assert metrics.max_momentary_lufs == pytest.approx(
        expected["momentary"],
        abs=_BLOCK_LOUDNESS_ABSOLUTE_TOLERANCE,
    )
    assert metrics.max_short_term_lufs == pytest.approx(
        expected["short"],
        abs=_BLOCK_LOUDNESS_ABSOLUTE_TOLERANCE,
    )
    assert metrics.loudness_range_lu == pytest.approx(expected["lra"], abs=5e-5)
