import numpy as np
import pytest

from definers.audio.mastering_loudness import (
    get_lufs,
    measure_mastering_loudness,
)

_REFERENCE_METRICS_BY_SAMPLE_RATE = {
    32000: {
        "integrated": -9.911759595162122,
        "momentary": -9.911687379542126,
        "short": -9.911676395345959,
        "lra": 0.9949244189635547,
    },
    44100: {
        "integrated": -9.929540611734835,
        "momentary": -9.929468529142024,
        "short": -9.929460694179822,
        "lra": 0.994932174406852,
    },
    48000: {
        "integrated": -9.933363362229574,
        "momentary": -9.933291306492228,
        "short": -9.93328366888048,
        "lra": 0.9949338285939238,
    },
    88200: {
        "integrated": -9.95307773582712,
        "momentary": -9.953005831393524,
        "short": -9.953001267151695,
        "lra": 0.9949422846096763,
    },
    96000: {
        "integrated": -9.954991686660025,
        "momentary": -9.954919799796267,
        "short": -9.954915200564015,
        "lra": 0.994943098726262,
    },
    176400: {
        "integrated": -9.964857685478519,
        "momentary": -9.964785879321855,
        "short": -9.964783641853328,
        "lra": 0.9949472770041528,
    },
    192000: {
        "integrated": -9.965814831686666,
        "momentary": -9.965743030415119,
        "short": -9.965742095811656,
        "lra": 0.9949476807751001,
    },
}

_BLOCK_LOUDNESS_ABSOLUTE_TOLERANCE = 2e-5


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
        abs=1e-5,
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
        expected["integrated"], abs=1e-5
    )
    assert metrics.max_momentary_lufs == pytest.approx(
        expected["momentary"],
        abs=_BLOCK_LOUDNESS_ABSOLUTE_TOLERANCE,
    )
    assert metrics.max_short_term_lufs == pytest.approx(
        expected["short"],
        abs=_BLOCK_LOUDNESS_ABSOLUTE_TOLERANCE,
    )
    assert metrics.loudness_range_lu == pytest.approx(expected["lra"], abs=1e-5)
