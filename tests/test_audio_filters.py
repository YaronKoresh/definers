import numpy as np

from definers.audio.filters import freq_cut


def test_freq_cut_preserves_signal_when_cutoffs_disabled() -> None:
    source = np.array([0.25, -0.5, 0.75, -1.0], dtype=np.float32)

    filtered = freq_cut(source, 44100, low_cut=None, high_cut=None)

    assert np.array_equal(filtered, source)


def test_freq_cut_removes_out_of_band_energy_when_cutoffs_are_explicit() -> (
    None
):
    sr = 4096
    samples = np.arange(sr, dtype=np.float32)
    source = (
        0.8 * np.sin(2.0 * np.pi * 30.0 * samples / sr)
        + 1.0 * np.sin(2.0 * np.pi * 220.0 * samples / sr)
        + 0.8 * np.sin(2.0 * np.pi * 1600.0 * samples / sr)
    ).astype(np.float32)

    filtered = freq_cut(source, sr, low_cut=100.0, high_cut=600.0)

    spectrum = np.abs(np.fft.rfft(filtered))
    frequencies = np.fft.rfftfreq(filtered.shape[-1], d=1.0 / sr)
    low_bin = int(np.argmin(np.abs(frequencies - 30.0)))
    mid_bin = int(np.argmin(np.abs(frequencies - 220.0)))
    high_bin = int(np.argmin(np.abs(frequencies - 1600.0)))

    assert spectrum[mid_bin] > spectrum[low_bin] * 20.0
    assert spectrum[mid_bin] > spectrum[high_bin] * 20.0
