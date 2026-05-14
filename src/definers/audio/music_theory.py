from __future__ import annotations

from definers.runtime_numpy import get_numpy_module

np = get_numpy_module()


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


def subdivide_beats(beat_times: np.ndarray, subdivision: int) -> np.ndarray:
    if subdivision <= 1 or len(beat_times) < 2:
        return np.array(beat_times)
    new_beats: list[float] = []
    for i in range(len(beat_times) - 1):
        start_beat = beat_times[i]
        end_beat = beat_times[i + 1]
        interval = (end_beat - start_beat) / subdivision
        for j in range(subdivision):
            new_beats.append(start_beat + j * interval)
    new_beats.append(beat_times[-1])
    return np.array(sorted(list(set(new_beats))))


def get_scale_notes(
    key: str = "C",
    scale: str = "major",
    start_octave: int = 1,
    end_octave: int = 9,
) -> np.ndarray:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    scales = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
    start_note_midi = (start_octave - 1) * 12 + notes.index(key.upper())
    scale_intervals = scales.get(scale.lower(), scales["major"])
    scale_notes: list[int] = []
    for i in range((end_octave - start_octave) * 12):
        if i % 12 in scale_intervals:
            scale_notes.append(start_note_midi + i)
    return np.array(scale_notes)


def create_sample_audio(duration_s: float, sr: int = 44100) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 440 * t)
