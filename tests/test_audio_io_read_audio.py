import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
IO_MODULE = _load_module(
    "_test_audio_io_read_audio",
    ROOT / "src" / "definers" / "audio" / "io.py",
)


class FakeAudioSegment:
    def __init__(self, samples, *, channels, frame_rate, sample_width):
        self._samples = samples
        self.channels = channels
        self.frame_rate = frame_rate
        self.sample_width = sample_width

    def get_array_of_samples(self):
        return self._samples


def test_read_audio_scales_using_sample_width():
    fake_segment = FakeAudioSegment(
        [8388607, -8388608, 4194304, -4194304],
        channels=2,
        frame_rate=48000,
        sample_width=3,
    )
    pydub_module = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(from_file=lambda path: fake_segment)
    )

    with patch.dict(
        sys.modules,
        {"pydub": pydub_module, "soundfile": types.SimpleNamespace()},
    ):
        with (
            patch.object(IO_MODULE, "exist", lambda path: True),
            patch("definers.system.install_ffmpeg", lambda: None),
        ):
            sr, audio = IO_MODULE.read_audio("demo.mp3")

    assert sr == 48000
    assert audio.shape == (2, 2)
    assert audio[0, 0] == np.float32(8388607 / 8388608)
    assert audio[1, 0] == np.float32(-8388608 / 8388608)


def test_read_audio_bootstraps_ffmpeg_before_loading():
    fake_segment = FakeAudioSegment(
        [1, 2],
        channels=1,
        frame_rate=16000,
        sample_width=2,
    )
    pydub_module = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(from_file=lambda path: fake_segment)
    )
    calls = []

    with patch.dict(
        sys.modules,
        {"pydub": pydub_module, "soundfile": types.SimpleNamespace()},
    ):
        with (
            patch.object(IO_MODULE, "exist", lambda path: True),
            patch(
                "definers.system.install_ffmpeg",
                lambda: calls.append("ffmpeg"),
            ),
        ):
            IO_MODULE.read_audio("demo.mp3")

    assert calls == ["ffmpeg"]


def test_read_audio_prefers_soundfile_for_lossless_formats():
    soundfile_module = types.SimpleNamespace(
        read=lambda path, dtype, always_2d: (
            np.array([[0.25, -0.25]], dtype=np.float32),
            48000,
        )
    )
    pydub_module = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(
            from_file=lambda path: (_ for _ in ()).throw(
                AssertionError("unexpected pydub path")
            )
        )
    )
    calls = []

    with patch.dict(
        sys.modules,
        {"pydub": pydub_module, "soundfile": soundfile_module},
    ):
        with (
            patch.object(IO_MODULE, "exist", lambda path: True),
            patch(
                "definers.system.install_ffmpeg",
                lambda: calls.append("ffmpeg"),
            ),
        ):
            sr, audio = IO_MODULE.read_audio("demo.wav")

    assert sr == 48000
    assert audio.shape == (2, 1)
    assert np.allclose(audio[:, 0], np.array([0.25, -0.25], dtype=np.float32))
    assert calls == []


def test_read_audio_missing_file_fails_before_bootstrapping_ffmpeg():
    calls = []

    with patch(
        "definers.system.install_ffmpeg",
        lambda: calls.append("ffmpeg"),
    ):
        try:
            IO_MODULE.read_audio("definitely_missing_audio_file.wav")
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("read_audio should raise FileNotFoundError")

    assert calls == []


def test_save_audio_writes_standard_pcm_wav_by_default():
    write_calls = []
    soundfile_module = types.SimpleNamespace(
        write=lambda path, data, sample_rate, subtype=None, format=None: (
            write_calls.append(
                (
                    str(path),
                    np.asarray(data).shape,
                    sample_rate,
                    subtype,
                    format,
                )
            )
        )
    )

    with patch.dict(
        sys.modules,
        {"pydub": types.SimpleNamespace(), "soundfile": soundfile_module},
    ):
        with patch("definers.system.install_ffmpeg", lambda: None):
            result = IO_MODULE.save_audio(
                "demo.wav",
                np.zeros((2, 8), dtype=np.float32),
                44100,
                bit_depth=32,
            )

    assert result == "demo.wav"
    assert write_calls == [("demo.wav", (8, 2), 44100, "FLOAT", "WAV")]


def test_save_audio_falls_back_to_rf64_when_standard_wav_write_fails():
    write_calls = []

    def fake_write(path, data, sample_rate, subtype=None, format=None):
        write_calls.append((str(path), subtype, format))
        if format == "WAV":
            raise RuntimeError("primary wav write failed")

    soundfile_module = types.SimpleNamespace(write=fake_write)

    with patch.dict(
        sys.modules,
        {"pydub": types.SimpleNamespace(), "soundfile": soundfile_module},
    ):
        with patch("definers.system.install_ffmpeg", lambda: None):
            result = IO_MODULE.save_audio(
                "fallback.wav",
                np.zeros((2, 8), dtype=np.float32),
                44100,
                bit_depth=32,
            )

    assert result == "fallback.wav"
    assert write_calls == [
        ("fallback.wav", "FLOAT", "WAV"),
        ("fallback.wav", "FLOAT", "RF64"),
    ]


def test_save_audio_uses_float_temp_wav_for_lossy_exports():
    write_calls = []
    export_calls = []

    class FakeSong:
        def export(
            self, destination_path, format, bitrate=None, parameters=None
        ):
            export_calls.append(
                (destination_path, format, bitrate, tuple(parameters))
            )

    soundfile_module = types.SimpleNamespace(
        write=lambda path, data, sample_rate, subtype=None, format=None: (
            write_calls.append(
                (
                    type(path).__name__,
                    np.asarray(data).shape,
                    sample_rate,
                    subtype,
                    format,
                )
            )
        )
    )
    pydub_module = types.SimpleNamespace(
        AudioSegment=types.SimpleNamespace(from_wav=lambda buffer: FakeSong())
    )

    with patch.dict(
        sys.modules,
        {"pydub": pydub_module, "soundfile": soundfile_module},
    ):
        with patch("definers.system.install_ffmpeg", lambda: None):
            result = IO_MODULE.save_audio(
                "demo.mp3",
                np.zeros((2, 8), dtype=np.float32),
                44100,
                bit_depth=32,
                bitrate=192,
            )

    assert result == "demo.mp3"
    assert write_calls == [("BytesIO", (8, 2), 44100, "FLOAT", "WAV")]
    assert export_calls == [
        (
            "demo.mp3",
            "mp3",
            "192k",
            ("-compression_level", "9", "-b:a", "192k"),
        )
    ]
