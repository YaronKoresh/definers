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

    with patch.dict(sys.modules, {"pydub": pydub_module}):
        sr, audio = IO_MODULE.read_audio("demo.wav")

    assert sr == 48000
    assert audio.shape == (2, 2)
    assert audio[0, 0] == np.float32(8388607 / 8388608)
    assert audio[1, 0] == np.float32(-8388608 / 8388608)
