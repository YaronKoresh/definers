import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = ROOT / "src" / "definers" / "audio"


def _load_mastering_module(package_name: str):
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    package = types.ModuleType(package_name)
    package.__path__ = [str(AUDIO_ROOT)]
    sys.modules[package_name] = package

    _load_module(f"{package_name}.config", AUDIO_ROOT / "config.py")
    sys.modules[f"{package_name}.dsp"] = types.SimpleNamespace(
        decoupled_envelope=lambda x, *_: x,
        limiter_smooth_env=lambda x, *_: x,
        resample=lambda y, *_: y,
    )
    sys.modules[f"{package_name}.effects"] = types.SimpleNamespace(
        apply_exciter=lambda y, *_: y,
        mix_audio=lambda *_, **__: None,
        pad_audio=lambda *_, **__: None,
        stereo=lambda y: y if getattr(y, "ndim", 1) > 1 else np.vstack([y, y]),
    )
    sys.modules[f"{package_name}.filters"] = types.SimpleNamespace(
        freq_cut=lambda y, *_ , **__: y,
    )
    return _load_module(f"{package_name}.mastering", AUDIO_ROOT / "mastering.py")


MASTERING_MODULE = _load_mastering_module("_test_audio_mastering_wrapper_pkg")


def test_master_reads_processes_and_saves(monkeypatch):
    saved: list[dict] = []
    io_module = types.SimpleNamespace(
        read_audio=lambda path: (22050, np.array([0.1, 0.2], dtype=float)),
        save_audio=lambda **kwargs: saved.append(kwargs),
    )
    sys.modules["_test_audio_mastering_wrapper_pkg.io"] = io_module
    sys.modules["_test_audio_mastering_wrapper_pkg"].tmp = (
        lambda audio_format, keep=False: f"mastered.{audio_format}"
    )

    class FakeMastering:
        def process(self, y, sr):
            assert sr == 22050
            return np.array([[0.5, 0.25]], dtype=float), 44100

    monkeypatch.setattr(MASTERING_MODULE, "SmartMastering", FakeMastering)

    result = MASTERING_MODULE.master("input.wav", "wav")

    assert result == "mastered.wav"
    assert len(saved) == 1
    assert saved[0]["destination_path"] == "mastered.wav"
    assert np.allclose(saved[0]["audio_signal"], np.array([[0.5, 0.25]], dtype=float))
    assert saved[0]["sample_rate"] == 44100
    assert saved[0]["output_format"] == "wav"


def test_master_catches_failures_and_returns_none(monkeypatch):
    caught: list[Exception] = []
    sys.modules["_test_audio_mastering_wrapper_pkg.io"] = types.SimpleNamespace(
        read_audio=lambda path: (_ for _ in ()).throw(RuntimeError("boom")),
        save_audio=lambda **kwargs: None,
    )
    sys.modules["_test_audio_mastering_wrapper_pkg"].tmp = (
        lambda audio_format, keep=False: f"mastered.{audio_format}"
    )

    definers_package = types.ModuleType("definers")
    definers_package.__path__ = []
    sys.modules["definers"] = definers_package
    sys.modules["definers.system"] = types.SimpleNamespace(
        catch=lambda exc: caught.append(exc)
    )

    class FakeMastering:
        def process(self, y, sr):
            raise AssertionError("process should not run")

    monkeypatch.setattr(MASTERING_MODULE, "SmartMastering", FakeMastering)

    result = MASTERING_MODULE.master("broken.wav")

    assert result is None
    assert len(caught) == 1
    assert isinstance(caught[0], RuntimeError)
    assert str(caught[0]) == "boom"