import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


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

    config_module = _load_module(f"{package_name}.config", AUDIO_ROOT / "config.py")
    sys.modules[f"{package_name}.dsp"] = types.SimpleNamespace(
        decoupled_envelope=lambda x, *_: np.zeros_like(x),
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
    mastering_module = _load_module(f"{package_name}.mastering", AUDIO_ROOT / "mastering.py")
    return config_module, mastering_module


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module("_test_audio_mastering_spectrum_pkg")


def test_measure_spectrum_pads_short_audio_and_clips_floor():
    mastering = MASTERING_MODULE.SmartMastering(
        CONFIG_MODULE.SmartMasteringConfig(sample_rate=8000)
    )
    mastering.nperseg = 16

    spectrum_db, freqs = mastering.measure_spectrum(np.zeros(4, dtype=float))

    assert spectrum_db.shape == freqs.shape
    assert len(spectrum_db) == 9
    assert np.all(spectrum_db == -120.0)


def test_compute_spectrum_clamps_frequency_bounds():
    mastering = MASTERING_MODULE.SmartMastering(
        CONFIG_MODULE.SmartMasteringConfig(
            sample_rate=48000,
            slope_db=6.0,
            slope_hz=1000.0,
            low_cut=100,
            high_cut=4000,
        )
    )

    target = mastering.compute_spectrum(np.array([10.0, 100.0, 1000.0, 16000.0]))
    expected = -6.0 * np.log2(np.array([100.0, 100.0, 1000.0, 4000.0]) / 1000.0)

    assert np.allclose(target, expected)


def test_smooth_curve_averages_local_bandwidth():
    mastering = MASTERING_MODULE.SmartMastering(
        CONFIG_MODULE.SmartMasteringConfig(sample_rate=8000, smoothing_fraction=1.0)
    )
    curve = np.array([0.0, 10.0, 20.0, 30.0])
    freqs = np.array([100.0, 200.0, 400.0, 800.0])

    smoothed = mastering.smooth_curve(curve, freqs, smoothing_fraction=1.0)

    assert smoothed[0] == pytest.approx(0.0)
    assert smoothed[1] == pytest.approx(5.0)
    assert smoothed[2] == pytest.approx(10.0)
    assert smoothed[3] == pytest.approx(20.0)