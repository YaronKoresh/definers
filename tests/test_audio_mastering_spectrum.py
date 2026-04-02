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

    parent_name, _, _ = package_name.rpartition(".")
    if parent_name:
        parent_package = types.ModuleType(parent_name)
        parent_package.__path__ = [str(ROOT / "src" / "definers")]
        sys.modules[parent_name] = parent_package

    package = types.ModuleType(package_name)
    package.__path__ = [str(AUDIO_ROOT)]
    sys.modules[package_name] = package

    config_module = _load_module(
        f"{package_name}.config", AUDIO_ROOT / "config.py"
    )
    sys.modules[f"{package_name}.dsp"] = types.SimpleNamespace(
        decoupled_envelope=lambda x, *_: np.zeros_like(x),
        limiter_smooth_env=lambda x, *_: x,
        remove_spectral_spikes=lambda y, *_: y,
        resample=lambda y, *_: y,
    )
    sys.modules[f"{package_name}.effects"] = types.SimpleNamespace(
        apply_exciter=lambda y, *_: y,
        mix_audio=lambda *_, **__: None,
        pad_audio=lambda *_, **__: None,
        stereo=lambda y: y if getattr(y, "ndim", 1) > 1 else np.vstack([y, y]),
    )
    sys.modules[f"{package_name}.filters"] = types.SimpleNamespace(
        freq_cut=lambda y, *_, **__: y,
    )
    sys.modules[f"{package_name}.utils"] = types.SimpleNamespace(
        apply_lufs=lambda y, *_, **__: y,
        generate_bands=lambda start, stop, count: np.geomspace(
            float(start), float(stop), int(count)
        ).tolist(),
        get_lufs=lambda y, *_: -14.0,
        stereo_widen=lambda y, *_, **__: y,
    )
    if parent_name:
        sys.modules[f"{parent_name}.file_ops"] = types.SimpleNamespace(
            log=lambda *_, **__: None,
        )
    mastering_module = _load_module(
        f"{package_name}.mastering", AUDIO_ROOT / "mastering.py"
    )
    return config_module, mastering_module


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module(
    "_test_audio_mastering_spectrum_pkg.audio"
)


def _make_mastering_instance(**overrides):
    config = {
        "resampling_target": 8000,
        "correction_strength": 0.5,
    }
    config.update(overrides)
    return MASTERING_MODULE.SmartMastering(8000, **config)


def test_measure_spectrum_pads_short_audio_and_clips_floor():
    mastering = _make_mastering_instance()
    mastering.analysis_nperseg = 16
    mastering.fft_n = 16

    spectrum_db, freqs = mastering.measure_spectrum(np.zeros(4, dtype=float))

    assert spectrum_db.shape == freqs.shape
    assert len(spectrum_db) == 9
    assert np.all(spectrum_db == -120.0)


def test_smooth_curve_averages_local_bandwidth():
    mastering = _make_mastering_instance(smoothing_fraction=1.0)
    curve = np.array([0.0, 10.0, 20.0, 30.0])
    freqs = np.array([100.0, 200.0, 400.0, 800.0])

    smoothed = mastering.smooth_curve(curve, freqs, smoothing_fraction=1.0)

    assert smoothed[0] == pytest.approx(0.0)
    assert smoothed[1] == pytest.approx(5.0)
    assert smoothed[2] == pytest.approx(10.0)
    assert smoothed[3] == pytest.approx(20.0)


def test_measure_spectrum_clips_frequency_axis_to_cut_range(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()

    monkeypatch.setattr(
        MASTERING_MODULE.signal,
        "welch",
        lambda *args, **kwargs: (
            np.array([0.0, 100.0, 10000.0], dtype=float),
            np.array([1e-30, 1.0, 1e3], dtype=float),
        ),
    )

    spectrum_db, freqs = mastering.measure_spectrum(np.ones(8, dtype=float))

    assert spectrum_db.shape == freqs.shape
    assert freqs[0] == pytest.approx(mastering.low_cut)
    assert freqs[-1] == pytest.approx(mastering.high_cut)
    assert np.all(freqs >= mastering.low_cut)
    assert np.all(freqs <= mastering.high_cut)
