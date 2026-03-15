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

    config_module = _load_module(
        f"{package_name}.config", AUDIO_ROOT / "config.py"
    )
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
        freq_cut=lambda y, *_, **__: y,
    )
    mastering_module = _load_module(
        f"{package_name}.mastering", AUDIO_ROOT / "mastering.py"
    )
    return config_module, mastering_module


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module(
    "_test_audio_mastering_phase_pkg"
)


def _make_mastering_instance():
    return MASTERING_MODULE.SmartMastering(
        CONFIG_MODULE.SmartMasteringConfig(
            sample_rate=8000, correction_strength=0.5
        )
    )


def test_apply_phase_correction_linear_pads_short_output(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    source = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=float)

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([0.0, -1.0, -2.0, -3.0]),
            np.array([50.0, 100.0, 200.0, 400.0]),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "compute_spectrum",
        lambda f_axis: np.array([0.0, 0.0, 0.0, 0.0]),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        MASTERING_MODULE.signal,
        "oaconvolve",
        lambda y, h, mode: np.array([[0.5, -0.5]], dtype=float),
    )

    corrected = mastering.apply_phase_correction(source, "linear")

    assert corrected.shape == source.shape
    assert np.allclose(corrected[0, :2], [0.5, -0.5])
    assert np.allclose(corrected[0, 2:], [0.0, 0.0])


def test_apply_phase_correction_rejects_unknown_phase_type(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (np.array([0.0, -1.0]), np.array([100.0, 200.0])),
    )
    monkeypatch.setattr(
        mastering,
        "compute_spectrum",
        lambda f_axis: np.array([0.0, 0.0]),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )

    with pytest.raises(ValueError, match="Unknown phase type"):
        mastering.apply_phase_correction(
            np.array([[0.1, 0.2]], dtype=float), "invalid"
        )
