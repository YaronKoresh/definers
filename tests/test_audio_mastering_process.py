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


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module("_test_audio_mastering_process_pkg")


def test_process_runs_pipeline_in_order_and_clips_output(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        CONFIG_MODULE.SmartMasteringConfig(sample_rate=22050, ceil_db=-6.0)
    )
    calls: list[str] = []

    def fake_resample(y, old_sr, new_sr):
        calls.append("resample")
        assert old_sr == 22050
        assert new_sr == mastering.resampling_target
        return np.array([0.2, 0.4, 0.6, 1.2], dtype=float)

    def fake_stereo(y):
        calls.append("stereo")
        return y if y.ndim > 1 else np.vstack([y, y])

    def fake_apply_exciter(y, sr):
        calls.append("exciter")
        assert sr == mastering.resampling_target
        return y + 0.1

    def fake_freq_cut(y, sr, low_cut=None, high_cut=None):
        calls.append("freq_cut")
        assert sr == mastering.resampling_target
        return y

    monkeypatch.setattr(MASTERING_MODULE, "resample", fake_resample)
    monkeypatch.setattr(MASTERING_MODULE, "stereo", fake_stereo)
    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", fake_apply_exciter)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", fake_freq_cut)
    monkeypatch.setattr(
        mastering,
        "multiband_compress",
        lambda y: calls.append("multiband") or y,
    )
    monkeypatch.setattr(
        mastering,
        "update_bands",
        lambda: calls.append("update_bands"),
    )
    monkeypatch.setattr(
        mastering,
        "apply_phase_correction",
        lambda y, phase_type: calls.append("phase") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_stereo_widening",
        lambda y: calls.append("widen") or y,
        raising=False,
    )
    monkeypatch.setattr(
        mastering,
        "apply_lufs",
        lambda y, target: calls.append("lufs") or y,
        raising=False,
    )
    monkeypatch.setattr(
        mastering,
        "apply_limiter",
        lambda y, drive_db, ceil_db: calls.append("limiter") or np.full_like(y, 1.0),
        raising=False,
    )

    output, sample_rate = mastering.process(np.array([0.1, 0.2, 0.3], dtype=float), 22050)

    assert calls == [
        "resample",
        "stereo",
        "exciter",
        "multiband",
        "update_bands",
        "phase",
        "stereo",
        "widen",
        "freq_cut",
        "lufs",
        "limiter",
        "lufs",
        "freq_cut",
    ]
    assert sample_rate == mastering.resampling_target
    assert output.shape == (2, 3)
    assert np.allclose(output, 10 ** (mastering.ceil_db / 20.0))


def test_process_pads_short_final_signal(monkeypatch: pytest.MonkeyPatch):
    mastering = MASTERING_MODULE.SmartMastering(
        CONFIG_MODULE.SmartMasteringConfig(sample_rate=44100, ceil_db=0.0)
    )

    monkeypatch.setattr(
        MASTERING_MODULE,
        "stereo",
        lambda y: y if getattr(y, "ndim", 1) > 1 else np.vstack([y, y]),
    )
    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, sr: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "freq_cut",
        lambda y, sr, low_cut=None, high_cut=None: y[..., :2],
    )
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "update_bands", lambda: None)
    monkeypatch.setattr(mastering, "apply_phase_correction", lambda y, phase_type: y)
    monkeypatch.setattr(mastering, "apply_stereo_widening", lambda y: y, raising=False)
    monkeypatch.setattr(mastering, "apply_lufs", lambda y, target: y, raising=False)
    monkeypatch.setattr(mastering, "apply_limiter", lambda y, drive_db, ceil_db: y, raising=False)

    output, sample_rate = mastering.process(np.array([0.1, 0.2, 0.3, 0.4], dtype=float))

    assert sample_rate == 44100
    assert output.shape == (2, 4)
    assert np.allclose(output[:, :2], [[0.1, 0.2], [0.1, 0.2]])
    assert np.allclose(output[:, 2:], 0.0)