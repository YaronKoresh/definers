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
    parent_name, _, _ = package_name.rpartition(".")

    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

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
    "_test_audio_mastering_generation_pkg.audio"
)


def test_make_bands_from_fcs_returns_expected_schema():
    bands = CONFIG_MODULE.SmartMasteringConfig.make_bands_from_fcs(
        [100.0, 1000.0],
        20,
        20000,
    )

    assert len(bands) == 2
    assert set(bands[0]) == {
        "fc",
        "base_threshold",
        "ratio",
        "attack_ms",
        "release_ms",
        "makeup_db",
        "knee_db",
    }
    assert bands[0]["fc"] == pytest.approx(100.0)
    assert bands[0]["release_ms"] > bands[1]["release_ms"]
    assert bands[0]["knee_db"] == pytest.approx(6.0)


def test_slope_property_triggers_band_refresh(monkeypatch: pytest.MonkeyPatch):
    update_calls: list[float] = []

    def fake_update_bands(self) -> None:
        update_calls.append(self._slope_db)
        self.bands = []

    monkeypatch.setattr(
        MASTERING_MODULE.SmartMastering, "update_bands", fake_update_bands
    )

    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    update_calls.clear()

    mastering.slope_db = 6.0

    assert mastering.slope_db == pytest.approx(6.0)
    assert update_calls == [6.0]
