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
    expected_position = (
        np.log2(100.0) - np.log2(20.0)
    ) / (
        np.log2(20000.0) - np.log2(20.0)
    )
    expected_knee = CONFIG_MODULE.SmartMasteringConfig.bass_knee_db + (
        CONFIG_MODULE.SmartMasteringConfig.treb_knee_db
        - CONFIG_MODULE.SmartMasteringConfig.bass_knee_db
    ) * expected_position
    assert bands[0]["knee_db"] == pytest.approx(expected_knee)


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


def test_apply_limiter_uses_forward_lookahead_without_output_delay(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(1000, resampling_target=1000)
    source = np.array([0.8, 2.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(
        MASTERING_MODULE.signal,
        "resample_poly",
        lambda y, up, down, axis=-1: np.array(y, copy=True),
    )

    limited = mastering.apply_limiter(
        source,
        drive_db=0.0,
        ceil_db=0.0,
        os_factor=1,
        lookahead_ms=1.0,
        attack_ms=1.0,
        release_ms_min=1.0,
        release_ms_max=1.0,
        soft_clip_ratio=0.0,
        window_ms=1.0,
    )

    assert limited[0] > 0.0
    assert limited[0] == pytest.approx(source[0] / 1.88, rel=1e-6)
    assert limited[1] == pytest.approx(1.0, rel=1e-6)


def test_multiband_compress_cascades_lr4_filters(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.arange(16, dtype=float)
    sosfilt_calls: list[tuple[str, np.ndarray]] = []

    mastering.bands = [
        {
            "fc": 200.0,
            "base_threshold": 100.0,
            "ratio": 2.0,
            "attack_ms": 1.0,
            "release_ms": 1.0,
            "makeup_db": 0.0,
            "knee_db": 1.0,
        },
        {
            "fc": 1000.0,
            "base_threshold": 100.0,
            "ratio": 2.0,
            "attack_ms": 1.0,
            "release_ms": 1.0,
            "makeup_db": 0.0,
            "knee_db": 1.0,
        },
    ]

    monkeypatch.setattr(
        MASTERING_MODULE.signal,
        "butter",
        lambda order, cutoff, btype, output: f"{btype}-sos",
    )

    def fake_sosfilt(sos, x):
        signal_slice = np.array(x, copy=True)
        sosfilt_calls.append((sos, signal_slice))
        delta = 1.0 if sos == "low-sos" else 10.0
        return signal_slice + delta

    monkeypatch.setattr(MASTERING_MODULE.signal, "sosfilt", fake_sosfilt)

    mastering.multiband_compress(source)

    assert [call[0] for call in sosfilt_calls] == [
        "low-sos",
        "low-sos",
        "high-sos",
        "high-sos",
    ]
    assert np.array_equal(sosfilt_calls[1][1], source + 1.0)
    assert np.array_equal(sosfilt_calls[3][1], source + 10.0)
