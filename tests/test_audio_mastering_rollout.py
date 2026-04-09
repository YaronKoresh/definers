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
MASTERING_ROOT = AUDIO_ROOT / "mastering"


def _install_scipy_stub() -> None:
    scipy_module = types.ModuleType("scipy")
    scipy_module.__version__ = "1.11.0"
    scipy_module.__path__ = []
    signal_module = types.ModuleType("scipy.signal")

    signal_module.lfilter = lambda _b, _a, y, axis=-1: np.array(
        y, dtype=np.float32, copy=True
    )
    signal_module.filtfilt = lambda _b, _a, y, axis=-1: np.array(
        y, dtype=np.float32, copy=True
    )
    signal_module.resample_poly = lambda y, up, down, axis=-1: np.array(
        y, dtype=np.float32, copy=True
    )
    signal_module.butter = lambda *args, **kwargs: "sos"
    signal_module.sosfiltfilt = lambda sos, x, axis=-1: np.array(
        x, dtype=np.float32, copy=True
    )

    scipy_module.signal = signal_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module


def _load_rollout_modules(package_name: str):
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
    mastering_package_name = f"{package_name}.mastering"
    mastering_package = types.ModuleType(mastering_package_name)
    mastering_package.__path__ = [str(MASTERING_ROOT)]
    sys.modules[mastering_package_name] = mastering_package
    _install_scipy_stub()
    config_module = _load_module(
        f"{package_name}.config", AUDIO_ROOT / "config.py"
    )
    _load_module(
        f"{mastering_package_name}.loudness",
        MASTERING_ROOT / "loudness.py",
    )
    reference_module = _load_module(
        f"{mastering_package_name}.reference",
        MASTERING_ROOT / "reference.py",
    )
    return config_module, reference_module


CONFIG_MODULE, REFERENCE_MODULE = _load_rollout_modules(
    "_test_audio_mastering_rollout_pkg.audio"
)


def _bass_heavy_fixture(sample_rate: int) -> np.ndarray:
    time_axis = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    return (0.8 * np.sin(2.0 * np.pi * 55.0 * time_axis)).astype(np.float32)


def _transient_heavy_fixture(sample_rate: int) -> np.ndarray:
    signal = np.zeros(sample_rate, dtype=np.float32)
    signal[:: max(sample_rate // 8, 1)] = 1.0
    return signal


def _dense_stereo_fixture(sample_rate: int) -> np.ndarray:
    time_axis = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    left = 0.35 * np.sin(2.0 * np.pi * 220.0 * time_axis) + 0.25 * np.sin(
        2.0 * np.pi * 440.0 * time_axis
    )
    right = 0.35 * np.sin(2.0 * np.pi * 230.0 * time_axis) + 0.25 * np.sin(
        2.0 * np.pi * 660.0 * time_axis
    )
    return np.stack([left, right], axis=0).astype(np.float32)


def _commercial_reference_fixture(sample_rate: int) -> np.ndarray:
    time_axis = np.linspace(0.0, 1.0, sample_rate, endpoint=False)
    return (
        0.65 * np.sin(2.0 * np.pi * 60.0 * time_axis)
        + 0.22 * np.sin(2.0 * np.pi * 180.0 * time_axis)
        + 0.18 * np.sin(2.0 * np.pi * 4000.0 * time_axis)
    ).astype(np.float32)


def test_rollout_fixture_reference_analysis_produces_finite_deltas():
    sample_rate = 8000
    reference = _commercial_reference_fixture(sample_rate)
    fixtures = [
        _bass_heavy_fixture(sample_rate),
        _transient_heavy_fixture(sample_rate),
        _dense_stereo_fixture(sample_rate),
        _commercial_reference_fixture(sample_rate),
    ]

    for fixture in fixtures:
        analysis = REFERENCE_MODULE.analyze_reference(
            reference, fixture, sample_rate
        )
        assert np.isfinite(analysis.integrated_lufs_delta_db)
        assert np.isfinite(analysis.spectral_tilt_delta_db_per_oct)
        assert np.isfinite(analysis.transient_density_delta)
        assert np.isfinite(analysis.stereo_motion_delta)


def test_rollout_reference_match_assist_returns_machine_readable_suggestions():
    sample_rate = 8000
    reference = _commercial_reference_fixture(sample_rate)
    config = CONFIG_MODULE.SmartMasteringConfig.edm()
    fixtures = [
        _bass_heavy_fixture(sample_rate),
        _transient_heavy_fixture(sample_rate),
        _dense_stereo_fixture(sample_rate),
    ]

    for fixture in fixtures:
        assist = REFERENCE_MODULE.reference_match_assist(
            reference,
            fixture,
            sample_rate,
            current_config=config,
        )
        assert "target_lufs" in assist.suggested_overrides
        assert "micro_dynamics_strength" in assist.suggested_overrides
        assert "stereo_tone_variation_db" in assist.suggested_overrides
        assert all(
            np.isfinite(value)
            for value in assist.remaining_delta_estimate.values()
        )
