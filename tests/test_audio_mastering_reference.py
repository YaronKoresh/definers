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


def _install_scipy_stub() -> None:
    scipy_module = types.ModuleType("scipy")
    scipy_module.__path__ = []
    signal_module = types.ModuleType("scipy.signal")

    signal_module.lfilter = lambda _b, _a, y, axis=-1: np.array(
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


def _load_reference_modules(package_name: str):
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
    _install_scipy_stub()
    config_module = _load_module(
        f"{package_name}.config", AUDIO_ROOT / "config.py"
    )
    _load_module(
        f"{package_name}.mastering_loudness",
        AUDIO_ROOT / "mastering_loudness.py",
    )
    reference_module = _load_module(
        f"{package_name}.mastering_reference",
        AUDIO_ROOT / "mastering_reference.py",
    )
    return config_module, reference_module


CONFIG_MODULE, REFERENCE_MODULE = _load_reference_modules(
    "_test_audio_mastering_reference_pkg.audio"
)


def test_analyze_reference_reports_quieter_and_wider_candidate():
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    reference = 0.8 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(np.float32)
    candidate = np.stack(
        [
            0.3 * np.sin(2.0 * np.pi * 110.0 * time_axis),
            -0.3 * np.sin(2.0 * np.pi * 110.0 * time_axis),
        ],
        axis=0,
    ).astype(np.float32)

    analysis = REFERENCE_MODULE.analyze_reference(reference, candidate, 8000)

    assert analysis.integrated_lufs_delta_db < 0.0
    assert analysis.stereo_width_delta > 0.0
    assert analysis.low_end_mono_ratio_delta < 0.0


def test_reference_match_assist_suggests_hotter_target_and_tighter_low_end():
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    motion = 1.0 + 0.25 * np.sin(2.0 * np.pi * 2.0 * time_axis)
    reference = np.stack(
        [
            0.75 * np.sin(2.0 * np.pi * 90.0 * time_axis) * motion,
            0.75 * np.sin(2.0 * np.pi * 90.0 * time_axis) / motion,
        ],
        axis=0,
    ).astype(np.float32)
    candidate = np.stack(
        [
            0.22 * np.sin(2.0 * np.pi * 90.0 * time_axis),
            -0.22 * np.sin(2.0 * np.pi * 90.0 * time_axis),
        ],
        axis=0,
    ).astype(np.float32)
    config = CONFIG_MODULE.SmartMasteringConfig.safe()

    assist = REFERENCE_MODULE.reference_match_assist(
        reference,
        candidate,
        8000,
        current_config=config,
        match_amount=0.5,
    )

    assert assist.suggested_overrides["target_lufs"] > config.target_lufs
    assert assist.suggested_overrides["stereo_width"] < config.stereo_width
    assert (
        assist.suggested_overrides["stereo_tone_variation_db"]
        > config.stereo_tone_variation_db
    )
    assert (
        assist.suggested_overrides["low_end_mono_tightening_amount"]
        >= config.low_end_mono_tightening_amount
    )
    assert (
        assist.remaining_delta_estimate["integrated_lufs_delta_db"]
        > assist.analysis.integrated_lufs_delta_db
    )
    assert np.isfinite(assist.analysis.stereo_motion_delta)
