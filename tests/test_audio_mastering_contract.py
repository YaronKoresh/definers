import importlib.util
import sys
import types
from pathlib import Path

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
MASTERING_ROOT = AUDIO_ROOT / "mastering"


def _install_scipy_stub() -> None:
    scipy_module = types.ModuleType("scipy")
    scipy_module.__version__ = "1.11.0"
    signal_module = types.ModuleType("scipy.signal")

    signal_module.lfilter = lambda _b, _a, y, axis=-1: y
    signal_module.resample_poly = lambda y, up, down, axis=-1: y
    signal_module.butter = lambda *args, **kwargs: "sos"
    signal_module.sosfiltfilt = lambda sos, x, axis=-1: x

    scipy_module.signal = signal_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module


def _load_contract_modules(package_name: str):
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
    loudness_module = _load_module(
        f"{mastering_package_name}.loudness",
        MASTERING_ROOT / "loudness.py",
    )
    contract_module = _load_module(
        f"{mastering_package_name}.contract",
        MASTERING_ROOT / "contract.py",
    )
    return loudness_module, contract_module


LOUDNESS_MODULE, CONTRACT_MODULE = _load_contract_modules(
    "_test_audio_mastering_contract_pkg.audio"
)


def test_resolve_mastering_contract_fills_default_limits_from_target():
    contract = CONTRACT_MODULE.resolve_mastering_contract(
        None,
        target_lufs=-8.0,
        ceil_db=-0.1,
    )

    assert contract.name == "default"
    assert contract.max_short_term_lufs == pytest.approx(-6.5)
    assert contract.max_momentary_lufs == pytest.approx(-5.5)
    assert contract.max_true_peak_dbfs == pytest.approx(-0.1)


def test_assess_mastering_contract_reports_multiple_failures():
    metrics = LOUDNESS_MODULE.MasteringLoudnessMetrics(
        integrated_lufs=-3.5,
        max_short_term_lufs=-2.5,
        max_momentary_lufs=-1.5,
        loudness_range_lu=2.0,
        sample_peak_dbfs=-0.2,
        true_peak_dbfs=0.1,
        crest_factor_db=2.0,
        stereo_width_ratio=0.8,
        low_end_mono_ratio=0.4,
    )
    contract = CONTRACT_MODULE.resolve_mastering_contract(
        "edm",
        target_lufs=-5.0,
        ceil_db=-0.1,
        target_lufs_tolerance_db=0.4,
        max_short_term_lufs=-3.8,
        max_momentary_lufs=-2.8,
        min_crest_factor_db=4.0,
        max_crest_factor_db=9.0,
        max_stereo_width_ratio=0.5,
        min_low_end_mono_ratio=0.85,
    )

    assessment = CONTRACT_MODULE.assess_mastering_contract(metrics, contract)

    assert assessment.passed is False
    assert assessment.true_peak_over_db > 0.0
    assert assessment.stereo_width_over_ratio > 0.0
    assert assessment.low_end_mono_under_ratio > 0.0
    assert len(assessment.issues) >= 4
