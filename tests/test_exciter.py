import importlib.util
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXCITER_PATH = ROOT / "src" / "definers" / "audio" / "effects" / "exciter.py"


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _register_package(name: str, path: Path) -> None:
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _load_exciter_module(package_name: str):
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    root_package = package_name
    definers_package = f"{root_package}.definers"
    audio_package = f"{definers_package}.audio"
    effects_package = f"{audio_package}.effects"

    _register_package(root_package, ROOT)
    _register_package(definers_package, ROOT / "src" / "definers")
    _register_package(audio_package, ROOT / "src" / "definers" / "audio")
    _register_package(
        effects_package, ROOT / "src" / "definers" / "audio" / "effects"
    )

    sys.modules[f"{definers_package}.file_ops"] = types.SimpleNamespace(
        catch=lambda *_, **__: None,
        log=lambda *_, **__: None,
    )
    sys.modules[f"{audio_package}.dsp"] = types.SimpleNamespace(
        remove_spectral_spikes=lambda values: np.asarray(
            values, dtype=np.float32
        ),
        resample=lambda values, *_args, **_kwargs: np.asarray(
            values, dtype=np.float32
        ),
    )
    sys.modules[f"{audio_package}.utils"] = types.SimpleNamespace(
        get_rms=lambda values: float(
            np.sqrt(np.mean(np.square(values), dtype=np.float32))
        ),
    )
    sys.modules[f"{effects_package}.mixing"] = types.SimpleNamespace(
        pad_audio=lambda dry, wet: (
            np.asarray(dry, dtype=np.float32),
            np.asarray(wet, dtype=np.float32),
        ),
    )

    return _load_module(f"{effects_package}.exciter", EXCITER_PATH)


def test_analyze_exciter_uses_oversampled_rate_for_cutoff(
    monkeypatch: pytest.MonkeyPatch,
):
    exciter = _load_exciter_module("_test_exciter_rate")
    captured: dict[str, int] = {}

    def fake_cutoff(signal: np.ndarray, sample_rate: int, config) -> float:
        captured["rate"] = sample_rate
        return 4200.0

    monkeypatch.setattr(exciter, "calculate_dynamic_cutoff", fake_cutoff)
    monkeypatch.setattr(
        exciter,
        "butter",
        lambda *args, **kwargs: np.array([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        exciter, "sosfiltfilt", lambda _sos, values, axis=-1: values
    )
    monkeypatch.setattr(
        exciter, "_apply_adaptive_gate", lambda values, *_args: values
    )
    monkeypatch.setattr(
        exciter,
        "_spectral_summary",
        lambda *_args, **_kwargs: (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.2,
            0.1,
        ),
    )

    signal = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
    analysis = exciter.analyze_exciter(signal, 24000)

    assert captured["rate"] == 96000
    assert analysis.cutoff_hz == pytest.approx(4200.0)


def test_spectral_summary_detects_bright_material():
    exciter = _load_exciter_module("_test_exciter_brightness")
    sample_rate = 44100
    timeline = np.arange(sample_rate // 4, dtype=np.float32) / sample_rate

    dark_signal = np.sin(2.0 * np.pi * 800.0 * timeline).astype(np.float32)
    bright_signal = dark_signal + 0.8 * np.sin(
        2.0 * np.pi * 9000.0 * timeline
    ).astype(np.float32)

    _, _, _, dark_ratio = exciter._spectral_summary(dark_signal, sample_rate)
    _, _, _, bright_ratio = exciter._spectral_summary(
        bright_signal, sample_rate
    )

    assert bright_ratio > dark_ratio
    assert bright_ratio - dark_ratio > 0.05


def test_analyze_exciter_caps_drive(monkeypatch: pytest.MonkeyPatch):
    exciter = _load_exciter_module("_test_exciter_drive")

    monkeypatch.setattr(
        exciter, "calculate_dynamic_cutoff", lambda *_args, **_kwargs: 4000.0
    )
    monkeypatch.setattr(
        exciter,
        "butter",
        lambda *args, **kwargs: np.array([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        exciter, "sosfiltfilt", lambda _sos, values, axis=-1: values
    )
    monkeypatch.setattr(
        exciter, "_apply_adaptive_gate", lambda values, *_args: values
    )
    monkeypatch.setattr(exciter, "get_rms", lambda _values: 1e-6)
    monkeypatch.setattr(
        exciter,
        "_spectral_summary",
        lambda *_args, **_kwargs: (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            0.0,
            0.0,
        ),
    )

    signal = np.zeros(512, dtype=np.float32)
    signal[256] = 1.0
    analysis = exciter.analyze_exciter(signal, 44100)

    assert analysis.drive == pytest.approx(exciter._DEFAULT_CONFIG.max_drive)


def test_analyze_exciter_reduces_mix_for_bright_transient_rich_material(
    monkeypatch: pytest.MonkeyPatch,
):
    exciter = _load_exciter_module("_test_exciter_retreat")

    monkeypatch.setattr(
        exciter, "calculate_dynamic_cutoff", lambda *_args, **_kwargs: 4000.0
    )
    monkeypatch.setattr(
        exciter,
        "butter",
        lambda *args, **kwargs: np.array([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        exciter, "sosfiltfilt", lambda _sos, values, axis=-1: values
    )
    monkeypatch.setattr(
        exciter, "_apply_adaptive_gate", lambda values, *_args: values
    )
    monkeypatch.setattr(exciter, "get_rms", lambda _values: 0.02)

    signal = np.zeros(512, dtype=np.float32)
    signal[256] = 1.0

    def analyze_with_profile(
        high_frequency_ratio: float, spectral_flatness: float
    ):
        monkeypatch.setattr(
            exciter,
            "_spectral_summary",
            lambda *_args, **_kwargs: (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
                spectral_flatness,
                high_frequency_ratio,
            ),
        )
        return exciter.analyze_exciter(signal, 44100, mix=0.8)

    dark = analyze_with_profile(0.05, 0.0)
    bright = analyze_with_profile(0.42, 0.2)

    assert bright.adaptive_mix < dark.adaptive_mix
    assert bright.drive <= dark.drive
    assert bright.transient_ducking_depth > dark.transient_ducking_depth


def test_build_transient_ducking_curve_reduces_wet_gain_around_impulse():
    exciter = _load_exciter_module("_test_exciter_ducking")
    signal = np.zeros(256, dtype=np.float32)
    signal[128] = 1.0

    curve = exciter._build_transient_ducking_curve(signal, 44100, 0.6)

    assert curve.shape == signal.shape
    assert float(np.min(curve)) < 1.0
    assert curve[128] < curve[0]


def test_calculate_dynamic_cutoff_enforces_safe_floor(
    monkeypatch: pytest.MonkeyPatch,
):
    exciter = _load_exciter_module("_test_exciter_cutoff_floor")

    monkeypatch.setattr(
        exciter,
        "_spectral_summary",
        lambda *_args, **_kwargs: (
            np.linspace(100.0, 4000.0, 64, dtype=np.float32),
            np.ones(64, dtype=np.float32),
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        exciter,
        "_calculate_spectral_features",
        lambda *_args, **_kwargs: (260.0, 180.0, 0.0),
    )

    cutoff = exciter.calculate_dynamic_cutoff(
        np.ones(1024, dtype=np.float32),
        44100,
    )

    assert cutoff == pytest.approx(
        exciter._DEFAULT_CONFIG.min_adaptive_cutoff_hz
    )


def test_apply_exciter_preserves_core_output_gain(
    monkeypatch: pytest.MonkeyPatch,
):
    exciter = _load_exciter_module("_test_exciter_output")
    expected = np.full(32, 0.25, dtype=np.float32)
    analysis = exciter.ExciterAnalysis(
        cutoff_hz=4000.0,
        drive=1.0,
        oversample_factor=1,
        adaptive_mix=0.5,
        band_rms=0.1,
        spectral_flatness=0.0,
        high_frequency_ratio=0.0,
    )

    monkeypatch.setattr(
        exciter, "analyze_exciter", lambda *_args, **_kwargs: analysis
    )
    monkeypatch.setattr(
        exciter, "_apply_exciter_core", lambda *_args, **_kwargs: expected
    )

    output = exciter.apply_exciter(np.ones(32, dtype=np.float32) * 0.1, 44100)

    assert np.allclose(output, expected)


def test_apply_exciter_core_bounds_extrapolated_spectrum(
    monkeypatch: pytest.MonkeyPatch,
):
    exciter = _load_exciter_module("_test_exciter_overflow")
    analysis = exciter.ExciterAnalysis(
        cutoff_hz=1500.0,
        drive=2.0,
        oversample_factor=1,
        adaptive_mix=0.8,
        band_rms=0.1,
        spectral_flatness=0.0,
        high_frequency_ratio=0.0,
    )

    monkeypatch.setattr(
        exciter,
        "butter",
        lambda *args, **kwargs: np.array([1.0], dtype=np.float32),
    )
    monkeypatch.setattr(
        exciter,
        "sosfiltfilt",
        lambda _sos, values, axis=-1: values,
    )
    monkeypatch.setattr(
        exciter,
        "_apply_adaptive_gate",
        lambda values, *_args: values,
    )
    monkeypatch.setattr(
        exciter.np, "polyfit", lambda *_args, **_kwargs: (800.0, 800.0)
    )

    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 2048, dtype=np.float32))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = exciter._apply_exciter_core(signal, 44100, analysis)

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]

    assert not runtime_warnings
    assert np.isfinite(output).all()
