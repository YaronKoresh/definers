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


def _install_scipy_stub() -> None:
    scipy_module = types.ModuleType("scipy")
    signal_module = types.ModuleType("scipy.signal")
    ndimage_module = types.ModuleType("scipy.ndimage")

    def _moving_last_axis(values, size, reducer):
        array = np.asarray(values, dtype=np.float32)
        out = np.empty_like(array)
        window = max(int(size), 1)
        for index in range(array.shape[-1]):
            start = max(0, index - window + 1)
            out[..., index] = reducer(array[..., start : index + 1], axis=-1)
        return out

    def welch(y, fs=1.0, nperseg=256, nfft=None, **kwargs):
        array = np.asarray(y, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=0)
        fft_size = int(nfft or nperseg or max(array.size, 1))
        if array.size < fft_size:
            array = np.pad(array, (0, fft_size - array.size))
        else:
            array = array[:fft_size]
        spectrum = np.fft.rfft(array)
        power = np.square(np.abs(spectrum)) / max(array.size, 1)
        freqs = np.fft.rfftfreq(array.size, d=1.0 / float(fs))
        return freqs.astype(np.float32), power.astype(np.float32)

    def stft(audio_data, fs=44100, nperseg=8192):
        array = np.asarray(audio_data, dtype=np.float32)
        if array.ndim > 1:
            array = np.mean(array, axis=0)
        freq_count = max(int(nperseg) // 2 + 1, 2)
        freqs = np.linspace(0.0, fs / 2.0, freq_count, dtype=np.float32)
        return (
            freqs,
            np.array([0.0], dtype=np.float32),
            np.tile(array, (freq_count, 1)),
        )

    def istft(Zxx_modified, fs=44100, nperseg=8192):
        output = np.mean(np.asarray(Zxx_modified, dtype=np.float32), axis=0)
        return np.array([0.0], dtype=np.float32), output

    signal_module.welch = welch
    signal_module.stft = stft
    signal_module.istft = istft
    signal_module.resample_poly = lambda y, up, down, axis=-1: np.array(
        y, copy=True
    )
    signal_module.butter = lambda *args, **kwargs: ("b", "a")
    signal_module.sosfilt = lambda sos, x: np.array(x, copy=True)
    signal_module.sosfiltfilt = lambda sos, x, axis=-1: np.array(x, copy=True)

    ndimage_module.maximum_filter1d = lambda values, size, mode="constant": (
        _moving_last_axis(values, size, np.max)
    )
    ndimage_module.uniform_filter1d = lambda values, size, mode="constant": (
        _moving_last_axis(values, size, np.mean)
    )

    scipy_module.signal = signal_module
    scipy_module.ndimage = ndimage_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module
    sys.modules["scipy.ndimage"] = ndimage_module


def _load_mastering_module(package_name: str):
    root_package_name, _, _ = package_name.rpartition(".")

    for name in list(sys.modules):
        if (
            name == package_name
            or name.startswith(f"{package_name}.")
            or name == root_package_name
            or name.startswith(f"{root_package_name}.")
        ):
            del sys.modules[name]

    root_package = types.ModuleType(root_package_name)
    root_package.__path__ = [str(ROOT / "src" / "definers")]
    sys.modules[root_package_name] = root_package

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
        remove_spectral_spikes=lambda y, *_: y,
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
    _install_scipy_stub()
    sys.modules[f"{package_name}.utils"] = types.SimpleNamespace(
        apply_lufs=lambda y, *_, **__: y,
        generate_bands=lambda *_, **__: [],
        get_lufs=lambda y, *_: -14.0,
        stereo_widen=lambda y, *_, **__: y,
    )
    sys.modules[f"{root_package_name}.file_ops"] = types.SimpleNamespace(
        log=lambda *_, **__: None,
    )
    mastering_module = _load_module(
        f"{package_name}.mastering", AUDIO_ROOT / "mastering.py"
    )
    return config_module, mastering_module


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module(
    "_test_audio_mastering_phase_pkg.audio"
)


def _make_mastering_instance():
    return MASTERING_MODULE.SmartMastering(
        8000,
        correction_strength=0.5,
    )


def test_apply_eq_handles_small_spectrum_and_preserves_samples(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    source = np.array([0.1, 0.2, 0.3, 0.4], dtype=float)
    calls: list[tuple[np.ndarray, object, int, int]] = []

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
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        calls.append(
            (
                np.array(audio_data, copy=True),
                anchors,
                sample_rate,
                nperseg,
            )
        )
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    corrected = mastering.apply_eq(source)

    assert np.array_equal(corrected, source)
    assert len(calls) == 2
    assert np.array_equal(calls[0][0], source)
    assert calls[0][2] == mastering.resampling_target
    assert calls[0][3] == mastering.analysis_nperseg


def test_apply_eq_sanitizes_non_finite_anchor_values(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[object] = []

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([np.nan, np.inf, -np.inf, 10.0]),
            np.array([50.0, 100.0, 200.0, 400.0]),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    corrected = mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3]))

    assert np.all(np.isfinite(corrected))
    assert len(captured) == 2
    flat_anchors = captured[0]
    assert np.all(np.isfinite(flat_anchors[:, 1]))
    assert flat_anchors[0, 1] == pytest.approx(0.0)
    assert flat_anchors[-1, 1] == pytest.approx(0.0)


def test_apply_eq_uses_mono_average_and_mastering_anchors(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    source = np.array(
        [[0.1, 0.3, 0.5, 0.7], [0.3, 0.5, 0.7, 0.9]],
        dtype=float,
    )
    measure_inputs: list[np.ndarray] = []
    calls: list[tuple[np.ndarray, object, int, int]] = []

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            measure_inputs.append(np.array(y, copy=True))
            or np.array([0.0, -1.0, -2.0, -3.0]),
            np.array([50.0, 100.0, 200.0, 400.0]),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        calls.append(
            (
                np.array(audio_data, copy=True),
                anchors,
                sample_rate,
                nperseg,
            )
        )
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    corrected = mastering.apply_eq(source)

    assert len(measure_inputs) == 1
    assert np.allclose(measure_inputs[0], np.mean(source, axis=0))
    assert len(calls) == 4
    assert np.allclose(calls[0][0], source[0])
    assert np.allclose(calls[2][0], source[1])
    assert calls[1][1] == mastering.anchors
    assert calls[3][1] == mastering.anchors
    assert calls[1][2] == mastering.resampling_target
    assert calls[1][3] == mastering.analysis_nperseg
    assert calls[3][3] == mastering.analysis_nperseg
    assert np.allclose(corrected, source)


def test_build_spectral_balance_profile_raises_repair_ceiling_for_edm_like_deficit():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        correction_strength=0.9,
        max_spectrum_boost_db=5.5,
        spectral_rescue_strength=0.18,
        spectral_rescue_boost_db=2.0,
        spectral_rescue_band_intensity=0.55,
    )

    correction_db = np.array([8.0, 9.0, 3.0, 8.5, 9.5], dtype=float)
    f_axis = np.array([40.0, 80.0, 1000.0, 6000.0, 12000.0], dtype=float)

    profile = mastering.build_spectral_balance_profile(correction_db, f_axis)

    assert profile.rescue_factor > 0.0
    assert profile.correction_strength > mastering.correction_strength
    assert profile.max_boost_db > mastering.max_spectrum_boost_db
    assert profile.band_intensity > mastering.config.intensity
