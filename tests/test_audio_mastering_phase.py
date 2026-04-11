import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_module(module_name: str, module_path: Path):
    spec_kwargs = {}
    if module_path.name == "__init__.py":
        spec_kwargs["submodule_search_locations"] = [str(module_path.parent)]
    spec = importlib.util.spec_from_file_location(
        module_name,
        module_path,
        **spec_kwargs,
    )
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
    signal_module.filtfilt = lambda b, a, y, axis=-1: np.array(y, copy=True)
    signal_module.sosfilt = lambda sos, x: np.array(x, copy=True)
    signal_module.sosfiltfilt = lambda sos, x, axis=-1: np.array(x, copy=True)

    ndimage_module.maximum_filter1d = lambda values, size, mode="constant": (
        _moving_last_axis(values, size, np.max)
    )
    ndimage_module.median_filter = lambda values, size, mode="nearest": (
        np.array(values, dtype=np.float32, copy=True)
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
    effects_package = types.ModuleType(f"{package_name}.effects")
    effects_package.__path__ = []
    exciter_module = types.ModuleType(f"{package_name}.effects.exciter")
    exciter_module.apply_exciter = lambda y, *_: y
    mixing_module = types.ModuleType(f"{package_name}.effects.mixing")
    mixing_module.stereo = lambda y: (
        y if getattr(y, "ndim", 1) > 1 else np.vstack([y, y])
    )
    effects_package.exciter = exciter_module
    effects_package.mixing = mixing_module
    sys.modules[f"{package_name}.effects"] = effects_package
    sys.modules[f"{package_name}.effects.exciter"] = exciter_module
    sys.modules[f"{package_name}.effects.mixing"] = mixing_module
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
        f"{package_name}.mastering", MASTERING_ROOT / "__init__.py"
    )
    return config_module, mastering_module


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module(
    "_test_audio_mastering_phase_pkg.audio"
)
EQ_MODULE = sys.modules["_test_audio_mastering_phase_pkg.audio.mastering.eq"]


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


def test_apply_eq_preserves_dual_end_legacy_repair_when_restoration_is_hot(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[np.ndarray] = []

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([-8.0, -4.0, 0.0, -5.0, -9.0], dtype=float),
            np.array([40.0, 120.0, 1000.0, 5000.0, 12000.0], dtype=float),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        mastering,
        "build_target_curve",
        lambda f_axis: np.zeros_like(f_axis, dtype=float),
    )
    monkeypatch.setattr(
        mastering,
        "build_spectral_balance_profile",
        lambda correction_db, f_axis: MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=12.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=1.0,
            body_restoration_factor=1.0,
        ),
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))

    flat_anchors = captured[0]

    assert flat_anchors[1, 1] > 0.0
    assert flat_anchors[-2, 1] > 0.0
    assert flat_anchors[2, 1] < flat_anchors[1, 1]
    assert flat_anchors[2, 1] < flat_anchors[-2, 1]


def test_apply_eq_pushes_presence_repair_for_closed_legacy_material(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[np.ndarray] = []

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([-4.0, -6.0, -6.5, -8.0, -9.2, -9.5], dtype=float),
            np.array(
                [120.0, 700.0, 2200.0, 5000.0, 9000.0, 15000.0], dtype=float
            ),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        mastering,
        "build_target_curve",
        lambda f_axis: np.zeros_like(f_axis, dtype=float),
    )
    monkeypatch.setattr(
        mastering,
        "build_spectral_balance_profile",
        lambda correction_db, f_axis: MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=12.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=1.0,
            body_restoration_factor=0.7,
        ),
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))

    flat_anchors = captured[0]

    assert flat_anchors[2, 1] > 2.0
    assert flat_anchors[3, 1] > flat_anchors[2, 1]
    assert flat_anchors[4, 1] > flat_anchors[2, 1]


def test_apply_eq_cuts_mud_and_restores_treble_for_old_closed_material(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[np.ndarray] = []

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([0.0, 4.6, 3.4, -3.8, -6.2, -8.4], dtype=float),
            np.array(
                [70.0, 220.0, 420.0, 2400.0, 7000.0, 12000.0], dtype=float
            ),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        mastering,
        "build_target_curve",
        lambda f_axis: np.zeros_like(f_axis, dtype=float),
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))

    flat_anchors = captured[0]

    assert flat_anchors[1, 1] < -3.0
    assert flat_anchors[2, 1] < -1.9
    assert flat_anchors[3, 1] > 3.0
    assert flat_anchors[4, 1] > 4.0
    assert flat_anchors[4, 1] > flat_anchors[3, 1]


def test_apply_eq_deharshes_upper_mids_while_preserving_air_repair(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[np.ndarray] = []

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([-1.0, -2.2, 1.8, -5.6, -7.2, -8.4], dtype=float),
            np.array(
                [120.0, 1800.0, 3600.0, 6200.0, 9000.0, 15000.0], dtype=float
            ),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        mastering,
        "build_target_curve",
        lambda f_axis: np.zeros_like(f_axis, dtype=float),
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))

    flat_anchors = captured[0]

    assert flat_anchors[2, 1] < 2.0
    assert flat_anchors[3, 1] > 3.5
    assert flat_anchors[4, 1] > flat_anchors[3, 1]


def test_apply_eq_restrains_legacy_low_end_excess_when_body_repair_is_hot(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[np.ndarray] = []
    low_end_restraint_values = iter([0.0, 1.0])

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([-5.8, 1.4, 2.2, -4.6, -7.3, -8.9], dtype=float),
            np.array(
                [60.0, 130.0, 260.0, 2400.0, 7000.0, 12000.0], dtype=float
            ),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        mastering,
        "build_target_curve",
        lambda f_axis: np.zeros_like(f_axis, dtype=float),
    )
    monkeypatch.setattr(
        mastering,
        "build_spectral_balance_profile",
        lambda correction_db, f_axis: MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=12.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=1.0,
            body_restoration_factor=1.0,
            closure_repair_factor=1.0,
            mud_cleanup_factor=0.8,
            low_end_restraint_factor=next(low_end_restraint_values),
        ),
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))
    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))

    unrestrained_anchors = captured[0]
    restrained_anchors = captured[2]

    assert restrained_anchors[1, 1] < unrestrained_anchors[1, 1] - 0.45
    assert restrained_anchors[2, 1] < unrestrained_anchors[2, 1] - 0.05
    assert restrained_anchors[4, 1] > 3.5


def test_apply_eq_builds_broader_high_shelf_for_closed_four_k_rolloff(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    captured: list[np.ndarray] = []
    closed_top_values = iter([0.0, 1.0])

    monkeypatch.setattr(
        mastering,
        "measure_spectrum",
        lambda y: (
            np.array([-1.0, -2.0, -3.2, -8.5, -12.4, -15.2], dtype=float),
            np.array(
                [180.0, 1100.0, 2600.0, 4200.0, 8000.0, 14000.0], dtype=float
            ),
        ),
    )
    monkeypatch.setattr(
        mastering,
        "smooth_curve",
        lambda curve, f_axis, smoothing_fraction=None: curve,
    )
    monkeypatch.setattr(
        mastering,
        "build_target_curve",
        lambda f_axis: np.zeros_like(f_axis, dtype=float),
    )
    monkeypatch.setattr(
        mastering,
        "build_spectral_balance_profile",
        lambda correction_db, f_axis: MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=18.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=1.0,
            body_restoration_factor=0.6,
            closure_repair_factor=1.0,
            closed_top_end_repair_factor=next(closed_top_values),
        ),
    )

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))
    mastering.apply_eq(np.array([0.2, -0.1, 0.4, -0.3], dtype=float))

    base_anchors = captured[0]
    repaired_anchors = captured[2]

    assert repaired_anchors[3, 1] > base_anchors[3, 1] + 0.3
    assert repaired_anchors[4, 1] > base_anchors[4, 1] + 0.6
    assert repaired_anchors[1, 1] < base_anchors[1, 1] + 0.15


def test_build_spectral_balance_profile_detects_legacy_low_end_restraint_pressure():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        correction_strength=0.9,
        max_spectrum_boost_db=5.5,
        spectral_rescue_strength=0.18,
        spectral_rescue_boost_db=2.0,
        spectral_rescue_band_intensity=0.55,
    )

    correction_db = np.array([3.7, -2.9, -4.4, 4.1, 7.3, 8.8], dtype=float)
    f_axis = np.array(
        [55.0, 130.0, 260.0, 2400.0, 7000.0, 12000.0], dtype=float
    )

    profile = mastering.build_spectral_balance_profile(correction_db, f_axis)

    assert profile.closure_repair_factor > 0.0
    assert profile.low_end_restraint_factor > 0.0
    assert profile.max_cut_db > mastering.max_spectrum_cut_db


def test_build_spectral_balance_profile_rebalances_legacy_low_end_loaded_material():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        correction_strength=0.9,
        max_spectrum_boost_db=5.5,
        spectral_rescue_strength=0.18,
        spectral_rescue_boost_db=2.0,
        spectral_rescue_band_intensity=0.55,
    )

    f_axis = np.array(
        [60.0, 220.0, 420.0, 2400.0, 7000.0, 12000.0], dtype=float
    )
    low_end_loaded_profile = mastering.build_spectral_balance_profile(
        np.array([1.2, -4.9, -3.6, 4.2, 7.6, 9.1], dtype=float),
        f_axis,
    )
    lighter_low_end_profile = mastering.build_spectral_balance_profile(
        np.array([1.2, -0.6, -0.4, 4.2, 7.6, 9.1], dtype=float),
        f_axis,
    )

    assert low_end_loaded_profile.low_end_restraint_factor > 0.0
    assert low_end_loaded_profile.legacy_tonal_rebalance_factor > 0.0
    assert (
        low_end_loaded_profile.band_intensity
        < lighter_low_end_profile.band_intensity
    )


def test_apply_stem_cleanup_shapes_closed_vocal_stem(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = _make_mastering_instance()
    mastering.spectral_balance_profile = (
        MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=12.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=1.0,
            body_restoration_factor=0.8,
            closure_repair_factor=1.0,
        )
    )
    captured: list[np.ndarray] = []

    def fake_audio_eq(audio_data, anchors, sample_rate, nperseg):
        captured.append(np.array(anchors, copy=True))
        return np.array(audio_data, copy=True)

    monkeypatch.setattr(MASTERING_MODULE, "audio_eq", fake_audio_eq)

    mastering.apply_stem_cleanup(
        np.array([0.2, -0.1, 0.4, -0.3], dtype=float),
        stem_role="vocals",
    )

    cleanup_anchors = captured[0]

    assert np.min(cleanup_anchors[:, 1]) < -0.5
    assert np.max(cleanup_anchors[:, 1]) > 0.6
    assert cleanup_anchors[-2, 1] > 0.8


def test_apply_stem_cleanup_suppresses_between_hit_drum_bleed():
    mastering = _make_mastering_instance()
    source = np.full(512, 0.08, dtype=np.float32)
    source[96:112] += 0.92
    source[256:272] += 0.84
    source[384:396] += 0.76

    cleaned = mastering.apply_stem_cleanup(source, stem_role="drums")

    idle_before = float(np.mean(np.abs(source[0:72])))
    idle_after = float(np.mean(np.abs(cleaned[0:72])))
    hit_before = float(np.max(np.abs(source[96:112])))
    hit_after = float(np.max(np.abs(cleaned[96:112])))

    assert idle_after < idle_before * 0.35
    assert hit_after > hit_before * 0.65
    assert float(np.mean(np.abs(cleaned[150:220]))) < hit_after * 0.12


def test_apply_stem_cleanup_suppresses_vocal_bleed_outside_phrase():
    mastering = _make_mastering_instance()
    source = np.full(480, 0.06, dtype=np.float32)
    source[120:320] += 0.44

    cleaned = mastering.apply_stem_cleanup(source, stem_role="vocals")

    lead_before = float(np.mean(np.abs(source[160:280])))
    lead_after = float(np.mean(np.abs(cleaned[160:280])))
    bleed_before = float(np.mean(np.abs(source[0:96])))
    bleed_after = float(np.mean(np.abs(cleaned[0:96])))

    assert bleed_after < bleed_before * 0.45
    assert lead_after > lead_before * 0.6
    assert bleed_after < lead_after * 0.2


def test_apply_stem_cleanup_noise_gate_cleans_low_level_tail():
    mastering = _make_mastering_instance()
    source = np.full(640, 0.012, dtype=np.float32)
    source[180:360] += 0.34
    source[480:560] += 0.02

    cleaned = mastering.apply_stem_cleanup(source, stem_role="vocals")

    tail_before = float(np.mean(np.abs(source[0:120])))
    tail_after = float(np.mean(np.abs(cleaned[0:120])))
    phrase_before = float(np.mean(np.abs(source[220:320])))
    phrase_after = float(np.mean(np.abs(cleaned[220:320])))

    assert tail_after < tail_before * 0.205
    assert phrase_after > phrase_before * 0.5
    assert tail_after < phrase_after * 0.08


def test_apply_stem_cleanup_preserves_drum_body_under_high_repair_pressure():
    mastering = _make_mastering_instance()
    mastering.spectral_balance_profile = (
        MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=12.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=0.92,
            body_restoration_factor=0.8,
            closure_repair_factor=1.0,
        )
    )
    source = np.full(640, 0.03, dtype=np.float32)
    source[120:164] += 0.78
    source[164:214] += 0.2
    source[332:376] += 0.74
    source[376:428] += 0.18

    cleaned = mastering.apply_stem_cleanup(source, stem_role="drums")

    idle_before = float(np.mean(np.abs(source[0:96])))
    idle_after = float(np.mean(np.abs(cleaned[0:96])))
    body_before = float(np.mean(np.abs(source[164:214])))
    body_after = float(np.mean(np.abs(cleaned[164:214])))
    peak_before = float(np.max(np.abs(source[120:164])))
    peak_after = float(np.max(np.abs(cleaned[120:164])))

    assert idle_after < idle_before * 0.45
    assert body_after > body_before * 0.5
    assert peak_after > peak_before * 0.7


def test_apply_stem_cleanup_does_not_hollow_sustained_bass_sections():
    mastering = _make_mastering_instance()
    mastering.spectral_balance_profile = (
        MASTERING_MODULE.SpectralBalanceProfile(
            rescue_factor=1.0,
            correction_strength=1.0,
            max_boost_db=12.0,
            max_cut_db=6.0,
            band_intensity=1.0,
            restoration_factor=1.0,
            air_restoration_factor=0.9,
            body_restoration_factor=0.9,
            closure_repair_factor=1.0,
        )
    )
    source = np.full(960, 0.01, dtype=np.float32)
    source[96:864] += 0.09
    source[160:800] += 0.04 * np.sin(
        np.linspace(0.0, 18.0 * np.pi, 640, endpoint=False, dtype=np.float32)
    )

    cleaned = mastering.apply_stem_cleanup(source, stem_role="bass")

    body_before = float(np.mean(np.abs(source[224:736])))
    body_after = float(np.mean(np.abs(cleaned[224:736])))
    preserved_ratio = float(
        np.mean(np.abs(cleaned[224:736]) >= np.abs(source[224:736]) * 0.45)
    )

    assert body_after > body_before * 0.72
    assert preserved_ratio > 0.8


def test_apply_stem_noise_gate_exposes_full_requested_controls():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_normalization_mode="dbfs",
        stem_noise_gate_normalization_target_dbfs=-9.0,
        stem_noise_gate_threshold_db=-12.0,
        stem_noise_gate_hysteresis_db=3.0,
        stem_noise_gate_reduction_range_db=36.0,
        stem_noise_gate_attack_ms=0.8,
        stem_noise_gate_hold_ms=44.0,
        stem_noise_gate_release_ms=28.0,
        stem_noise_gate_lookahead_ms=2.5,
        stem_noise_gate_soft_knee_db=6.0,
        stem_noise_gate_oversampling=2,
        stem_noise_gate_sidechain_hpf_hz=120.0,
        stem_noise_gate_sidechain_lpf_hz=5200.0,
        stem_noise_gate_stereo_link_percent=35.0,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_zero_crossing_window_ms=2.0,
        stem_noise_gate_adaptive_release_enabled=True,
        stem_noise_gate_adaptive_release_strength=0.7,
        stem_noise_gate_rms_window_ms=12.0,
        stem_noise_gate_dc_offset_compensation=True,
        stem_noise_gate_delay_compensation_enabled=True,
        stem_noise_gate_inter_sample_peak_awareness=True,
        stem_noise_gate_analysis_peak_dbfs=-1.0,
    )
    source = np.zeros(320, dtype=np.float32)
    source[80:160] = 0.25

    EQ_MODULE._apply_stem_noise_gate(
        mastering,
        source,
        stem_role="vocals",
        cleanup_pressure=0.0,
    )

    profile = mastering.last_stem_noise_gate_profile

    assert profile["normalization_mode"] == "dbfs"
    assert profile["normalization_target_dbfs"] == pytest.approx(-9.0)
    assert profile["threshold_db"] == pytest.approx(-12.0)
    assert profile["hysteresis_db"] == pytest.approx(3.0)
    assert profile["reduction_range_db"] == pytest.approx(36.0)
    assert profile["attack_ms"] == pytest.approx(0.8)
    assert profile["hold_ms"] == pytest.approx(44.0)
    assert profile["release_ms"] == pytest.approx(28.0)
    assert profile["lookahead_ms"] == pytest.approx(2.5)
    assert profile["soft_knee_db"] == pytest.approx(6.0)
    assert profile["oversampling"] == 2
    assert profile["sidechain_hpf_hz"] == pytest.approx(120.0)
    assert profile["sidechain_lpf_hz"] == pytest.approx(5200.0)
    assert profile["stereo_link_percent"] == pytest.approx(35.0)
    assert profile["zero_crossing_enabled"] is False
    assert profile["zero_crossing_window_ms"] == pytest.approx(2.0)
    assert profile["adaptive_release_enabled"] is True
    assert profile["adaptive_release_strength"] == pytest.approx(0.7)
    assert profile["rms_window_ms"] == pytest.approx(12.0)
    assert profile["dc_offset_compensation"] is True
    assert profile["delay_compensation_enabled"] is True
    assert profile["inter_sample_peak_awareness"] is True
    assert profile["analysis_peak_dbfs"] == pytest.approx(-1.0)
    assert mastering.last_stem_noise_gate_analysis_peak_dbfs == pytest.approx(
        -1.0,
        abs=0.05,
    )


def test_apply_stem_noise_gate_sidechain_hpf_ignores_low_rumble_trigger():
    time_axis = np.arange(640, dtype=np.float32) / 8000.0
    source = 0.18 * np.sin(2.0 * np.pi * 40.0 * time_axis).astype(np.float32)
    source[240:320] += 0.09 * np.sin(
        2.0 * np.pi * 1400.0 * time_axis[240:320]
    ).astype(np.float32)

    no_hpf = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_threshold_db=-12.0,
        stem_noise_gate_reduction_range_db=48.0,
        stem_noise_gate_attack_ms=0.3,
        stem_noise_gate_hold_ms=0.0,
        stem_noise_gate_release_ms=6.0,
        stem_noise_gate_lookahead_ms=0.0,
        stem_noise_gate_rms_window_ms=2.0,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_sidechain_hpf_hz=0.0,
    )
    with_hpf = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_threshold_db=-12.0,
        stem_noise_gate_reduction_range_db=48.0,
        stem_noise_gate_attack_ms=0.3,
        stem_noise_gate_hold_ms=0.0,
        stem_noise_gate_release_ms=6.0,
        stem_noise_gate_lookahead_ms=0.0,
        stem_noise_gate_rms_window_ms=2.0,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_sidechain_hpf_hz=160.0,
    )

    output_no_hpf = EQ_MODULE._apply_stem_noise_gate(
        no_hpf,
        source,
        stem_role="vocals",
        cleanup_pressure=0.0,
    )
    output_with_hpf = EQ_MODULE._apply_stem_noise_gate(
        with_hpf,
        source,
        stem_role="vocals",
        cleanup_pressure=0.0,
    )

    idle_no_hpf = float(np.mean(np.abs(output_no_hpf[0:160])))
    idle_with_hpf = float(np.mean(np.abs(output_with_hpf[0:160])))
    burst_with_hpf = float(np.mean(np.abs(output_with_hpf[252:308])))

    assert idle_with_hpf < idle_no_hpf * 0.35
    assert burst_with_hpf > idle_with_hpf * 4.0


def test_apply_stem_noise_gate_stereo_link_percent_controls_linking():
    source = np.full((2, 480), 0.002, dtype=np.float32)
    source[0, 120:320] += 0.42
    source[1, 120:320] += 0.07

    unlinked = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_threshold_db=-10.0,
        stem_noise_gate_reduction_range_db=42.0,
        stem_noise_gate_attack_ms=0.5,
        stem_noise_gate_hold_ms=0.0,
        stem_noise_gate_release_ms=8.0,
        stem_noise_gate_lookahead_ms=0.0,
        stem_noise_gate_rms_window_ms=3.0,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_stereo_link_percent=0.0,
    )
    linked = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_threshold_db=-10.0,
        stem_noise_gate_reduction_range_db=42.0,
        stem_noise_gate_attack_ms=0.5,
        stem_noise_gate_hold_ms=0.0,
        stem_noise_gate_release_ms=8.0,
        stem_noise_gate_lookahead_ms=0.0,
        stem_noise_gate_rms_window_ms=3.0,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_stereo_link_percent=100.0,
    )

    output_unlinked = EQ_MODULE._apply_stem_noise_gate(
        unlinked,
        source,
        stem_role="vocals",
        cleanup_pressure=0.0,
    )
    output_linked = EQ_MODULE._apply_stem_noise_gate(
        linked,
        source,
        stem_role="vocals",
        cleanup_pressure=0.0,
    )

    right_phrase_unlinked = float(np.mean(np.abs(output_unlinked[1, 160:280])))
    right_phrase_linked = float(np.mean(np.abs(output_linked[1, 160:280])))
    right_idle_linked = float(np.mean(np.abs(output_linked[1, 0:96])))

    assert right_phrase_linked > right_phrase_unlinked * 1.35
    assert right_idle_linked < right_phrase_linked * 0.45


def test_apply_stem_noise_gate_delay_compensation_keeps_transient_alignment():
    source = np.zeros(400, dtype=np.float32)
    source[80:86] = 1.0

    delayed = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_threshold_db=-18.0,
        stem_noise_gate_reduction_range_db=54.0,
        stem_noise_gate_attack_ms=0.1,
        stem_noise_gate_hold_ms=0.0,
        stem_noise_gate_release_ms=2.0,
        stem_noise_gate_lookahead_ms=5.0,
        stem_noise_gate_rms_window_ms=0.5,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_delay_compensation_enabled=False,
    )
    compensated = MASTERING_MODULE.SmartMastering(
        8000,
        stem_noise_gate_threshold_db=-18.0,
        stem_noise_gate_reduction_range_db=54.0,
        stem_noise_gate_attack_ms=0.1,
        stem_noise_gate_hold_ms=0.0,
        stem_noise_gate_release_ms=2.0,
        stem_noise_gate_lookahead_ms=5.0,
        stem_noise_gate_rms_window_ms=0.5,
        stem_noise_gate_zero_crossing_enabled=False,
        stem_noise_gate_delay_compensation_enabled=True,
    )

    output_delayed = EQ_MODULE._apply_stem_noise_gate(
        delayed,
        source,
        stem_role="drums",
        cleanup_pressure=0.0,
    )
    output_compensated = EQ_MODULE._apply_stem_noise_gate(
        compensated,
        source,
        stem_role="drums",
        cleanup_pressure=0.0,
    )

    delayed_peak_index = int(np.argmax(np.abs(output_delayed)))
    compensated_peak_index = int(np.argmax(np.abs(output_compensated)))

    assert output_delayed.shape == source.shape
    assert output_compensated.shape == source.shape
    assert delayed_peak_index > compensated_peak_index
    assert abs(compensated_peak_index - 80) <= 4


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
    assert profile.restoration_factor > 0.0
    assert profile.air_restoration_factor > 0.0
    assert profile.body_restoration_factor > 0.0
    assert profile.closure_repair_factor > 0.0


def test_build_spectral_balance_profile_detects_mud_and_closed_top_end_pressure():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        correction_strength=0.9,
        max_spectrum_boost_db=5.5,
        spectral_rescue_strength=0.18,
        spectral_rescue_boost_db=2.0,
        spectral_rescue_band_intensity=0.55,
    )

    correction_db = np.array([1.2, -4.9, -3.6, 4.2, 7.6, 9.1], dtype=float)
    f_axis = np.array(
        [60.0, 220.0, 420.0, 2400.0, 7000.0, 12000.0], dtype=float
    )

    profile = mastering.build_spectral_balance_profile(correction_db, f_axis)

    assert profile.mud_cleanup_factor > 0.0
    assert profile.closure_repair_factor > 0.0
    assert profile.air_restoration_factor > 0.0
    assert profile.max_cut_db > mastering.max_spectrum_cut_db


def test_build_spectral_balance_profile_detects_harshness_risk_under_closed_top_end():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        correction_strength=0.9,
        max_spectrum_boost_db=5.5,
        spectral_rescue_strength=0.18,
        spectral_rescue_boost_db=2.0,
        spectral_rescue_band_intensity=0.55,
    )

    correction_db = np.array([0.8, 2.5, -2.8, 5.9, 7.6, 8.9], dtype=float)
    f_axis = np.array(
        [80.0, 1800.0, 3600.0, 6200.0, 9000.0, 12000.0], dtype=float
    )

    profile = mastering.build_spectral_balance_profile(correction_db, f_axis)

    assert profile.closure_repair_factor > 0.0
    assert profile.air_restoration_factor > 0.0
    assert profile.harshness_restraint_factor > 0.0
