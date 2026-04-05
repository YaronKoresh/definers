import importlib.util
import json
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
    effects_package = types.ModuleType(f"{package_name}.effects")
    effects_package.__path__ = []
    exciter_module = types.ModuleType(f"{package_name}.effects.exciter")
    exciter_module.apply_exciter = lambda y, *_: y
    mixing_module = types.ModuleType(f"{package_name}.effects.mixing")
    mixing_module.stereo = (
        lambda y: y if getattr(y, "ndim", 1) > 1 else np.vstack([y, y])
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


def test_default_intensity_remains_neutral_one():
    config = CONFIG_MODULE.SmartMasteringConfig()

    assert config.intensity == pytest.approx(1.0)
    assert config.preset_name == "balanced"
    assert CONFIG_MODULE.SmartMasteringConfig.preset_names() == (
        "balanced",
        "edm",
        "vocal",
    )


def test_default_mastering_disables_frequency_cuts():
    config = CONFIG_MODULE.SmartMasteringConfig()

    assert config.low_cut is None
    assert config.high_cut is None
    assert CONFIG_MODULE.SmartMasteringConfig.from_preset(None).preset_name == (
        "balanced"
    )


def test_config_presets_span_density_and_stereo_motion_profiles():
    edm = CONFIG_MODULE.SmartMasteringConfig.edm()
    balanced = CONFIG_MODULE.SmartMasteringConfig.balanced()
    vocal = CONFIG_MODULE.SmartMasteringConfig.vocal()

    assert edm.preset_name == "edm"
    assert balanced.preset_name == "balanced"
    assert vocal.preset_name == "vocal"
    assert edm.target_lufs > balanced.target_lufs > vocal.target_lufs
    assert (
        edm.limiter_soft_clip_ratio
        > balanced.limiter_soft_clip_ratio
        > vocal.limiter_soft_clip_ratio
    )
    assert (
        edm.pre_limiter_saturation_ratio
        > balanced.pre_limiter_saturation_ratio
        > vocal.pre_limiter_saturation_ratio
    )
    assert edm.spectral_drive_bias_db > balanced.spectral_drive_bias_db
    assert edm.exciter_mix > balanced.exciter_mix > vocal.exciter_mix
    assert edm.bass_boost_db_per_oct > balanced.bass_boost_db_per_oct
    assert balanced.bass_boost_db_per_oct > vocal.bass_boost_db_per_oct
    assert balanced.treble_boost_db_per_oct > vocal.treble_boost_db_per_oct
    assert vocal.treble_boost_db_per_oct > edm.treble_boost_db_per_oct
    assert vocal.micro_dynamics_strength > balanced.micro_dynamics_strength
    assert balanced.micro_dynamics_strength > edm.micro_dynamics_strength
    assert balanced.stereo_tone_variation_db > edm.stereo_tone_variation_db
    assert vocal.stereo_tone_variation_db > balanced.stereo_tone_variation_db
    assert vocal.stereo_motion_high_amount > balanced.stereo_motion_high_amount
    assert balanced.stereo_motion_high_amount > edm.stereo_motion_high_amount
    assert (
        balanced.contract_max_stereo_width_ratio
        > edm.contract_max_stereo_width_ratio
    )
    assert vocal.contract_max_stereo_width_ratio > (
        balanced.contract_max_stereo_width_ratio
    )
    assert vocal.mono_bass_hz < balanced.mono_bass_hz < edm.mono_bass_hz
    assert edm.limiter_recovery_style == "tight"
    assert vocal.codec_headroom_margin_db > balanced.codec_headroom_margin_db


def test_smart_mastering_defaults_to_balanced_preset():
    balanced = CONFIG_MODULE.SmartMasteringConfig.balanced()

    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)

    assert mastering.preset_name == "balanced"
    assert mastering.target_lufs == pytest.approx(balanced.target_lufs)
    assert mastering.stereo_width == pytest.approx(balanced.stereo_width)


def test_smart_mastering_accepts_preset_name():
    edm = CONFIG_MODULE.SmartMasteringConfig.edm()

    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="edm",
    )

    assert mastering.preset_name == "edm"
    assert mastering.target_lufs == pytest.approx(edm.target_lufs)
    assert (
        mastering.true_peak_oversample_factor == edm.true_peak_oversample_factor
    )


def test_mastering_facade_keeps_public_exports_stable():
    profile = MASTERING_MODULE.SpectralBalanceProfile(
        rescue_factor=0.1,
        correction_strength=0.2,
        max_boost_db=1.0,
        max_cut_db=0.5,
        band_intensity=1.0,
    )

    assert callable(MASTERING_MODULE.audio_eq)
    assert profile.rescue_factor == pytest.approx(0.1)
    assert profile.band_intensity == pytest.approx(1.0)


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
    assert bands[0]["knee_db"] > bands[1]["knee_db"]
    assert bands[0]["ratio"] > bands[1]["ratio"]
    assert bands[0]["makeup_db"] >= 0.0


def test_build_bands_from_fcs_uses_instance_overrides():
    config = CONFIG_MODULE.SmartMasteringConfig(
        intensity=1.0,
        bass_ratio=4.0,
        treb_ratio=2.0,
        bass_threshold_db=-10.0,
        treb_threshold_db=-20.0,
    )

    bands = config.build_bands_from_fcs([20.0, 20000.0], 20.0, 20000.0)

    assert bands[0]["ratio"] == pytest.approx(4.0)
    assert bands[-1]["ratio"] == pytest.approx(2.0)
    assert bands[0]["base_threshold"] == pytest.approx(-10.0)
    assert bands[-1]["base_threshold"] == pytest.approx(-20.0)


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


def test_slope_property_refreshes_mastering_profile():
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    anchors_before = np.array(mastering.anchors, dtype=float)

    mastering.slope_db = mastering.slope_db + 1.5

    anchors_after = np.array(mastering.anchors, dtype=float)

    assert anchors_after[1, 1] > anchors_before[1, 1]


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


def test_multiband_compress_links_stereo_gain(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.array(
        [[1.0, 1.0, 1.0, 1.0], [0.1, 0.1, 0.1, 0.1]],
        dtype=float,
    )

    mastering.bands = [
        {
            "fc": 100.0,
            "base_threshold": -18.0,
            "ratio": 4.0,
            "attack_ms": 1.0,
            "release_ms": 1.0,
            "makeup_db": 0.0,
            "knee_db": 0.5,
        }
    ]

    monkeypatch.setattr(
        MASTERING_MODULE,
        "decoupled_envelope",
        lambda env, *_: np.array(env, copy=True),
    )

    compressed = mastering.multiband_compress(source)

    assert np.all(compressed[0] < source[0])
    assert np.all(compressed[1] < source[1])
    assert np.allclose(compressed[0] / source[0], compressed[1] / source[1])


def test_process_runs_follow_up_limiter_when_loudness_is_still_low(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        target_lufs=-10.0,
        final_lufs_tolerance=0.5,
        max_final_boost_db=2.0,
        drive_db=0.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    limiter_calls: list[tuple[float, float, int, float]] = []
    loudness_values = iter([-18.0])
    measured_metrics = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-11.0,
                max_short_term_lufs=-11.5,
                max_momentary_lufs=-11.0,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
            ),
            types.SimpleNamespace(
                integrated_lufs=-10.1,
                max_short_term_lufs=-10.5,
                max_momentary_lufs=-10.1,
                crest_factor_db=6.5,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
            ),
            types.SimpleNamespace(
                integrated_lufs=-10.1,
                max_short_term_lufs=-10.5,
                max_momentary_lufs=-10.1,
                crest_factor_db=6.5,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
            ),
            types.SimpleNamespace(
                integrated_lufs=-10.1,
                max_short_term_lufs=-10.5,
                max_momentary_lufs=-10.1,
                crest_factor_db=6.5,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
            ),
        ]
    )
    follow_up_actions = iter(
        [
            types.SimpleNamespace(
                should_apply=True,
                gain_db=1.0,
                soft_clip_ratio=0.33,
                stereo_width_scale=1.0,
                reasons=("gain",),
                integrated_gap_db=1.0,
            ),
            types.SimpleNamespace(
                should_apply=False,
                gain_db=0.0,
                soft_clip_ratio=mastering.limiter_soft_clip_ratio,
                stereo_width_scale=1.0,
                reasons=(),
                integrated_gap_db=0.0,
            ),
        ]
    )

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: next(measured_metrics),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: next(follow_up_actions),
    )

    def fake_apply_limiter(y, drive_db, ceil_db, **kwargs):
        limiter_calls.append(
            (
                float(drive_db),
                float(ceil_db),
                int(kwargs["os_factor"]),
                float(kwargs["soft_clip_ratio"]),
            )
        )
        return y

    monkeypatch.setattr(mastering, "apply_limiter", fake_apply_limiter)

    sr_out, y_out = mastering.process(source, 8000)

    assert sr_out == 8000
    assert np.allclose(y_out, np.vstack([source, source]))
    assert len(limiter_calls) == 2
    assert limiter_calls[0][0] == pytest.approx(8.0)
    assert limiter_calls[0][2] == mastering.limiter_oversample_factor
    assert limiter_calls[0][3] > mastering.limiter_soft_clip_ratio
    assert limiter_calls[1][0] == pytest.approx(1.0)
    assert limiter_calls[1][3] == pytest.approx(0.33)


def test_plan_follow_up_drives_splits_large_remaining_loudness_budget():
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        max_final_boost_db=4.5,
        final_lufs_tolerance=0.25,
        max_follow_up_passes=3,
    )

    follow_up_drives = mastering.plan_follow_up_drives(4.0, rescue_factor=0.5)

    assert len(follow_up_drives) == 3
    assert follow_up_drives[0] > follow_up_drives[1] > follow_up_drives[2] > 0.0
    assert sum(follow_up_drives) == pytest.approx(4.0)


def test_process_increases_primary_soft_clip_when_rescue_profile_is_hot(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        target_lufs=-10.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    limiter_calls: list[tuple[float, float, int, float]] = []
    loudness_values = iter([-18.0, -10.0])

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: types.SimpleNamespace(
            integrated_lufs=-10.0,
            max_short_term_lufs=-10.5,
            max_momentary_lufs=-10.0,
            crest_factor_db=6.0,
            stereo_width_ratio=0.2,
            low_end_mono_ratio=0.95,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )

    def fake_apply_eq(y):
        mastering.spectral_balance_profile = (
            MASTERING_MODULE.SpectralBalanceProfile(
                rescue_factor=1.0,
                correction_strength=mastering.correction_strength,
                max_boost_db=mastering.max_spectrum_boost_db + 2.0,
                max_cut_db=mastering.max_spectrum_cut_db + 1.0,
                band_intensity=mastering.config.intensity + 0.5,
            )
        )
        return y

    def fake_apply_limiter(y, drive_db, ceil_db, **kwargs):
        limiter_calls.append(
            (
                float(drive_db),
                float(ceil_db),
                int(kwargs["os_factor"]),
                float(kwargs["soft_clip_ratio"]),
            )
        )
        return y

    monkeypatch.setattr(mastering, "apply_eq", fake_apply_eq)
    monkeypatch.setattr(mastering, "apply_limiter", fake_apply_limiter)

    mastering.process(source, 8000)

    assert limiter_calls[0][3] > mastering.limiter_soft_clip_ratio


def test_process_passes_exciter_controls_into_exciter(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        exciter_mix=0.6,
        exciter_cutoff_hz=2400.0,
        exciter_max_drive=4.0,
        exciter_high_frequency_cutoff_hz=6800.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    exciter_calls: list[
        tuple[int, float | None, float, float, float | None]
    ] = []
    loudness_values = iter([-10.0])

    monkeypatch.setattr(
        MASTERING_MODULE,
        "apply_exciter",
        lambda y, sample_rate, cutoff_hz=None, mix=1.0, max_drive=None, high_frequency_cutoff_hz=None: (
            exciter_calls.append(
                (
                    int(sample_rate),
                    None if cutoff_hz is None else float(cutoff_hz),
                    float(mix),
                    None if max_drive is None else float(max_drive),
                    None
                    if high_frequency_cutoff_hz is None
                    else float(high_frequency_cutoff_hz),
                )
            )
            or y
        ),
    )
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(
        mastering, "apply_safety_clamp", lambda y, ceil_db=-0.1: y
    )
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE, "get_lufs", lambda y, sr: next(loudness_values)
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: types.SimpleNamespace(
            integrated_lufs=-10.0,
            max_short_term_lufs=-10.0,
            max_momentary_lufs=-10.0,
            crest_factor_db=6.0,
            stereo_width_ratio=0.2,
            low_end_mono_ratio=0.95,
            true_peak_dbfs=-0.4,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )
    monkeypatch.setattr(
        mastering, "apply_limiter", lambda y, drive_db, ceil_db, **kwargs: y
    )

    mastering.process(source, 8000)

    assert exciter_calls == [(8000, 2400.0, 0.6, 4.0, 6800.0)]


def test_process_adds_rescue_drive_bias_when_profile_is_hot(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        target_lufs=-10.0,
        drive_db=0.0,
        spectral_drive_bias_db=1.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    limiter_calls: list[float] = []
    loudness_values = iter([-18.0, -10.0])

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: types.SimpleNamespace(
            integrated_lufs=-10.0,
            max_short_term_lufs=-10.5,
            max_momentary_lufs=-10.0,
            crest_factor_db=6.0,
            stereo_width_ratio=0.2,
            low_end_mono_ratio=0.95,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )

    def fake_apply_eq(y):
        mastering.spectral_balance_profile = (
            MASTERING_MODULE.SpectralBalanceProfile(
                rescue_factor=1.0,
                correction_strength=mastering.correction_strength,
                max_boost_db=mastering.max_spectrum_boost_db,
                max_cut_db=mastering.max_spectrum_cut_db,
                band_intensity=mastering.config.intensity,
            )
        )
        return y

    def fake_apply_limiter(y, drive_db, ceil_db, **kwargs):
        limiter_calls.append(float(drive_db))
        return y

    monkeypatch.setattr(mastering, "apply_eq", fake_apply_eq)
    monkeypatch.setattr(mastering, "apply_limiter", fake_apply_limiter)

    mastering.process(source, 8000)

    assert limiter_calls[0] > 8.0


def test_process_passes_no_filter_cutoffs_by_default(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    freq_cut_calls: list[tuple[float | None, float | None]] = []
    loudness_values = iter([-10.0, -10.0])

    def fake_freq_cut(y, sr, low_cut, high_cut):
        freq_cut_calls.append((low_cut, high_cut))
        return y

    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", fake_freq_cut)
    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: types.SimpleNamespace(
            integrated_lufs=-10.0,
            max_short_term_lufs=-10.0,
            max_momentary_lufs=-10.0,
            crest_factor_db=6.0,
            stereo_width_ratio=0.2,
            low_end_mono_ratio=0.95,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_limiter",
        lambda y, drive_db, ceil_db, **kwargs: y,
    )

    mastering.process(source, 8000)

    assert freq_cut_calls == [(None, None), (None, None)]


def test_process_calls_finalization_stages(monkeypatch: pytest.MonkeyPatch):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="balanced",
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    stage_calls: list[tuple[str, float | None]] = []
    loudness_values = iter([-10.0])

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_low_end_mono_tightening",
        lambda y: stage_calls.append(("mono_tightening", None)) or y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: types.SimpleNamespace(
            integrated_lufs=-10.0,
            max_short_term_lufs=-10.0,
            max_momentary_lufs=-10.0,
            crest_factor_db=6.0,
            stereo_width_ratio=0.1,
            low_end_mono_ratio=0.95,
            true_peak_dbfs=-1.0,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: (
            stage_calls.append(("saturation", float(dynamic_drive_db))) or y
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_limiter",
        lambda y, drive_db, ceil_db, **kwargs: (
            stage_calls.append(("limiter", float(drive_db))) or y
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_delivery_trim",
        lambda y: stage_calls.append(("trim", None)) or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_final_headroom_recovery",
        lambda y: stage_calls.append(("headroom_recovery", None)) or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_micro_dynamics_finish",
        lambda y: stage_calls.append(("micro_dynamics", None)) or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_safety_clamp",
        lambda y, ceil_db=-0.1: (
            stage_calls.append(("clamp", float(ceil_db))) or y
        ),
    )

    mastering.process(source, 8000)

    stage_names = [name for name, _value in stage_calls]

    assert stage_names[:3] == ["mono_tightening", "saturation", "limiter"]
    assert stage_names[-3:] == ["trim", "clamp", "headroom_recovery"]
    micro_dynamics_index = stage_names.index("micro_dynamics")
    trim_index = stage_names.index("trim")
    assert micro_dynamics_index < trim_index
    assert all(
        name == "limiter"
        for name in stage_names[micro_dynamics_index + 1 : trim_index]
    )


def test_process_applies_final_peak_catch_after_character_stage_when_true_peak_is_hot(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="edm",
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    limiter_calls: list[tuple[float, float]] = []
    stage_calls: list[tuple[str, float]] = []
    loudness_values = iter([-10.0])
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-10.0,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-10.0,
                crest_factor_db=6.0,
                stereo_width_ratio=0.1,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-0.2,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.8,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-10.0,
                crest_factor_db=6.0,
                stereo_width_ratio=0.1,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=0.8,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.9,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-10.0,
                crest_factor_db=6.0,
                stereo_width_ratio=0.1,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-0.6,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.9,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-10.0,
                crest_factor_db=6.0,
                stereo_width_ratio=0.1,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-0.6,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.9,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-10.0,
                crest_factor_db=6.0,
                stereo_width_ratio=0.1,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-0.6,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.9,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-10.0,
                crest_factor_db=6.0,
                stereo_width_ratio=0.1,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-0.6,
            ),
        ]
    )

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_micro_dynamics_finish",
        lambda y: stage_calls.append(("micro_dynamics", 0.0)) or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_delivery_trim",
        lambda y: stage_calls.append(("trim", 0.0)) or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_safety_clamp",
        lambda y, ceil_db=-0.1: (
            stage_calls.append(("clamp", float(ceil_db))) or y
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_final_headroom_recovery",
        lambda y: stage_calls.append(("headroom_recovery", 0.0)) or y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: next(metrics_values),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_limiter",
        lambda y, drive_db, ceil_db, **kwargs: (
            limiter_calls.append((float(drive_db), float(ceil_db)))
            or stage_calls.append(("limiter", float(drive_db)))
            or y
        ),
    )

    mastering.process(source, 8000)

    assert len(limiter_calls) == 2
    assert limiter_calls[0][0] > 0.0
    assert limiter_calls[1][0] == pytest.approx(0.0)
    assert (
        limiter_calls[1][1]
        < mastering.last_resolved_final_true_peak_target_dbfs
    )
    assert len(mastering.last_peak_catch_events) == 1
    assert mastering.last_peak_catch_events[0].ceil_db == pytest.approx(
        limiter_calls[1][1]
    )
    assert stage_calls[-1][0] == "headroom_recovery"
    assert mastering.last_stage_signals["post_spatial"].shape == (2, 4)
    assert mastering.last_stage_signals["post_peak_catch"].shape == (2, 4)
    assert mastering.last_stage_signals["post_delivery_trim"].shape == (2, 4)
    assert mastering.last_stage_signals["post_clamp"].shape == (2, 4)


def test_process_reverts_micro_dynamics_when_character_stage_breaks_contract(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="edm",
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    loudness_values = iter([-10.0])
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-5.3,
                max_short_term_lufs=-3.9,
                max_momentary_lufs=-2.9,
                crest_factor_db=7.8,
                stereo_width_ratio=0.3,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-0.4,
            ),
            types.SimpleNamespace(
                integrated_lufs=-4.9,
                max_short_term_lufs=-3.3,
                max_momentary_lufs=-2.5,
                crest_factor_db=7.4,
                stereo_width_ratio=0.3,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=0.7,
            ),
            types.SimpleNamespace(
                integrated_lufs=-4.9,
                max_short_term_lufs=-3.3,
                max_momentary_lufs=-2.5,
                crest_factor_db=7.4,
                stereo_width_ratio=0.3,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=0.7,
            ),
            types.SimpleNamespace(
                integrated_lufs=-4.9,
                max_short_term_lufs=-3.3,
                max_momentary_lufs=-2.5,
                crest_factor_db=7.4,
                stereo_width_ratio=0.3,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=0.7,
            ),
            types.SimpleNamespace(
                integrated_lufs=-4.9,
                max_short_term_lufs=-3.3,
                max_momentary_lufs=-2.5,
                crest_factor_db=7.4,
                stereo_width_ratio=0.3,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=0.7,
            ),
        ]
    )

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: next(metrics_values),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: types.SimpleNamespace(
            should_apply=False,
            gain_db=0.0,
            soft_clip_ratio=mastering.limiter_soft_clip_ratio,
            stereo_width_scale=1.0,
            reasons=(),
            integrated_gap_db=0.0,
        ),
    )
    monkeypatch.setattr(
        mastering, "apply_limiter", lambda y, drive_db, ceil_db, **kwargs: y
    )
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(
        mastering, "apply_safety_clamp", lambda y, ceil_db=-0.1: y
    )
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        mastering, "apply_micro_dynamics_finish", lambda y: y * 0.5
    )

    mastering.process(source, 8000)

    assert np.allclose(
        mastering.last_stage_signals["post_character"],
        mastering.last_stage_signals["post_limiter"],
    )
    assert mastering.last_character_stage_decision is not None
    assert mastering.last_character_stage_decision.reverted is True


def test_process_applies_stereo_restraint_when_follow_up_action_requests_it(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    loudness_values = iter([-14.0])
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-11.5,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.8,
                low_end_mono_ratio=0.4,
            ),
            types.SimpleNamespace(
                integrated_lufs=-11.0,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=6.8,
                stereo_width_ratio=0.4,
                low_end_mono_ratio=0.9,
            ),
            types.SimpleNamespace(
                integrated_lufs=-11.0,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=6.8,
                stereo_width_ratio=0.4,
                low_end_mono_ratio=0.9,
            ),
            types.SimpleNamespace(
                integrated_lufs=-11.0,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=6.8,
                stereo_width_ratio=0.4,
                low_end_mono_ratio=0.9,
            ),
        ]
    )
    follow_up_actions = iter(
        [
            types.SimpleNamespace(
                should_apply=True,
                gain_db=0.5,
                soft_clip_ratio=0.3,
                stereo_width_scale=0.75,
                reasons=("gain", "stereo_restraint"),
                integrated_gap_db=1.0,
            ),
            types.SimpleNamespace(
                should_apply=False,
                gain_db=0.0,
                soft_clip_ratio=mastering.limiter_soft_clip_ratio,
                stereo_width_scale=1.0,
                reasons=(),
                integrated_gap_db=0.0,
            ),
        ]
    )
    restraint_calls: list[float] = []

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: next(metrics_values),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: next(follow_up_actions),
    )
    monkeypatch.setattr(
        mastering,
        "apply_stereo_width_restraint",
        lambda y, stereo_width_scale=1.0: (
            restraint_calls.append(float(stereo_width_scale)) or y
        ),
    )
    monkeypatch.setattr(
        mastering, "apply_limiter", lambda y, drive_db, ceil_db, **kwargs: y
    )

    mastering.process(source, 8000)

    assert restraint_calls == [pytest.approx(0.75)]
    assert set(mastering.last_stage_signals) == {
        "post_eq",
        "post_spatial",
        "post_character",
        "post_limiter",
        "post_peak_catch",
        "post_delivery_trim",
        "post_clamp",
        "final_in_memory",
    }
    assert mastering.last_mastering_contract is not None


def test_process_skips_relimit_when_follow_up_action_only_restrains_stereo(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    loudness_values = iter([-14.0])
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-11.5,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.8,
                low_end_mono_ratio=0.4,
            ),
            types.SimpleNamespace(
                integrated_lufs=-11.2,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=6.9,
                stereo_width_ratio=0.4,
                low_end_mono_ratio=0.9,
            ),
            types.SimpleNamespace(
                integrated_lufs=-11.2,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=6.9,
                stereo_width_ratio=0.4,
                low_end_mono_ratio=0.9,
            ),
            types.SimpleNamespace(
                integrated_lufs=-11.2,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.5,
                crest_factor_db=6.9,
                stereo_width_ratio=0.4,
                low_end_mono_ratio=0.9,
            ),
        ]
    )
    follow_up_actions = iter(
        [
            types.SimpleNamespace(
                should_apply=True,
                gain_db=0.0,
                soft_clip_ratio=mastering.limiter_soft_clip_ratio,
                stereo_width_scale=0.75,
                reasons=("stereo_restraint",),
                integrated_gap_db=0.0,
            ),
            types.SimpleNamespace(
                should_apply=False,
                gain_db=0.0,
                soft_clip_ratio=mastering.limiter_soft_clip_ratio,
                stereo_width_scale=1.0,
                reasons=(),
                integrated_gap_db=0.0,
            ),
        ]
    )
    restraint_calls: list[float] = []
    limiter_calls: list[tuple[float, float]] = []

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(mastering, "apply_low_end_mono_tightening", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(mastering, "apply_final_headroom_recovery", lambda y: y)
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: next(loudness_values),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: next(metrics_values),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: next(follow_up_actions),
    )
    monkeypatch.setattr(
        mastering,
        "apply_stereo_width_restraint",
        lambda y, stereo_width_scale=1.0: (
            restraint_calls.append(float(stereo_width_scale)) or y
        ),
    )

    def fake_apply_limiter(y, drive_db, ceil_db, **kwargs):
        limiter_calls.append(
            (float(drive_db), float(kwargs["soft_clip_ratio"]))
        )
        return y

    monkeypatch.setattr(mastering, "apply_limiter", fake_apply_limiter)

    mastering.process(source, 8000)

    assert restraint_calls == [pytest.approx(0.75)]
    assert len(limiter_calls) == 1


def test_master_with_report_writes_report_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "mastering-report.json"
    output_path = tmp_path / "mastered.wav"
    io_module_name = f"{MASTERING_MODULE.__package__}.io"
    original_io_module = sys.modules.get(io_module_name)

    fake_report = types.SimpleNamespace(
        preset_name="edm",
        to_dict=lambda: {
            "preset_name": "edm",
            "delivery_profile_name": "lossless",
        },
    )

    class FakeMastering:
        def __init__(self, sr, **kwargs):
            self.target_lufs = -5.0
            self.ceil_db = -0.1
            self.preset_name = kwargs.get("preset")
            self.delivery_profile = "lossless"
            self.delivery_decoded_true_peak_dbfs = -0.1
            self.codec_headroom_margin_db = 0.0
            self.delivery_lufs_tolerance_db = 0.35
            self.true_peak_oversample_factor = 8
            self.delivery_bitrate = 320
            self.last_stage_signals = {
                "post_eq": np.zeros(8, dtype=np.float32),
                "post_limiter": np.zeros(8, dtype=np.float32),
                "post_character": np.zeros(8, dtype=np.float32),
            }
            self.last_mastering_contract = None

        def process(self, y, sr):
            return sr, np.array(y, copy=True)

        def resolve_mastering_contract(self):
            return types.SimpleNamespace(max_true_peak_dbfs=self.ceil_db)

    try:
        sys.modules[io_module_name] = types.SimpleNamespace(
            read_audio=lambda path: (8000, np.zeros(8, dtype=np.float32)),
            save_audio=lambda **kwargs: kwargs["destination_path"],
        )
        monkeypatch.setitem(
            sys.modules,
            "definers.system",
            types.SimpleNamespace(
                tmp=lambda suffix, keep=False: str(output_path)
            ),
        )
        monkeypatch.setattr(MASTERING_MODULE, "SmartMastering", FakeMastering)
        monkeypatch.setattr(
            MASTERING_MODULE,
            "save_verified_audio",
            lambda **kwargs: (
                kwargs["destination_path"],
                kwargs["audio_signal"],
                types.SimpleNamespace(report=fake_report),
            ),
        )

        mastered_path, report = MASTERING_MODULE.master(
            "input.wav",
            output_path=str(output_path),
            preset="edm",
            report_path=str(report_path),
        )
    finally:
        if original_io_module is None:
            sys.modules.pop(io_module_name, None)
        else:
            sys.modules[io_module_name] = original_io_module

    assert mastered_path == str(output_path)
    assert report is fake_report
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["preset_name"] == "edm"
    assert payload["delivery_profile_name"] == "lossless"
