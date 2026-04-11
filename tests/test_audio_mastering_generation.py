import importlib.util
import json
import sys
import types
from dataclasses import replace
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
        f"{package_name}.mastering", MASTERING_ROOT / "__init__.py"
    )
    return config_module, mastering_module


CONFIG_MODULE, MASTERING_MODULE = _load_mastering_module(
    "_test_audio_mastering_generation_pkg.audio"
)
MASTERING_AUDIO_PACKAGE = MASTERING_MODULE.__package__.rpartition(".")[0]
MASTERING_PACKAGE = MASTERING_MODULE.__package__


def test_default_intensity_remains_neutral_one():
    config = CONFIG_MODULE.SmartMasteringConfig()

    assert config.intensity == pytest.approx(
        CONFIG_MODULE.SmartMasteringConfig.balanced().intensity
    )
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
    assert edm.target_lufs - balanced.target_lufs > 3.0
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
    assert 0.0 <= balanced.exciter_mix <= 1.0
    assert 0.0 <= edm.exciter_mix <= 1.0
    assert 0.0 <= vocal.exciter_mix <= 1.0
    assert edm.exciter_max_drive > balanced.exciter_max_drive
    assert balanced.exciter_max_drive > vocal.exciter_max_drive
    assert (
        edm.exciter_high_frequency_cutoff_hz
        < balanced.exciter_high_frequency_cutoff_hz
        < vocal.exciter_high_frequency_cutoff_hz
    )
    assert edm.bass_boost_db_per_oct > balanced.bass_boost_db_per_oct
    assert balanced.bass_boost_db_per_oct > vocal.bass_boost_db_per_oct
    assert edm.bass_boost_db_per_oct - balanced.bass_boost_db_per_oct > 0.35
    assert vocal.treble_boost_db_per_oct > balanced.treble_boost_db_per_oct
    assert balanced.treble_boost_db_per_oct > edm.treble_boost_db_per_oct
    assert balanced.treble_boost_db_per_oct - edm.treble_boost_db_per_oct > 0.2
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


def test_preset_name_constructor_derives_matching_preset_profile():
    edm = CONFIG_MODULE.SmartMasteringConfig.edm()
    derived = CONFIG_MODULE.SmartMasteringConfig(preset_name="edm")

    assert derived.preset_name == "edm"
    assert derived.intensity == pytest.approx(edm.intensity)
    assert derived.target_lufs == pytest.approx(edm.target_lufs)
    assert derived.drive_db == pytest.approx(edm.drive_db)
    assert derived.bass_ratio == pytest.approx(edm.bass_ratio)
    assert derived.limiter_recovery_style == edm.limiter_recovery_style


def test_macro_controls_derive_low_level_parameters():
    balanced = CONFIG_MODULE.SmartMasteringConfig.balanced()
    config = CONFIG_MODULE.SmartMasteringConfig(
        bass=0.25,
        volume=0.8,
        effects=0.2,
    )

    assert config.target_lufs > balanced.target_lufs
    assert config.bass_ratio > balanced.bass_ratio
    assert config.stereo_width < balanced.stereo_width
    assert config.treble_boost_db_per_oct < balanced.treble_boost_db_per_oct
    assert config.limiter_recovery_style == "tight"
    assert config.low_end_mono_tightening == "balanced"
    assert config.delivery_decoded_true_peak_dbfs is None

    base_bass_boost = config.bass_boost_db_per_oct
    base_treble_boost = config.treble_boost_db_per_oct
    base_exciter_high_cutoff = config.exciter_high_frequency_cutoff_hz
    config.bass = 1.0
    assert config.bass_boost_db_per_oct > base_bass_boost + 0.2
    assert config.treble_boost_db_per_oct < base_treble_boost - 0.1
    assert config.exciter_high_frequency_cutoff_hz < (
        base_exciter_high_cutoff - 300.0
    )

    base_stereo_width = config.stereo_width
    base_micro_dynamics = config.micro_dynamics_strength
    config.effects = 1.0
    assert config.stereo_width > base_stereo_width
    assert config.micro_dynamics_strength > base_micro_dynamics

    base_drive = config.drive_db
    base_target_lufs = config.target_lufs
    base_final_boost = config.max_final_boost_db
    config.volume = 1.0
    assert config.drive_db > base_drive + 0.15
    assert config.target_lufs > base_target_lufs + 0.4
    assert config.max_final_boost_db > base_final_boost + 0.2


def test_low_level_overrides_take_precedence_over_macro_derivation():
    config = CONFIG_MODULE.SmartMasteringConfig(
        volume=1.0,
        bass_ratio=9.0,
        treb_threshold_db=-31.0,
    )

    assert config.bass_ratio == pytest.approx(9.0)
    assert config.treb_threshold_db == pytest.approx(-31.0)
    assert config.treb_ratio == pytest.approx(
        CONFIG_MODULE.SmartMasteringConfig(volume=1.0).treb_ratio
    )


def test_dataclass_replace_preserves_low_level_overrides():
    config = CONFIG_MODULE.SmartMasteringConfig(
        volume=1.0,
        bass_ratio=9.0,
        effects=0.0,
    )

    updated = replace(config, effects=1.0)

    assert updated.effects == pytest.approx(1.0)
    assert updated.bass_ratio == pytest.approx(9.0)
    assert updated.stereo_width > config.stereo_width


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


def test_select_mastering_preset_prefers_edm_for_dense_bass_heavy_material(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda signal, sample_rate: types.SimpleNamespace(
            integrated_lufs=-8.0,
            crest_factor_db=7.2,
            stereo_width_ratio=0.16,
            low_end_mono_ratio=0.91,
        ),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_spectral_tilt",
        lambda signal, sample_rate: -6.2,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_transient_density",
        lambda signal, sample_rate: 0.19,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_stereo_motion",
        lambda signal, sample_rate: 0.03,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_preset_band_profile",
        lambda signal, sample_rate: {
            "bass_share": 0.41,
            "low_mid_share": 0.26,
            "presence_share": 0.12,
            "air_share": 0.06,
        },
    )

    assert (
        MASTERING_MODULE._select_mastering_preset(
            np.zeros((2, 1024), dtype=np.float32),
            8000,
        )
        == "edm"
    )


def test_select_mastering_preset_prefers_vocal_for_wide_dynamic_material(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda signal, sample_rate: types.SimpleNamespace(
            integrated_lufs=-17.5,
            crest_factor_db=15.2,
            stereo_width_ratio=0.56,
            low_end_mono_ratio=0.62,
        ),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_spectral_tilt",
        lambda signal, sample_rate: -3.4,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_transient_density",
        lambda signal, sample_rate: 0.03,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_stereo_motion",
        lambda signal, sample_rate: 0.16,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_preset_band_profile",
        lambda signal, sample_rate: {
            "bass_share": 0.08,
            "low_mid_share": 0.22,
            "presence_share": 0.34,
            "air_share": 0.18,
        },
    )

    assert (
        MASTERING_MODULE._select_mastering_preset(
            np.zeros((2, 1024), dtype=np.float32),
            8000,
        )
        == "vocal"
    )


def test_select_mastering_preset_prefers_vocal_for_closed_bassy_legacy_ensemble(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda signal, sample_rate: types.SimpleNamespace(
            integrated_lufs=-17.2,
            crest_factor_db=12.8,
            stereo_width_ratio=0.11,
            low_end_mono_ratio=0.96,
        ),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_spectral_tilt",
        lambda signal, sample_rate: -10.4,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_transient_density",
        lambda signal, sample_rate: 0.025,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_stereo_motion",
        lambda signal, sample_rate: 0.015,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_preset_band_profile",
        lambda signal, sample_rate: {
            "bass_share": 0.27,
            "low_mid_share": 0.42,
            "presence_share": 0.08,
            "air_share": 0.025,
        },
    )

    assert (
        MASTERING_MODULE._select_mastering_preset(
            np.zeros((2, 1024), dtype=np.float32),
            8000,
        )
        == "vocal"
    )


def test_select_mastering_preset_keeps_balanced_for_general_midweight_material(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda signal, sample_rate: types.SimpleNamespace(
            integrated_lufs=-13.0,
            crest_factor_db=10.2,
            stereo_width_ratio=0.24,
            low_end_mono_ratio=0.86,
        ),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_spectral_tilt",
        lambda signal, sample_rate: -5.6,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_transient_density",
        lambda signal, sample_rate: 0.08,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_stereo_motion",
        lambda signal, sample_rate: 0.05,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_preset_band_profile",
        lambda signal, sample_rate: {
            "bass_share": 0.19,
            "low_mid_share": 0.24,
            "presence_share": 0.18,
            "air_share": 0.09,
        },
    )

    assert (
        MASTERING_MODULE._select_mastering_preset(
            np.zeros((2, 1024), dtype=np.float32),
            8000,
        )
        == "balanced"
    )


def test_resolve_mastering_kwargs_for_input_preserves_explicit_preset_override(
    monkeypatch: pytest.MonkeyPatch,
):
    called = []

    monkeypatch.setattr(
        MASTERING_MODULE,
        "_select_mastering_preset",
        lambda signal, sample_rate: called.append("auto") or "vocal",
    )

    explicit = MASTERING_MODULE._resolve_mastering_kwargs_for_input(
        np.zeros(64, dtype=np.float32),
        8000,
        {"preset": "edm"},
    )
    automatic = MASTERING_MODULE._resolve_mastering_kwargs_for_input(
        np.zeros(64, dtype=np.float32),
        8000,
        {"preset": "auto"},
    )

    assert explicit == {"preset": "edm"}
    assert automatic == {"preset": "vocal"}
    assert called == ["auto"]


def test_resolve_mastering_kwargs_for_input_reuses_input_analysis_preset():
    analysis = MASTERING_MODULE.MasteringInputAnalysis(
        preset_name="edm",
        quality_flags=("Low-Quality",),
        target_sample_rate=44100,
        metrics=None,
    )

    resolved = MASTERING_MODULE._resolve_mastering_kwargs_for_input(
        np.zeros(64, dtype=np.float32),
        44100,
        {},
        input_analysis=analysis,
    )

    assert resolved == {"preset": "edm"}


def test_analyze_mastering_input_flags_legacy_and_low_quality(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_collect_mastering_input_metrics",
        lambda signal, sample_rate: MASTERING_MODULE.MasteringInputMetrics(
            integrated_lufs=-18.5,
            crest_factor_db=12.8,
            stereo_width_ratio=0.1,
            low_end_mono_ratio=0.95,
            spectral_tilt=-11.0,
            transient_density=0.025,
            stereo_motion=0.01,
            bass_share=0.28,
            low_mid_share=0.48,
            presence_share=0.06,
            air_share=0.02,
        ),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_select_mastering_preset_from_metrics",
        lambda metrics: "vocal",
    )

    analysis = MASTERING_MODULE._analyze_mastering_input(
        np.zeros((2, 64), dtype=np.float32),
        22050,
    )

    assert analysis.preset_name == "vocal"
    assert analysis.quality_flags == ("Old-Recording", "Low-Quality")
    assert analysis.target_sample_rate == 44100


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


def test_multiband_compress_uses_zero_phase_lr4_filters(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.arange(16, dtype=float)
    sosfiltfilt_calls: list[tuple[str, np.ndarray]] = []

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

    def fake_sosfiltfilt(sos, x, axis=-1):
        signal_slice = np.array(x, copy=True)
        sosfiltfilt_calls.append((sos, signal_slice))
        delta = 1.0 if sos == "low-sos" else 10.0
        return signal_slice + delta

    monkeypatch.setattr(
        MASTERING_MODULE.signal,
        "sosfiltfilt",
        fake_sosfiltfilt,
    )

    mastering.multiband_compress(source)

    assert [call[0] for call in sosfiltfilt_calls] == [
        "low-sos",
        "high-sos",
    ]
    assert all(
        np.array_equal(signal_slice, source)
        for _sos, signal_slice in sosfiltfilt_calls
    )


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


def test_process_stem_skips_master_bus_stages(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(8000, resampling_target=8000)
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    stage_calls: list[str] = []

    monkeypatch.setattr(
        MASTERING_MODULE,
        "apply_exciter",
        lambda y, *_: stage_calls.append("exciter") or y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "freq_cut",
        lambda y, *_, **__: stage_calls.append("filter") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_eq",
        lambda y: stage_calls.append("eq") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_stem_cleanup",
        lambda y, stem_role=None: stage_calls.append("stem_cleanup") or y,
    )
    monkeypatch.setattr(
        mastering,
        "multiband_compress",
        lambda y: stage_calls.append("multiband") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_spatial_enhancement",
        lambda y: stage_calls.append("spatial") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_low_end_mono_tightening",
        lambda y: stage_calls.append("mono") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: stage_calls.append("saturation") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_limiter",
        lambda y, drive_db, ceil_db, **kwargs: (
            stage_calls.append("limiter") or y
        ),
    )
    monkeypatch.setattr(
        mastering,
        "apply_micro_dynamics_finish",
        lambda y: stage_calls.append("micro") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_delivery_trim",
        lambda y: stage_calls.append("delivery") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_safety_clamp",
        lambda y, ceil_db=-0.1: stage_calls.append("clamp") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_final_headroom_recovery",
        lambda y: stage_calls.append("headroom") or y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: (_ for _ in ()).throw(
            AssertionError("stem path should not read LUFS")
        ),
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: (_ for _ in ()).throw(
            AssertionError("stem path should not measure mastering loudness")
        ),
    )
    monkeypatch.setattr(
        mastering,
        "plan_follow_up_action",
        lambda metrics, contract: (_ for _ in ()).throw(
            AssertionError("stem path should not plan follow-up actions")
        ),
    )

    sr_out, y_out = mastering.process_stem(source, 8000)

    assert sr_out == 8000
    assert y_out.shape == (2, 4)
    assert stage_calls == [
        "filter",
        "eq",
        "stem_cleanup",
        "filter",
    ]
    assert set(mastering.last_stage_signals) == {"post_eq", "final_in_memory"}
    assert mastering.last_finalization_actions == ()
    assert mastering.last_mastering_contract is None
    assert mastering.last_peak_catch_events == ()
    assert mastering.last_character_stage_decision is None
    assert mastering.last_resolved_final_true_peak_target_dbfs is None


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


def test_process_scales_exciter_controls_for_legacy_repair(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        exciter_mix=0.3,
        exciter_cutoff_hz=2400.0,
        exciter_max_drive=2.6,
        exciter_high_frequency_cutoff_hz=7200.0,
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
            stereo_width_ratio=0.02,
            low_end_mono_ratio=0.99,
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

    def fake_apply_eq(y):
        mastering.spectral_balance_profile = (
            MASTERING_MODULE.SpectralBalanceProfile(
                rescue_factor=1.0,
                correction_strength=mastering.correction_strength,
                max_boost_db=mastering.max_spectrum_boost_db,
                max_cut_db=mastering.max_spectrum_cut_db,
                band_intensity=mastering.config.intensity,
                restoration_factor=1.0,
                air_restoration_factor=1.0,
                body_restoration_factor=0.8,
                closure_repair_factor=1.0,
            )
        )
        return y

    monkeypatch.setattr(mastering, "apply_eq", fake_apply_eq)

    mastering.process(source, 8000)

    assert exciter_calls[0][2] > mastering.exciter_mix
    assert exciter_calls[0][3] > mastering.exciter_max_drive
    assert exciter_calls[0][4] < mastering.exciter_high_frequency_cutoff_hz


def test_process_lowers_exciter_cutoff_for_closed_legacy_repair(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        exciter_mix=1.0,
        exciter_cutoff_hz=None,
        exciter_max_drive=3.0,
        exciter_high_frequency_cutoff_hz=7200.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    exciter_calls: list[tuple[float | None, float, float, float | None]] = []
    loudness_values = iter([-10.0])

    monkeypatch.setattr(
        MASTERING_MODULE,
        "apply_exciter",
        lambda y, sample_rate, cutoff_hz=None, mix=1.0, max_drive=None, high_frequency_cutoff_hz=None: (
            exciter_calls.append(
                (
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
            stereo_width_ratio=0.02,
            low_end_mono_ratio=0.99,
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

    def fake_apply_eq(y):
        mastering.spectral_balance_profile = (
            MASTERING_MODULE.SpectralBalanceProfile(
                rescue_factor=1.0,
                correction_strength=mastering.correction_strength,
                max_boost_db=mastering.max_spectrum_boost_db,
                max_cut_db=mastering.max_spectrum_cut_db,
                band_intensity=mastering.config.intensity,
                restoration_factor=1.0,
                air_restoration_factor=1.0,
                body_restoration_factor=0.75,
                closure_repair_factor=1.0,
            )
        )
        return y

    monkeypatch.setattr(mastering, "apply_eq", fake_apply_eq)

    mastering.process(source, 8000)

    assert exciter_calls[0][0] is not None
    assert exciter_calls[0][0] < 2500.0
    assert exciter_calls[0][1] == pytest.approx(1.0)


def test_process_retains_more_exciter_air_for_low_end_loaded_legacy_rebalance(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        exciter_mix=1.0,
        exciter_cutoff_hz=None,
        exciter_max_drive=3.0,
        exciter_high_frequency_cutoff_hz=7200.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    exciter_calls: list[tuple[float | None, float, float, float | None]] = []
    loudness_values = iter([-10.0, -10.0])
    rebalance_levels = iter([0.0, 1.0])

    monkeypatch.setattr(
        MASTERING_MODULE,
        "apply_exciter",
        lambda y, sample_rate, cutoff_hz=None, mix=1.0, max_drive=None, high_frequency_cutoff_hz=None: (
            exciter_calls.append(
                (
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
            stereo_width_ratio=0.02,
            low_end_mono_ratio=0.99,
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

    def fake_apply_eq(y):
        mastering.spectral_balance_profile = (
            MASTERING_MODULE.SpectralBalanceProfile(
                rescue_factor=1.0,
                correction_strength=mastering.correction_strength,
                max_boost_db=mastering.max_spectrum_boost_db,
                max_cut_db=mastering.max_spectrum_cut_db,
                band_intensity=mastering.config.intensity,
                restoration_factor=1.0,
                air_restoration_factor=1.0,
                body_restoration_factor=0.75,
                closure_repair_factor=1.0,
                legacy_tonal_rebalance_factor=next(rebalance_levels),
            )
        )
        return y

    monkeypatch.setattr(mastering, "apply_eq", fake_apply_eq)

    mastering.process(source, 8000)
    mastering.process(source, 8000)

    assert exciter_calls[0][3] is not None
    assert exciter_calls[1][3] is not None
    assert exciter_calls[1][3] > exciter_calls[0][3]
    assert exciter_calls[1][3] <= mastering.high_cut


def test_process_deharshes_exciter_controls_when_harshness_risk_is_high(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        44100,
        resampling_target=44100,
        exciter_mix=0.65,
        exciter_cutoff_hz=2400.0,
        exciter_max_drive=3.1,
        exciter_high_frequency_cutoff_hz=7200.0,
    )
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    exciter_calls: list[tuple[float | None, float, float, float | None]] = []
    loudness_values = iter([-10.0, -10.0])
    harshness_levels = iter([0.0, 1.0])

    monkeypatch.setattr(
        MASTERING_MODULE,
        "apply_exciter",
        lambda y, sample_rate, cutoff_hz=None, mix=1.0, max_drive=None, high_frequency_cutoff_hz=None: (
            exciter_calls.append(
                (
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
            stereo_width_ratio=0.02,
            low_end_mono_ratio=0.99,
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

    def fake_apply_eq(y):
        mastering.spectral_balance_profile = (
            MASTERING_MODULE.SpectralBalanceProfile(
                rescue_factor=1.0,
                correction_strength=mastering.correction_strength,
                max_boost_db=mastering.max_spectrum_boost_db,
                max_cut_db=mastering.max_spectrum_cut_db,
                band_intensity=mastering.config.intensity,
                restoration_factor=1.0,
                air_restoration_factor=1.0,
                body_restoration_factor=0.8,
                closure_repair_factor=1.0,
                harshness_restraint_factor=next(harshness_levels),
            )
        )
        return y

    monkeypatch.setattr(mastering, "apply_eq", fake_apply_eq)

    mastering.process(source, 44100)
    mastering.process(source, 44100)

    assert exciter_calls[1][0] > exciter_calls[0][0]
    assert exciter_calls[1][1] < exciter_calls[0][1]
    assert exciter_calls[1][2] < exciter_calls[0][2]
    assert exciter_calls[1][3] < exciter_calls[0][3]


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


def test_process_keeps_post_clamp_to_final_output_linear_when_recovery_runs(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="balanced",
        target_lufs=-8.0,
        drive_db=0.0,
        spectral_drive_bias_db=0.0,
        final_lufs_tolerance=0.2,
        max_final_boost_db=1.0,
        max_follow_up_passes=1,
        limiter_soft_clip_ratio=0.0,
        pre_limiter_saturation_ratio=0.0,
        micro_dynamics_strength=0.0,
        ceil_db=-0.1,
        contract_min_crest_factor_db=0.0,
    )
    source = np.array([0.03, -0.03, 0.05, -0.05, 0.04, -0.04], dtype=np.float32)
    loudness_values = iter([-8.0])
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-9.2,
                max_short_term_lufs=-10.0,
                max_momentary_lufs=-9.8,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-18.0,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.7,
                max_short_term_lufs=-9.5,
                max_momentary_lufs=-9.2,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-17.0,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.7,
                max_short_term_lufs=-9.5,
                max_momentary_lufs=-9.2,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-17.0,
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
        MASTERING_MODULE,
        "_measure_true_peak",
        lambda y, sr, oversample_factor=4: float(
            20.0 * np.log10(max(np.max(np.abs(y)), 1e-12))
        ),
    )

    sr_out, y_out = mastering.process(source, 8000)

    post_clamp = mastering.last_stage_signals["post_clamp"]
    final_in_memory = mastering.last_stage_signals["final_in_memory"]
    mask = np.abs(post_clamp) > 1e-6
    ratios = final_in_memory[mask] / post_clamp[mask]

    assert sr_out == 8000
    assert np.max(np.abs(final_in_memory)) > np.max(np.abs(post_clamp))
    assert np.max(ratios) - np.min(ratios) < 1e-5
    assert mastering.last_headroom_recovery_gain_db > 0.0
    assert np.allclose(y_out, final_in_memory)


def test_process_stem_mastered_input_skips_multiband_and_trims_hot_output(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="balanced",
        target_lufs=-8.0,
        micro_dynamics_strength=0.0,
        pre_limiter_saturation_ratio=0.0,
    )
    mastering.stem_mastered_input = True
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    multiband_calls: list[bool] = []
    loudness_values = iter([-8.0])
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-5.0,
                max_short_term_lufs=-6.0,
                max_momentary_lufs=-5.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-1.0,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.0,
                max_short_term_lufs=-9.0,
                max_momentary_lufs=-8.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-4.0,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.0,
                max_short_term_lufs=-9.0,
                max_momentary_lufs=-8.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-4.0,
            ),
        ]
    )

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "multiband_compress",
        lambda y: multiband_calls.append(True) or y,
    )
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_low_end_mono_tightening",
        lambda y: y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: y,
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_safety_clamp",
        lambda y, ceil_db=-0.1: y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_final_headroom_recovery",
        lambda y: y,
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
        lambda y, drive_db, ceil_db, **kwargs: y,
    )

    sr_out, y_out = mastering.process(source, 8000)

    expected_scale = float(10.0 ** (-3.0 / 20.0))

    assert sr_out == 8000
    assert multiband_calls == []
    assert np.allclose(y_out, np.vstack([source, source]) * expected_scale)
    assert np.allclose(mastering.last_stage_signals["final_in_memory"], y_out)


def test_process_stem_mastered_input_preserves_tonal_stages_for_final_glue(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="balanced",
        target_lufs=-8.0,
        micro_dynamics_strength=0.14,
        pre_limiter_saturation_ratio=0.08,
    )
    mastering.stem_mastered_input = True
    source = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
    stage_calls: list[object] = []
    metrics_values = iter(
        [
            types.SimpleNamespace(
                integrated_lufs=-8.0,
                max_short_term_lufs=-9.0,
                max_momentary_lufs=-8.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-4.0,
            ),
            types.SimpleNamespace(
                integrated_lufs=-8.0,
                max_short_term_lufs=-9.0,
                max_momentary_lufs=-8.5,
                crest_factor_db=7.0,
                stereo_width_ratio=0.2,
                low_end_mono_ratio=0.95,
                true_peak_dbfs=-4.0,
            ),
        ]
    )

    monkeypatch.setattr(
        MASTERING_MODULE,
        "apply_exciter",
        lambda y, *_: stage_calls.append("exciter") or y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "freq_cut",
        lambda y, *_, **__: stage_calls.append("filter") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_eq",
        lambda y: stage_calls.append("eq") or (y * 2.0),
    )
    monkeypatch.setattr(
        mastering,
        "multiband_compress",
        lambda y: stage_calls.append("multiband") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_spatial_enhancement",
        lambda y: stage_calls.append("spatial") or (y * 1.1),
    )
    monkeypatch.setattr(
        mastering,
        "apply_low_end_mono_tightening",
        lambda y: stage_calls.append("mono") or (y * 0.9),
    )
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: stage_calls.append("saturation") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_micro_dynamics_finish",
        lambda y: stage_calls.append("micro") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_delivery_trim",
        lambda y: stage_calls.append("delivery") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_safety_clamp",
        lambda y, ceil_db=-0.1: stage_calls.append("clamp") or y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_final_headroom_recovery",
        lambda y: stage_calls.append("headroom") or y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: -8.0,
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
            stage_calls.append(
                (
                    "limiter",
                    float(drive_db),
                    float(kwargs.get("soft_clip_ratio", -1.0)),
                )
            )
            or y
        ),
    )

    sr_out, y_out = mastering.process(source, 8000)

    assert sr_out == 8000
    assert np.allclose(y_out, np.vstack([source, source]) * 0.99)
    assert stage_calls.count("eq") == 0
    assert stage_calls.count("exciter") == 1
    assert stage_calls.count("multiband") == 0
    assert stage_calls.count("spatial") == 1
    assert stage_calls.count("mono") == 1
    assert stage_calls.count("saturation") == 1
    assert stage_calls.count("micro") == 0
    limiter_calls = [call for call in stage_calls if isinstance(call, tuple)]
    assert len(limiter_calls) == 1
    assert (
        mastering.drive_db + 0.2
        < limiter_calls[0][1]
        <= max(
            mastering.drive_db * 1.48,
            3.35,
        )
    )
    assert (
        mastering.limiter_soft_clip_ratio + 0.02
        < limiter_calls[0][2]
        <= max(
            mastering.limiter_soft_clip_ratio * 1.08,
            0.24,
        )
    )
    assert np.array_equal(
        mastering.last_stage_signals["post_eq"],
        np.vstack([source, source]),
    )
    assert np.allclose(
        mastering.last_stage_signals["post_spatial"],
        np.vstack([source, source]) * 1.1,
    )


def test_process_applies_premaster_true_peak_trim_before_lufs_and_limiter(
    monkeypatch: pytest.MonkeyPatch,
):
    mastering = MASTERING_MODULE.SmartMastering(
        8000,
        resampling_target=8000,
        preset="balanced",
        target_lufs=-8.0,
        micro_dynamics_strength=0.0,
        pre_limiter_saturation_ratio=0.0,
    )
    source = np.array([0.8, -0.8, 0.9, -0.9], dtype=np.float32)
    lufs_input_peaks: list[float] = []
    saturation_input_peaks: list[float] = []
    metrics = types.SimpleNamespace(
        integrated_lufs=-8.0,
        max_short_term_lufs=-9.0,
        max_momentary_lufs=-8.5,
        crest_factor_db=7.0,
        stereo_width_ratio=0.2,
        low_end_mono_ratio=0.95,
        true_peak_dbfs=-4.0,
    )

    monkeypatch.setattr(MASTERING_MODULE, "apply_exciter", lambda y, *_: y)
    monkeypatch.setattr(MASTERING_MODULE, "freq_cut", lambda y, *_, **__: y)
    monkeypatch.setattr(mastering, "apply_eq", lambda y: y)
    monkeypatch.setattr(mastering, "multiband_compress", lambda y: y)
    monkeypatch.setattr(mastering, "apply_spatial_enhancement", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_low_end_mono_tightening",
        lambda y: y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_pre_limiter_saturation",
        lambda y, dynamic_drive_db=0.0: (
            saturation_input_peaks.append(float(np.max(np.abs(y)))) or y
        ),
    )
    monkeypatch.setattr(mastering, "apply_micro_dynamics_finish", lambda y: y)
    monkeypatch.setattr(mastering, "apply_delivery_trim", lambda y: y)
    monkeypatch.setattr(
        mastering,
        "apply_safety_clamp",
        lambda y, ceil_db=-0.1: y,
    )
    monkeypatch.setattr(
        mastering,
        "apply_final_headroom_recovery",
        lambda y: y,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "get_lufs",
        lambda y, sr: lufs_input_peaks.append(float(np.max(np.abs(y)))) or -8.0,
    )
    monkeypatch.setattr(
        MASTERING_MODULE,
        "_measure_mastering_loudness",
        lambda y, sr, **kwargs: metrics,
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

    sr_out, y_out = mastering.process(source, 8000)

    expected_peak = float(10.0 ** (-3.0 / 20.0))

    assert sr_out == 8000
    assert lufs_input_peaks == [pytest.approx(expected_peak, abs=1e-4)]
    assert saturation_input_peaks == [pytest.approx(expected_peak, abs=1e-4)]
    assert float(np.max(np.abs(y_out))) == pytest.approx(
        expected_peak,
        abs=1e-4,
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
    io_module_name = f"{MASTERING_AUDIO_PACKAGE}.io"
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
            stem_mastering=False,
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


def test_master_stems_forces_stem_mastering(monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        MASTERING_MODULE,
        "_master_internal",
        lambda input_path, *, output_path, stem_mastering, **kwargs: (
            captured.update(
                {
                    "input_path": input_path,
                    "output_path": output_path,
                    "stem_mastering": stem_mastering,
                    "kwargs": dict(kwargs),
                }
            )
            or ("out.wav", None)
        ),
    )

    mastered_path, report = MASTERING_MODULE.master(
        "input.wav",
        output_path="out.wav",
        preset="vocal",
    )

    assert mastered_path == "out.wav"
    assert report is None
    assert captured == {
        "input_path": "input.wav",
        "output_path": "out.wav",
        "stem_mastering": True,
        "kwargs": {"preset": "vocal"},
    }


def test_master_routes_stem_mastering_through_audio_separator_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    report_path = tmp_path / "stem-mastering-report.json"
    output_path = tmp_path / "stem-mastered.wav"
    io_module_name = f"{MASTERING_AUDIO_PACKAGE}.io"
    stems_module_name = f"{MASTERING_AUDIO_PACKAGE}.stems"
    mastering_stems_module_name = f"{MASTERING_PACKAGE}.stems"
    original_io_module = sys.modules.get(io_module_name)
    original_stems_module = sys.modules.get(stems_module_name)
    original_mastering_stems_module = sys.modules.get(
        mastering_stems_module_name
    )
    processed_inputs: list[np.ndarray] = []
    helper_calls: list[
        tuple[
            str,
            str,
            int,
            tuple[str, ...],
            float,
            dict[str, object],
            bool,
            str | None,
            str,
        ]
    ] = []
    mixed_signal = np.full((2, 8), 0.25, dtype=np.float32)
    processed_stem_flags: list[bool] = []

    fake_report = types.SimpleNamespace(
        preset_name="balanced",
        to_dict=lambda: {
            "preset_name": "balanced",
            "delivery_profile_name": "lossless",
        },
    )

    class FakeMastering:
        def __init__(self, sr, **kwargs):
            self.target_lufs = -6.0
            self.ceil_db = -0.1
            self.preset_name = kwargs.get("preset", "balanced")
            self.delivery_profile = "lossless"
            self.delivery_decoded_true_peak_dbfs = -0.1
            self.codec_headroom_margin_db = 0.0
            self.delivery_lufs_tolerance_db = 0.6
            self.true_peak_oversample_factor = 1
            self.delivery_bitrate = None
            self.last_stage_signals = {}
            self.stem_mastered_input = False
            self.config = CONFIG_MODULE.SmartMasteringConfig.from_preset(
                kwargs.get("preset")
            )

        def process(self, y, sr):
            processed_stem_flags.append(bool(self.stem_mastered_input))
            processed_inputs.append(np.array(y, copy=True))
            return sr, np.array(y, copy=True)

        def resolve_mastering_contract(self):
            return types.SimpleNamespace(
                max_true_peak_dbfs=self.ceil_db,
                low_end_mono_cutoff_hz=140.0,
            )

    def fake_process_stem_layers(
        audio_path,
        *,
        base_config,
        base_mastering_kwargs,
        process_stem_fn,
        separate_stems_fn,
        read_audio_fn,
        delete_fn,
        model_name,
        shifts,
        quality_flags=(),
        mix_headroom_db,
        resample_fn=None,
        save_mastered_stems=True,
        mastered_stems_output_dir=None,
        save_audio_fn=None,
        mastered_stems_format="wav",
        mastered_stems_bit_depth=32,
        mastered_stems_bitrate=320,
        mastered_stems_compression_level=9,
    ):
        helper_calls.append(
            (
                audio_path,
                model_name,
                shifts,
                tuple(quality_flags),
                mix_headroom_db,
                dict(base_mastering_kwargs),
                save_mastered_stems,
                mastered_stems_output_dir,
                mastered_stems_format,
            )
        )
        return 8000, np.array(mixed_signal, copy=True)

    try:
        sys.modules[io_module_name] = types.SimpleNamespace(
            read_audio=lambda path: (8000, np.zeros((2, 8), dtype=np.float32)),
            save_audio=lambda **kwargs: kwargs["destination_path"],
        )
        sys.modules[stems_module_name] = types.SimpleNamespace(
            separate_stem_layers=lambda *args, **kwargs: ({}, "unused"),
        )
        sys.modules[mastering_stems_module_name] = types.SimpleNamespace(
            process_stem_layers=fake_process_stem_layers,
        )
        monkeypatch.setitem(
            sys.modules,
            "definers.system",
            types.SimpleNamespace(
                tmp=lambda suffix, keep=False: str(output_path),
                delete=lambda path: None,
            ),
        )
        monkeypatch.setattr(MASTERING_MODULE, "SmartMastering", FakeMastering)
        monkeypatch.setattr(
            MASTERING_MODULE,
            "_analyze_mastering_input",
            lambda signal, sample_rate: MASTERING_MODULE.MasteringInputAnalysis(
                preset_name="balanced",
                quality_flags=("Low-Quality",),
                target_sample_rate=48000,
                metrics=None,
            ),
        )
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
            stem_mastering=True,
            stem_model_name="htdemucs_6s",
            stem_shifts=4,
            stem_mix_headroom_db=7.0,
            preset="balanced",
            report_path=str(report_path),
        )
    finally:
        if original_io_module is None:
            sys.modules.pop(io_module_name, None)
        else:
            sys.modules[io_module_name] = original_io_module
        if original_stems_module is None:
            sys.modules.pop(stems_module_name, None)
        else:
            sys.modules[stems_module_name] = original_stems_module
        if original_mastering_stems_module is None:
            sys.modules.pop(mastering_stems_module_name, None)
        else:
            sys.modules[mastering_stems_module_name] = (
                original_mastering_stems_module
            )

    assert mastered_path == str(output_path)
    assert report is fake_report
    assert helper_calls == [
        (
            "input.wav",
            "htdemucs_6s",
            4,
            ("Low-Quality",),
            7.0,
            {"preset": "balanced", "resampling_target": 48000},
            True,
            str(output_path.with_suffix("")) + "_stems",
            "wav",
        )
    ]
    assert len(processed_inputs) == 1
    assert processed_stem_flags == [True]
    assert np.allclose(processed_inputs[0], mixed_signal)


def test_master_auto_selects_preset_before_stem_mastering_helper(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output_path = tmp_path / "stem-auto-mastered.wav"
    io_module_name = f"{MASTERING_AUDIO_PACKAGE}.io"
    stems_module_name = f"{MASTERING_AUDIO_PACKAGE}.stems"
    mastering_stems_module_name = f"{MASTERING_PACKAGE}.stems"
    original_io_module = sys.modules.get(io_module_name)
    original_stems_module = sys.modules.get(stems_module_name)
    original_mastering_stems_module = sys.modules.get(
        mastering_stems_module_name
    )
    helper_calls: list[dict[str, object]] = []

    class FakeMastering:
        def __init__(self, sr, **kwargs):
            self.target_lufs = -8.9
            self.ceil_db = -0.1
            self.preset_name = kwargs.get("preset", "balanced")
            self.delivery_profile = "lossless"
            self.delivery_decoded_true_peak_dbfs = -0.1
            self.codec_headroom_margin_db = 0.0
            self.delivery_lufs_tolerance_db = 0.6
            self.true_peak_oversample_factor = 1
            self.delivery_bitrate = None
            self.last_stage_signals = {}
            self.config = CONFIG_MODULE.SmartMasteringConfig.from_preset(
                kwargs.get("preset")
            )

        def process(self, y, sr):
            return sr, np.array(y, copy=True)

        def resolve_mastering_contract(self):
            return types.SimpleNamespace(
                max_true_peak_dbfs=self.ceil_db,
                low_end_mono_cutoff_hz=140.0,
            )

    def fake_process_stem_layers(
        audio_path,
        *,
        base_config,
        base_mastering_kwargs,
        process_stem_fn,
        separate_stems_fn,
        read_audio_fn,
        delete_fn,
        model_name,
        shifts,
        quality_flags=(),
        mix_headroom_db,
        resample_fn=None,
        save_mastered_stems=True,
        mastered_stems_output_dir=None,
        save_audio_fn=None,
        mastered_stems_format="wav",
        mastered_stems_bit_depth=32,
        mastered_stems_bitrate=320,
        mastered_stems_compression_level=9,
    ):
        helper_calls.append(
            {
                **dict(base_mastering_kwargs),
                "quality_flags": tuple(quality_flags),
                "save_mastered_stems": save_mastered_stems,
                "mastered_stems_output_dir": mastered_stems_output_dir,
                "mastered_stems_format": mastered_stems_format,
            }
        )
        return 8000, np.zeros((2, 8), dtype=np.float32)

    try:
        sys.modules[io_module_name] = types.SimpleNamespace(
            read_audio=lambda path: (8000, np.zeros((2, 8), dtype=np.float32)),
            save_audio=lambda **kwargs: kwargs["destination_path"],
        )
        sys.modules[stems_module_name] = types.SimpleNamespace(
            separate_stem_layers=lambda *args, **kwargs: ({}, "unused"),
        )
        sys.modules[mastering_stems_module_name] = types.SimpleNamespace(
            process_stem_layers=fake_process_stem_layers,
        )
        monkeypatch.setitem(
            sys.modules,
            "definers.system",
            types.SimpleNamespace(
                tmp=lambda suffix, keep=False: str(output_path),
                delete=lambda path: None,
            ),
        )
        monkeypatch.setattr(MASTERING_MODULE, "SmartMastering", FakeMastering)
        monkeypatch.setattr(
            MASTERING_MODULE,
            "_select_mastering_preset",
            lambda signal, sample_rate: "vocal",
        )
        monkeypatch.setattr(
            MASTERING_MODULE,
            "_analyze_mastering_input",
            lambda signal, sample_rate: MASTERING_MODULE.MasteringInputAnalysis(
                preset_name="vocal",
                quality_flags=("Old-Recording",),
                target_sample_rate=48000,
                metrics=None,
            ),
        )
        monkeypatch.setattr(
            MASTERING_MODULE,
            "save_verified_audio",
            lambda **kwargs: (
                kwargs["destination_path"],
                kwargs["audio_signal"],
                types.SimpleNamespace(
                    report=types.SimpleNamespace(
                        preset_name=kwargs["preset_name"],
                        to_dict=lambda: {
                            "preset_name": kwargs["preset_name"],
                            "delivery_profile_name": "lossless",
                        },
                    )
                ),
            ),
        )

        mastered_path, report = MASTERING_MODULE.master(
            "input.wav",
            output_path=str(output_path),
            stem_mastering=True,
        )
    finally:
        if original_io_module is None:
            sys.modules.pop(io_module_name, None)
        else:
            sys.modules[io_module_name] = original_io_module
        if original_stems_module is None:
            sys.modules.pop(stems_module_name, None)
        else:
            sys.modules[stems_module_name] = original_stems_module
        if original_mastering_stems_module is None:
            sys.modules.pop(mastering_stems_module_name, None)
        else:
            sys.modules[mastering_stems_module_name] = (
                original_mastering_stems_module
            )

    assert mastered_path == str(output_path)
    assert report.preset_name == "vocal"
    assert helper_calls == [
        {
            "preset": "vocal",
            "resampling_target": 48000,
            "quality_flags": ("Old-Recording",),
            "save_mastered_stems": True,
            "mastered_stems_output_dir": str(output_path.with_suffix(""))
            + "_stems",
            "mastered_stems_format": "wav",
        }
    ]


def test_process_stem_signal_uses_dedicated_stem_path(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[str] = []
    stem_roles: list[str | None] = []

    class FakeMastering:
        def __init__(self, sr, **kwargs):
            self.sample_rate = sr
            self.kwargs = kwargs

        def process(self, y, sr):
            calls.append("process")
            return sr, np.array(y, copy=True) * 2.0

        def process_stem(self, y, sr, stem_role=None):
            calls.append("process_stem")
            stem_roles.append(stem_role)
            return sr, np.array(y, copy=True) * 0.5

    monkeypatch.setattr(MASTERING_MODULE, "SmartMastering", FakeMastering)

    sample_rate, processed = MASTERING_MODULE._process_stem_signal(
        np.array([0.2, -0.2], dtype=np.float32),
        8000,
        {"preset": "balanced", "stem_role": "vocals"},
    )

    assert sample_rate == 8000
    assert calls == ["process_stem"]
    assert stem_roles == ["vocals"]
    assert np.allclose(processed, np.array([0.1, -0.1], dtype=np.float32))
