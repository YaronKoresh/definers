import importlib.util
import sys
import types
from dataclasses import replace
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
_SCIPY_MODULE_NAMES = ("scipy", "scipy.signal", "scipy.io", "scipy.io.wavfile")


def _restore_modules(backup: dict[str, object | None]) -> None:
    for name, module in backup.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_scipy_stub() -> None:
    scipy_module = types.ModuleType("scipy")
    scipy_module.__version__ = "1.11.0"
    scipy_module.__path__ = []
    signal_module = types.ModuleType("scipy.signal")
    io_module = types.ModuleType("scipy.io")
    io_module.__path__ = []
    wavfile_module = types.ModuleType("scipy.io.wavfile")

    def lfilter(_b, _a, y, axis=-1):
        return np.array(y, dtype=np.float32, copy=True)

    def resample_poly(y, up, down, axis=-1):
        array = np.asarray(y, dtype=np.float32)
        if max(int(up), 1) == 1 and max(int(down), 1) == 1:
            return np.array(array, copy=True)
        repeated = np.repeat(array, max(int(up), 1), axis=axis)
        return repeated[..., :: max(int(down), 1)]

    signal_module.lfilter = lfilter
    signal_module.resample_poly = resample_poly
    signal_module.butter = lambda *args, **kwargs: "sos"
    signal_module.sosfiltfilt = lambda sos, x, axis=-1: np.array(
        x, dtype=np.float32, copy=True
    )
    wavfile_module.write = lambda *args, **kwargs: None

    scipy_module.signal = signal_module
    scipy_module.io = io_module
    io_module.wavfile = wavfile_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module
    sys.modules["scipy.io"] = io_module
    sys.modules["scipy.io.wavfile"] = wavfile_module


def _load_reporting_modules(package_name: str):
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
    backup = {name: sys.modules.get(name) for name in _SCIPY_MODULE_NAMES}
    _install_scipy_stub()

    try:
        config_module = _load_module(
            f"{package_name}.config", AUDIO_ROOT / "config.py"
        )
        loudness_module = _load_module(
            f"{package_name}.mastering_loudness",
            AUDIO_ROOT / "mastering_loudness.py",
        )
        contract_module = _load_module(
            f"{package_name}.mastering_contract",
            AUDIO_ROOT / "mastering_contract.py",
        )
        metrics_module = _load_module(
            f"{package_name}.mastering_metrics",
            AUDIO_ROOT / "mastering_metrics.py",
        )
        presets_module = _load_module(
            f"{package_name}.mastering_presets",
            AUDIO_ROOT / "mastering_presets.py",
        )
        return (
            config_module,
            loudness_module,
            contract_module,
            metrics_module,
            presets_module,
        )
    finally:
        _restore_modules(backup)


(
    CONFIG_MODULE,
    LOUDNESS_MODULE,
    CONTRACT_MODULE,
    METRICS_MODULE,
    PRESETS_MODULE,
) = _load_reporting_modules("_test_audio_mastering_reporting_pkg.audio")


def test_measure_mastering_loudness_reports_silence_metrics():
    metrics = LOUDNESS_MODULE.measure_mastering_loudness(
        np.zeros(128, dtype=np.float32), 8000
    )

    assert metrics.integrated_lufs == pytest.approx(-70.0)
    assert metrics.sample_peak_dbfs == pytest.approx(-120.0)
    assert metrics.true_peak_dbfs == pytest.approx(-120.0)
    assert metrics.loudness_range_lu == pytest.approx(0.0)
    assert metrics.stereo_width_ratio == pytest.approx(0.0)
    assert metrics.low_end_mono_ratio == pytest.approx(1.0)


def test_measure_mastering_loudness_reports_finite_values_for_tone():
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    tone = 0.5 * np.sin(2.0 * np.pi * 220.0 * time_axis).astype(np.float32)

    metrics = LOUDNESS_MODULE.measure_mastering_loudness(tone, 8000)

    assert np.isfinite(metrics.integrated_lufs)
    assert metrics.sample_peak_dbfs == pytest.approx(-6.0206, abs=0.2)
    assert metrics.true_peak_dbfs >= metrics.sample_peak_dbfs
    assert metrics.crest_factor_db >= 0.0
    assert metrics.stereo_width_ratio == pytest.approx(0.0)
    assert metrics.low_end_mono_ratio == pytest.approx(1.0)


def test_measure_mastering_loudness_reports_stereo_width_and_mono_bass_ratio():
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    left = 0.5 * np.sin(2.0 * np.pi * 80.0 * time_axis).astype(np.float32)
    right = -left
    stereo_signal = np.stack([left, right], axis=0)

    metrics = LOUDNESS_MODULE.measure_mastering_loudness(stereo_signal, 8000)

    assert metrics.stereo_width_ratio > 0.95
    assert metrics.low_end_mono_ratio < 0.05


def test_measure_metric_batch_reuses_results_for_identical_signal_objects():
    calls: list[tuple[int, int]] = []
    original_measure = METRICS_MODULE.measure_mastering_loudness
    signal = np.ones((2, 64), dtype=np.float32)

    def fake_measure(y, sr, **kwargs):
        calls.append((id(y), int(sr)))
        return types.SimpleNamespace(signal_id=id(y), sample_rate=int(sr))

    METRICS_MODULE.measure_mastering_loudness = fake_measure
    try:
        metrics = METRICS_MODULE._measure_metric_batch(
            {
                "input": (signal, 8000),
                "output": (signal, 8000),
                "decoded": (signal, 8000),
            },
            true_peak_oversample_factor=4,
            low_end_mono_cutoff_hz=160.0,
            signal_module=types.SimpleNamespace(),
        )
    finally:
        METRICS_MODULE.measure_mastering_loudness = original_measure

    assert calls == [(id(signal), 8000)]
    assert metrics["input"] is metrics["output"]
    assert metrics["output"] is metrics["decoded"]


def test_generate_mastering_report_tracks_gain_deltas():
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    input_signal = 0.2 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(
        np.float32
    )
    output_signal = 0.5 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(
        np.float32
    )
    contract = CONTRACT_MODULE.resolve_mastering_contract(
        "edm",
        target_lufs=-8.0,
        ceil_db=-0.1,
        max_short_term_lufs=-5.5,
        max_momentary_lufs=-4.5,
    )
    decoded_contract = replace(
        contract,
        name="edm:decoded",
        target_lufs_tolerance_db=1.0,
        max_true_peak_dbfs=-1.0,
    )

    report = METRICS_MODULE.generate_mastering_report(
        input_signal,
        output_signal,
        8000,
        post_eq_signal=input_signal * 1.2,
        post_spatial_signal=output_signal * 0.98,
        post_limiter_signal=output_signal * 0.95,
        post_character_signal=output_signal * 0.92,
        post_peak_catch_signal=output_signal * 0.9,
        post_delivery_trim_signal=output_signal * 0.88,
        post_clamp_signal=output_signal * 0.87,
        decoded_signal=output_signal * 0.9,
        decoded_sample_rate=8000,
        target_lufs=-8.0,
        ceil_db=-0.1,
        preset_name="edm",
        contract=contract,
        decoded_contract=decoded_contract,
        character_stage_decision=types.SimpleNamespace(
            applied=True,
            reverted=False,
            reasons=("kept",),
            input_integrated_lufs=-8.5,
            output_integrated_lufs=-8.0,
            input_true_peak_dbfs=-0.4,
            output_true_peak_dbfs=-0.2,
        ),
        peak_catch_events=(
            types.SimpleNamespace(
                attempt_index=1,
                drive_db=0.0,
                ceil_db=-0.2,
                soft_clip_ratio=0.2,
                peak_over_db=0.3,
                before_integrated_lufs=-8.0,
                after_integrated_lufs=-8.02,
                before_true_peak_dbfs=0.2,
                after_true_peak_dbfs=-0.05,
            ),
        ),
        resolved_true_peak_target_dbfs=-0.1,
        stereo_motion_activity=0.26,
        stereo_motion_correlation_guard=0.82,
        delivery_trim_attenuation_db=0.15,
        delivery_trim_input_true_peak_dbfs=0.05,
        delivery_trim_target_dbfs=-0.1,
        delivery_trim_output_true_peak_dbfs=-0.12,
        post_clamp_true_peak_dbfs=-0.09,
        post_clamp_true_peak_delta_db=0.03,
        headroom_recovery_gain_db=0.28,
        headroom_recovery_input_true_peak_dbfs=-0.38,
        headroom_recovery_output_true_peak_dbfs=-0.1,
        headroom_recovery_failure_reasons=(),
        headroom_recovery_mode="guarded",
        headroom_recovery_integrated_gap_db=0.9,
        headroom_recovery_transient_density=0.18,
        headroom_recovery_closed_margin_db=0.28,
        headroom_recovery_unused_margin_db=0.0,
    )

    assert report.preset_name == "edm"
    assert report.contract is not None
    assert report.post_eq_metrics is not None
    assert report.post_spatial_metrics is not None
    assert report.post_limiter_metrics is not None
    assert report.post_character_metrics is not None
    assert report.post_peak_catch_metrics is not None
    assert report.post_delivery_trim_metrics is not None
    assert report.post_clamp_metrics is not None
    assert report.final_in_memory_metrics.integrated_lufs == pytest.approx(
        report.output_metrics.integrated_lufs
    )
    assert report.output_contract_assessment is not None
    assert report.decoded_contract_assessment is not None
    assert report.character_stage_decision is not None
    assert len(report.peak_catch_events) == 1
    assert report.delivery_trim_attenuation_db == pytest.approx(0.15)
    assert report.delivery_trim_output_true_peak_dbfs == pytest.approx(-0.12)
    assert report.post_clamp_true_peak_dbfs == pytest.approx(-0.09)
    assert report.post_clamp_true_peak_delta_db == pytest.approx(0.03)
    assert report.headroom_recovery_gain_db == pytest.approx(0.28)
    assert report.headroom_recovery_output_true_peak_dbfs == pytest.approx(-0.1)
    assert report.headroom_recovery_mode == "guarded"
    assert report.headroom_recovery_closed_margin_db == pytest.approx(0.28)
    assert report.stereo_motion_activity == pytest.approx(0.26)
    assert report.stereo_motion_correlation_guard == pytest.approx(0.82)
    assert report.post_spatial_stereo_motion is not None
    assert report.output_stereo_motion is not None
    assert report.resolved_true_peak_target_dbfs == pytest.approx(-0.1)
    assert report.integrated_lufs_delta > 0.0
    assert report.sample_peak_delta_db > 0.0
    assert report.true_peak_delta_db > 0.0
    assert report.true_peak_margin_db is not None
    assert report.to_dict()["preset_name"] == "edm"


def test_generate_mastering_report_uses_resolved_true_peak_target_for_margin():
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    input_signal = 0.2 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(
        np.float32
    )
    output_signal = 0.5 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(
        np.float32
    )

    report = METRICS_MODULE.generate_mastering_report(
        input_signal,
        output_signal,
        8000,
        ceil_db=-0.1,
        resolved_true_peak_target_dbfs=-1.0,
    )

    assert report.true_peak_margin_db == pytest.approx(
        -1.0 - report.output_metrics.true_peak_dbfs
    )


def test_write_mastering_report_serializes_json(tmp_path: Path):
    time_axis = np.linspace(0.0, 1.0, 8000, endpoint=False)
    input_signal = 0.2 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(
        np.float32
    )
    output_signal = 0.5 * np.sin(2.0 * np.pi * 110.0 * time_axis).astype(
        np.float32
    )

    report = METRICS_MODULE.generate_mastering_report(
        input_signal,
        output_signal,
        8000,
        post_spatial_signal=output_signal * 0.96,
        post_character_signal=output_signal * 0.95,
        post_peak_catch_signal=output_signal * 0.94,
        post_delivery_trim_signal=output_signal * 0.93,
        post_clamp_signal=output_signal * 0.92,
        preset_name="edm",
        stereo_motion_activity=0.24,
        delivery_trim_attenuation_db=0.25,
        delivery_trim_output_true_peak_dbfs=-0.3,
        post_clamp_true_peak_dbfs=-0.2,
        headroom_recovery_gain_db=0.2,
        headroom_recovery_mode="makeup_only",
        headroom_recovery_failure_reasons=("loudness_already_within_tolerance",),
    )
    destination = tmp_path / "mastering-report.json"

    written_path = METRICS_MODULE.write_mastering_report(
        report, str(destination)
    )

    assert written_path == str(destination)
    payload = destination.read_text(encoding="utf-8")
    assert '"preset_name": "edm"' in payload
    assert '"post_character_metrics"' in payload
    assert '"post_peak_catch_metrics"' in payload
    assert '"post_delivery_trim_metrics"' in payload
    assert '"post_clamp_metrics"' in payload
    assert '"post_spatial_metrics"' in payload
    assert '"delivery_trim_attenuation_db"' in payload
    assert '"headroom_recovery_gain_db"' in payload


def test_mastering_presets_wrap_config_factories():
    balanced = PRESETS_MODULE.balanced()
    edm = PRESETS_MODULE.edm()
    vocal = PRESETS_MODULE.mastering_preset("vocal")

    assert balanced.preset_name == "balanced"
    assert edm.preset_name == "edm"
    assert vocal.preset_name == "vocal"
    assert PRESETS_MODULE.MasteringPresets.names() == (
        "balanced",
        "edm",
        "vocal",
    )
    assert edm.target_lufs > balanced.target_lufs > vocal.target_lufs
    assert edm.drive_db > balanced.drive_db > vocal.drive_db
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
    assert vocal.treble_boost_db_per_oct > balanced.treble_boost_db_per_oct
    assert balanced.treble_boost_db_per_oct > edm.treble_boost_db_per_oct
    assert vocal.micro_dynamics_strength > balanced.micro_dynamics_strength
    assert balanced.micro_dynamics_strength > edm.micro_dynamics_strength
    assert balanced.stereo_tone_variation_db > edm.stereo_tone_variation_db
    assert vocal.stereo_tone_variation_db > balanced.stereo_tone_variation_db
    assert vocal.stereo_motion_high_amount > balanced.stereo_motion_high_amount
    assert balanced.stereo_motion_high_amount > edm.stereo_motion_high_amount
    assert vocal.mono_bass_hz < balanced.mono_bass_hz < edm.mono_bass_hz


def test_unknown_mastering_preset_raises_value_error():
    with pytest.raises(ValueError, match="Unknown mastering preset"):
        CONFIG_MODULE.SmartMasteringConfig.from_preset("not_a_preset")
