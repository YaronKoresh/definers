import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


def _load_finalization_module(package_name: str):
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
    _load_module(
        f"{package_name}.mastering_loudness",
        AUDIO_ROOT / "mastering_loudness.py",
    )
    _load_module(
        f"{package_name}.mastering_contract",
        AUDIO_ROOT / "mastering_contract.py",
    )
    _load_module(
        f"{package_name}.mastering_dynamics",
        AUDIO_ROOT / "mastering_dynamics.py",
    )
    return _load_module(
        f"{package_name}.mastering_finalization",
        AUDIO_ROOT / "mastering_finalization.py",
    )


FINALIZATION_MODULE = _load_finalization_module(
    "_test_audio_mastering_finalization_pkg.audio"
)


def _mastering_stub(**overrides):
    data = {
        "target_lufs": -6.0,
        "drive_db": 1.0,
        "spectral_drive_bias_db": 1.5,
        "limiter_soft_clip_ratio": 0.2,
        "spectral_balance_profile": SimpleNamespace(rescue_factor=0.5),
        "pre_limiter_saturation_ratio": 0.0,
        "ceil_db": -0.1,
        "delivery_decoded_true_peak_dbfs": None,
        "true_peak_oversample_factor": 4,
    }
    data.update(overrides)
    return SimpleNamespace(**data)


def test_compute_dynamic_drive_includes_rescue_bias():
    mastering = _mastering_stub(
        target_lufs=-5.0,
        drive_db=1.0,
        spectral_drive_bias_db=2.0,
        spectral_balance_profile=SimpleNamespace(rescue_factor=0.5),
    )

    dynamic_drive = FINALIZATION_MODULE.compute_dynamic_drive(mastering, -10.0)

    assert dynamic_drive == pytest.approx(7.0)


def test_compute_primary_soft_clip_ratio_grows_with_drive():
    mastering = _mastering_stub(
        limiter_soft_clip_ratio=0.2,
        spectral_balance_profile=SimpleNamespace(rescue_factor=1.0),
    )

    ratio = FINALIZATION_MODULE.compute_primary_soft_clip_ratio(mastering, 10.0)

    assert ratio > mastering.limiter_soft_clip_ratio


def test_apply_pre_limiter_saturation_is_noop_when_disabled():
    mastering = _mastering_stub(pre_limiter_saturation_ratio=0.0)
    source = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    saturated = FINALIZATION_MODULE.apply_pre_limiter_saturation(
        mastering,
        source,
        dynamic_drive_db=6.0,
    )

    assert np.allclose(saturated, source)


def test_apply_pre_limiter_saturation_softens_hot_signal():
    mastering = _mastering_stub(pre_limiter_saturation_ratio=0.2)
    source = np.array([0.0, 0.6, 0.95, -0.95], dtype=np.float32)

    saturated = FINALIZATION_MODULE.apply_pre_limiter_saturation(
        mastering,
        source,
        dynamic_drive_db=8.0,
    )

    assert saturated.shape == source.shape
    assert np.max(np.abs(saturated)) <= 1.0
    assert not np.allclose(saturated, source)


def test_resolve_final_true_peak_target_prefers_stricter_delivery_target():
    mastering = _mastering_stub(
        ceil_db=-0.1,
        delivery_decoded_true_peak_dbfs=-1.0,
    )

    target_db = FINALIZATION_MODULE.resolve_final_true_peak_target(mastering)

    assert target_db == pytest.approx(-1.0)


def test_apply_delivery_trim_reduces_signal_when_true_peak_exceeds_target():
    mastering = _mastering_stub(
        ceil_db=-0.1,
        delivery_decoded_true_peak_dbfs=-1.0,
        true_peak_oversample_factor=8,
    )
    source = np.array([0.2, -0.4, 0.98], dtype=np.float32)

    trimmed = FINALIZATION_MODULE.apply_delivery_trim(
        mastering,
        source,
        sample_rate=44100,
        measure_true_peak_fn=lambda y, sr, oversample_factor=4: float(
            20.0 * np.log10(max(np.max(np.abs(y)), 1e-12))
        ),
    )

    assert np.max(np.abs(trimmed)) < np.max(np.abs(source))
    assert mastering.last_delivery_trim_attenuation_db > 0.8
    assert mastering.last_delivery_trim_input_true_peak_dbfs == pytest.approx(
        float(20.0 * np.log10(0.98)),
        abs=1e-5,
    )
    assert mastering.last_delivery_trim_target_dbfs == pytest.approx(-1.0)
    assert mastering.last_delivery_trim_output_true_peak_dbfs == pytest.approx(
        -1.0,
        abs=1e-4,
    )


def test_apply_final_headroom_recovery_closes_remaining_true_peak_margin():
    mastering = _mastering_stub(
        ceil_db=-0.1,
        true_peak_oversample_factor=1,
    )
    source = np.array([0.2, -0.25, 0.3], dtype=np.float32)

    recovered = FINALIZATION_MODULE.apply_final_headroom_recovery(
        mastering,
        source,
        sample_rate=44100,
        measure_true_peak_fn=lambda y, sr, oversample_factor=4: float(
            20.0 * np.log10(max(np.max(np.abs(y)), 1e-12))
        ),
    )

    assert np.max(np.abs(recovered)) > np.max(np.abs(source))
    assert mastering.last_headroom_recovery_gain_db > 0.0
    assert mastering.last_headroom_recovery_input_true_peak_dbfs is not None
    assert (
        mastering.last_headroom_recovery_output_true_peak_dbfs
        == pytest.approx(
            -0.1,
            abs=1e-4,
        )
    )
    assert (
        "unused_headroom_remains"
        not in mastering.last_headroom_recovery_failure_reasons
    )


def test_apply_final_headroom_recovery_shaves_peaks_when_direct_boost_overshoots():
    mastering = _mastering_stub(
        ceil_db=-0.1,
        true_peak_oversample_factor=1,
        target_lufs=-6.0,
        final_lufs_tolerance=0.2,
        max_final_boost_db=6.0,
        last_post_clamp_metrics=SimpleNamespace(
            integrated_lufs=-7.4,
            crest_factor_db=8.5,
        ),
        last_mastering_contract=SimpleNamespace(min_crest_factor_db=4.0),
    )
    source = np.array([0.82, -0.82, 0.9, -0.9], dtype=np.float32)

    def fake_true_peak(y, sr, oversample_factor=4):
        peak = float(np.max(np.abs(y)))
        intersample_multiplier = 1.0 + max(peak - 0.9, 0.0) * 0.9
        return float(20.0 * np.log10(max(peak * intersample_multiplier, 1e-12)))

    recovered = FINALIZATION_MODULE.apply_final_headroom_recovery(
        mastering,
        source,
        sample_rate=44100,
        measure_true_peak_fn=fake_true_peak,
    )

    assert fake_true_peak(recovered, 44100) <= mastering.ceil_db + 0.05
    assert mastering.last_headroom_recovery_mode == "transient_shave"
    assert mastering.last_headroom_recovery_transient_density is not None
    assert mastering.last_headroom_recovery_closed_margin_db is not None


def test_apply_final_headroom_recovery_prefers_makeup_only_for_dense_material():
    mastering = _mastering_stub(
        ceil_db=-0.1,
        true_peak_oversample_factor=1,
        target_lufs=-6.0,
        final_lufs_tolerance=0.2,
        max_final_boost_db=4.0,
        last_post_clamp_metrics=SimpleNamespace(
            integrated_lufs=-6.1,
            crest_factor_db=4.6,
        ),
        last_mastering_contract=SimpleNamespace(min_crest_factor_db=4.5),
    )
    source = np.linspace(-0.48, 0.48, 64, dtype=np.float32)

    recovered = FINALIZATION_MODULE.apply_final_headroom_recovery(
        mastering,
        source,
        sample_rate=44100,
        measure_true_peak_fn=lambda y, sr, oversample_factor=4: float(
            20.0 * np.log10(max(np.max(np.abs(y)), 1e-12))
        ),
    )

    assert np.max(np.abs(recovered)) > np.max(np.abs(source))
    assert mastering.last_headroom_recovery_mode == "makeup_only"
    assert mastering.last_headroom_recovery_unused_margin_db is not None


def test_apply_stereo_width_restraint_reduces_side_energy():
    source = np.array(
        [[1.0, -1.0, 1.0, -1.0], [-1.0, 1.0, -1.0, 1.0]],
        dtype=np.float32,
    )

    narrowed = FINALIZATION_MODULE.apply_stereo_width_restraint(
        source,
        stereo_width_scale=0.5,
    )

    assert np.max(np.abs(narrowed[0] - narrowed[1])) < np.max(
        np.abs(source[0] - source[1])
    )


def test_plan_follow_up_action_requests_gain_and_stereo_restraint_when_needed():
    mastering = _mastering_stub(
        limiter_soft_clip_ratio=0.22,
        final_lufs_tolerance=0.2,
        max_final_boost_db=2.0,
        compute_primary_soft_clip_ratio=lambda drive_db: 0.22 + drive_db * 0.01,
    )
    metrics = SimpleNamespace(
        integrated_lufs=-8.0,
        max_short_term_lufs=-5.0,
        max_momentary_lufs=-4.0,
        crest_factor_db=7.5,
        stereo_width_ratio=0.75,
        low_end_mono_ratio=0.45,
    )
    contract = SimpleNamespace(
        target_lufs=-6.0,
        max_short_term_lufs=-5.5,
        max_momentary_lufs=-4.5,
        min_crest_factor_db=4.0,
        max_crest_factor_db=10.0,
        max_stereo_width_ratio=0.45,
        min_low_end_mono_ratio=0.8,
    )

    action = FINALIZATION_MODULE.plan_follow_up_action(
        mastering, metrics, contract
    )

    assert action.should_apply is True
    assert action.gain_db > 0.0
    assert action.stereo_width_scale < 1.0
    assert "gain" in action.reasons
    assert "stereo_restraint" in action.reasons


def test_plan_follow_up_action_skips_clip_density_without_gain():
    mastering = _mastering_stub(
        limiter_soft_clip_ratio=0.22,
        final_lufs_tolerance=0.2,
        max_final_boost_db=2.0,
        compute_primary_soft_clip_ratio=lambda drive_db: 0.22 + drive_db * 0.02,
    )
    metrics = SimpleNamespace(
        integrated_lufs=-5.0,
        max_short_term_lufs=-5.2,
        max_momentary_lufs=-4.8,
        crest_factor_db=6.5,
        stereo_width_ratio=0.25,
        low_end_mono_ratio=0.92,
    )
    contract = SimpleNamespace(
        target_lufs=-6.0,
        max_short_term_lufs=-4.5,
        max_momentary_lufs=-3.5,
        min_crest_factor_db=4.0,
        max_crest_factor_db=10.0,
        max_stereo_width_ratio=0.5,
        min_low_end_mono_ratio=0.8,
    )

    action = FINALIZATION_MODULE.plan_follow_up_action(
        mastering, metrics, contract
    )

    assert action.should_apply is False
    assert action.gain_db == pytest.approx(0.0)
    assert action.soft_clip_ratio == pytest.approx(
        mastering.limiter_soft_clip_ratio
    )
    assert action.reasons == ()
