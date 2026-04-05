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
_SCIPY_MODULE_NAMES = ("scipy", "scipy.signal")


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
    scipy_module.signal = signal_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module


def _load_delivery_modules(package_name: str):
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
        _load_module(
            f"{package_name}.mastering_loudness",
            AUDIO_ROOT / "mastering_loudness.py",
        )
        contract_module = _load_module(
            f"{package_name}.mastering_contract",
            AUDIO_ROOT / "mastering_contract.py",
        )
        _load_module(
            f"{package_name}.mastering_metrics",
            AUDIO_ROOT / "mastering_metrics.py",
        )
        delivery_module = _load_module(
            f"{package_name}.mastering_delivery",
            AUDIO_ROOT / "mastering_delivery.py",
        )
        return contract_module, delivery_module
    finally:
        _restore_modules(backup)


CONTRACT_MODULE, DELIVERY_MODULE = _load_delivery_modules(
    "_test_audio_mastering_delivery_pkg.audio"
)


def test_resolve_delivery_profile_uses_lossy_defaults_for_mp3():
    profile = DELIVERY_MODULE.resolve_delivery_profile(None, "track.mp3")

    assert profile.name == "lossy"
    assert profile.is_lossy is True
    assert profile.decoded_true_peak_dbfs == pytest.approx(-0.6)


def test_resolve_delivery_profile_preserves_explicit_lossless_profile_for_mp3():
    profile = DELIVERY_MODULE.resolve_delivery_profile("lossless", "track.mp3")

    assert profile.name == "lossy"
    assert profile.decoded_true_peak_dbfs == pytest.approx(-0.6)


def test_save_verified_audio_retries_with_attenuation_until_profile_passes():
    saved_signals: list[np.ndarray] = []
    decoded_signals = iter(
        [
            np.full(32, 0.98, dtype=np.float32),
            np.full(32, 0.84, dtype=np.float32),
        ]
    )

    def fake_save_audio(**kwargs):
        saved_signals.append(np.array(kwargs["audio_signal"], copy=True))
        return kwargs["destination_path"]

    final_path, final_signal, verification = (
        DELIVERY_MODULE.save_verified_audio(
            destination_path="track.mp3",
            audio_signal=np.full(32, 0.9, dtype=np.float32),
            sample_rate=8000,
            input_signal=np.zeros(32, dtype=np.float32),
            save_audio_fn=fake_save_audio,
            read_audio_fn=lambda path: (8000, next(decoded_signals)),
            target_lufs=-10.0,
            ceil_db=-1.0,
            preset_name="balanced",
            delivery_profile_name="streaming_lossy",
            true_peak_oversample_factor=1,
        )
    )

    assert final_path == "track.mp3"
    assert len(saved_signals) == 2
    assert np.max(np.abs(saved_signals[1])) < np.max(np.abs(saved_signals[0]))
    assert np.max(np.abs(final_signal)) == pytest.approx(
        np.max(np.abs(saved_signals[-1]))
    )
    assert verification.report.decoded_metrics is not None


def test_save_verified_audio_preserves_signal_without_remastering(
    monkeypatch: pytest.MonkeyPatch,
):
    saved_signals: list[np.ndarray] = []
    source = np.array([0.12, -0.24, 0.31, -0.28], dtype=np.float32)

    monkeypatch.setattr(
        DELIVERY_MODULE,
        "apply_lufs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("apply_lufs should not run during export")
        ),
        raising=False,
    )

    final_path, final_signal, verification = DELIVERY_MODULE.save_verified_audio(
        destination_path="track.wav",
        audio_signal=source,
        sample_rate=8000,
        input_signal=np.zeros_like(source),
        save_audio_fn=lambda **kwargs: saved_signals.append(
            np.array(kwargs["audio_signal"], copy=True)
        )
        or kwargs["destination_path"],
        read_audio_fn=lambda path: (8000, np.array(source, copy=True)),
        target_lufs=-10.0,
        ceil_db=-0.1,
        preset_name="balanced",
        delivery_profile_name="lossless",
        true_peak_oversample_factor=1,
    )

    assert final_path == "track.wav"
    assert len(saved_signals) == 1
    assert np.allclose(saved_signals[0], source)
    assert np.allclose(final_signal, source)
    assert verification.report.output_metrics is not None


def test_verify_delivery_export_attaches_stage_metrics_and_contract_assessments():
    profile = DELIVERY_MODULE.resolve_delivery_profile(
        "streaming_lossy", "track.mp3"
    )
    contract = CONTRACT_MODULE.resolve_mastering_contract(
        "balanced",
        target_lufs=-10.0,
        ceil_db=-1.0,
        max_short_term_lufs=-8.5,
        max_momentary_lufs=-7.0,
        max_stereo_width_ratio=0.4,
        min_low_end_mono_ratio=0.8,
    )
    input_signal = np.zeros((2, 64), dtype=np.float32)
    output_signal = np.vstack(
        [
            np.full(64, 0.5, dtype=np.float32),
            np.full(64, -0.5, dtype=np.float32),
        ]
    )

    result = DELIVERY_MODULE.verify_delivery_export(
        input_signal,
        output_signal,
        8000,
        post_eq_signal=output_signal * 0.8,
        post_spatial_signal=output_signal * 0.83,
        post_limiter_signal=output_signal * 0.9,
        post_character_signal=output_signal * 0.85,
        post_peak_catch_signal=output_signal * 0.82,
        post_delivery_trim_signal=output_signal * 0.81,
        post_clamp_signal=output_signal * 0.8,
        output_path="track.mp3",
        profile=profile,
        read_audio_fn=lambda path: (8000, output_signal),
        target_lufs=-10.0,
        ceil_db=-1.0,
        preset_name="balanced",
        contract=contract,
        character_stage_decision=types.SimpleNamespace(
            applied=True,
            reverted=False,
            reasons=(),
            input_integrated_lufs=-10.5,
            output_integrated_lufs=-10.0,
            input_true_peak_dbfs=-0.4,
            output_true_peak_dbfs=-0.2,
        ),
        peak_catch_events=(
            types.SimpleNamespace(
                attempt_index=1,
                drive_db=0.0,
                ceil_db=-1.2,
                soft_clip_ratio=0.18,
                peak_over_db=0.22,
                before_integrated_lufs=-10.0,
                after_integrated_lufs=-10.02,
                before_true_peak_dbfs=0.05,
                after_true_peak_dbfs=-0.1,
            ),
        ),
        resolved_true_peak_target_dbfs=-1.0,
        stereo_motion_activity=0.22,
        stereo_motion_correlation_guard=0.9,
        delivery_trim_attenuation_db=0.2,
        delivery_trim_input_true_peak_dbfs=-0.8,
        delivery_trim_target_dbfs=-1.0,
        delivery_trim_output_true_peak_dbfs=-1.02,
        post_clamp_true_peak_dbfs=-0.99,
        post_clamp_true_peak_delta_db=0.03,
        headroom_recovery_gain_db=0.3,
        headroom_recovery_input_true_peak_dbfs=-1.3,
        headroom_recovery_output_true_peak_dbfs=-1.0,
        headroom_recovery_failure_reasons=("linear_recovery_stalled",),
        headroom_recovery_mode="guarded",
        headroom_recovery_integrated_gap_db=0.7,
        headroom_recovery_transient_density=0.16,
        headroom_recovery_closed_margin_db=0.3,
        headroom_recovery_unused_margin_db=0.0,
        true_peak_oversample_factor=1,
    )

    assert result.report.post_eq_metrics is not None
    assert result.report.post_spatial_metrics is not None
    assert result.report.post_limiter_metrics is not None
    assert result.report.post_character_metrics is not None
    assert result.report.post_peak_catch_metrics is not None
    assert result.report.post_delivery_trim_metrics is not None
    assert result.report.post_clamp_metrics is not None
    assert result.report.character_stage_decision is not None
    assert len(result.report.peak_catch_events) == 1
    assert result.report.delivery_trim_attenuation_db == pytest.approx(0.2)
    assert result.report.delivery_trim_output_true_peak_dbfs == pytest.approx(
        -1.02
    )
    assert result.report.post_clamp_true_peak_dbfs == pytest.approx(-0.99)
    assert result.report.post_clamp_true_peak_delta_db == pytest.approx(0.03)
    assert result.report.headroom_recovery_gain_db == pytest.approx(0.3)
    assert result.report.headroom_recovery_failure_reasons == (
        "linear_recovery_stalled",
    )
    assert result.report.headroom_recovery_mode == "guarded"
    assert result.report.stereo_motion_activity == pytest.approx(0.22)
    assert result.report.output_contract_assessment is not None
    assert result.report.decoded_contract_assessment is not None
    assert any("contract" in issue for issue in result.issues)
