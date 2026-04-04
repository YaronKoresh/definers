import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
CHARACTER_MODULE = _load_module(
    "_test_audio_mastering_character",
    ROOT / "src" / "definers" / "audio" / "mastering_character.py",
)


def test_resolve_limiter_recovery_settings_changes_timing_by_style():
    tight = CHARACTER_MODULE.resolve_limiter_recovery_settings(
        SimpleNamespace(limiter_recovery_style="tight"),
        attack_ms=2.0,
        release_ms_min=30.0,
        release_ms_max=130.0,
        window_ms=4.0,
    )
    glue = CHARACTER_MODULE.resolve_limiter_recovery_settings(
        SimpleNamespace(limiter_recovery_style="glue"),
        attack_ms=2.0,
        release_ms_min=30.0,
        release_ms_max=130.0,
        window_ms=4.0,
    )

    assert tight.attack_ms < glue.attack_ms
    assert tight.release_ms_max < glue.release_ms_max
    assert tight.window_ms < glue.window_ms


def test_apply_low_end_mono_tightening_reduces_channel_difference():
    mastering = SimpleNamespace(
        low_end_mono_tightening="firm",
        low_end_mono_tightening_amount=1.0,
        contract_low_end_mono_cutoff_hz=120.0,
    )
    source = np.array(
        [[1.0, -1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, -1.0, 1.0, -1.0]],
        dtype=np.float32,
    )

    tightened = CHARACTER_MODULE.apply_low_end_mono_tightening(
        mastering,
        source,
        sample_rate=8000,
    )

    assert np.max(np.abs(tightened[0] - tightened[1])) < np.max(
        np.abs(source[0] - source[1])
    )


def test_apply_micro_dynamics_finish_changes_sustain_more_than_transient():
    mastering = SimpleNamespace(
        micro_dynamics_strength=0.2,
        micro_dynamics_fast_window_ms=5.0,
        micro_dynamics_slow_window_ms=25.0,
        micro_dynamics_transient_bias=0.8,
    )
    source = np.array([0.0, 1.0, 0.7, 0.7, 0.7, 0.7, 0.0], dtype=np.float32)

    finished = CHARACTER_MODULE.apply_micro_dynamics_finish(
        mastering,
        source,
        sample_rate=1000,
    )

    assert finished.shape == source.shape
    assert not np.allclose(finished, source)
    assert abs(finished[1] - source[1]) < abs(finished[3] - source[3])
