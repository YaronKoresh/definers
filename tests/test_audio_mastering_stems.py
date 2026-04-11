import importlib.util
import sys
import threading
import types
from pathlib import Path

import numpy as np

from definers.system.download_activity import (
    bind_download_activity_scope,
    clear_download_activity_scope,
    create_download_activity_scope,
    get_download_activity_snapshot,
    report_download_activity,
)


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


def _load_mastering_stems_module(package_name: str):
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

    config_module = _load_module(
        f"{package_name}.config", AUDIO_ROOT / "config.py"
    )
    mastering_stems_module = _load_module(
        f"{mastering_package_name}.stems",
        MASTERING_ROOT / "stems.py",
    )
    return config_module, mastering_stems_module


CONFIG_MODULE, MASTERING_STEMS_MODULE = _load_mastering_stems_module(
    "_test_audio_mastering_stems_pkg.audio"
)


def test_resolve_stem_mastering_plan_scales_roles_differently():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()

    drums = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("drums", base)
    bass = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("bass", base)
    vocals = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("vocals", base)

    assert drums.mix_gain_db > bass.mix_gain_db
    assert bass.mix_gain_db > vocals.mix_gain_db
    assert drums.mix_gain_db - vocals.mix_gain_db > 2.3
    assert bass.mix_gain_db >= 0.9
    assert bass.overrides["stereo_width"] < vocals.overrides["stereo_width"]
    assert (
        bass.overrides["bass_boost_db_per_oct"]
        > base.bass_boost_db_per_oct + 0.5
    )
    assert drums.overrides["low_end_mono_tightening"] == "firm"
    assert vocals.overrides["exciter_mix"] > bass.overrides["exciter_mix"]
    assert bass.overrides["exciter_mix"] > 0.5
    assert drums.overrides["target_lufs"] < base.target_lufs
    assert drums.overrides["stem_noise_gate_enabled"] is True
    assert (
        drums.overrides["stem_cleanup_strength"]
        < vocals.overrides["stem_cleanup_strength"]
    )
    assert (
        drums.overrides["stem_noise_gate_strength"]
        < vocals.overrides["stem_noise_gate_strength"]
    )


def test_resolve_stem_mastering_plan_clips_exciter_mix_and_staggers_drive():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()

    drums = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("drums", base)
    vocals = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("vocals", base)
    piano = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("piano", base)

    assert drums.overrides["exciter_mix"] <= 1.0
    assert vocals.overrides["exciter_mix"] <= 1.0
    assert piano.overrides["exciter_mix"] <= 1.0
    assert (
        vocals.overrides["exciter_max_drive"]
        < drums.overrides["exciter_max_drive"]
    )
    assert "exciter_max_drive" not in piano.overrides


def test_mix_stem_layers_aligns_sample_rates_and_applies_headroom():
    stems = {
        "drums": (
            4000,
            np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32),
        ),
        "vocals": (
            8000,
            np.array(
                [[0.7, 0.7, 0.7, 0.7], [0.7, 0.7, 0.7, 0.7]], dtype=np.float32
            ),
        ),
    }

    mixed_sr, mixed_signal = MASTERING_STEMS_MODULE.mix_stem_layers(
        stems,
        mix_headroom_db=6.0,
        resample_fn=lambda y, sr, target_sr: np.repeat(
            y, target_sr // sr, axis=-1
        ),
    )

    assert mixed_sr == 8000
    assert mixed_signal.shape == (2, 4)
    assert (
        float(np.max(np.abs(mixed_signal)))
        <= float(10.0 ** (-6.0 / 20.0)) + 1e-6
    )


def test_mix_stem_layers_synchronizes_shapes_and_rescales_hot_sum():
    stems = {
        "drums": (
            8000,
            np.array([2.0, 0.2, 0.1], dtype=np.float32),
        ),
        "vocals": (
            8000,
            np.array([[2.0, 0.2], [2.0, 0.2]], dtype=np.float32),
        ),
    }

    mixed_sr, mixed_signal = MASTERING_STEMS_MODULE.mix_stem_layers(
        stems,
        mix_headroom_db=0.0,
    )

    float(10.0 ** (-6.0 / 20.0))

    assert mixed_sr == 8000
    assert mixed_signal.dtype == np.float32
    assert mixed_signal.shape == (2, 3)
    assert np.allclose(mixed_signal[:, 0], 1.0)
    assert np.allclose(mixed_signal[:, 1], 0.1, atol=1e-6)
    assert np.allclose(mixed_signal[:, 2], 0.025, atol=1e-6)


def test_process_stem_layers_processes_each_stem_and_cleans_temp_dir(
    tmp_path: Path,
):
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    processed_calls: list[tuple[int, dict[str, object]]] = []
    deleted_paths: list[str] = []
    saved_stems: list[tuple[str, int, tuple[int, ...]]] = []
    stems_dir = tmp_path / "saved-stems"

    mixed_sr, mixed_signal = MASTERING_STEMS_MODULE.process_stem_layers(
        "song.wav",
        base_config=base,
        base_mastering_kwargs={"preset": "balanced"},
        process_stem_fn=lambda signal, sample_rate, mastering_kwargs: (
            processed_calls.append((sample_rate, dict(mastering_kwargs)))
            or (sample_rate, np.array(signal, copy=True) * 0.5)
        ),
        separate_stems_fn=lambda audio_path, model_name, shifts, quality_flags=(): (
            {
                "drums": "drums.wav",
                "vocals": "vocals.wav",
            },
            "temp-demucs-dir",
        ),
        read_audio_fn=lambda path: (
            8000,
            np.array(
                [[0.2, 0.2, 0.2], [0.2, 0.2, 0.2]],
                dtype=np.float32,
            )
            if path == "drums.wav"
            else np.array(
                [[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]],
                dtype=np.float32,
            ),
        ),
        delete_fn=lambda path: deleted_paths.append(path),
        mix_headroom_db=3.0,
        mastered_stems_output_dir=str(stems_dir),
        save_audio_fn=lambda **kwargs: (
            saved_stems.append(
                (
                    kwargs["destination_path"],
                    kwargs["sample_rate"],
                    tuple(np.asarray(kwargs["audio_signal"]).shape),
                )
            )
            or kwargs["destination_path"]
        ),
    )

    assert mixed_sr == 8000
    assert mixed_signal.shape == (2, 3)
    assert len(processed_calls) == 2
    assert {call[1]["preset"] for call in processed_calls} == {"balanced"}
    assert {call[1]["stem_role"] for call in processed_calls} == {
        "drums",
        "vocals",
    }
    assert saved_stems == [
        (str(stems_dir / "drums_mastered.wav"), 8000, (2, 3)),
        (str(stems_dir / "vocals_mastered.wav"), 8000, (2, 3)),
    ]
    assert deleted_paths == ["temp-demucs-dir"]


def test_process_stem_layers_saves_pre_mix_mastered_stems(tmp_path: Path):
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    stems_dir = tmp_path / "saved-stems"
    saved_audio: dict[str, np.ndarray] = {}
    drums_signal = np.full((2, 64), 0.2, dtype=np.float32)
    vocals_signal = np.full((2, 64), 0.1, dtype=np.float32)

    MASTERING_STEMS_MODULE.process_stem_layers(
        "song.wav",
        base_config=base,
        base_mastering_kwargs={"preset": "balanced"},
        process_stem_fn=lambda signal, sample_rate, mastering_kwargs: (
            sample_rate,
            np.array(signal, copy=True),
        ),
        separate_stems_fn=lambda audio_path, model_name, shifts, quality_flags=(): (
            {
                "drums": "drums.wav",
                "vocals": "vocals.wav",
            },
            "temp-demucs-dir",
        ),
        read_audio_fn=lambda path: (
            8000,
            np.array(drums_signal, copy=True)
            if path == "drums.wav"
            else np.array(vocals_signal, copy=True),
        ),
        delete_fn=lambda path: None,
        mix_headroom_db=6.0,
        mastered_stems_output_dir=str(stems_dir),
        save_audio_fn=lambda **kwargs: (
            saved_audio.__setitem__(
                Path(kwargs["destination_path"]).name,
                np.asarray(kwargs["audio_signal"], dtype=np.float32),
            )
            or kwargs["destination_path"]
        ),
    )

    drums_plan = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan(
        "drums", base
    )
    vocals_plan = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan(
        "vocals", base
    )
    finished_drums = MASTERING_STEMS_MODULE._apply_stem_role_finish(
        drums_signal,
        8000,
        "drums",
        base,
        stem_overrides=drums_plan.overrides,
    )
    finished_vocals = MASTERING_STEMS_MODULE._apply_stem_role_finish(
        vocals_signal,
        8000,
        "vocals",
        base,
        pitch_shift_fn=lambda channel, sample_rate, semitones: np.array(
            channel,
            copy=True,
        ),
        stem_overrides=vocals_plan.overrides,
    )
    expected_drums, _ = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        finished_drums,
        "drums",
        drums_plan.mix_gain_db,
    )
    expected_vocals, _ = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        finished_vocals,
        "vocals",
        vocals_plan.mix_gain_db,
    )
    _, aligned_layers, _ = MASTERING_STEMS_MODULE._prepare_mixed_stem_layers(
        {
            "drums": (8000, expected_drums),
            "vocals": (8000, expected_vocals),
        },
        mix_headroom_db=6.0,
    )

    assert np.allclose(saved_audio["drums_mastered.wav"], expected_drums)
    assert np.allclose(
        saved_audio["vocals_mastered.wav"],
        expected_vocals,
    )
    assert not np.allclose(
        saved_audio["drums_mastered.wav"],
        aligned_layers["drums"][1],
    )


def test_process_stem_layers_preserves_output_order_when_threads_finish_out_of_order(
    tmp_path: Path,
):
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    vocals_started = threading.Event()
    saved_stems: list[str] = []
    stems_dir = tmp_path / "saved-stems"

    def process_stem(signal, sample_rate, mastering_kwargs):
        stem_role = mastering_kwargs["stem_role"]
        if stem_role == "drums":
            assert vocals_started.wait(timeout=1.0)
        else:
            vocals_started.set()
        return sample_rate, np.array(signal, copy=True)

    MASTERING_STEMS_MODULE.process_stem_layers(
        "song.wav",
        base_config=base,
        base_mastering_kwargs={"preset": "balanced"},
        process_stem_fn=process_stem,
        separate_stems_fn=lambda audio_path, model_name, shifts, quality_flags=(): (
            {
                "drums": "drums.wav",
                "vocals": "vocals.wav",
            },
            "temp-demucs-dir",
        ),
        read_audio_fn=lambda path: (
            8000,
            np.full(
                (2, 32), 0.2 if path == "drums.wav" else 0.1, dtype=np.float32
            ),
        ),
        delete_fn=lambda path: None,
        mix_headroom_db=3.0,
        mastered_stems_output_dir=str(stems_dir),
        save_audio_fn=lambda **kwargs: saved_stems.append(
            kwargs["destination_path"]
        ),
    )

    assert saved_stems == [
        str(stems_dir / "drums_mastered.wav"),
        str(stems_dir / "vocals_mastered.wav"),
    ]


def test_apply_stem_mix_balance_prioritizes_drums_over_vocals():
    drums, drums_gain_db = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        np.full((2, 256), 0.18, dtype=np.float32),
        "drums",
        0.95,
    )
    vocals, vocals_gain_db = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        np.full((2, 256), 0.18, dtype=np.float32),
        "vocals",
        -0.12,
    )

    assert drums_gain_db > vocals_gain_db
    assert float(np.max(np.abs(drums))) > float(np.max(np.abs(vocals)))


def test_process_stem_layers_forwards_quality_flags_to_separator():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    separator_calls: list[tuple[str, str, int, tuple[str, ...]]] = []

    MASTERING_STEMS_MODULE.process_stem_layers(
        "song.wav",
        base_config=base,
        base_mastering_kwargs={"preset": "balanced"},
        process_stem_fn=lambda signal, sample_rate, mastering_kwargs: (
            sample_rate,
            np.array(signal, copy=True),
        ),
        separate_stems_fn=lambda audio_path, model_name, shifts, quality_flags=(): (
            separator_calls.append(
                (audio_path, model_name, shifts, tuple(quality_flags))
            )
            or {
                "vocals": "vocals.wav",
            },
            "temp-demucs-dir",
        ),
        read_audio_fn=lambda path: (
            8000,
            np.full((2, 16), 0.1, dtype=np.float32),
        ),
        delete_fn=lambda path: None,
        quality_flags=("Low-Quality", "Old-Recording"),
        save_mastered_stems=False,
    )

    assert separator_calls == [
        ("song.wav", "mastering", 2, ("Low-Quality", "Old-Recording"))
    ]


def test_process_stem_layers_rebinds_activity_scope_for_worker_threads(
    monkeypatch,
):
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    scope_id = create_download_activity_scope()

    def process_stem(signal, sample_rate, mastering_kwargs):
        report_download_activity(
            f"Stem {mastering_kwargs['stem_role']}",
            phase="step",
        )
        return sample_rate, np.array(signal, copy=True)

    monkeypatch.setattr(
        MASTERING_STEMS_MODULE,
        "_resolve_parallel_stem_workers",
        lambda stem_count: 2,
    )

    with bind_download_activity_scope(scope_id):
        MASTERING_STEMS_MODULE.process_stem_layers(
            "song.wav",
            base_config=base,
            base_mastering_kwargs={"preset": "balanced"},
            process_stem_fn=process_stem,
            separate_stems_fn=lambda audio_path, model_name, shifts, quality_flags=(): (
                {
                    "drums": "drums.wav",
                    "vocals": "vocals.wav",
                },
                "temp-demucs-dir",
            ),
            read_audio_fn=lambda path: (
                8000,
                np.full(
                    (2, 32),
                    0.2 if path == "drums.wav" else 0.1,
                    dtype=np.float32,
                ),
            ),
            delete_fn=lambda path: None,
            save_mastered_stems=False,
        )

    snapshot = get_download_activity_snapshot(scope_id)
    clear_download_activity_scope(scope_id)

    assert snapshot is not None
    assert snapshot.item_label in {"Stem drums", "Stem vocals"}


def test_apply_stem_mix_balance_gives_drums_a_clear_level_lead():
    drums, drums_gain_db = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        np.full((2, 256), 0.18, dtype=np.float32),
        "drums",
        1.35,
    )
    vocals, vocals_gain_db = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        np.full((2, 256), 0.18, dtype=np.float32),
        "vocals",
        -0.3,
    )

    assert drums_gain_db - vocals_gain_db > 3.0
    assert float(np.max(np.abs(drums))) > float(np.max(np.abs(vocals))) * 1.4


def test_apply_stem_mix_balance_keeps_bass_substantial_against_vocals():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    bass_plan = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("bass", base)
    vocals_plan = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan(
        "vocals", base
    )
    bass, bass_gain_db = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        np.full((2, 256), 0.18, dtype=np.float32),
        "bass",
        bass_plan.mix_gain_db,
    )
    vocals, vocals_gain_db = MASTERING_STEMS_MODULE._apply_stem_mix_balance(
        np.full((2, 256), 0.18, dtype=np.float32),
        "vocals",
        vocals_plan.mix_gain_db,
    )

    assert bass_gain_db - vocals_gain_db > 2.8
    assert float(np.max(np.abs(bass))) > float(np.max(np.abs(vocals))) * 1.25


def test_apply_stem_dynamics_controls_peaks_without_hollowing_bass():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    bass_plan = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan("bass", base)
    signal = np.full((2, 2048), 0.11, dtype=np.float32)
    signal[:, :24] = 0.42
    signal[:, 256:280] = -0.36

    finished = MASTERING_STEMS_MODULE._apply_stem_dynamics(
        signal,
        8000,
        "bass",
        base,
        stem_overrides=bass_plan.overrides,
    )

    input_peak = float(np.max(np.abs(signal)))
    output_peak = float(np.max(np.abs(finished)))
    input_body = float(np.mean(np.abs(signal[:, 512:1536])))
    output_body = float(np.mean(np.abs(finished[:, 512:1536])))

    assert output_peak < input_peak
    assert output_body > input_body * 0.95


def test_apply_stem_stereo_width_finish_widens_narrow_vocals():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    vocals_plan = MASTERING_STEMS_MODULE.resolve_stem_mastering_plan(
        "vocals", base
    )
    time = np.linspace(
        0.0, 12.0 * np.pi, 2048, endpoint=False, dtype=np.float32
    )
    mono = 0.18 * np.sin(time) + 0.05 * np.sin(time * 3.7)
    signal = np.stack([mono, mono], axis=0).astype(np.float32)

    widened = MASTERING_STEMS_MODULE._apply_stem_stereo_width_finish(
        signal,
        8000,
        "vocals",
        base,
        stem_overrides=vocals_plan.overrides,
    )

    input_width = float(
        np.sqrt(np.mean(np.square(0.5 * (signal[0] - signal[1]))))
    )
    output_width = float(
        np.sqrt(np.mean(np.square(0.5 * (widened[0] - widened[1]))))
    )

    assert input_width == 0.0
    assert output_width > 0.015


def test_apply_stem_tone_enrichment_focuses_on_bass_by_default():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    calls: list[float] = []
    signal = np.full((2, 192), 0.2, dtype=np.float32)

    enriched = MASTERING_STEMS_MODULE._apply_stem_tone_enrichment(
        signal,
        8000,
        "bass",
        base,
        pitch_shift_fn=lambda channel, sample_rate, semitones: (
            calls.append(float(semitones))
            or np.array(channel, copy=True) * (1.0 + float(semitones) / 48.0)
        ),
    )

    assert set(calls) == {-0.07, 12.0}
    assert len(calls) == 4
    assert enriched.shape == signal.shape
    assert float(np.max(np.abs(enriched))) > float(np.max(np.abs(signal)))


def test_constrain_stem_peak_growth_limits_hot_stage_output():
    reference = np.full((2, 256), 0.4, dtype=np.float32)
    hot_signal = np.full((2, 256), 2.0, dtype=np.float32)

    constrained = MASTERING_STEMS_MODULE._constrain_stem_peak_growth(
        hot_signal,
        reference_signal=reference,
    )

    assert float(np.max(np.abs(constrained))) <= 0.4 * 1.08 + 1e-6
    assert float(np.max(np.abs(constrained))) <= 0.98 + 1e-6


def test_resolve_stem_tone_layers_varies_by_role():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()

    bass_layers = MASTERING_STEMS_MODULE._resolve_stem_tone_layers("bass", base)
    vocal_layers = MASTERING_STEMS_MODULE._resolve_stem_tone_layers(
        "vocals", base
    )
    guitar_layers = MASTERING_STEMS_MODULE._resolve_stem_tone_layers(
        "guitar", base
    )
    piano_layers = MASTERING_STEMS_MODULE._resolve_stem_tone_layers(
        "piano", base
    )
    other_layers = MASTERING_STEMS_MODULE._resolve_stem_tone_layers(
        "other", base
    )

    assert tuple(semitones for semitones, _mix in bass_layers) == (-0.07, 12.0)
    assert vocal_layers == ()
    assert guitar_layers == ()
    assert piano_layers == ()
    assert other_layers == ()


def test_apply_stem_tone_enrichment_skips_drums_by_default():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    signal = np.full((2, 8), 0.2, dtype=np.float32)

    enriched = MASTERING_STEMS_MODULE._apply_stem_tone_enrichment(
        signal,
        8000,
        "drums",
        base,
        pitch_shift_fn=lambda channel, sample_rate, semitones: np.zeros_like(
            channel
        ),
    )

    assert np.array_equal(enriched, signal)


def test_apply_stem_glue_reverb_only_colors_vocals_and_other():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    signal = np.zeros((2, 1024), dtype=np.float32)
    signal[:, :96] = 0.2

    vocals = MASTERING_STEMS_MODULE._apply_stem_glue_reverb(
        signal,
        8000,
        "vocals",
        base,
    )
    other = MASTERING_STEMS_MODULE._apply_stem_glue_reverb(
        signal,
        8000,
        "other",
        base,
    )
    drums = MASTERING_STEMS_MODULE._apply_stem_glue_reverb(
        signal,
        8000,
        "drums",
        base,
    )

    assert vocals.shape == signal.shape
    assert other.shape == signal.shape
    assert not np.allclose(vocals, signal)
    assert not np.allclose(other, signal)
    assert np.allclose(drums, signal)


def test_apply_stem_glue_reverb_extends_sparse_tail_when_amount_is_higher():
    low_glue = CONFIG_MODULE.SmartMasteringConfig.balanced()
    high_glue = CONFIG_MODULE.SmartMasteringConfig.balanced()
    low_glue.stem_glue_reverb_amount = 0.4
    high_glue.stem_glue_reverb_amount = 1.5
    signal = np.zeros((2, 4096), dtype=np.float32)
    signal[:, :96] = 0.2

    low_result = MASTERING_STEMS_MODULE._apply_stem_glue_reverb(
        signal,
        8000,
        "vocals",
        low_glue,
    )
    high_result = MASTERING_STEMS_MODULE._apply_stem_glue_reverb(
        signal,
        8000,
        "vocals",
        high_glue,
    )

    low_tail_energy = float(np.sum(np.abs(low_result[:, 1800:3400])))
    high_tail_energy = float(np.sum(np.abs(high_result[:, 1800:3400])))

    assert low_tail_energy > 0.0
    assert high_tail_energy > low_tail_energy * 1.9
    assert float(np.max(np.abs(high_result))) <= 0.98 + 1e-6


def test_apply_drum_edge_finish_adds_controlled_transient_bite():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    signal = np.zeros((2, 2048), dtype=np.float32)
    signal[:, ::128] = 0.3
    signal[:, 1::128] = -0.18

    finished = MASTERING_STEMS_MODULE._apply_drum_edge_finish(
        signal,
        8000,
        base,
    )

    assert finished.shape == signal.shape
    assert not np.allclose(finished, signal)
    assert float(np.max(np.abs(finished))) <= 0.98 + 1e-6
