import importlib.util
import sys
import threading
import types
from pathlib import Path

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = ROOT / "src" / "definers" / "audio"


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

    config_module = _load_module(
        f"{package_name}.config", AUDIO_ROOT / "config.py"
    )
    mastering_stems_module = _load_module(
        f"{package_name}.mastering_stems",
        AUDIO_ROOT / "mastering_stems.py",
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
    assert bass.overrides["stereo_width"] < vocals.overrides["stereo_width"]
    assert drums.overrides["low_end_mono_tightening"] == "firm"
    assert vocals.overrides["exciter_mix"] > bass.overrides["exciter_mix"]
    assert bass.overrides["exciter_mix"] > 0.5
    assert drums.overrides["target_lufs"] < base.target_lufs


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


def test_apply_stem_tone_enrichment_adds_default_octave_layers_to_vocals():
    base = CONFIG_MODULE.SmartMasteringConfig.balanced()
    calls: list[float] = []
    signal = np.full((2, 192), 0.2, dtype=np.float32)

    enriched = MASTERING_STEMS_MODULE._apply_stem_tone_enrichment(
        signal,
        8000,
        "vocals",
        base,
        pitch_shift_fn=lambda channel, sample_rate, semitones: (
            calls.append(float(semitones))
            or np.array(channel, copy=True) * (1.0 + float(semitones) / 48.0)
        ),
    )

    assert set(calls) == {-12.0, -0.18, 0.18, 12.0}
    assert len(calls) == 8
    assert enriched.shape == signal.shape
    assert float(np.max(np.abs(enriched))) > float(np.max(np.abs(signal)))


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
    assert tuple(semitones for semitones, _mix in vocal_layers) == (
        -12.0,
        -0.18,
        0.18,
        12.0,
    )
    assert tuple(semitones for semitones, _mix in guitar_layers) == (
        -12.0,
        -0.12,
        0.12,
        12.0,
    )
    assert tuple(semitones for semitones, _mix in piano_layers) == (
        -12.0,
        -0.08,
        0.08,
        12.0,
    )
    assert tuple(semitones for semitones, _mix in other_layers) == (
        -12.0,
        -0.1,
        0.1,
        12.0,
    )


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
