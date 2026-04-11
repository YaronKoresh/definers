import importlib
import json
import types
from pathlib import Path

import numpy as np

from definers.ui.apps import audio_app_services as services


def _patch_audio_symbol(
    monkeypatch,
    symbol_name: str,
    module_name: str,
    replacement,
):
    audio_facade = importlib.import_module("definers.audio")
    source_module = importlib.import_module(module_name)
    monkeypatch.setattr(audio_facade, symbol_name, replacement, raising=False)
    monkeypatch.setattr(source_module, symbol_name, replacement, raising=False)


def test_run_mastering_tool_returns_report_and_stems(monkeypatch):
    captured = {}
    fake_report = types.SimpleNamespace(
        preset_name="vocal",
        delivery_profile_name="lossless",
        input_metrics=types.SimpleNamespace(integrated_lufs=-19.4),
        output_metrics=types.SimpleNamespace(
            integrated_lufs=-10.2,
            true_peak_dbfs=-0.7,
            crest_factor_db=9.8,
            stereo_width_ratio=0.31,
        ),
        headroom_recovery_mode="adaptive",
        headroom_recovery_gain_db=0.6,
        post_spatial_stereo_motion=0.08,
        post_clamp_true_peak_dbfs=-0.75,
        delivery_issues=(),
        to_dict=lambda: {"preset_name": "vocal"},
    )

    def fake_master(audio_path, output_path=None, **kwargs):
        captured.update(kwargs)
        resolved_output_path = Path(str(output_path))
        resolved_output_path.write_text("audio", encoding="utf-8")
        report_path = Path(str(kwargs["report_path"]))
        report_path.write_text("{}", encoding="utf-8")
        stem_dir = (
            resolved_output_path.parent / f"{resolved_output_path.stem}_stems"
        )
        stem_dir.mkdir(parents=True, exist_ok=True)
        (stem_dir / "vocals_mastered.wav").write_text(
            "vocals", encoding="utf-8"
        )
        (stem_dir / "drums_mastered.wav").write_text("drums", encoding="utf-8")
        return str(resolved_output_path), fake_report

    _patch_audio_symbol(
        monkeypatch,
        "master",
        "definers.audio.mastering",
        fake_master,
    )

    mastered_path, report_path, summary_text, stem_files = (
        services.run_mastering_tool(
            "song.wav",
            "WAV",
            "vocal",
            0.55,
            0.6,
            0.65,
            True,
            "mastering",
            3,
            6.0,
            True,
        )
    )

    assert Path(mastered_path).suffix == ".wav"
    assert report_path is not None
    assert Path(report_path).exists()
    assert len(stem_files) == 2
    assert captured["preset"] == "vocal"
    assert "bass" not in captured
    assert "volume" not in captured
    assert "effects" not in captured
    assert captured["stem_model_name"] == "mastering"
    assert Path(report_path).suffix == ".md"
    assert "**Verdict:**" in summary_text
    assert "**Control Mode:** Vocal Focus" in summary_text
    assert "**Preset:** vocal" in summary_text
    assert "**Mastered Stems:** 2 files" in summary_text


def test_run_mastering_tool_collects_stems_for_gui_preview(monkeypatch):
    captured = {}
    fake_report = types.SimpleNamespace(
        preset_name="balanced",
        delivery_profile_name="lossless",
        input_metrics=None,
        output_metrics=None,
        headroom_recovery_mode="adaptive",
        headroom_recovery_gain_db=0.0,
        post_spatial_stereo_motion=0.0,
        post_clamp_true_peak_dbfs=-1.0,
        delivery_issues=(),
        to_dict=lambda: {"preset_name": "balanced"},
    )

    def fake_master(audio_path, output_path=None, **kwargs):
        captured.update(kwargs)
        resolved_output_path = Path(str(output_path))
        resolved_output_path.write_text("audio", encoding="utf-8")
        report_path = Path(str(kwargs["report_path"]))
        report_path.write_text("{}", encoding="utf-8")
        stem_dir = (
            resolved_output_path.parent / f"{resolved_output_path.stem}_stems"
        )
        stem_dir.mkdir(parents=True, exist_ok=True)
        (stem_dir / "bass_mastered.wav").write_text("bass", encoding="utf-8")
        return str(resolved_output_path), fake_report

    _patch_audio_symbol(
        monkeypatch,
        "master",
        "definers.audio.mastering",
        fake_master,
    )

    _mastered_path, _report_path, _summary_text, stem_files = (
        services.run_mastering_tool(
            "song.wav",
            "WAV",
            "Balanced",
            0.5,
            0.5,
            0.5,
            True,
            "mastering",
            2,
            6.0,
            False,
        )
    )

    assert captured["save_mastered_stems"] is True
    assert stem_files == [
        str(Path(stem_files[0]).with_name("bass_mastered.wav"))
    ]


def test_run_mastering_tool_writes_outputs_under_managed_root(
    monkeypatch, tmp_path
):
    fake_report = types.SimpleNamespace(
        preset_name="balanced",
        delivery_profile_name="lossless",
        input_metrics=None,
        output_metrics=None,
        headroom_recovery_mode="adaptive",
        headroom_recovery_gain_db=0.0,
        post_spatial_stereo_motion=0.0,
        post_clamp_true_peak_dbfs=-1.0,
        delivery_issues=(),
        to_dict=lambda: {"preset_name": "balanced"},
    )

    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    def fake_master(audio_path, output_path=None, **kwargs):
        resolved_output_path = Path(str(output_path))
        assert resolved_output_path.is_relative_to(tmp_path)
        resolved_output_path.write_text("audio", encoding="utf-8")
        report_path = Path(str(kwargs["report_path"]))
        assert report_path.is_relative_to(tmp_path)
        report_path.write_text("{}", encoding="utf-8")
        stem_dir = (
            resolved_output_path.parent / f"{resolved_output_path.stem}_stems"
        )
        stem_dir.mkdir(parents=True, exist_ok=True)
        (stem_dir / "vocals_mastered.wav").write_text(
            "vocals", encoding="utf-8"
        )
        return str(resolved_output_path), fake_report

    _patch_audio_symbol(
        monkeypatch,
        "master",
        "definers.audio.mastering",
        fake_master,
    )

    mastered_path, report_path, _summary_text, stem_files = (
        services.run_mastering_tool(
            "song.wav",
            "WAV",
            "Balanced",
            0.5,
            0.5,
            0.5,
            True,
            "mastering",
            2,
            6.0,
            True,
        )
    )

    assert Path(mastered_path).is_relative_to(tmp_path / "audio")
    assert Path(report_path).is_relative_to(tmp_path / "audio" / "reports")
    assert Path(report_path).suffix == ".md"
    assert stem_files == [
        str(
            Path(mastered_path).parent
            / f"{Path(mastered_path).stem}_stems"
            / "vocals_mastered.wav"
        )
    ]


def test_run_stem_separation_tool_passes_managed_output_dir(
    monkeypatch, tmp_path
):
    captured = {}

    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    def fake_separate_stem_layers(
        audio_path,
        model_name="mastering",
        shifts=2,
        output_dir=None,
    ):
        captured.update(
            {
                "audio_path": audio_path,
                "model_name": model_name,
                "shifts": shifts,
                "output_dir": output_dir,
            }
        )
        resolved_output_dir = Path(str(output_dir))
        resolved_output_dir.mkdir(parents=True, exist_ok=True)
        return (
            {
                "vocals": str(resolved_output_dir / "vocals.wav"),
                "drums": str(resolved_output_dir / "drums.wav"),
            },
            str(resolved_output_dir),
        )

    _patch_audio_symbol(
        monkeypatch,
        "separate_stem_layers",
        "definers.audio.stems",
        fake_separate_stem_layers,
    )

    primary_output, output_files, _summary_text = (
        services.run_stem_separation_tool(
            "song.wav",
            "mastering_layers",
            "WAV",
            "htdemucs_ft.yaml",
            4,
        )
    )

    assert primary_output == str(Path(captured["output_dir"]) / "vocals.wav")
    assert output_files == [
        str(Path(captured["output_dir"]) / "vocals.wav"),
        str(Path(captured["output_dir"]) / "drums.wav"),
    ]
    assert Path(str(captured["output_dir"])).is_relative_to(
        tmp_path / "audio" / "stems"
    )


def test_resolve_mastering_stem_previews_maps_common_stems_and_extras():
    previews, extras = services.resolve_mastering_stem_previews(
        [
            "vocals_mastered.wav",
            "drums_mastered.wav",
            "guitar_room.wav",
            "fx_print.wav",
        ]
    )

    assert previews == {
        "vocals": "vocals_mastered.wav",
        "drums": "drums_mastered.wav",
        "bass": None,
        "other": None,
        "guitar": "guitar_room.wav",
        "piano": None,
    }
    assert extras == ["fx_print.wav"]


def test_get_mastering_profile_ui_state_locks_named_profiles():
    state = services.get_mastering_profile_ui_state("EDM", 0.2, 0.3, 0.4)

    assert state["controls_enabled"] is False
    assert state["bass"] == 1.0
    assert state["volume"] == 1.0
    assert state["effects"] == 1.0


def test_get_mastering_profile_ui_state_custom_keeps_manual_macros():
    state = services.get_mastering_profile_ui_state(
        "Custom Macro Blend",
        0.2,
        0.3,
        0.4,
    )

    assert state["controls_enabled"] is True
    assert state["bass"] == 0.2
    assert state["volume"] == 0.3
    assert state["effects"] == 0.4


def test_run_mastering_tool_custom_profile_sends_manual_macros(monkeypatch):
    captured = {}
    fake_report = types.SimpleNamespace(
        preset_name="balanced",
        delivery_profile_name="lossless",
        input_metrics=None,
        output_metrics=None,
        headroom_recovery_mode="adaptive",
        headroom_recovery_gain_db=0.0,
        post_spatial_stereo_motion=0.0,
        post_clamp_true_peak_dbfs=-1.0,
        delivery_issues=(),
        to_dict=lambda: {"preset_name": "balanced"},
    )

    def fake_master(audio_path, output_path=None, **kwargs):
        captured.update(kwargs)
        resolved_output_path = Path(str(output_path))
        resolved_output_path.write_text("audio", encoding="utf-8")
        report_path = Path(str(kwargs["report_path"]))
        report_path.write_text("{}", encoding="utf-8")
        return str(resolved_output_path), fake_report

    _patch_audio_symbol(
        monkeypatch,
        "master",
        "definers.audio.mastering",
        fake_master,
    )

    _, _, summary_text, _ = services.run_mastering_tool(
        "song.wav",
        "WAV",
        "Custom Macro Blend",
        0.25,
        0.35,
        0.45,
        False,
        "mastering",
        2,
        6.0,
        False,
    )

    assert captured["preset"] == "balanced"
    assert captured["bass"] == 0.25
    assert captured["volume"] == 0.35
    assert captured["effects"] == 0.45
    assert "**Control Mode:** Custom Macro Blend" in summary_text


def test_run_stem_separation_tool_supports_mastering_layers(monkeypatch):
    captured = {}

    def fake_separate_stem_layers(audio_path, model_name="mastering", shifts=2):
        captured.update(
            {
                "audio_path": audio_path,
                "model_name": model_name,
                "shifts": shifts,
            }
        )
        return (
            {
                "vocals": "vocals.wav",
                "drums": "drums.wav",
                "bass": "bass.wav",
                "other": "other.wav",
            },
            "temp-stems",
        )

    _patch_audio_symbol(
        monkeypatch,
        "separate_stem_layers",
        "definers.audio.stems",
        fake_separate_stem_layers,
    )

    primary_output, output_files, summary_text = (
        services.run_stem_separation_tool(
            "song.wav",
            "mastering_layers",
            "WAV",
            "htdemucs_6s",
            4,
        )
    )

    assert primary_output == "vocals.wav"
    assert output_files == [
        "vocals.wav",
        "drums.wav",
        "bass.wav",
        "other.wav",
    ]
    assert "vocals" in summary_text.lower()
    assert captured == {
        "audio_path": "song.wav",
        "model_name": "htdemucs_6s",
        "shifts": 4,
    }


def test_resolve_stem_model_name_supports_labels_and_custom_override():
    assert (
        services.resolve_stem_model_name("Demucs fine-tuned 4-stem")
        == "htdemucs_ft.yaml"
    )
    assert (
        services.resolve_stem_model_name(
            "Custom checkpoint override",
            "htdemucs_6s",
        )
        == "htdemucs_6s"
    )
    assert services.is_custom_stem_model_strategy("Custom checkpoint override")


def test_run_audio_analysis_tool_writes_json_summary(monkeypatch):
    _patch_audio_symbol(
        monkeypatch,
        "analyze_audio_features",
        "definers.audio.analysis",
        lambda path: "C major (120 bpm)",
    )
    _patch_audio_symbol(
        monkeypatch,
        "analyze_audio",
        "definers.audio.analysis",
        lambda audio_path, hop_length=1024, duration=None, offset=0.0: {
            "sr": 44100,
            "duration": 12.5,
            "hop_length": hop_length,
            "bpm": 120,
            "beat_frames": np.array([0, 10, 20]),
            "spectral_centroid": np.array([1000.0, 1200.0, 1400.0]),
            "rms": np.array([0.1, 0.2, 0.3]),
            "rms_low": np.array([0.2, 0.2, 0.2]),
            "rms_mid": np.array([0.15, 0.16, 0.17]),
            "rms_high": np.array([0.05, 0.06, 0.07]),
        },
    )

    bpm_key_text, diagnostics_text, report_path = (
        services.run_audio_analysis_tool(
            "song.wav",
            2048,
            8.0,
            1.5,
        )
    )

    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    assert bpm_key_text == "C major (120 bpm)"
    assert payload["sample_rate_hz"] == 44100
    assert payload["beat_count"] == 3
    assert payload["requested_window_seconds"] == 8.0
    assert "**Key / Tempo:** C major (120 bpm)" in diagnostics_text


def test_run_audio_preview_tool_reports_source_and_preview_lengths(
    monkeypatch,
):
    _patch_audio_symbol(
        monkeypatch,
        "audio_preview",
        "definers.audio.preview",
        lambda file_path, max_duration=30: "preview.wav",
    )
    _patch_audio_symbol(
        monkeypatch,
        "get_audio_duration",
        "definers.audio.preview",
        lambda file_path: 120.0 if file_path == "song.wav" else 30.0,
    )

    preview_path, summary_text = services.run_audio_preview_tool(
        "song.wav",
        30,
        "WAV",
    )

    assert preview_path == "preview.wav"
    assert "**Source Duration:** 120.00 s" in summary_text
    assert "**Preview Duration:** 30.00 s" in summary_text


def test_run_split_audio_tool_returns_first_chunk_and_summary(monkeypatch):
    _patch_audio_symbol(
        monkeypatch,
        "split_audio",
        "definers.audio.io",
        lambda *args, **kwargs: ["chunk_0000.mp3", "chunk_0001.mp3"],
    )

    preview_path, output_files, summary_text = services.run_split_audio_tool(
        "song.wav",
        15,
        "MP3",
        0,
        5,
        0,
    )

    assert preview_path == "chunk_0000.mp3"
    assert output_files == ["chunk_0000.mp3", "chunk_0001.mp3"]
    assert "**Chunks Created:** 2" in summary_text


def test_prepare_mastering_job_persists_manifest_and_input_copy(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        services,
        "_report_audio_activity",
        lambda *args, **kwargs: None,
    )

    source_path = tmp_path / "mix.wav"
    source_path.write_text("audio", encoding="utf-8")

    audio_io = importlib.import_module("definers.audio.io")
    mastering_module = importlib.import_module("definers.audio.mastering")
    analysis = types.SimpleNamespace(
        preset_name="balanced",
        quality_flags=("dense_low_end",),
        target_sample_rate=48000,
    )
    monkeypatch.setattr(
        audio_io,
        "read_audio",
        lambda path: (44100, np.zeros(16, dtype=np.float32)),
        raising=False,
    )
    monkeypatch.setattr(
        mastering_module,
        "_analyze_mastering_input",
        lambda signal, sample_rate: analysis,
        raising=False,
    )
    monkeypatch.setattr(
        mastering_module,
        "_resolve_mastering_kwargs_for_input",
        lambda signal, sample_rate, kwargs, input_analysis=None: {
            **kwargs,
            "preset": "balanced",
        },
        raising=False,
    )

    manifest = services.prepare_mastering_job(
        str(source_path),
        "wav",
        "Custom Macro Blend",
        0.25,
        0.35,
        0.45,
        True,
        "Demucs fine-tuned 4-stem",
        4,
        7.5,
        True,
        stem_glue_reverb_amount=2.25,
        stem_drum_edge_amount=-0.5,
        stem_vocal_pullback_db=4.0,
    )

    job_dir = Path(str(manifest["job_dir"]))
    payload = services.read_manifest(str(job_dir))

    assert job_dir.is_relative_to(tmp_path / "audio" / "mastering_jobs")
    assert Path(str(payload["input"]["path"])).exists()
    assert payload["analysis"]["quality_flags"] == ["dense_low_end"]
    assert payload["settings"]["stem_mastering"] is True
    assert payload["settings"]["stem_model_name"] == "htdemucs_ft.yaml"
    assert payload["settings"]["stem_glue_reverb_amount"] == 1.5
    assert payload["settings"]["stem_drum_edge_amount"] == 0.0
    assert payload["settings"]["stem_vocal_pullback_db"] == 3.0
    assert payload["resolved_mastering_kwargs"]["preset"] == "balanced"
    assert (
        payload["resolved_mastering_kwargs"]["stem_glue_reverb_amount"] == 1.5
    )
    assert payload["resolved_mastering_kwargs"]["stem_drum_edge_amount"] == 0.0
    assert payload["resolved_mastering_kwargs"]["stem_vocal_pullback_db"] == 3.0

    status_text = services.format_mastering_job_status(payload)

    assert "**Glue reverb:** 1.50x" in status_text
    assert "**Drum edge:** 0.00x" in status_text
    assert "**Extra vocal pullback:** 3.00 dB" in status_text


def test_render_mastering_job_view_discovers_saved_artifacts(tmp_path):
    job_dir = tmp_path / "mastering_job"
    input_path = job_dir / "input.wav"
    raw_dir = job_dir / "raw_stems"
    processed_dir = job_dir / "processed_stems"
    mixed_path = job_dir / "stem_mix.wav"
    report_path = job_dir / "report.md"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    input_path.write_text("audio", encoding="utf-8")
    (raw_dir / "vocals.wav").write_text("vocals", encoding="utf-8")
    (raw_dir / "drums.wav").write_text("drums", encoding="utf-8")
    (processed_dir / "vocals_mastered.wav").write_text(
        "vocals",
        encoding="utf-8",
    )
    mixed_path.write_text("mix", encoding="utf-8")
    report_path.write_text("# report", encoding="utf-8")

    services.write_manifest(
        str(job_dir),
        {
            "job_type": "audio-mastering-job",
            "job_version": 1,
            "job_dir": str(job_dir),
            "input": {"path": str(input_path), "sample_rate": 44100},
            "settings": {
                "format_choice": "wav",
                "profile_name": "Balanced",
                "control_mode": "Balanced",
                "stem_mastering": True,
            },
            "analysis": {
                "suggested_preset": "balanced",
                "quality_flags": ["dense_low_end"],
            },
            "resolved_mastering_kwargs": {"preset": "balanced"},
            "artifacts": {
                "raw_stems_dir": str(raw_dir),
                "raw_stems": {},
                "processed_stems_dir": str(processed_dir),
                "processed_stems": {},
                "mixed_path": str(mixed_path),
                "mastered_path": None,
                "report_path": str(report_path),
                "report_summary": "Ready for final delivery.",
            },
        },
    )

    view = services.render_mastering_job_view(str(job_dir))

    assert view[0] == str(job_dir)
    assert "Stem mix ready" in view[1]
    assert set(view[2] or []) == {
        str(raw_dir / "drums.wav"),
        str(raw_dir / "vocals.wav"),
    }
    assert set(view[3] or []) == {str(processed_dir / "vocals_mastered.wav")}
    assert view[4] == str(mixed_path)
    assert view[6] == str(report_path)
    assert view[7] == "Ready for final delivery."
    assert '"processed_stems"' in view[8]
