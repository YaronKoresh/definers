import importlib
import json
import types
from pathlib import Path

import numpy as np

from definers.presentation.apps import audio_app_services as services


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
    assert "**Preset:** vocal" in summary_text
    assert "**Mastered Stems:** 2 files" in summary_text


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
