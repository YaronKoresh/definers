import os
import time
from pathlib import Path

import definers.system.output_paths as output_paths


def test_managed_output_path_uses_configured_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    output_path = output_paths.managed_output_path(
        "wav",
        section="audio",
        stem="preview_clip",
    )

    assert Path(output_path).parent == tmp_path / "audio"
    assert Path(output_path).name == "preview_clip.wav"


def test_managed_output_session_dir_creates_unique_subdirectory(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    session_dir = output_paths.managed_output_session_dir(
        "animation",
        stem="chunks",
    )

    assert Path(session_dir).is_dir()
    assert Path(session_dir).parent == tmp_path / "animation"
    assert Path(session_dir).name.startswith("chunks_")
    assert (Path(session_dir) / ".definers_session").exists()


def test_managed_output_path_supports_nested_sections(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    output_path = output_paths.managed_output_path(
        "json",
        section="audio/reports",
        stem="report",
    )

    assert Path(output_path).parent == tmp_path / "audio" / "reports"
    assert Path(output_path).name == "report.json"


def test_cleanup_managed_output_root_prunes_stale_session_dirs(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    stale_session_dir = Path(
        output_paths.managed_output_session_dir("audio/split", stem="chunks")
    )
    fresh_session_dir = Path(
        output_paths.managed_output_session_dir("audio/split", stem="chunks")
    )
    stale_timestamp = time.time() - 90000
    os.utime(stale_session_dir, (stale_timestamp, stale_timestamp))
    os.utime(
        stale_session_dir / ".definers_session",
        (stale_timestamp, stale_timestamp),
    )

    output_paths.cleanup_managed_output_root()

    assert stale_session_dir.exists() is False
    assert fresh_session_dir.exists() is True


def test_managed_output_path_uses_isolated_pytest_root(monkeypatch):
    monkeypatch.delenv("DEFINERS_GUI_OUTPUT_ROOT", raising=False)
    monkeypatch.delenv("DEFINERS_DATA_ROOT", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests::output-path")
    monkeypatch.setattr(output_paths, "_PYTEST_SESSION_ROOT", None)

    output_path = Path(
        output_paths.managed_output_path(
            "wav",
            section="audio",
            stem="preview_clip",
        )
    )

    assert "gui_outputs_pytest" in str(output_path)


def test_cleanup_managed_output_path_removes_directory_under_root(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    session_dir = Path(
        output_paths.managed_output_session_dir("animation", stem="chunks")
    )

    removed = output_paths.cleanup_managed_output_path(str(session_dir))

    assert removed is True
    assert session_dir.exists() is False
