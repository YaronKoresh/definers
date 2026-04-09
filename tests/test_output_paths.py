from pathlib import Path

from definers.system.output_paths import (
    managed_output_path,
    managed_output_session_dir,
)


def test_managed_output_path_uses_configured_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    output_path = managed_output_path(
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

    session_dir = managed_output_session_dir("animation", stem="chunks")

    assert Path(session_dir).is_dir()
    assert Path(session_dir).parent == tmp_path / "animation"
    assert Path(session_dir).name.startswith("chunks_")
