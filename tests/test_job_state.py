from pathlib import Path

from definers.ui import job_state


def test_create_job_dir_uses_managed_output_root(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    job_dir = Path(
        job_state.create_job_dir("image/generate_jobs", stem="poster")
    )

    assert job_dir.exists()
    assert job_dir.is_relative_to(tmp_path / "image" / "generate_jobs")
    assert job_dir.name.startswith("poster_")


def test_manifest_roundtrip_and_file_scanning(tmp_path):
    job_dir = tmp_path / "job"

    manifest = job_state.write_manifest(
        str(job_dir),
        {
            "job_dir": job_dir,
            "settings": {"count": 2},
        },
    )

    artifacts_dir = job_dir / "artifacts"
    nested_dir = artifacts_dir / "nested"
    nested_dir.mkdir(parents=True, exist_ok=True)
    wav_path = artifacts_dir / "mix.wav"
    flac_path = nested_dir / "stem.flac"
    txt_path = artifacts_dir / "ignore.txt"
    wav_path.write_text("mix", encoding="utf-8")
    flac_path.write_text("stem", encoding="utf-8")
    txt_path.write_text("ignore", encoding="utf-8")

    loaded_manifest = job_state.read_manifest(str(job_dir))
    scanned_files = job_state.scan_files(
        artifacts_dir,
        suffixes=("wav", ".flac"),
        recursive=True,
    )
    scanned_map = job_state.scan_file_map(
        artifacts_dir,
        suffixes=("wav", ".flac"),
        recursive=True,
    )
    markdown = job_state.manifest_markdown(loaded_manifest)

    assert loaded_manifest == manifest
    assert job_state.existing_path(wav_path) == str(wav_path)
    assert job_state.existing_path(artifacts_dir / "missing.wav") is None
    assert set(scanned_files) == {str(wav_path), str(flac_path)}
    assert scanned_map == {
        "mix": str(wav_path),
        "stem": str(flac_path),
    }
    assert "```json" in markdown
    assert '"job_dir"' in markdown
