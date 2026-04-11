from pathlib import Path

from definers.ui.apps import image_generate_jobs as jobs
from definers.ui.apps.image import ImageApp
from definers.ui.job_state import read_manifest


def test_prepare_image_generate_job_persists_manifest(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    manifest = jobs.prepare_image_generate_job(
        "cinematic skyline",
        8,
        6,
        "Top",
        "Middle",
        "Bottom",
    )

    job_dir = Path(str(manifest["job_dir"]))
    payload = read_manifest(str(job_dir))

    assert job_dir.is_relative_to(tmp_path / "image" / "generate_jobs")
    assert payload["settings"]["prompt"] == "cinematic skyline"
    assert payload["settings"]["width"] == 8
    assert payload["artifacts"]["generated_path"] is None
    assert jobs.resolve_image_job_status(payload)[0] == "Job prepared"


def test_image_job_flow_updates_artifacts_and_status(monkeypatch, tmp_path):
    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))

    generated_source = tmp_path / "generated-source.png"
    upscaled_source = tmp_path / "upscaled-source.png"
    titled_source = tmp_path / "titled-source.png"
    generated_source.write_text("generated", encoding="utf-8")
    upscaled_source.write_text("upscaled", encoding="utf-8")
    titled_source.write_text("titled", encoding="utf-8")

    monkeypatch.setattr(
        ImageApp,
        "generate_image",
        lambda *args: str(generated_source),
    )
    monkeypatch.setattr(
        ImageApp,
        "upscale_image",
        lambda *args: str(upscaled_source),
    )
    monkeypatch.setattr(
        ImageApp,
        "title_image",
        lambda *args: str(titled_source),
    )

    manifest = jobs.prepare_image_generate_job(
        "album cover",
        10,
        10,
        "Top",
        "Middle",
        "Bottom",
    )
    job_dir = str(manifest["job_dir"])

    generated_manifest = jobs.generate_image_job(job_dir)
    upscaled_manifest = jobs.upscale_image_job(job_dir)
    titled_manifest = jobs.title_image_job(
        job_dir,
        "Top",
        "Middle",
        "Bottom",
    )
    view = jobs.render_image_job_view(job_dir)

    assert Path(str(generated_manifest["artifacts"]["generated_path"])).exists()
    assert Path(str(upscaled_manifest["artifacts"]["upscaled_path"])).exists()
    assert Path(str(titled_manifest["artifacts"]["titled_path"])).exists()
    assert view[0] == job_dir
    assert "Image ready" in view[1]
    assert view[2] is not None
    assert view[3] is not None
    assert view[4] is not None
    assert '"titled_path"' in view[5]


def test_run_full_image_generate_job_runs_all_stages(monkeypatch):
    calls = []

    monkeypatch.setattr(
        jobs,
        "prepare_image_generate_job",
        lambda *args, **kwargs: {"job_dir": "image-job"},
    )
    monkeypatch.setattr(
        jobs,
        "generate_image_job",
        lambda job_dir: calls.append(("generate", job_dir)) or {},
    )
    monkeypatch.setattr(
        jobs,
        "upscale_image_job",
        lambda job_dir: calls.append(("upscale", job_dir)) or {},
    )
    monkeypatch.setattr(
        jobs,
        "title_image_job",
        lambda job_dir, top, middle, bottom: (
            calls.append(("title", job_dir, top, middle, bottom))
            or {"job_dir": job_dir}
        ),
    )

    manifest = jobs.run_full_image_generate_job(
        "album cover",
        8,
        8,
        "Top",
        "Middle",
        "Bottom",
    )

    assert manifest == {"job_dir": "image-job"}
    assert calls == [
        ("generate", "image-job"),
        ("upscale", "image-job"),
        ("title", "image-job", "Top", "Middle", "Bottom"),
    ]
