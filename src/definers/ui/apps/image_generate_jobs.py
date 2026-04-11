from __future__ import annotations

from pathlib import Path
from shutil import copy2

from definers.ui.gradio_shared import status_card_markdown
from definers.ui.job_state import (
    create_job_dir,
    existing_path,
    manifest_markdown,
    read_manifest,
    write_manifest,
)

_IMAGE_JOB_TYPE = "image-generate-job"


def _image_compute_note() -> str:
    return (
        "**Compute Profile:** Generate Image is the heaviest stage, Upscale Result is the heavier enhancement stage, "
        "and Add Titles is the light overlay step."
    )


def _job_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _read_image_job(job_dir: str) -> dict[str, object]:
    manifest = read_manifest(job_dir)
    if str(manifest.get("job_type", "")).strip() != _IMAGE_JOB_TYPE:
        raise ValueError(
            "The selected job folder is not an image generation job."
        )
    return manifest


def _job_settings(manifest: dict[str, object]) -> dict[str, object]:
    return _job_dict(manifest.get("settings"))


def _job_artifacts(manifest: dict[str, object]) -> dict[str, object]:
    return _job_dict(manifest.get("artifacts"))


def _refresh_image_job_artifacts(
    manifest: dict[str, object],
) -> dict[str, object]:
    artifacts = _job_artifacts(manifest)
    manifest["artifacts"] = {
        **artifacts,
        "generated_path": existing_path(artifacts.get("generated_path")),
        "upscaled_path": existing_path(artifacts.get("upscaled_path")),
        "titled_path": existing_path(artifacts.get("titled_path")),
    }
    return manifest


def refresh_image_generate_job(job_dir: str) -> dict[str, object]:
    manifest = _read_image_job(job_dir)
    refreshed_manifest = _refresh_image_job_artifacts(manifest)
    return write_manifest(job_dir, refreshed_manifest)


def resolve_image_job_status(
    manifest: dict[str, object],
) -> tuple[str, str]:
    artifacts = _job_artifacts(manifest)
    if existing_path(artifacts.get("titled_path")) is not None:
        return (
            "Image ready",
            "Done. Download the generated, upscaled, or titled result from the job artifacts.",
        )
    if existing_path(artifacts.get("upscaled_path")) is not None:
        return (
            "Upscaled image ready",
            "Next: Add Titles or stop here. Upscaling is done and the lighter title pass can run any time.",
        )
    if existing_path(artifacts.get("generated_path")) is not None:
        return (
            "Generated image ready",
            "Next: Upscale Result or Add Titles. Generation is complete and the job can continue later.",
        )
    return (
        "Job prepared",
        "Next: Generate Image. Generation is the heaviest stage and the job folder keeps the workflow resumable.",
    )


def format_image_job_status(
    manifest: dict[str, object],
    *,
    title: str | None = None,
    detail: str | None = None,
) -> str:
    resolved_title, resolved_detail = resolve_image_job_status(manifest)
    settings = _job_settings(manifest)
    artifacts = _job_artifacts(manifest)
    prompt_preview = str(settings.get("prompt", "")).strip()
    if len(prompt_preview) > 96:
        prompt_preview = prompt_preview[:93].rstrip() + "..."
    return status_card_markdown(
        title or resolved_title,
        detail or resolved_detail,
        [
            ("Job folder", manifest.get("job_dir", "")),
            ("Prompt", prompt_preview or "n/a"),
            (
                "Target size",
                f"{int(settings.get('width', 1))} x {int(settings.get('height', 1))}",
            ),
            (
                "Generated image ready",
                existing_path(artifacts.get("generated_path")) is not None,
            ),
            (
                "Upscaled image ready",
                existing_path(artifacts.get("upscaled_path")) is not None,
            ),
            (
                "Titled image ready",
                existing_path(artifacts.get("titled_path")) is not None,
            ),
        ],
    )


def render_image_job_view(
    job_dir: str,
    *,
    title: str | None = None,
    detail: str | None = None,
) -> tuple[str, str, str | None, str | None, str | None, str]:
    manifest = refresh_image_generate_job(job_dir)
    artifacts = _job_artifacts(manifest)
    return (
        str(manifest.get("job_dir", job_dir)),
        format_image_job_status(manifest, title=title, detail=detail),
        existing_path(artifacts.get("generated_path")),
        existing_path(artifacts.get("upscaled_path")),
        existing_path(artifacts.get("titled_path")),
        manifest_markdown(manifest),
    )


def prepare_image_generate_job(
    prompt: str,
    width: int,
    height: int,
    top_title: str,
    middle_title: str,
    bottom_title: str,
) -> dict[str, object]:
    job_dir = create_job_dir("image/generate_jobs", stem="image")
    manifest = {
        "job_type": _IMAGE_JOB_TYPE,
        "job_version": 1,
        "job_dir": job_dir,
        "settings": {
            "prompt": str(prompt),
            "width": int(width),
            "height": int(height),
            "top_title": str(top_title),
            "middle_title": str(middle_title),
            "bottom_title": str(bottom_title),
        },
        "artifacts": {
            "generated_path": None,
            "upscaled_path": None,
            "titled_path": None,
        },
    }
    return write_manifest(job_dir, manifest)


def _copy_job_image_artifact(
    source_path: str,
    job_dir: str,
    stem: str,
) -> str:
    resolved_source_path = existing_path(source_path)
    if resolved_source_path is None:
        raise RuntimeError(
            "The image workflow did not produce a saved artifact."
        )
    source = Path(resolved_source_path)
    destination = (
        Path(str(job_dir)) / f"{stem}{source.suffix.lower() or '.png'}"
    )
    copy2(source, destination)
    return str(destination)


def _latest_image_artifact(manifest: dict[str, object]) -> str | None:
    artifacts = _job_artifacts(manifest)
    for key in ("titled_path", "upscaled_path", "generated_path"):
        resolved_path = existing_path(artifacts.get(key))
        if resolved_path is not None:
            return resolved_path
    return None


def generate_image_job(job_dir: str) -> dict[str, object]:
    from definers.ui.apps.image import ImageApp

    manifest = _read_image_job(job_dir)
    settings = _job_settings(manifest)
    generated_source = ImageApp.generate_image(
        str(settings.get("prompt", "")),
        int(settings.get("width", 1)),
        int(settings.get("height", 1)),
    )
    generated_path = _copy_job_image_artifact(
        str(generated_source),
        job_dir,
        "generated",
    )
    manifest["artifacts"] = {
        **_job_artifacts(manifest),
        "generated_path": generated_path,
    }
    return write_manifest(job_dir, manifest)


def upscale_image_job(job_dir: str) -> dict[str, object]:
    from definers.ui.apps.image import ImageApp

    manifest = _read_image_job(job_dir)
    source_path = _latest_image_artifact(manifest)
    if source_path is None:
        raise ValueError("Generate an image before running the upscale stage.")
    upscaled_source = ImageApp.upscale_image(source_path)
    upscaled_path = _copy_job_image_artifact(
        str(upscaled_source),
        job_dir,
        "upscaled",
    )
    manifest["artifacts"] = {
        **_job_artifacts(manifest),
        "upscaled_path": upscaled_path,
    }
    return write_manifest(job_dir, manifest)


def title_image_job(
    job_dir: str,
    top_title: str,
    middle_title: str,
    bottom_title: str,
) -> dict[str, object]:
    from definers.ui.apps.image import ImageApp

    manifest = _read_image_job(job_dir)
    source_path = _latest_image_artifact(manifest)
    if source_path is None:
        raise ValueError("Generate an image before adding titles.")

    settings = {
        **_job_settings(manifest),
        "top_title": str(top_title),
        "middle_title": str(middle_title),
        "bottom_title": str(bottom_title),
    }
    titled_source = ImageApp.title_image(
        source_path,
        str(top_title),
        str(middle_title),
        str(bottom_title),
    )
    titled_path = _copy_job_image_artifact(
        str(titled_source),
        job_dir,
        "titled",
    )
    manifest["settings"] = settings
    manifest["artifacts"] = {
        **_job_artifacts(manifest),
        "titled_path": titled_path,
    }
    return write_manifest(job_dir, manifest)


def run_full_image_generate_job(
    prompt: str,
    width: int,
    height: int,
    top_title: str,
    middle_title: str,
    bottom_title: str,
) -> dict[str, object]:
    manifest = prepare_image_generate_job(
        prompt,
        width,
        height,
        top_title,
        middle_title,
        bottom_title,
    )
    job_dir = str(manifest["job_dir"])
    generate_image_job(job_dir)
    upscale_image_job(job_dir)
    if any(
        str(value).strip() for value in (top_title, middle_title, bottom_title)
    ):
        return title_image_job(
            job_dir,
            top_title,
            middle_title,
            bottom_title,
        )
    return refresh_image_generate_job(job_dir)


__all__ = (
    "format_image_job_status",
    "generate_image_job",
    "prepare_image_generate_job",
    "refresh_image_generate_job",
    "render_image_job_view",
    "resolve_image_job_status",
    "run_full_image_generate_job",
    "title_image_job",
    "upscale_image_job",
)
