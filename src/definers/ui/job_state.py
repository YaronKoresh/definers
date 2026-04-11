from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path


def _json_safe(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    try:
        return float(value)
    except Exception:
        return str(value)


def _resolve_job_dir(job_dir: str) -> Path:
    resolved_path = Path(str(job_dir or "").strip()).expanduser()
    if not str(resolved_path).strip():
        raise ValueError("A job folder is required.")
    resolved_path.mkdir(parents=True, exist_ok=True)
    return resolved_path


def create_job_dir(section: str, *, stem: str | None = None) -> str:
    from definers.system.output_paths import managed_output_session_dir

    return managed_output_session_dir(section, stem=stem)


def manifest_path(job_dir: str, *, name: str = "job.json") -> str:
    return str(_resolve_job_dir(job_dir) / str(name).strip())


def write_manifest(
    job_dir: str,
    manifest: Mapping[str, object],
    *,
    name: str = "job.json",
) -> dict[str, object]:
    resolved_manifest = dict(_json_safe(dict(manifest)))
    destination_path = Path(manifest_path(job_dir, name=name))
    destination_path.write_text(
        json.dumps(resolved_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return resolved_manifest


def read_manifest(
    job_dir: str,
    *,
    name: str = "job.json",
) -> dict[str, object]:
    source_path = Path(manifest_path(job_dir, name=name))
    if not source_path.exists():
        raise FileNotFoundError(f"Job manifest was not found: {source_path}")
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Job manifest must contain a JSON object.")
    return dict(payload)


def update_manifest(
    job_dir: str,
    updates: Mapping[str, object],
    *,
    name: str = "job.json",
) -> dict[str, object]:
    manifest = read_manifest(job_dir, name=name)
    manifest.update(dict(_json_safe(dict(updates))))
    return write_manifest(job_dir, manifest, name=name)


def existing_path(value: object) -> str | None:
    if value is None:
        return None
    candidate = Path(str(value).strip())
    if not str(candidate):
        return None
    return str(candidate) if candidate.exists() else None


def scan_files(
    directory: object,
    *,
    suffixes: Iterable[str] = (),
    recursive: bool = False,
) -> list[str]:
    resolved_directory = existing_path(directory)
    if resolved_directory is None:
        return []
    root_path = Path(resolved_directory)
    normalized_suffixes = {
        str(suffix).strip().lower()
        if str(suffix).startswith(".")
        else f".{str(suffix).strip().lower()}"
        for suffix in suffixes
        if str(suffix).strip()
    }
    iterator = root_path.rglob("*") if recursive else root_path.iterdir()
    results: list[str] = []
    for file_path in sorted(iterator):
        if not file_path.is_file():
            continue
        if (
            normalized_suffixes
            and file_path.suffix.lower() not in normalized_suffixes
        ):
            continue
        results.append(str(file_path))
    return results


def scan_file_map(
    directory: object,
    *,
    suffixes: Iterable[str] = (),
    recursive: bool = False,
) -> dict[str, str]:
    return {
        Path(file_path).stem: file_path
        for file_path in scan_files(
            directory,
            suffixes=suffixes,
            recursive=recursive,
        )
    }


def manifest_markdown(manifest: Mapping[str, object]) -> str:
    resolved_manifest = _json_safe(dict(manifest))
    return (
        "```json\n"
        + json.dumps(resolved_manifest, indent=2, ensure_ascii=False)
        + "\n```"
    )


__all__ = (
    "create_job_dir",
    "existing_path",
    "manifest_markdown",
    "manifest_path",
    "read_manifest",
    "scan_file_map",
    "scan_files",
    "update_manifest",
    "write_manifest",
)
