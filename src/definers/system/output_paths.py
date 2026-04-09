from __future__ import annotations

import os
import re
import tempfile
import uuid
from pathlib import Path


def _sanitize_segment(value: object, default: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    normalized = normalized.strip("._-")
    return normalized or default


def _normalize_suffix(suffix: str | None) -> str:
    normalized = str(suffix or "").strip()
    if not normalized:
        return ""
    if normalized.startswith("."):
        return normalized.lower()
    return f".{normalized.lower()}"


def _root_path() -> Path:
    configured_root = os.environ.get("DEFINERS_GUI_OUTPUT_ROOT", "").strip()
    configured_data_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
    if configured_root:
        root_path = Path(configured_root).expanduser()
    elif configured_data_root:
        root_path = Path(configured_data_root).expanduser() / "gui_outputs"
    else:
        root_path = Path(tempfile.gettempdir()) / "definers" / "gui_outputs"
    root_path.mkdir(parents=True, exist_ok=True)
    return root_path.resolve()


def managed_output_root() -> str:
    return str(_root_path())


def managed_output_dir(*segments: object) -> str:
    directory_path = _root_path()
    for segment in segments:
        if segment is None:
            continue
        text = str(segment).strip()
        if not text:
            continue
        directory_path = directory_path / _sanitize_segment(text, "item")
    directory_path.mkdir(parents=True, exist_ok=True)
    return str(directory_path)


def managed_output_session_dir(
    section: str,
    *,
    stem: str | None = None,
) -> str:
    directory_path = Path(managed_output_dir(section)) / (
        f"{_sanitize_segment(stem, 'session')}_{uuid.uuid4().hex[:8]}"
    )
    directory_path.mkdir(parents=True, exist_ok=True)
    return str(directory_path)


def managed_output_path(
    suffix: str | None = None,
    *,
    section: str,
    stem: str | None = None,
    filename: str | None = None,
    unique: bool = True,
) -> str:
    directory_path = Path(managed_output_dir(section))
    if filename is not None and str(filename).strip():
        filename_path = Path(str(filename).strip())
        stem_name = _sanitize_segment(filename_path.stem, "artifact")
        suffix_text = _normalize_suffix(filename_path.suffix)
    else:
        stem_name = _sanitize_segment(stem, "artifact")
        suffix_text = _normalize_suffix(suffix)

    candidate_path = directory_path / f"{stem_name}{suffix_text}"
    if unique:
        while candidate_path.exists():
            candidate_path = (
                directory_path
                / f"{stem_name}_{uuid.uuid4().hex[:8]}{suffix_text}"
            )
    return str(candidate_path)


__all__ = [
    "managed_output_dir",
    "managed_output_path",
    "managed_output_root",
    "managed_output_session_dir",
]
