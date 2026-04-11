from __future__ import annotations

import atexit
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

_SESSION_MARKER_NAME = ".definers_session"
_SESSION_DIR_PATTERN = re.compile(r"^[A-Za-z0-9._-]+_[0-9a-f]{8}$")
_CLEANUP_LOCK = threading.Lock()
_LAST_CLEANUP_AT = 0.0
_PYTEST_SESSION_ROOT: Path | None = None
_REGISTERED_PYTEST_ROOTS: set[str] = set()


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


def _cleanup_interval_seconds() -> float:
    configured_value = os.environ.get(
        "DEFINERS_GUI_OUTPUT_CLEANUP_INTERVAL_SECONDS",
        "300",
    ).strip()
    try:
        return max(float(configured_value), 0.0)
    except Exception:
        return 300.0


def _session_retention_seconds() -> float:
    try:
        from definers.system.runtime_budget import (
            estimate_session_retention_seconds,
        )

        return estimate_session_retention_seconds(os.environ)
    except Exception:
        configured_value = os.environ.get(
            "DEFINERS_GUI_SESSION_RETENTION_SECONDS",
            "86400",
        ).strip()
        try:
            return max(float(configured_value), 0.0)
        except Exception:
            return 86400.0


def _pytest_output_retention_seconds() -> float:
    configured_value = os.environ.get(
        "DEFINERS_GUI_TEST_OUTPUT_RETENTION_SECONDS",
        "3600",
    ).strip()
    try:
        return max(float(configured_value), 0.0)
    except Exception:
        return 3600.0


def _is_pytest_runtime() -> bool:
    return (
        bool(os.environ.get("PYTEST_CURRENT_TEST")) or "pytest" in sys.modules
    )


def _pytest_root_base_path() -> Path:
    base_path = Path(tempfile.gettempdir()) / "definers" / "gui_outputs_pytest"
    base_path.mkdir(parents=True, exist_ok=True)
    return base_path.resolve()


def _register_pytest_root_cleanup(root_path: Path) -> None:
    root_text = str(root_path.resolve())
    if root_text in _REGISTERED_PYTEST_ROOTS:
        return
    _REGISTERED_PYTEST_ROOTS.add(root_text)
    atexit.register(shutil.rmtree, root_text, True)


def _pytest_root_path() -> Path:
    global _PYTEST_SESSION_ROOT

    if _PYTEST_SESSION_ROOT is None:
        _PYTEST_SESSION_ROOT = (
            _pytest_root_base_path() / f"session_{os.getpid()}"
        )
    _PYTEST_SESSION_ROOT.mkdir(parents=True, exist_ok=True)
    _register_pytest_root_cleanup(_PYTEST_SESSION_ROOT)
    return _PYTEST_SESSION_ROOT.resolve()


def _current_time() -> float:
    return time.time()


def _path_last_modified(path: Path) -> float:
    try:
        latest_modified = path.stat().st_mtime
    except Exception:
        return 0.0
    try:
        for child_path in path.rglob("*"):
            try:
                latest_modified = max(
                    latest_modified, child_path.stat().st_mtime
                )
            except Exception:
                continue
    except Exception:
        return latest_modified
    return latest_modified


def _remove_directory_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _remove_empty_parents(root_path: Path) -> None:
    child_directories = sorted(
        (path for path in root_path.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory_path in child_directories:
        if directory_path == root_path:
            continue
        try:
            directory_path.rmdir()
        except Exception:
            continue


def _iter_session_directories(root_path: Path) -> tuple[Path, ...]:
    session_directories: list[Path] = []
    seen_paths: set[str] = set()
    marker_paths = list(root_path.rglob(_SESSION_MARKER_NAME))
    for marker_path in marker_paths:
        session_path = marker_path.parent.resolve()
        session_text = str(session_path)
        if session_text in seen_paths:
            continue
        seen_paths.add(session_text)
        session_directories.append(session_path)
    for candidate_path in root_path.rglob("*"):
        if not candidate_path.is_dir():
            continue
        if not _SESSION_DIR_PATTERN.fullmatch(candidate_path.name):
            continue
        candidate_text = str(candidate_path.resolve())
        if candidate_text in seen_paths:
            continue
        seen_paths.add(candidate_text)
        session_directories.append(candidate_path.resolve())
    return tuple(session_directories)


def _cleanup_stale_session_directories(root_path: Path) -> None:
    retention_seconds = _session_retention_seconds()
    if retention_seconds <= 0:
        return
    expiration_cutoff = _current_time() - retention_seconds
    for session_path in _iter_session_directories(root_path):
        if _path_last_modified(session_path) >= expiration_cutoff:
            continue
        _remove_directory_tree(session_path)
    _remove_empty_parents(root_path)


def _cleanup_stale_pytest_roots(active_root: Path | None = None) -> None:
    pytest_root_base = (
        Path(tempfile.gettempdir()) / "definers" / "gui_outputs_pytest"
    )
    if not pytest_root_base.exists():
        return
    retention_seconds = _pytest_output_retention_seconds()
    if retention_seconds <= 0:
        return
    expiration_cutoff = _current_time() - retention_seconds
    active_root_text = (
        str(active_root.resolve()) if active_root is not None else None
    )
    for child_path in pytest_root_base.iterdir():
        if not child_path.is_dir():
            continue
        child_text = str(child_path.resolve())
        if active_root_text is not None and child_text == active_root_text:
            continue
        if _path_last_modified(child_path) >= expiration_cutoff:
            continue
        _remove_directory_tree(child_path)


def cleanup_managed_output_root() -> str:
    root_path = _root_path(cleanup=False)
    _cleanup_stale_session_directories(root_path)
    _cleanup_stale_pytest_roots(root_path if _is_pytest_runtime() else None)
    return str(root_path)


def cleanup_managed_output_path(path: str | None) -> bool:
    if path is None:
        return False
    candidate_path = Path(str(path).strip()).expanduser()
    if not str(candidate_path).strip():
        return False
    try:
        resolved_path = candidate_path.resolve()
    except Exception:
        return False
    root_path = _root_path(cleanup=False)
    try:
        resolved_path.relative_to(root_path)
    except ValueError:
        return False
    if resolved_path.is_dir():
        _remove_directory_tree(resolved_path)
        _remove_empty_parents(root_path)
        return True
    if resolved_path.exists():
        try:
            resolved_path.unlink(missing_ok=True)
        except Exception:
            return False
        _remove_empty_parents(root_path)
        return True
    return False


def _maybe_cleanup_output_root(root_path: Path) -> None:
    global _LAST_CLEANUP_AT

    cleanup_interval = _cleanup_interval_seconds()
    now = _current_time()
    with _CLEANUP_LOCK:
        if cleanup_interval > 0 and now - _LAST_CLEANUP_AT < cleanup_interval:
            return
        _LAST_CLEANUP_AT = now
    _cleanup_stale_session_directories(root_path)
    _cleanup_stale_pytest_roots(root_path if _is_pytest_runtime() else None)


def _default_root_path() -> Path:
    if _is_pytest_runtime():
        return _pytest_root_path()
    return Path(tempfile.gettempdir()) / "definers" / "gui_outputs"


def _root_path(*, cleanup: bool = True) -> Path:
    configured_root = os.environ.get("DEFINERS_GUI_OUTPUT_ROOT", "").strip()
    configured_data_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
    if configured_root:
        root_path = Path(configured_root).expanduser()
    elif configured_data_root:
        root_path = Path(configured_data_root).expanduser() / "gui_outputs"
    else:
        root_path = _default_root_path()
    root_path.mkdir(parents=True, exist_ok=True)
    resolved_root = root_path.resolve()
    if cleanup:
        _maybe_cleanup_output_root(resolved_root)
    return resolved_root


def managed_output_root() -> str:
    return str(_root_path())


def _iter_output_segments(*segments: object):
    for segment in segments:
        if segment is None:
            continue
        normalized_text = str(segment).strip().replace("\\", "/")
        if not normalized_text:
            continue
        for part in normalized_text.split("/"):
            resolved_part = str(part).strip()
            if resolved_part:
                yield _sanitize_segment(resolved_part, "item")


def managed_output_dir(*segments: object) -> str:
    directory_path = _root_path()
    for segment in _iter_output_segments(*segments):
        directory_path = directory_path / segment
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
    (directory_path / _SESSION_MARKER_NAME).touch(exist_ok=True)
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
    "cleanup_managed_output_path",
    "cleanup_managed_output_root",
    "managed_output_dir",
    "managed_output_path",
    "managed_output_root",
    "managed_output_session_dir",
]
