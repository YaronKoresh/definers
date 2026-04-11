from __future__ import annotations

import contextlib
import contextvars
import threading
import time
import uuid
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DownloadActivitySnapshot:
    scope_id: str
    phase: str
    message: str
    item_label: str | None
    completed: int | None
    total: int | None
    bytes_downloaded: int | None
    bytes_total: int | None
    sequence: int
    updated_at: float


@dataclass(slots=True)
class DownloadActivityTask:
    scope_id: str
    worker: threading.Thread
    result_box: dict[str, object]
    error_box: dict[str, BaseException]


_ACTIVE_SCOPE = contextvars.ContextVar(
    "definers_download_activity_scope",
    default=None,
)
_ACTIVE_LABEL = contextvars.ContextVar(
    "definers_download_activity_label",
    default=None,
)
_SNAPSHOT_LOCK = threading.Lock()
_SNAPSHOTS: dict[str, DownloadActivitySnapshot] = {}


def _normalize_count(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        normalized = int(value)
    except Exception:
        return None
    return normalized if normalized >= 0 else None


def _format_bytes(value: int | None) -> str | None:
    normalized = _normalize_count(value)
    if normalized is None:
        return None
    suffixes = ("B", "KiB", "MiB", "GiB", "TiB")
    scaled = float(normalized)
    suffix = suffixes[0]
    for suffix in suffixes:
        if scaled < 1024.0 or suffix == suffixes[-1]:
            break
        scaled /= 1024.0
    if suffix == "B":
        return f"{int(scaled)} {suffix}"
    return f"{scaled:.1f} {suffix}"


def _activity_prefix(
    phase: str,
    item_label: str | None,
    completed: int | None,
    total: int | None,
) -> str:
    prefix = {
        "artifact": "Downloading artifact",
        "download": "Downloading model",
        "extract": "Extracting archive",
        "index": "Resolving model index",
        "model": "Initializing model",
        "step": "Running step",
        "transfer": "Transferring artifact",
    }.get(str(phase).strip().lower(), "Running task")
    if completed is not None and total:
        prefix = f"{prefix} {completed}/{total}"
    if item_label:
        return f"{prefix}: {item_label}"
    return prefix


def _build_message(
    *,
    phase: str,
    item_label: str | None,
    detail: str | None,
    completed: int | None,
    total: int | None,
    bytes_downloaded: int | None,
    bytes_total: int | None,
) -> str:
    message = _activity_prefix(phase, item_label, completed, total)
    detail_text = str(detail or "").strip()
    if detail_text:
        message = f"{message}. {detail_text}"
    downloaded_text = _format_bytes(bytes_downloaded)
    total_text = _format_bytes(bytes_total)
    if downloaded_text and total_text:
        message = f"{message} ({downloaded_text} / {total_text})"
    elif downloaded_text:
        message = f"{message} ({downloaded_text})"
    return message


def create_download_activity_scope() -> str:
    scope_id = uuid.uuid4().hex
    with _SNAPSHOT_LOCK:
        _SNAPSHOTS.pop(scope_id, None)
    return scope_id


@contextlib.contextmanager
def bind_download_activity_scope(scope_id: str):
    token = _ACTIVE_SCOPE.set(str(scope_id))
    try:
        yield str(scope_id)
    finally:
        _ACTIVE_SCOPE.reset(token)


def get_download_activity_snapshot(
    scope_id: str,
) -> DownloadActivitySnapshot | None:
    with _SNAPSHOT_LOCK:
        return _SNAPSHOTS.get(str(scope_id))


def clear_download_activity_scope(scope_id: str) -> None:
    with _SNAPSHOT_LOCK:
        _SNAPSHOTS.pop(str(scope_id), None)


def current_download_activity_scope() -> str | None:
    scope_id = _ACTIVE_SCOPE.get()
    if not scope_id:
        return None
    return str(scope_id)


@contextlib.contextmanager
def bind_download_activity_label(label: str | None):
    normalized_label = str(label).strip() or None if label is not None else None
    token = _ACTIVE_LABEL.set(normalized_label)
    try:
        yield normalized_label
    finally:
        _ACTIVE_LABEL.reset(token)


def current_download_activity_label() -> str | None:
    label = _ACTIVE_LABEL.get()
    if not label:
        return None
    return str(label)


def report_download_activity(
    item_label: str | None = None,
    *,
    detail: str | None = None,
    phase: str = "download",
    completed: int | None = None,
    total: int | None = None,
    bytes_downloaded: int | None = None,
    bytes_total: int | None = None,
) -> DownloadActivitySnapshot | None:
    scope_id = _ACTIVE_SCOPE.get()
    if not scope_id:
        return None
    normalized_label = (
        str(item_label).strip() or None if item_label is not None else None
    )
    normalized_completed = _normalize_count(completed)
    normalized_total = _normalize_count(total)
    normalized_downloaded = _normalize_count(bytes_downloaded)
    normalized_bytes_total = _normalize_count(bytes_total)
    message = _build_message(
        phase=phase,
        item_label=normalized_label,
        detail=detail,
        completed=normalized_completed,
        total=normalized_total,
        bytes_downloaded=normalized_downloaded,
        bytes_total=normalized_bytes_total,
    )
    now = time.monotonic()
    with _SNAPSHOT_LOCK:
        previous = _SNAPSHOTS.get(scope_id)
        if (
            previous is not None
            and previous.phase == str(phase).strip().lower()
            and previous.message == message
            and previous.item_label == normalized_label
            and previous.completed == normalized_completed
            and previous.total == normalized_total
            and previous.bytes_downloaded == normalized_downloaded
            and previous.bytes_total == normalized_bytes_total
        ):
            return previous
        snapshot = DownloadActivitySnapshot(
            scope_id=scope_id,
            phase=str(phase).strip().lower() or "download",
            message=message,
            item_label=normalized_label,
            completed=normalized_completed,
            total=normalized_total,
            bytes_downloaded=normalized_downloaded,
            bytes_total=normalized_bytes_total,
            sequence=0 if previous is None else previous.sequence + 1,
            updated_at=now,
        )
        _SNAPSHOTS[scope_id] = snapshot
        return snapshot


def create_activity_reporter(
    total: int,
    *,
    phase: str = "step",
):
    resolved_total = _normalize_count(total)
    if resolved_total is None or resolved_total == 0:
        resolved_total = 1

    def reporter(
        completed: int,
        item_label: str | None = None,
        *,
        detail: str | None = None,
        phase_override: str | None = None,
        total_override: int | None = None,
        bytes_downloaded: int | None = None,
        bytes_total: int | None = None,
    ) -> DownloadActivitySnapshot | None:
        normalized_completed = _normalize_count(completed)
        if normalized_completed is None:
            normalized_completed = 0
        normalized_total = _normalize_count(total_override)
        if normalized_total is None or normalized_total == 0:
            normalized_total = resolved_total
        return report_download_activity(
            item_label,
            detail=detail,
            phase=phase_override or phase,
            completed=normalized_completed,
            total=normalized_total,
            bytes_downloaded=bytes_downloaded,
            bytes_total=bytes_total,
        )

    return reporter


def create_download_activity_task(
    handler,
    *args,
    **kwargs,
) -> DownloadActivityTask:
    scope_id = create_download_activity_scope()
    result_box: dict[str, object] = {}
    error_box: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            with bind_download_activity_scope(scope_id):
                result_box["value"] = handler(*args, **kwargs)
        except BaseException as error:
            error_box["error"] = error

    worker = threading.Thread(target=runner, daemon=False)
    worker.start()
    return DownloadActivityTask(
        scope_id=scope_id,
        worker=worker,
        result_box=result_box,
        error_box=error_box,
    )


def wait_for_download_activity_task(
    task: DownloadActivityTask,
    timeout_seconds: float,
) -> bool:
    task.worker.join(timeout=max(float(timeout_seconds), 0.0))
    return not task.worker.is_alive()


def resolve_download_activity_task(
    task: DownloadActivityTask,
) -> tuple[object, DownloadActivitySnapshot | None]:
    task.worker.join()
    snapshot = get_download_activity_snapshot(task.scope_id)
    error = task.error_box.get("error")
    if error is not None:
        setattr(error, "download_activity_snapshot", snapshot)
        clear_download_activity_scope(task.scope_id)
        raise error
    clear_download_activity_scope(task.scope_id)
    return task.result_box.get("value"), snapshot


__all__ = (
    "DownloadActivitySnapshot",
    "DownloadActivityTask",
    "bind_download_activity_scope",
    "clear_download_activity_scope",
    "current_download_activity_scope",
    "create_activity_reporter",
    "create_download_activity_scope",
    "create_download_activity_task",
    "get_download_activity_snapshot",
    "report_download_activity",
    "resolve_download_activity_task",
    "wait_for_download_activity_task",
)
