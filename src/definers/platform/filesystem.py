import datetime
import errno
import os
import shutil
import stat
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from definers.platform.paths import full_path, parent_directory, secure_path
from definers.platform.runtime import get_os_name
from definers.shared_kernel.observability import catch_exception, log_message


def _joined_path(path_parts: Sequence[str]) -> str:
    return os.path.join(*[str(path_part).strip() for path_part in path_parts])


def exist(*p: str) -> bool:
    joined = _joined_path(p)
    if not joined or not joined.strip():
        return False
    expanded = os.path.expanduser(joined)
    absolute = os.path.abspath(expanded)
    return os.path.exists(absolute)


def is_text(data: bytes) -> bool:
    if not data:
        return True
    text_chars = set(range(32, 127)) | {9, 10, 13} | set(range(128, 256))
    return all(byte_value in text_chars for byte_value in data[:8192])


def load(path: str):
    resolved_path = full_path(str(path))
    permit(resolved_path)
    if not exist(resolved_path):
        return None
    if is_directory(resolved_path):
        return sorted(child.name for child in Path(resolved_path).iterdir())
    raw = Path(resolved_path).read_bytes()
    if b"\x00" in raw or not is_text(raw):
        return raw
    try:
        return raw.decode("utf-8").replace("\r\n", "\n")
    except (UnicodeDecodeError, ValueError):
        return raw


def read(path: str):
    return load(path)


def save(path: str, text: Any = ""):
    resolved_path = full_path(str(path))
    os.makedirs(parent_directory(resolved_path), exist_ok=True)
    with open(resolved_path, "w+", encoding="utf8") as file_handle:
        file_handle.write(str(text))


def write(path: str, txt: Any = ""):
    return save(path, txt)


def directory(dir_path: str, exist_ok: bool = True):
    os.makedirs(full_path(str(dir_path)), exist_ok=exist_ok)


def is_directory(*p: str) -> bool:
    return Path(_joined_path(p)).is_dir()


def is_symlink(*p: str) -> bool:
    return Path(_joined_path(p)).is_symlink()


def remove_readonly(func, path, excinfo):
    exception_instance = excinfo[1]
    if exception_instance.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
        return
    raise exception_instance


def shutil_rmtree_readonly_handler(func, path, exc_info):
    exception_instance = (
        exc_info[1] if isinstance(exc_info, tuple) else exc_info
    )
    if exception_instance.errno == errno.EACCES:
        os.chmod(path, stat.S_IWRITE)
        func(path)
        return
    raise exception_instance


def delete(path: str | list[str]):
    secure_path(path)
    raw_path = _joined_path(path) if isinstance(path, list) else str(path)
    expanded = Path(raw_path).expanduser()
    unresolved = Path(os.path.abspath(str(expanded)))
    if unresolved.is_symlink():
        unresolved.unlink()
        return
    resolved = full_path(raw_path)
    if not exist(resolved):
        return
    if is_directory(resolved):
        shutil.rmtree(resolved, onerror=shutil_rmtree_readonly_handler)
        return
    Path(resolved).unlink(missing_ok=True)


def remove(path: str | list[str]):
    delete(path)


def copy(src: str, dst: str):
    source_path = Path(full_path(src))
    if source_path.is_symlink():
        resolved_source = source_path.resolve()
        if os.path.isdir(str(resolved_source)):
            shutil.copytree(
                str(src),
                str(dst),
                symlinks=False,
                ignore_dangling_symlinks=True,
            )
            return
        shutil.copy(str(src), str(dst))
        return
    if os.path.isdir(str(source_path)):
        shutil.copytree(
            str(src),
            str(dst),
            symlinks=False,
            ignore_dangling_symlinks=True,
        )
        return
    shutil.copy(str(src), str(dst))


def move(src: str, dest: str):
    source_path = full_path(str(src))
    if not exist(source_path):
        raise FileNotFoundError(f"Source path not found: {src}")
    copy(src, dest)
    delete(src)


def glob(pattern: str, recursive: bool = False):
    from glob import glob as _glob

    return _glob(pattern, recursive=recursive)


def permit(
    path: str,
    *,
    exists_func=exist,
    get_os_name_func=get_os_name,
    subprocess_module=subprocess,
) -> bool:
    try:
        if not exists_func(path):
            return False
        os_name = get_os_name_func()
        if os_name == "linux":
            subprocess_module.run(["chmod", "-R", "a+xrw", path], check=True)
            return True
        if os_name == "windows":
            subprocess_module.run(
                ["icacls", path, "/grant", "Everyone:F", "/T"], check=True
            )
            return True
        return False
    except Exception:
        return False
