import os
import re
import shlex
import shutil
import sys
import tempfile
import unicodedata
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path


def full_path(*p: str) -> str:
    joined = os.path.join(*[str(_p).strip() for _p in p])
    return str(Path(joined).expanduser().resolve())


def normalize_path(p: str) -> str:
    normalized = os.path.normpath(p)
    return os.path.expanduser(normalized)


def parent_directory(p: str, levels: int = 1) -> str:
    path = Path(str(p))
    for _ in range(levels):
        path = path.parent
    return os.path.normpath(str(path))


def path_end(p: str) -> str:
    return Path(p.rstrip("/").rstrip("\\")).name


def path_ext(p: str):
    if Path(p).is_dir():
        return None
    try:
        return "".join(Path(p).suffixes)
    except Exception:
        return None


def path_name(p: str) -> str:
    return str(Path(p).stem)


def _is_absolute_path(path_text: str) -> bool:
    expanded_path = os.path.expanduser(path_text)
    return Path(expanded_path).is_absolute()


def _is_relative_to(path: Path, base: Path) -> bool:
    return path.is_relative_to(base)


@contextmanager
def cwd(dir_path: str | None = None) -> Generator[str, None, None]:
    target = dir_path or "."
    if not target.startswith("~") and not _is_absolute_path(target):
        package_root = str(Path(__file__).resolve().parent.parent)
        target = full_path(package_root, target)
    else:
        target = full_path(target)
    original_working_directory = full_path(os.getcwd())
    try:
        os.chdir(target)
        yield target
    finally:
        try:
            os.chdir(original_working_directory)
        except Exception:
            pass


def _resolve_tmp_suffix(suffix: str | None) -> str:
    normalized_suffix = suffix
    if not isinstance(normalized_suffix, str) or not normalized_suffix.strip():
        normalized_suffix = "data"
    normalized_suffix = str(normalized_suffix).strip().strip(".").lower()

    from definers.constants import SAFE_EXTENSIONS

    allowed_suffixes = {
        str(extension).strip().strip(".").lower(): "."
        + str(extension).strip().strip(".").lower()
        for extension in SAFE_EXTENSIONS
    }
    resolved_suffix = allowed_suffixes.get(normalized_suffix)
    if resolved_suffix is None:
        raise ValueError(
            "Invalid suffix for tmp file. Allowed extensions are: "
            + ", ".join(SAFE_EXTENSIONS)
        )
    return resolved_suffix


def tmp(suffix: str | None = None, keep: bool = True, dir: bool = False):
    if dir:
        directory_path = tempfile.mkdtemp()
        if not keep:
            if os.path.isdir(directory_path):
                shutil.rmtree(directory_path, ignore_errors=True)
            else:
                try:
                    os.remove(directory_path)
                except Exception:
                    pass
        return directory_path

    resolved_suffix = _resolve_tmp_suffix(suffix)

    with tempfile.NamedTemporaryFile(
        suffix=resolved_suffix, delete=False
    ) as temporary_file:
        temporary_name = temporary_file.name

    if not keep:
        try:
            os.remove(temporary_name)
        except Exception:
            pass

    return temporary_name


def _contains_glob_wildcards(path_part: str) -> bool:
    return any(character in path_part for character in "*?[")


def _safe_glob_pattern(pattern: str) -> str | None:
    if not isinstance(pattern, str):
        return None
    clean_pattern = unicodedata.normalize("NFKC", pattern)
    clean_pattern = " ".join(clean_pattern.split())
    if not clean_pattern or "\x00" in clean_pattern:
        return None
    if clean_pattern.startswith(("http://", "https://", "file://")):
        return None

    normalized_pattern = normalize_path(clean_pattern)
    pattern_path = Path(os.path.expanduser(normalized_pattern))
    anchor_parts: list[str] = []
    wildcard_parts: list[str] = []
    wildcard_seen = False

    for part in pattern_path.parts:
        if wildcard_seen or _contains_glob_wildcards(part):
            wildcard_seen = True
            wildcard_parts.append(part)
            continue
        anchor_parts.append(part)

    anchor_path = full_path(*anchor_parts) if anchor_parts else full_path(".")
    try:
        secure_path(anchor_path)
    except Exception:
        return None
    if not wildcard_parts:
        return anchor_path
    return normalize_path(os.path.join(anchor_path, *wildcard_parts))


def paths(*patterns: str) -> list[str]:
    from glob import glob as _glob

    collected_paths: list[str] = []
    for candidate_pattern in patterns:
        pattern = _safe_glob_pattern(candidate_pattern)
        if pattern is None:
            continue
        try:
            collected_paths.extend(list(_glob(pattern, recursive=True)))
        except Exception:
            pass
    return sorted(set(collected_paths))


def unique(arr: list[str]) -> list[str]:
    return sorted(set(arr))


def secure_path(
    path: list[str] | str,
    trust: list[str] | str | None = None,
    *,
    basename: bool = False,
    shell: bool = False,
) -> str:
    if not path or not isinstance(path, (str, list)):
        raise ValueError("Invalid path: must be a non-empty string or list.")

    if isinstance(path, list):
        path = full_path(*path)

    clean_str = unicodedata.normalize("NFKC", str(path))
    clean_str = " ".join(clean_str.split())
    if not clean_str:
        raise ValueError("Path is empty after cleaning.")

    strict_basename_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
    traversal_pattern = re.compile(r"(?:\.\.[/\\])|(?:\A[/\\]\.\.)")

    if basename:
        if not strict_basename_pattern.fullmatch(clean_str):
            raise ValueError(
                f"Security Error: Invalid basename format: {clean_str!r}"
            )
        result = clean_str
    else:
        if traversal_pattern.search(clean_str):
            raise ValueError(
                "Security Error: Path traversal characters detected."
            )

        try:
            result = full_path(clean_str)
        except Exception as error:
            raise ValueError(f"Failed to resolve path: {error}")

        resolved_path = Path(result)
        if trust is None:
            trust_bases: list[str] = []
        elif isinstance(trust, str):
            trust_bases = [trust]
        else:
            trust_bases = trust

        bases = [Path(full_path(base)) for base in trust_bases if base.strip()]
        if trust is None:
            current_directory = Path.cwd().resolve()
            if current_directory not in bases:
                bases.append(current_directory)

            temp_directory = Path(full_path(tempfile.gettempdir())).resolve()
            if temp_directory not in bases:
                bases.append(temp_directory)

        is_safe = any(_is_relative_to(resolved_path, base) for base in bases)
        if not is_safe:
            raise ValueError(
                f"Security Error: Path escapes allowed directories: {resolved_path}"
            )

    if shell:
        return shlex.quote(result)
    return result
