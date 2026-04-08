from __future__ import annotations

import shutil
from pathlib import Path

from definers.constants import KNOWN_EXTENSIONS, ai_model_formats

from . import paths as _paths


def secure_path(
    path: list | str,
    trust: list | str | None = None,
    *,
    basename: bool = False,
    shell: bool = False,
) -> str:
    return _paths.secure_path(path, trust, basename=basename, shell=shell)


def get_ext(input_path):
    ext = str(Path(str(input_path)).suffix).strip(".").lower()
    if not ext:
        raise ValueError(
            f"Could not determine file extension for: {input_path}"
        )
    if ext in KNOWN_EXTENSIONS:
        return ext
    raise ValueError("Unsupported file extension")


def is_ai_model(input_path):
    extension = get_ext(input_path)
    return extension in ai_model_formats


def compress(dir: str, format: str = "zip", keep_name: bool = True):
    if keep_name:
        target = str(Path(dir).parent) + "/" + str(Path(dir).name)
    else:
        from definers.text import random_string

        target = str(Path(dir).parent) + "/" + random_string()
    shutil.make_archive(
        target,
        format,
        str(Path(dir).parent),
        str(Path(dir).name),
    )
    return target + "." + format


def extract(arcv, dest=None, format=None):
    if not dest:
        dest = str(Path(arcv).parent)
    if format:
        shutil.unpack_archive(arcv, dest, format)
        return
    shutil.unpack_archive(arcv, dest)
