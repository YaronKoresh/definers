import contextlib
import importlib
import importlib.metadata as importlib_metadata
import io
import subprocess
import sys
from types import ModuleType
from typing import Any

from .optional_dependencies import install_import_hook

install_import_hook()


def _resolve_version() -> str:
    try:
        return importlib_metadata.version("definers")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


class MissingTransformer:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ImportError("sox is not available")


class MissingSoxModule:
    Transformer = MissingTransformer

    def __getattr__(self, name: str) -> Any:
        raise ImportError("sox module is not available")


def load_sox_module() -> ModuleType | MissingSoxModule:
    cached_module = sys.modules.get("sox")
    if cached_module is not None:
        return cached_module
    buffer = io.StringIO()
    original_run = subprocess.run
    original_popen = subprocess.Popen

    def silent_run(*args: object, **kwargs: object) -> Any:
        kwargs.setdefault("stderr", subprocess.DEVNULL)
        kwargs.setdefault("stdout", subprocess.DEVNULL)
        return original_run(*args, **kwargs)

    class SilentPopen(original_popen):
        def __init__(self, *args: object, **kwargs: object) -> None:
            kwargs.setdefault("stderr", subprocess.DEVNULL)
            kwargs.setdefault("stdout", subprocess.DEVNULL)
            super().__init__(*args, **kwargs)

    try:
        subprocess.run = silent_run
        subprocess.Popen = SilentPopen
        with (
            contextlib.redirect_stderr(buffer),
            contextlib.redirect_stdout(buffer),
        ):
            return importlib.import_module("sox")
    except Exception:
        return MissingSoxModule()
    finally:
        subprocess.run = original_run
        subprocess.Popen = original_popen


def has_sox() -> bool:
    return not isinstance(sox, MissingSoxModule)


__version__ = _resolve_version()
sox = load_sox_module()

_LAZY_SUBMODULES = {
    "audio",
    "cuda",
    "logger",
    "ml",
    "system",
    "text",
    "ui",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = (
    "__version__",
    "has_sox",
    "load_sox_module",
    "sox",
)
