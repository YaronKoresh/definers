import contextlib
import importlib
import io
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from types import ModuleType
from typing import Any


def _resolve_version() -> str:
    try:
        return _pkg_version("definers")
    except PackageNotFoundError:
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


def _install_optional_module_alias(
    module_name: str, fallback_module_name: str
) -> None:
    if module_name in sys.modules:
        return
    try:
        importlib.import_module(module_name)
        return
    except Exception:
        pass
    sys.modules[module_name] = importlib.import_module(
        f"{__name__}.{fallback_module_name}"
    )


def install_optional_module_aliases() -> None:
    _install_optional_module_alias("cv2", "opencv_compat")
    _install_optional_module_alias("datasets", "datasets_compat")
    _install_optional_module_alias(
        "googledrivedownloader", "googledrivedownloader_compat"
    )
    _install_optional_module_alias("playwright", "playwright_compat")
    _install_optional_module_alias("refiners", "refiners_compat")


__version__ = _resolve_version()
sox = load_sox_module()
install_optional_module_aliases()

_LAZY_SUBMODULES = {
    "application_ml",
    "audio",
    "cuda",
    "image",
    "logger",
    "media",
    "ml",
    "platform",
    "system",
    "text",
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
    "install_optional_module_aliases",
    "load_sox_module",
    "sox",
)
