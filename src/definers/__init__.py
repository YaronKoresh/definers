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

_LEGACY_MODULE_ALIASES = {
    "application_data": "data",
    "ml_health": "ml.health_api",
    "ml_regression": "ml.regression_api",
    "ml_text": "ml.text.api",
    "platform": "system",
    "system_archives": "system.archives",
    "system_installation": "system.installation",
    "system_threads": "system.threads",
    "video_gui": "video.gui",
}


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


class LazyModuleAlias(ModuleType):
    def __init__(self, alias_name: str, target_name: str) -> None:
        super().__init__(alias_name)
        self.__dict__["_target_name"] = target_name

    def _load_target_module(self) -> ModuleType:
        module = importlib.import_module(self.__dict__["_target_name"])
        sys.modules[self.__name__] = module
        return module

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load_target_module(), name)


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
        module_spec = importlib.util.find_spec(module_name)
    except Exception:
        module_spec = None
    if module_spec is not None:
        return
    sys.modules.setdefault(
        module_name,
        LazyModuleAlias(
            module_name,
            f"{__name__}.{fallback_module_name}",
        ),
    )


def install_optional_module_aliases() -> None:
    _install_optional_module_alias("cv2", "opencv_compat")
    _install_optional_module_alias("datasets", "datasets_compat")
    _install_optional_module_alias(
        "googledrivedownloader", "googledrivedownloader_compat"
    )
    _install_optional_module_alias("playwright", "playwright_compat")
    _install_optional_module_alias("refiners", "refiners_compat")


def install_legacy_module_aliases() -> None:
    for alias_name, target_name in _LEGACY_MODULE_ALIASES.items():
        sys.modules.setdefault(
            f"{__name__}.{alias_name}",
            LazyModuleAlias(
                f"{__name__}.{alias_name}",
                f"{__name__}.{target_name}",
            ),
        )


__version__ = _resolve_version()
sox = load_sox_module()
install_optional_module_aliases()
install_legacy_module_aliases()

_LAZY_SUBMODULES = {
    "application_data",
    "audio",
    "catalogs",
    "chat",
    "cli",
    "cuda",
    "data",
    "database",
    "image",
    "logger",
    "media",
    "ml",
    "observability",
    "optional_dependencies",
    "platform",
    "resilience",
    "state",
    "system",
    "text",
    "ui",
    "video",
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
