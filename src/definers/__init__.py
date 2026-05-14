import contextlib
import importlib
import importlib.metadata as importlib_metadata
import io
import subprocess
import sys
from types import ModuleType
from typing import Any

from . import optional_dependencies
from .runtime_numpy import (
    bootstrap_runtime_numpy,
    get_array_module,
    get_numpy_module,
    is_cupy_backend,
    patch_numpy_runtime,
    runtime_backend_info,
    runtime_backend_name,
)

install_import_hook = optional_dependencies.install_import_hook
install_import_hook()
try:
    bootstrap_runtime_numpy()
except Exception:
    pass


def _resolve_version() -> str:
    try:
        return importlib_metadata.version("definers")
    except importlib_metadata.PackageNotFoundError:
        return "0.0.0"


class MissingTransformer:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ImportError("sox is not available")


class MissingSoxModule:
    __definers_missing_sox__ = True

    Transformer = MissingTransformer

    def __getattr__(self, name: str) -> Any:
        raise ImportError("sox module is not available")


def _is_trusted_sox_cache_entry(cached_module: object) -> bool:
    if getattr(cached_module, "__definers_missing_sox__", False) is True:
        return True
    if not isinstance(cached_module, ModuleType):
        return True
    return getattr(cached_module, "__spec__", None) is None


def load_sox_module() -> ModuleType | MissingSoxModule:
    cached_module = sys.modules.get("sox")
    if cached_module is not None and _is_trusted_sox_cache_entry(cached_module):
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
    return getattr(sox, "__definers_missing_sox__", False) is not True


__version__ = _resolve_version()
sox = load_sox_module()

from . import data, image, model_installation

__all__ = [glb for glb in globals() if not glb.startswith("_")]

_LAZY_SUBMODULES = frozenset(
    {
        "audio",
        "capabilities",
        "catalogs",
        "chat",
        "cli",
        "constants",
        "core",
        "cuda",
        "data",
        "database",
        "file_ops",
        "image",
        "logger",
        "media",
        "ml",
        "model_installation",
        "observability",
        "optional_dependencies",
        "os_utils",
        "path_utils",
        "regex_utils",
        "resilience",
        "runtime_numpy",
        "state",
        "system",
        "text",
        "ui",
        "video",
    }
)


def _load_public_submodule(name: str) -> Any:
    qualified_name = f"{__name__}.{name}"
    module = sys.modules.get(qualified_name)
    if module is None:
        module = importlib.import_module(qualified_name)
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return _load_public_submodule(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_SUBMODULES))


class _DefinersModule(ModuleType):
    def __getattribute__(self, name: str) -> Any:
        namespace = ModuleType.__getattribute__(self, "__dict__")
        lazy_submodules = namespace.get("_LAZY_SUBMODULES", ())
        if name in lazy_submodules:
            qualified_name = (
                f"{ModuleType.__getattribute__(self, '__name__')}.{name}"
            )
            bound_module = namespace.get(name)
            module = sys.modules.get(qualified_name)
            if module is None:
                if isinstance(bound_module, ModuleType):
                    return bound_module
                module = importlib.import_module(qualified_name)
            namespace[name] = module
            return module
        return ModuleType.__getattribute__(self, name)


sys.modules[__name__].__class__ = _DefinersModule
