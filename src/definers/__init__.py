import contextlib
import importlib
import importlib.metadata as importlib_metadata
import io
import subprocess
import sys
from types import ModuleType
from typing import Any

from .optional_dependencies import install_import_hook
from .runtime_numpy import (
    bootstrap_runtime_numpy,
    get_array_module,
    get_numpy_module,
    is_cupy_backend,
    patch_numpy_runtime,
    runtime_backend_info,
    runtime_backend_name,
)

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
    return not getattr(sox, "__definers_missing_sox__", False)


__version__ = _resolve_version()
sox = load_sox_module()

_LAZY_SUBMODULES = {
    "audio",
    "chat",
    "cuda",
    "data",
    "file_ops",
    "image",
    "logger",
    "media",
    "ml",
    "model_installation",
    "optional_dependencies",
    "runtime_numpy",
    "system",
    "text",
    "ui",
    "video",
}


def _load_lazy_submodule(name: str) -> Any:
    module_name = f"{__name__}.{name}"
    module = importlib.import_module(module_name)
    sys.modules.setdefault(module_name, module)
    globals()[name] = module
    return module


class _DefinersModule(ModuleType):
    def __getattribute__(self, name: str) -> Any:
        if name in _LAZY_SUBMODULES:
            namespace = ModuleType.__getattribute__(self, "__dict__")
            package_name = ModuleType.__getattribute__(self, "__name__")
            module_name = f"{package_name}.{name}"
            loaded_module = sys.modules.get(module_name)
            cached_module = namespace.get(name)
            if loaded_module is not None:
                if cached_module is not loaded_module:
                    namespace[name] = loaded_module
                return loaded_module
            if isinstance(cached_module, ModuleType):
                if (
                    ModuleType.__getattribute__(cached_module, "__name__")
                    == module_name
                ):
                    sys.modules[module_name] = cached_module
                    return cached_module
        return ModuleType.__getattribute__(self, name)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        return _load_lazy_submodule(name)
    raise AttributeError(name)


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_SUBMODULES))


__all__ = (
    "__version__",
    "bootstrap_runtime_numpy",
    "get_array_module",
    "has_sox",
    "get_numpy_module",
    "is_cupy_backend",
    "load_sox_module",
    "patch_numpy_runtime",
    "runtime_backend_info",
    "runtime_backend_name",
    "sox",
)


sys.modules[__name__].__class__ = _DefinersModule
