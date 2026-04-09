import importlib
from typing import Any

_LAZY_SUBMODULES = {
    "apps",
    "chat_handlers",
    "gui_entrypoints",
    "launchers",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = (
    "apps",
    "chat_handlers",
    "gui_entrypoints",
    "launchers",
)
