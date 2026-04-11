from __future__ import annotations

import importlib
from typing import Any

_MODULE_EXPORTS = {
    "image_helpers": "definers.image.helpers",
    "video_helpers": "definers.video.helpers",
}

_LAZY_SUBMODULES = {
    "transfer",
    "web_transfer",
}

__all__ = (
    "image_helpers",
    "transfer",
    "video_helpers",
    "web_transfer",
)


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    module_name = _MODULE_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(module_name)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()).union(_LAZY_SUBMODULES).union(__all__))
