from __future__ import annotations

import importlib
from typing import Any

from .exports_registry import AUDIO_EXPORTS, __all__


def __getattr__(name: str) -> Any:
    module_name = AUDIO_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(name)
    module = importlib.import_module(f"{__name__}.{module_name}")
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
