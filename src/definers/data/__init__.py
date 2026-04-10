from __future__ import annotations

import importlib
from typing import Any

__all__ = (
    "arrays",
    "contracts",
    "datasets",
    "exports",
    "lightweight_datasets",
    "loader_runtime",
    "loaders",
    "preparation",
    "runtime_patches",
    "text",
    "tokenization",
    "vectorizers",
)


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
