import importlib
from typing import Any

_LAZY_SUBMODULES = {
    "arrays",
    "contracts",
    "dataset_shape_service",
    "dataset_source_loader",
    "dataset_tensor_builder",
    "dataset_value_loader",
    "exports",
    "loader_runtime",
    "loaders",
    "preparation",
    "runtime_patches",
    "tokenization",
    "vectorizers",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = tuple(sorted(_LAZY_SUBMODULES))
