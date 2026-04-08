import importlib
import sys
from typing import Any

_SUBMODULES = {
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
}

_LEGACY_MODULE_ALIASES = {
    "dataset_shape_service": "datasets.shape",
    "dataset_source_loader": "datasets.source",
    "dataset_tensor_builder": "datasets.tensor",
    "dataset_value_loader": "datasets.value",
    "text_vectorizer": "text.vectorizer",
}


def _import_child(name: str):
    return importlib.import_module(f"{__name__}.{name}")


def _install_legacy_module_aliases() -> None:
    for alias_name, target_name in _LEGACY_MODULE_ALIASES.items():
        sys.modules.setdefault(
            f"{__name__}.{alias_name}",
            _import_child(target_name),
        )


arrays = _import_child("arrays")
contracts = _import_child("contracts")
datasets = _import_child("datasets")
exports = _import_child("exports")
lightweight_datasets = _import_child("lightweight_datasets")
loader_runtime = _import_child("loader_runtime")
loaders = _import_child("loaders")
preparation = _import_child("preparation")
runtime_patches = _import_child("runtime_patches")
text = _import_child("text")
tokenization = _import_child("tokenization")
vectorizers = _import_child("vectorizers")

_install_legacy_module_aliases()


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = _import_child(name)
        globals()[name] = module
        return module
    target_name = _LEGACY_MODULE_ALIASES.get(name)
    if target_name is not None:
        module = _import_child(target_name)
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = tuple(sorted(set(_SUBMODULES).union(_LEGACY_MODULE_ALIASES)))
