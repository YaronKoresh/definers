import importlib

_original_import_module = importlib.import_module


def _test_import_module(name, package=None):
    if name == "sox":
        raise ImportError
    return _original_import_module(name, package)


importlib.import_module = _test_import_module
