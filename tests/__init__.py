import builtins
import importlib
import sys

_original_import_module = importlib.import_module
_original_import = builtins.__import__


def _install_optional_module_alias(
    module_name: str, fallback_module_name: str
) -> None:
    if module_name in sys.modules:
        return
    try:
        _original_import_module(module_name)
        return
    except Exception:
        pass
    sys.modules[module_name] = _original_import_module(
        f"definers.{fallback_module_name}"
    )


def _install_optional_shims() -> None:
    _install_optional_module_alias("cv2", "opencv_compat")
    _install_optional_module_alias("datasets", "datasets_compat")


def _is_stub_module(module):
    if module is None:
        return False
    return (
        getattr(module, "__file__", None) is None
        and getattr(module, "__spec__", None) is None
    )


def _clear_scipy_stubs():
    if not _is_stub_module(sys.modules.get("scipy")):
        return
    for name in tuple(sys.modules):
        if name == "scipy" or name.startswith("scipy."):
            sys.modules.pop(name, None)


def _test_import_module(name, package=None):
    if name == "sox":
        raise ImportError
    if name == "sklearn" or name.startswith("sklearn."):
        _clear_scipy_stubs()
    return _original_import_module(name, package)


def _test_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "sox":
        raise ImportError
    if name == "sklearn" or name.startswith("sklearn."):
        _clear_scipy_stubs()
    return _original_import(name, globals, locals, fromlist, level)


_install_optional_shims()
importlib.import_module = _test_import_module
builtins.__import__ = _test_import
