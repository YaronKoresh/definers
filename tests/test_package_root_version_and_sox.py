import builtins
import contextlib
import importlib
import io
import sys
import types
from importlib.metadata import PackageNotFoundError
from unittest import mock

import pytest


def unload_package_root() -> None:
    sys.modules.pop("definers", None)
    sys.modules.pop("definers.optional_dependencies", None)
    sys.modules.pop("sox", None)


def test_package_root_uses_installed_version_and_available_sox():
    unload_package_root()
    original_import_module = importlib.import_module
    sox_module = types.ModuleType("sox")

    def fake_import_module(name: str, package: str | None = None):
        if name == "sox":
            return sox_module
        return original_import_module(name, package)

    with (
        mock.patch("importlib.metadata.version", return_value="9.8.7"),
        mock.patch("importlib.import_module", side_effect=fake_import_module),
    ):
        import definers

    assert definers.__version__ == "9.8.7"
    assert definers.sox is sox_module
    assert definers.has_sox() is True
    unload_package_root()


def test_package_root_falls_back_to_default_version_and_missing_sox_module():
    unload_package_root()
    original_import_module = importlib.import_module
    redirected_output = io.StringIO()

    def fake_import_module(name: str, package: str | None = None):
        if name == "sox":
            print("noisy stdout from sox import")
            print("noisy stderr from sox import", file=sys.stderr)
            raise RuntimeError("sox import failed")
        return original_import_module(name, package)

    with (
        mock.patch(
            "importlib.metadata.version",
            side_effect=PackageNotFoundError,
        ),
        mock.patch("importlib.import_module", side_effect=fake_import_module),
        contextlib.redirect_stdout(redirected_output),
        contextlib.redirect_stderr(redirected_output),
    ):
        import definers

    assert definers.__version__ == "0.0.0"
    assert definers.has_sox() is False
    assert redirected_output.getvalue() == ""
    with pytest.raises(ImportError, match="sox is not available"):
        definers.sox.Transformer()
    with pytest.raises(ImportError, match="sox module is not available"):
        getattr(definers.sox, "missing")
    unload_package_root()


def test_load_sox_module_uses_cached_sys_module_without_reimporting():
    unload_package_root()
    cached_sox = types.ModuleType("sox")
    sys.modules["sox"] = cached_sox

    with mock.patch("importlib.import_module", side_effect=AssertionError):
        import definers

    assert definers.sox is cached_sox
    assert definers.load_sox_module() is cached_sox
    assert definers.has_sox() is True
    unload_package_root()


def test_package_root_restores_optional_auto_install_hook():
    unload_package_root()
    original_import_module = importlib.import_module
    original_import = builtins.__import__
    sox_module = types.ModuleType("sox")

    def fake_import_module(name: str, package: str | None = None):
        if name == "sox":
            return sox_module
        return original_import_module(name, package)

    try:
        with mock.patch(
            "importlib.import_module", side_effect=fake_import_module
        ):
            import definers
            from definers import optional_dependencies

        assert (
            builtins.__import__
            is optional_dependencies._hooked_auto_install_import
        )
    finally:
        builtins.__import__ = original_import
        unload_package_root()
