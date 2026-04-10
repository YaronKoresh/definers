import importlib
import sys
import types
from unittest import mock

import pytest

TEST_LAZY_SUBMODULES = (
    "audio",
    "cuda",
    "logger",
    "ml",
    "system",
    "text",
)


def unload_package_root() -> None:
    sys.modules.pop("definers", None)
    sys.modules.pop("sox", None)
    for submodule_name in TEST_LAZY_SUBMODULES:
        sys.modules.pop(f"definers.{submodule_name}", None)


def test_lazy_attribute_imports_once_and_caches_module():
    unload_package_root()
    original_import_module = importlib.import_module
    sox_module = types.ModuleType("sox")
    audio_module = types.ModuleType("definers.audio")
    imported_names: list[str] = []

    def fake_import_module(name: str, package: str | None = None):
        imported_names.append(name)
        if name == "sox":
            return sox_module
        if name == "definers.audio":
            return audio_module
        return original_import_module(name, package)

    with mock.patch("importlib.import_module", side_effect=fake_import_module):
        import definers

        assert "audio" not in definers.__dict__

        first_value = definers.audio
        second_value = definers.audio

    assert first_value is audio_module
    assert second_value is audio_module
    assert definers.__dict__["audio"] is audio_module
    assert imported_names.count("definers.audio") == 1
    unload_package_root()


def test_unknown_attribute_raises_attribute_error_without_caching():
    unload_package_root()
    sox_module = types.ModuleType("sox")

    with mock.patch("importlib.import_module", return_value=sox_module):
        import definers

    with pytest.raises(AttributeError, match="missing_feature"):
        getattr(definers, "missing_feature")

    assert "missing_feature" not in definers.__dict__
    unload_package_root()


def test_multiple_lazy_attributes_use_package_qualified_import_names():
    unload_package_root()
    original_import_module = importlib.import_module
    sox_module = types.ModuleType("sox")
    text_module = types.ModuleType("definers.text")
    system_module = types.ModuleType("definers.system")
    imported_names: list[str] = []

    def fake_import_module(name: str, package: str | None = None):
        imported_names.append(name)
        if name == "sox":
            return sox_module
        if name == "definers.text":
            return text_module
        if name == "definers.system":
            return system_module
        return original_import_module(name, package)

    with mock.patch("importlib.import_module", side_effect=fake_import_module):
        import definers

        assert definers.text is text_module
        assert definers.system is system_module

    assert imported_names.count("definers.text") == 1
    assert imported_names.count("definers.system") == 1
    unload_package_root()


def test_lazy_attribute_refreshes_after_submodule_sys_modules_churn():
    unload_package_root()

    import definers

    first_data_module = definers.data
    sys.modules.pop("definers.data", None)

    assert definers.data is first_data_module
    assert definers.data is importlib.import_module("definers.data")
    assert sys.modules["definers.data"] is definers.data
    unload_package_root()


def test_data_package_exposes_submodules_without_root_routing():
    unload_package_root()

    import definers.data as data_package

    first_value = data_package.arrays
    second_value = data_package.arrays

    assert first_value is second_value
    assert first_value.__name__ == "definers.data.arrays"
    assert data_package.__dict__["arrays"] is first_value
    unload_package_root()


def test_removed_legacy_root_aliases_raise_attribute_error():
    unload_package_root()

    import definers

    with pytest.raises(AttributeError, match="application_data"):
        getattr(definers, "application_data")
    with pytest.raises(AttributeError, match="platform"):
        getattr(definers, "platform")

    unload_package_root()


@pytest.mark.parametrize(
    "attribute_name",
    ["logger", "cuda", "ml", "system"],
)
def test_lazy_attribute_caches_module_for_rca_regressions(attribute_name: str):
    unload_package_root()
    original_import_module = importlib.import_module
    sox_module = types.ModuleType("sox")
    lazy_module = types.ModuleType(f"definers.{attribute_name}")
    imported_names: list[str] = []

    def fake_import_module(name: str, package: str | None = None):
        imported_names.append(name)
        if name == "sox":
            return sox_module
        if name == f"definers.{attribute_name}":
            return lazy_module
        return original_import_module(name, package)

    with mock.patch("importlib.import_module", side_effect=fake_import_module):
        import definers

        assert attribute_name not in definers.__dict__

        first_value = getattr(definers, attribute_name)
        second_value = getattr(definers, attribute_name)

    assert first_value is lazy_module
    assert second_value is lazy_module
    assert definers.__dict__[attribute_name] is lazy_module
    assert imported_names.count(f"definers.{attribute_name}") == 1
    unload_package_root()
