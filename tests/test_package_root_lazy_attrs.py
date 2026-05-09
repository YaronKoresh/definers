import importlib
import sys

import pytest

TEST_ROOT_EXPORTS = (
    "data",
    "image",
    "model_installation",
)
TEST_EXPLICIT_SUBMODULES = (
    "logger",
    "cuda",
    "ml",
    "system",
    "text",
)
TEST_PACKAGE_MODULES = (
    "definers",
    "sox",
    *(f"definers.{name}" for name in TEST_ROOT_EXPORTS),
    *(f"definers.{name}" for name in TEST_EXPLICIT_SUBMODULES),
)
MISSING_MODULE = object()


def snapshot_package_modules():
    return {
        module_name: sys.modules.get(module_name, MISSING_MODULE)
        for module_name in TEST_PACKAGE_MODULES
    }


def restore_package_modules(snapshot) -> None:
    for module_name, module in snapshot.items():
        if module is MISSING_MODULE:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = module


@pytest.fixture(autouse=True)
def preserve_package_modules():
    snapshot = snapshot_package_modules()
    try:
        yield
    finally:
        unload_package_root()
        restore_package_modules(snapshot)


def unload_package_root() -> None:
    sys.modules.pop("definers", None)
    sys.modules.pop("sox", None)
    for submodule_name in TEST_ROOT_EXPORTS:
        sys.modules.pop(f"definers.{submodule_name}", None)


def test_root_package_binds_direct_exports_on_import():
    unload_package_root()

    import definers

    for submodule_name in TEST_ROOT_EXPORTS:
        exported_module = getattr(definers, submodule_name)
        owner_module = importlib.import_module(f"definers.{submodule_name}")

        assert exported_module is owner_module
        assert definers.__dict__[submodule_name] is owner_module

    unload_package_root()


def test_unknown_attribute_raises_attribute_error_without_caching():
    unload_package_root()

    import definers

    with pytest.raises(AttributeError, match="missing_feature"):
        getattr(definers, "missing_feature")

    assert "missing_feature" not in definers.__dict__
    unload_package_root()


def test_root_direct_exports_are_in_all():
    unload_package_root()

    import definers

    for submodule_name in TEST_ROOT_EXPORTS:
        assert submodule_name in definers.__all__

    unload_package_root()


def test_root_direct_export_stays_bound_after_sys_modules_churn():
    unload_package_root()

    import definers

    first_data_module = definers.data
    sys.modules.pop("definers.data", None)

    assert definers.data is first_data_module
    reimported_data_module = importlib.import_module("definers.data")

    assert reimported_data_module is not first_data_module
    assert definers.data is reimported_data_module
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
    TEST_EXPLICIT_SUBMODULES,
)
def test_unbound_root_subpackages_are_imported_explicitly(attribute_name: str):
    unload_package_root()

    import definers

    assert attribute_name not in definers.__dict__

    imported_module = importlib.import_module(f"definers.{attribute_name}")

    assert imported_module.__name__ == f"definers.{attribute_name}"
    unload_package_root()
