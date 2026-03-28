import importlib
import sys
import types
from unittest import mock

import pytest


def unload_audio_package() -> None:
    for module_name in list(sys.modules):
        if module_name == "definers.audio" or module_name.startswith(
            "definers.audio."
        ):
            sys.modules.pop(module_name, None)


def test_lazy_audio_export_imports_once_and_caches_value():
    unload_audio_package()
    original_import_module = importlib.import_module
    preview_module = types.ModuleType("definers.audio.preview")
    preview_module.audio_preview = object()
    imported_names: list[str] = []

    def fake_import_module(name: str, package: str | None = None):
        imported_names.append(name)
        if name == "definers.audio.preview":
            return preview_module
        return original_import_module(name, package)

    with mock.patch("importlib.import_module", side_effect=fake_import_module):
        audio_module = original_import_module("definers.audio")

        assert "audio_preview" not in audio_module.__dict__

        first_value = audio_module.audio_preview
        second_value = audio_module.audio_preview

    assert first_value is preview_module.audio_preview
    assert second_value is preview_module.audio_preview
    assert audio_module.__dict__["audio_preview"] is preview_module.audio_preview
    assert imported_names.count("definers.audio.preview") == 1
    unload_audio_package()


def test_lazy_audio_export_handles_multiple_modules_independently():
    unload_audio_package()
    original_import_module = importlib.import_module
    preview_module = types.ModuleType("definers.audio.preview")
    preview_module.get_audio_duration = object()
    production_module = types.ModuleType("definers.audio.production")
    production_module.value_to_keys = object()
    imported_names: list[str] = []

    def fake_import_module(name: str, package: str | None = None):
        imported_names.append(name)
        if name == "definers.audio.preview":
            return preview_module
        if name == "definers.audio.production":
            return production_module
        return original_import_module(name, package)

    with mock.patch("importlib.import_module", side_effect=fake_import_module):
        audio_module = original_import_module("definers.audio")

        assert audio_module.get_audio_duration is preview_module.get_audio_duration
        assert audio_module.value_to_keys is production_module.value_to_keys

    assert imported_names.count("definers.audio.preview") == 1
    assert imported_names.count("definers.audio.production") == 1
    unload_audio_package()


def test_unknown_audio_export_raises_attribute_error_without_caching():
    unload_audio_package()
    audio_module = importlib.import_module("definers.audio")

    with pytest.raises(AttributeError, match="missing_audio_feature"):
        getattr(audio_module, "missing_audio_feature")

    assert "missing_audio_feature" not in audio_module.__dict__
    unload_audio_package()