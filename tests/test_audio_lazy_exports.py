import importlib
import sys
import types

import pytest


def snapshot_audio_package() -> dict[str, types.ModuleType]:
    return {
        module_name: module
        for module_name, module in sys.modules.items()
        if module_name == "definers.audio"
        or module_name.startswith("definers.audio.")
    }


def unload_audio_package() -> None:
    for module_name in list(sys.modules):
        if module_name == "definers.audio" or module_name.startswith(
            "definers.audio."
        ):
            sys.modules.pop(module_name, None)


def restore_audio_package(snapshot: dict[str, types.ModuleType]) -> None:
    unload_audio_package()
    sys.modules.update(snapshot)


def test_audio_exports_are_bound_on_package_import():
    original_snapshot = snapshot_audio_package()
    unload_audio_package()

    audio_module = importlib.import_module("definers.audio")

    assert "audio_preview" in audio_module.__dict__
    assert "get_audio_duration" in audio_module.__dict__
    assert "value_to_keys" in audio_module.__dict__
    assert callable(audio_module.audio_preview)
    assert callable(audio_module.get_audio_duration)
    assert callable(audio_module.value_to_keys)
    restore_audio_package(original_snapshot)


def test_audio_exports_match_owner_modules():
    original_snapshot = snapshot_audio_package()
    unload_audio_package()

    audio_module = importlib.import_module("definers.audio")
    preview_module = importlib.import_module("definers.audio.preview")
    voice_module = importlib.import_module("definers.audio.voice")

    assert audio_module.audio_preview is preview_module.audio_preview
    assert audio_module.get_audio_duration is preview_module.get_audio_duration
    assert audio_module.value_to_keys is voice_module.value_to_keys
    restore_audio_package(original_snapshot)


def test_unknown_audio_export_raises_attribute_error_without_caching():
    original_snapshot = snapshot_audio_package()
    unload_audio_package()
    audio_module = importlib.import_module("definers.audio")

    with pytest.raises(AttributeError, match="missing_audio_feature"):
        getattr(audio_module, "missing_audio_feature")

    assert "missing_audio_feature" not in audio_module.__dict__
    restore_audio_package(original_snapshot)
