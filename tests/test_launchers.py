from unittest.mock import patch

from definers.ui.gui_registry import register_gui_launchers
from definers.ui.launchers import (
    get_gui_project_names,
    launch_installed_project,
    start_project,
)


def test_register_gui_launchers_normalizes_names():
    def launch_chat():
        return "chat"

    registry = register_gui_launchers({" Chat ": launch_chat})

    assert registry == {"chat": launch_chat}


def test_register_gui_launchers_rejects_unsupported_launcher_types():
    try:
        register_gui_launchers({"chat": None})
    except ValueError as exc:
        assert str(exc) == "Unsupported GUI launcher for project chat"
    else:
        raise AssertionError("expected ValueError")


def test_register_gui_launchers_rejects_string_launcher_without_namespace():
    try:
        register_gui_launchers({"chat": "_gui_chat"})
    except ValueError as exc:
        assert str(exc) == "GUI launcher chat requires a namespace"
    else:
        raise AssertionError("expected ValueError")


def test_start_project_uses_explicit_registry_first():
    calls: list[str] = []

    def launch_from_registry():
        calls.append("registry")
        return "registry"

    def launch_from_namespace():
        calls.append("namespace")
        return "namespace"

    registry = register_gui_launchers({"chat": launch_from_registry})
    namespace = {"_gui_chat": launch_from_namespace}

    result = start_project(
        "chat",
        namespace,
        lambda project_name: f"missing:{project_name}",
        registry=registry,
    )

    assert result == "registry"
    assert calls == ["registry"]


def test_start_project_falls_back_to_namespace():
    def launch_video():
        return "video"

    result = start_project(
        "video",
        {"_gui_video": launch_video},
        lambda project_name: f"missing:{project_name}",
    )

    assert result == "video"


def test_start_project_calls_missing_handler_for_unknown_project():
    result = start_project(
        "unknown",
        {},
        lambda project_name: f"missing:{project_name}",
        registry=register_gui_launchers({"chat": lambda: "chat"}),
    )

    assert result == "missing:unknown"


def test_get_gui_project_names_prefers_registry_when_available():
    registry = register_gui_launchers({" chat ": lambda: "chat"})

    names = get_gui_project_names(
        {"_gui_video": lambda: "video", "other": object()},
        registry=registry,
    )

    assert names == ("chat",)


def test_get_gui_project_names_uses_namespace_when_registry_missing():
    names = get_gui_project_names(
        {"_gui_video": lambda: "video", "other": object()},
    )

    assert names == ("video",)


def test_launch_installed_project_delegates_to_gui_entrypoints_start():
    with patch("definers.ui.launchers.import_module") as mock_import_module:
        mock_import_module.return_value.start.return_value = "started"

        result = launch_installed_project("chat")

    assert result == "started"
    mock_import_module.return_value.start.assert_called_once_with("chat")


def test_gui_entrypoints_registers_focused_task_surfaces():
    from definers.ui.gui_entrypoints import GUI_LAUNCHERS

    assert {
        "audio",
        "audio-mastering",
        "video",
        "video-composer",
        "image",
        "image-generate",
        "image-upscale",
        "image-title",
        "train",
    }.issubset(set(GUI_LAUNCHERS))
