from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from definers.cli.cli_command_definition import (
    CliCommandDefinition,
)
from definers.cli.command_registry import create_cli_command_registry


@dataclass(frozen=True, slots=True)
class CliRuntimeState:
    namespace: dict[str, Any]
    gui_registry: dict[str, Any]
    command_registry: dict[str, CliCommandDefinition]


def resolve_gui_registry():
    from importlib import import_module

    gui_entrypoints_module = import_module("definers.ui.gui_entrypoints")
    namespace = vars(gui_entrypoints_module)
    registry = getattr(gui_entrypoints_module, "GUI_LAUNCHERS", {})
    if not isinstance(registry, dict):
        registry = {}
    return registry, namespace


def build_cli_command_registry(namespace, *, registry):
    from definers.ui.launchers import get_gui_project_names

    gui_project_names = get_gui_project_names(namespace, registry=registry)
    return create_cli_command_registry(gui_project_names)


def resolve_cli_runtime_state() -> CliRuntimeState:
    registry, namespace = resolve_gui_registry()
    command_registry = build_cli_command_registry(
        namespace,
        registry=registry,
    )
    return CliRuntimeState(
        namespace=namespace,
        gui_registry=registry,
        command_registry=command_registry,
    )


def resolve_cli_command_registry():
    return resolve_cli_runtime_state().command_registry


def resolve_cli_handlers():
    from definers.ui.gui_entrypoints import (
        lyric_video,
        music_video,
        start,
    )

    return start, music_video, lyric_video


def resolve_optional_install_handler():
    from definers.cli.install import (
        run_optional_install_command,
    )

    return run_optional_install_command
