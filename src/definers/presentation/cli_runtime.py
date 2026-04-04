from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from definers.application_shell.cli_command_definition import (
    CliCommandDefinition,
)


@dataclass(frozen=True, slots=True)
class CliRuntimeState:
    namespace: dict[str, Any]
    gui_registry: dict[str, Any]
    command_registry: dict[str, CliCommandDefinition]


class CliRuntimeService:
    @staticmethod
    def resolve_gui_registry():
        from importlib import import_module

        chat_module = import_module("definers.chat")
        namespace = vars(chat_module)
        registry = getattr(chat_module, "GUI_LAUNCHERS", {})
        if not isinstance(registry, dict):
            registry = {}
        return registry, namespace

    @staticmethod
    def build_cli_command_registry(namespace, *, registry):
        from definers.application_shell.commands import (
            create_cli_command_registry,
        )
        from definers.presentation.launchers import get_gui_project_names

        gui_project_names = get_gui_project_names(namespace, registry=registry)
        return create_cli_command_registry(gui_project_names)

    @classmethod
    def resolve_cli_runtime_state(cls) -> CliRuntimeState:
        registry, namespace = cls.resolve_gui_registry()
        command_registry = cls.build_cli_command_registry(
            namespace,
            registry=registry,
        )
        return CliRuntimeState(
            namespace=namespace,
            gui_registry=registry,
            command_registry=command_registry,
        )

    @classmethod
    def resolve_cli_command_registry(cls):
        return cls.resolve_cli_runtime_state().command_registry

    @staticmethod
    def resolve_cli_handlers():
        from definers.chat import lyric_video, music_video, start

        return start, music_video, lyric_video


resolve_gui_registry = CliRuntimeService.resolve_gui_registry
build_cli_command_registry = CliRuntimeService.build_cli_command_registry
resolve_cli_runtime_state = CliRuntimeService.resolve_cli_runtime_state
resolve_cli_command_registry = CliRuntimeService.resolve_cli_command_registry
resolve_cli_handlers = CliRuntimeService.resolve_cli_handlers
