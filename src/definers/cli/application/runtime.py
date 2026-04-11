from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from definers.cli.application.catalog import create_cli_command_registry
from definers.cli.cli_command_definition import CliCommandDefinition
from definers.cli.install_runtime_port import InstallRuntimePort
from definers.cli.lyric_video_port import LyricVideoPort
from definers.cli.music_video_port import MusicVideoPort
from definers.cli.start_project_port import StartProjectPort


@dataclass(frozen=True, slots=True)
class CliRuntimeState:
    namespace: dict[str, Any]
    gui_registry: dict[str, Any]
    command_registry: dict[str, CliCommandDefinition]
    start: StartProjectPort
    music_video: MusicVideoPort
    lyric_video: LyricVideoPort
    install: InstallRuntimePort


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
    from definers.cli.install import run_optional_install_command

    registry, namespace = resolve_gui_registry()
    command_registry = build_cli_command_registry(
        namespace,
        registry=registry,
    )
    start = namespace["start"]
    music_video = namespace["music_video"]
    lyric_video = namespace["lyric_video"]
    return CliRuntimeState(
        namespace=namespace,
        gui_registry=registry,
        command_registry=command_registry,
        start=start,
        music_video=music_video,
        lyric_video=lyric_video,
        install=run_optional_install_command,
    )


def resolve_cli_command_registry():
    return resolve_cli_runtime_state().command_registry


def resolve_cli_handlers():
    runtime_state = resolve_cli_runtime_state()
    return (
        runtime_state.start,
        runtime_state.music_video,
        runtime_state.lyric_video,
    )


def resolve_optional_install_handler():
    return resolve_cli_runtime_state().install
