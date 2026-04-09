from __future__ import annotations

from dataclasses import dataclass

from definers.cli.command_registry import get_known_cli_names


@dataclass(frozen=True, slots=True)
class CliHealthSnapshot:
    command_names: tuple[str, ...]
    gui_project_names: tuple[str, ...]
    direct_gui_command_names: tuple[str, ...]
    media_command_names: tuple[str, ...]
    known_names_with_options: tuple[str, ...]
    command_count: int
    gui_project_count: int


def collect_cli_health_snapshot(*, command_registry, gui_project_names):
    command_names = tuple(sorted(command_registry))
    normalized_gui_project_names = tuple(sorted(gui_project_names))
    direct_gui_command_names = tuple(
        sorted(
            name
            for name, definition in command_registry.items()
            if definition.kind == "start" and name != "start"
        )
    )
    media_command_names = tuple(
        sorted(
            name
            for name, definition in command_registry.items()
            if definition.kind != "start"
        )
    )
    known_names_with_options = tuple(get_known_cli_names(command_registry))
    return CliHealthSnapshot(
        command_names=command_names,
        gui_project_names=normalized_gui_project_names,
        direct_gui_command_names=direct_gui_command_names,
        media_command_names=media_command_names,
        known_names_with_options=known_names_with_options,
        command_count=len(command_names),
        gui_project_count=len(normalized_gui_project_names),
    )


def validate_cli_health_snapshot(snapshot: CliHealthSnapshot):
    required_commands = {"start", "music-video", "lyric-video"}
    missing_required_commands = tuple(
        sorted(required_commands.difference(snapshot.command_names))
    )
    if missing_required_commands:
        raise LookupError(
            "Missing required CLI commands: "
            + ", ".join(missing_required_commands)
        )
    missing_direct_gui_commands = tuple(
        sorted(
            set(snapshot.gui_project_names).difference(
                snapshot.direct_gui_command_names
            )
        )
    )
    if missing_direct_gui_commands:
        raise LookupError(
            "Missing direct GUI commands: "
            + ", ".join(missing_direct_gui_commands)
        )
    return snapshot


def collect_live_cli_health_snapshot():
    from definers.cli.runtime import resolve_cli_runtime_state
    from definers.ui.launchers import get_gui_project_names

    runtime_state = resolve_cli_runtime_state()
    gui_project_names = get_gui_project_names(
        runtime_state.namespace,
        registry=runtime_state.gui_registry,
    )
    return collect_cli_health_snapshot(
        command_registry=runtime_state.command_registry,
        gui_project_names=gui_project_names,
    )


def run_cli_health_check():
    return validate_cli_health_snapshot(collect_live_cli_health_snapshot())
