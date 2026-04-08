from collections.abc import Iterable, Mapping

from definers.cli.cli_command_definition import (
    CliCommandDefinition,
)


class CliCommandRegistry:
    @staticmethod
    def normalize_cli_name(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        return normalized or None

    @classmethod
    def normalize_gui_commands(
        cls,
        gui_commands: Iterable[str],
    ) -> frozenset[str]:
        return frozenset(
            normalized
            for command in gui_commands
            if (normalized := cls.normalize_cli_name(command)) is not None
        )

    @classmethod
    def create_cli_command_registry(
        cls,
        gui_commands: Iterable[str],
    ) -> dict[str, CliCommandDefinition]:
        registry = {
            "start": CliCommandDefinition(name="start", kind="start"),
            "music-video": CliCommandDefinition(
                name="music-video",
                kind="music-video",
            ),
            "lyric-video": CliCommandDefinition(
                name="lyric-video",
                kind="lyric-video",
            ),
        }
        for command in sorted(cls.normalize_gui_commands(gui_commands)):
            if command in registry:
                raise ValueError(f"CLI command name conflict: {command}")
            registry[command] = CliCommandDefinition(
                name=command,
                kind="start",
                project=command,
            )
        return registry

    @staticmethod
    def get_known_cli_names(
        command_registry: Mapping[str, CliCommandDefinition],
        *,
        include_options: bool = True,
    ) -> tuple[str, ...]:
        known_names = tuple(command_registry)
        if not include_options:
            return known_names
        return known_names + ("--help", "--version")
