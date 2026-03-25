from collections.abc import Callable, Mapping

from definers.application_shell.command_registry import CliCommandRegistry
from definers.application_shell.cli_command_definition import CliCommandDefinition
from definers.application_shell.cli_request import CliRequest
from definers.application_shell.command_execution_metadata import CommandExecutionMetadata
from definers.application_shell.lyric_video_command import LyricVideoCommand
from definers.application_shell.music_video_command import MusicVideoCommand
from definers.application_shell.request_coercer import CliRequestCoercer
from definers.application_shell.start_command import StartCommand
from definers.application_shell.unknown_command import UnknownCommand


class CliCommandParser:
    @classmethod
    def parse_cli_command(
        cls,
        args: object | CliRequest,
        *,
        read_lyrics_text: Callable[[str], str],
        command_registry: Mapping[str, CliCommandDefinition],
    ) -> StartCommand | MusicVideoCommand | LyricVideoCommand | UnknownCommand:
        request = CliRequestCoercer.coerce_cli_request(args)
        command = CliCommandRegistry.normalize_cli_name(request.command)
        project = CliCommandRegistry.normalize_cli_name(request.project) or "chat"
        if command in (None, "start"):
            return StartCommand(
                project=project,
                metadata=CommandExecutionMetadata(
                    requested_name=request.command,
                    resolved_name="start",
                ),
            )
        definition = command_registry.get(command)
        if definition is None:
            return UnknownCommand(
                name=command,
                metadata=CommandExecutionMetadata(
                    requested_name=request.command,
                    resolved_name=None,
                ),
            )
        metadata = CommandExecutionMetadata(
            requested_name=request.command,
            resolved_name=definition.name,
        )
        if definition.kind == "start":
            return StartCommand(
                project=definition.project or command,
                metadata=metadata,
            )
        if definition.kind == "music-video":
            return MusicVideoCommand(
                audio=request.audio,
                width=request.width,
                height=request.height,
                fps=request.fps,
                metadata=metadata,
            )
        if definition.kind == "lyric-video":
            return LyricVideoCommand(
                audio=request.audio,
                background=request.background,
                lyrics=read_lyrics_text(request.lyrics),
                position=request.position,
                max_dim=request.max_dim,
                font_size=request.font_size,
                text_color=request.text_color,
                stroke_color=request.stroke_color,
                stroke_width=request.stroke_width,
                fade=request.fade,
                metadata=metadata,
            )
        return UnknownCommand(name=command, metadata=metadata)