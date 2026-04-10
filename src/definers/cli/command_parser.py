from collections.abc import Callable, Mapping

from definers.cli.application.catalog import (
    normalize_cli_name,
    resolve_cli_command_definition,
)
from definers.cli.cli_command_definition import (
    DEFAULT_START_PROJECT,
    CliCommandDefinition,
)
from definers.cli.cli_request import CliRequest
from definers.cli.command_execution_metadata import (
    CommandExecutionMetadata,
)
from definers.cli.install_command import InstallCommand
from definers.cli.lyric_video_command import LyricVideoCommand
from definers.cli.music_video_command import MusicVideoCommand
from definers.cli.request_coercer import coerce_cli_request
from definers.cli.start_command import StartCommand
from definers.cli.unknown_command import UnknownCommand


def parse_cli_command(
    args: object | CliRequest,
    *,
    read_lyrics_text: Callable[[str], str],
    command_registry: Mapping[str, CliCommandDefinition],
) -> (
    StartCommand
    | MusicVideoCommand
    | LyricVideoCommand
    | InstallCommand
    | UnknownCommand
):
    request = coerce_cli_request(args)
    command = normalize_cli_name(request.command)
    definition = resolve_cli_command_definition(
        command_registry,
        request.command,
    )
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
    if definition.kind == "install":
        return InstallCommand(
            target=request.install_target,
            target_kind=request.install_kind,
            list_only=bool(request.install_list),
            metadata=metadata,
        )
    if definition.kind == "start":
        return StartCommand(
            project=(
                definition.project
                or normalize_cli_name(request.project)
                or DEFAULT_START_PROJECT
            ),
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
