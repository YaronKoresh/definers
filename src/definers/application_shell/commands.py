from definers.application_shell.cli_command_definition import (
    CliCommandDefinition,
)
from definers.application_shell.cli_request import CliRequest
from definers.application_shell.command_execution_metadata import (
    CommandExecutionMetadata,
)
from definers.application_shell.lyric_video_command import LyricVideoCommand
from definers.application_shell.lyric_video_port import LyricVideoPort
from definers.application_shell.music_video_command import MusicVideoCommand
from definers.application_shell.music_video_port import MusicVideoPort
from definers.application_shell.output_port import OutputPort
from definers.application_shell.start_command import StartCommand
from definers.application_shell.start_project_port import StartProjectPort
from definers.application_shell.unknown_command import UnknownCommand

CliCommand = (
    StartCommand | MusicVideoCommand | LyricVideoCommand | UnknownCommand
)


from definers.application_shell.command_dispatcher import CliCommandDispatcher
from definers.application_shell.command_parser import CliCommandParser
from definers.application_shell.command_registry import CliCommandRegistry
from definers.application_shell.request_coercer import CliRequestCoercer

create_cli_command_registry = CliCommandRegistry.create_cli_command_registry
get_known_cli_names = CliCommandRegistry.get_known_cli_names
coerce_cli_request = CliRequestCoercer.coerce_cli_request
parse_cli_command = CliCommandParser.parse_cli_command
dispatch_cli_command = CliCommandDispatcher.dispatch_cli_command
