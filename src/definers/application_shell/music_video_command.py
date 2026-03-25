from dataclasses import dataclass

from definers.application_shell.command_execution_metadata import CommandExecutionMetadata


@dataclass(frozen=True, slots=True)
class MusicVideoCommand:
    audio: str
    width: int
    height: int
    fps: int
    metadata: CommandExecutionMetadata | None = None