from dataclasses import dataclass

from definers.application_shell.command_execution_metadata import CommandExecutionMetadata


@dataclass(frozen=True, slots=True)
class StartCommand:
    project: str
    metadata: CommandExecutionMetadata | None = None