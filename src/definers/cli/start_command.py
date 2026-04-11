from dataclasses import dataclass

from definers.cli.command_execution_metadata import (
    CommandExecutionMetadata,
)


@dataclass(frozen=True, slots=True)
class StartCommand:
    project: str
    metadata: CommandExecutionMetadata | None = None
