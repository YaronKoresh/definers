from dataclasses import dataclass

from definers.cli.command_execution_metadata import (
    CommandExecutionMetadata,
)


@dataclass(frozen=True, slots=True)
class UnknownCommand:
    name: str | None
    metadata: CommandExecutionMetadata | None = None
