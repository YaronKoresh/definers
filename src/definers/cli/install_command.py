from dataclasses import dataclass

from definers.cli.command_execution_metadata import (
    CommandExecutionMetadata,
)


@dataclass(frozen=True, slots=True)
class InstallCommand:
    target: str
    target_kind: str
    list_only: bool
    metadata: CommandExecutionMetadata | None = None
