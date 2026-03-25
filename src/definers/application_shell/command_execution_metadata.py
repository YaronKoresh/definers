from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CommandExecutionMetadata:
    requested_name: str | None
    resolved_name: str | None