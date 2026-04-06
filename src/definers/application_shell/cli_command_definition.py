from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CliCommandDefinition:
    name: str
    kind: str
    project: str | None = None
