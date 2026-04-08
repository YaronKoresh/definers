from dataclasses import dataclass

from definers.cli.command_execution_metadata import (
    CommandExecutionMetadata,
)


@dataclass(frozen=True, slots=True)
class LyricVideoCommand:
    audio: str
    background: str
    lyrics: str
    position: str
    max_dim: int
    font_size: int
    text_color: str
    stroke_color: str
    stroke_width: int
    fade: float
    metadata: CommandExecutionMetadata | None = None
