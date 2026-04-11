from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_START_PROJECT = "chat"
DEFAULT_INSTALL_KIND = "group"
INSTALL_KIND_CHOICES = (
    "group",
    "task",
    "module",
    "model-domain",
    "model-task",
)
DEFAULT_LYRIC_POSITION = "bottom"
DEFAULT_LYRIC_MAX_DIM = 640
DEFAULT_LYRIC_FONT_SIZE = 70
DEFAULT_LYRIC_TEXT_COLOR = "white"
DEFAULT_LYRIC_STROKE_COLOR = "black"
DEFAULT_LYRIC_STROKE_WIDTH = 2
DEFAULT_LYRIC_FADE = 0.5


@dataclass(frozen=True, slots=True)
class CliCommandDefinition:
    name: str
    kind: str
    help_text: str
    project: str | None = None
    configure_parser: Callable[[Any], None] | None = None
