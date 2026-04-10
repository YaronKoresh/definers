from dataclasses import dataclass

from definers.cli.cli_command_definition import (
    DEFAULT_INSTALL_KIND,
    DEFAULT_LYRIC_FADE,
    DEFAULT_LYRIC_FONT_SIZE,
    DEFAULT_LYRIC_MAX_DIM,
    DEFAULT_LYRIC_POSITION,
    DEFAULT_LYRIC_STROKE_COLOR,
    DEFAULT_LYRIC_STROKE_WIDTH,
    DEFAULT_LYRIC_TEXT_COLOR,
    DEFAULT_START_PROJECT,
)


@dataclass(frozen=True, slots=True)
class CliRequest:
    command: str | None
    project: str = DEFAULT_START_PROJECT
    install_target: str = ""
    install_kind: str = DEFAULT_INSTALL_KIND
    install_list: bool = False
    audio: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0
    background: str = ""
    lyrics: str = ""
    position: str = DEFAULT_LYRIC_POSITION
    max_dim: int = DEFAULT_LYRIC_MAX_DIM
    font_size: int = DEFAULT_LYRIC_FONT_SIZE
    text_color: str = DEFAULT_LYRIC_TEXT_COLOR
    stroke_color: str = DEFAULT_LYRIC_STROKE_COLOR
    stroke_width: int = DEFAULT_LYRIC_STROKE_WIDTH
    fade: float = DEFAULT_LYRIC_FADE
