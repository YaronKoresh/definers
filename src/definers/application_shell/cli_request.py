from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CliRequest:
    command: str | None
    project: str = "chat"
    audio: str = ""
    width: int = 0
    height: int = 0
    fps: int = 0
    background: str = ""
    lyrics: str = ""
    position: str = "bottom"
    max_dim: int = 640
    font_size: int = 70
    text_color: str = "white"
    stroke_color: str = "black"
    stroke_width: int = 2
    fade: float = 0.5
