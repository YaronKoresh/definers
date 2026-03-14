from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any, Protocol


class StartProjectPort(Protocol):
    def __call__(self, project: str) -> Any: ...


class MusicVideoPort(Protocol):
    def __call__(self, audio: str, width: int, height: int, fps: int) -> Any: ...


class LyricVideoPort(Protocol):
    def __call__(
        self,
        audio: str,
        background: str,
        lyrics: str,
        position: str,
        *,
        max_dim: int,
        font_size: int,
        text_color: str,
        stroke_color: str,
        stroke_width: int,
        fade_duration: float,
    ) -> Any: ...


class OutputPort(Protocol):
    def __call__(self, value: object) -> None: ...


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


@dataclass(frozen=True, slots=True)
class StartCommand:
    project: str


@dataclass(frozen=True, slots=True)
class MusicVideoCommand:
    audio: str
    width: int
    height: int
    fps: int


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


@dataclass(frozen=True, slots=True)
class UnknownCommand:
    name: str | None


CliCommand = StartCommand | MusicVideoCommand | LyricVideoCommand | UnknownCommand


def _normalize_cli_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def _normalize_gui_commands(gui_commands: Iterable[str]) -> frozenset[str]:
    return frozenset(
        normalized
        for command in gui_commands
        if (normalized := _normalize_cli_name(command)) is not None
    )


def coerce_cli_request(source: object | CliRequest) -> CliRequest:
    if isinstance(source, CliRequest):
        return source
    return CliRequest(
        command=getattr(source, "command", None),
        project=getattr(source, "project", "chat"),
        audio=getattr(source, "audio", ""),
        width=getattr(source, "width", 0),
        height=getattr(source, "height", 0),
        fps=getattr(source, "fps", 0),
        background=getattr(source, "background", ""),
        lyrics=getattr(source, "lyrics", ""),
        position=getattr(source, "position", "bottom"),
        max_dim=getattr(source, "max_dim", 640),
        font_size=getattr(source, "font_size", 70),
        text_color=getattr(source, "text_color", "white"),
        stroke_color=getattr(source, "stroke_color", "black"),
        stroke_width=getattr(source, "stroke_width", 2),
        fade=getattr(source, "fade", 0.5),
    )


def parse_cli_command(
    args: object | CliRequest,
    *,
    read_lyrics_text: Callable[[str], str],
    gui_commands: Iterable[str],
) -> CliCommand:
    request = coerce_cli_request(args)
    command = _normalize_cli_name(request.command)
    project = _normalize_cli_name(request.project) or "chat"
    normalized_gui_commands = _normalize_gui_commands(gui_commands)
    if command in (None, "start"):
        return StartCommand(project=project)
    if command in normalized_gui_commands:
        return StartCommand(project=command)
    if command == "music-video":
        return MusicVideoCommand(
            audio=request.audio,
            width=request.width,
            height=request.height,
            fps=request.fps,
        )
    if command == "lyric-video":
        return LyricVideoCommand(
            audio=request.audio,
            background=request.background,
            lyrics=read_lyrics_text(request.lyrics),
            position=request.position,
            max_dim=request.max_dim,
            font_size=request.font_size,
            text_color=request.text_color,
            stroke_color=request.stroke_color,
            stroke_width=request.stroke_width,
            fade=request.fade,
        )
    return UnknownCommand(name=command)


def dispatch_cli_command(
    command: CliCommand,
    *,
    start: StartProjectPort,
    music_video: MusicVideoPort,
    lyric_video: LyricVideoPort,
    output: OutputPort,
) -> int:
    if isinstance(command, StartCommand):
        return int(start(command.project))
    if isinstance(command, MusicVideoCommand):
        output(
            music_video(
                command.audio,
                command.width,
                command.height,
                command.fps,
            )
        )
        return 0
    if isinstance(command, LyricVideoCommand):
        output(
            lyric_video(
                command.audio,
                command.background,
                command.lyrics,
                command.position,
                max_dim=command.max_dim,
                font_size=command.font_size,
                text_color=command.text_color,
                stroke_color=command.stroke_color,
                stroke_width=command.stroke_width,
                fade_duration=command.fade,
            )
        )
        return 0
    output(f"unknown command {command.name}")
    return 1
