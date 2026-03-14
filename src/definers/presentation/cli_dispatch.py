import argparse
from pathlib import Path
from collections.abc import Sequence

from definers.application_shell.commands import (
    CliRequest,
    LyricVideoPort,
    MusicVideoPort,
    OutputPort,
    StartProjectPort,
    coerce_cli_request,
    dispatch_cli_command,
    parse_cli_command,
)
from definers.presentation.gui_registry import normalize_gui_project_name

GUI_PROJECTS = (
    "translate",
    "animation",
    "image",
    "chat",
    "faiss",
    "video",
    "audio",
    "train",
)

GUI_COMMANDS = tuple(normalize_gui_project_name(command) for command in GUI_PROJECTS)

KNOWN_COMMANDS = GUI_COMMANDS + (
    "start",
    "music-video",
    "lyric-video",
    "--help",
    "--version",
)


def build_parser(version: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="definers")
    parser.add_argument("--version", action="version", version=version)

    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="launch a GUI by name")
    start_parser.add_argument(
        "project", nargs="?", default="chat", help="project to launch"
    )

    for name in GUI_COMMANDS:
        subparsers.add_parser(name, help=f"launch the {name} interface")

    music_video_parser = subparsers.add_parser(
        "music-video", help="create a music visualizer video"
    )
    music_video_parser.add_argument("audio", help="input audio file path")
    music_video_parser.add_argument("width", type=int, help="video width")
    music_video_parser.add_argument("height", type=int, help="video height")
    music_video_parser.add_argument("fps", type=int, help="frames per second")

    lyric_video_parser = subparsers.add_parser(
        "lyric-video", help="create a lyric video"
    )
    lyric_video_parser.add_argument("audio", help="input audio file")
    lyric_video_parser.add_argument("background", help="background video/image")
    lyric_video_parser.add_argument("lyrics", help="lyrics text or file")
    lyric_video_parser.add_argument(
        "position", choices=["top", "center", "bottom"], default="bottom"
    )
    lyric_video_parser.add_argument("--max-dim", type=int, default=640)
    lyric_video_parser.add_argument("--font-size", type=int, default=70)
    lyric_video_parser.add_argument("--text-color", default="white")
    lyric_video_parser.add_argument("--stroke-color", default="black")
    lyric_video_parser.add_argument("--stroke-width", type=int, default=2)
    lyric_video_parser.add_argument("--fade", type=float, default=0.5)
    return parser


def find_unknown_command(argv: Sequence[str]) -> str | None:
    if not argv:
        return None
    first = normalize_gui_project_name(argv[0])
    if first in KNOWN_COMMANDS or first.startswith("-"):
        return None
    return first


def read_lyrics_text(lyrics: str) -> str:
    lyrics_path = Path(lyrics)
    if lyrics_path.is_file():
        try:
            return lyrics_path.read_text(encoding="utf-8")
        except OSError:
            return lyrics
    return lyrics


def build_cli_request(args: argparse.Namespace) -> CliRequest:
    return coerce_cli_request(args)


def resolve_cli_handlers() -> tuple[StartProjectPort, MusicVideoPort, LyricVideoPort]:
    from definers.chat import lyric_video, music_video, start

    return start, music_video, lyric_video


def dispatch_request(
    request: CliRequest,
    *,
    start: StartProjectPort,
    music_video: MusicVideoPort,
    lyric_video: LyricVideoPort,
    output: OutputPort,
) -> int:
    command = parse_cli_command(
        request,
        read_lyrics_text=read_lyrics_text,
        gui_commands=GUI_COMMANDS,
    )
    return dispatch_cli_command(
        command,
        start=start,
        music_video=music_video,
        lyric_video=lyric_video,
        output=output,
    )


def run_cli(argv: Sequence[str], *, version: str) -> int:
    parser = build_parser(version)

    unknown_command = find_unknown_command(argv)
    if unknown_command is not None:
        print(f"unknown command {unknown_command}")
        return 1

    args = parser.parse_args(list(argv))
    request = build_cli_request(args)
    start, music_video, lyric_video = resolve_cli_handlers()
    return dispatch_request(
        request,
        start=start,
        music_video=music_video,
        lyric_video=lyric_video,
        output=print,
    )