from __future__ import annotations

from definers.cli.command_registry import get_known_cli_names
from definers.cli.request_coercer import coerce_cli_request

BUILTIN_COMMANDS = frozenset({"install"})


def build_parser(version, *, command_registry):
    import argparse

    parser = argparse.ArgumentParser(prog="definers")
    parser.add_argument("--version", action="version", version=version)

    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="launch a GUI by name")
    start_parser.add_argument(
        "project", nargs="?", default="chat", help="project to launch"
    )

    install_parser = subparsers.add_parser(
        "install",
        help="install optional runtime dependencies or prewarm model assets",
    )
    install_parser.add_argument(
        "install_target",
        nargs="?",
        default="",
        help="group, task, or module target to install",
    )
    install_parser.add_argument(
        "--type",
        choices=["group", "task", "module", "model-domain", "model-task"],
        default="group",
        dest="install_kind",
        help="target kind for the install command",
    )
    install_parser.add_argument(
        "--list",
        action="store_true",
        dest="install_list",
        help="list available install targets",
    )

    for name, definition in command_registry.items():
        if definition.kind != "start" or name == "start":
            continue
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


def find_unknown_command(argv, *, command_registry):
    from definers.ui.gui_registry import (
        normalize_gui_project_name,
    )

    if not argv:
        return None
    first = normalize_gui_project_name(argv[0])
    if first in BUILTIN_COMMANDS:
        return None
    if first in get_known_cli_names(command_registry) or first.startswith("-"):
        return None
    return first


def read_lyrics_text(lyrics):
    from pathlib import Path

    lyrics_path = Path(lyrics)
    if lyrics_path.is_file():
        try:
            return lyrics_path.read_text(encoding="utf-8")
        except OSError:
            return lyrics
    return lyrics


def build_cli_request(args):
    return coerce_cli_request(args)
