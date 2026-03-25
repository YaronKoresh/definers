class CliDispatchService:
    @staticmethod
    def resolve_gui_registry():
        from importlib import import_module

        chat_module = import_module("definers.chat")
        namespace = vars(chat_module)
        registry = getattr(chat_module, "GUI_LAUNCHERS", {})
        if not isinstance(registry, dict):
            registry = {}
        return registry, namespace

    @staticmethod
    def resolve_cli_command_registry():
        from definers.application_shell.commands import (
            create_cli_command_registry,
        )
        from definers.presentation.launchers import get_gui_project_names

        registry, namespace = CliDispatchService.resolve_gui_registry()
        gui_project_names = get_gui_project_names(namespace, registry=registry)
        return create_cli_command_registry(gui_project_names)

    @staticmethod
    def build_parser(version, *, command_registry):
        import argparse

        parser = argparse.ArgumentParser(prog="definers")
        parser.add_argument("--version", action="version", version=version)

        subparsers = parser.add_subparsers(dest="command")

        start_parser = subparsers.add_parser(
            "start", help="launch a GUI by name"
        )
        start_parser.add_argument(
            "project", nargs="?", default="chat", help="project to launch"
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
        music_video_parser.add_argument(
            "fps", type=int, help="frames per second"
        )

        lyric_video_parser = subparsers.add_parser(
            "lyric-video", help="create a lyric video"
        )
        lyric_video_parser.add_argument("audio", help="input audio file")
        lyric_video_parser.add_argument(
            "background", help="background video/image"
        )
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

    @staticmethod
    def find_unknown_command(argv, *, command_registry):
        from definers.application_shell.commands import get_known_cli_names
        from definers.presentation.gui_registry import (
            normalize_gui_project_name,
        )

        if not argv:
            return None
        first = normalize_gui_project_name(argv[0])
        if first in get_known_cli_names(command_registry) or first.startswith(
            "-"
        ):
            return None
        return first

    @staticmethod
    def read_lyrics_text(lyrics):
        from pathlib import Path

        lyrics_path = Path(lyrics)
        if lyrics_path.is_file():
            try:
                return lyrics_path.read_text(encoding="utf-8")
            except OSError:
                return lyrics
        return lyrics

    @staticmethod
    def build_cli_request(args):
        from definers.application_shell.commands import coerce_cli_request

        return coerce_cli_request(args)

    @staticmethod
    def resolve_cli_handlers():
        from definers.chat import lyric_video, music_video, start

        return start, music_video, lyric_video

    @staticmethod
    def dispatch_request(
        request,
        *,
        command_registry,
        start,
        music_video,
        lyric_video,
        output,
    ):
        from definers.application_shell.commands import (
            dispatch_cli_command,
            parse_cli_command,
        )

        command = parse_cli_command(
            request,
            read_lyrics_text=CliDispatchService.read_lyrics_text,
            command_registry=command_registry,
        )
        return dispatch_cli_command(
            command,
            start=start,
            music_video=music_video,
            lyric_video=lyric_video,
            output=output,
        )

    @staticmethod
    def run_cli(argv, *, version):
        command_registry = CliDispatchService.resolve_cli_command_registry()
        parser = CliDispatchService.build_parser(
            version,
            command_registry=command_registry,
        )

        unknown_command = CliDispatchService.find_unknown_command(
            argv,
            command_registry=command_registry,
        )
        if unknown_command is not None:
            print(f"unknown command {unknown_command}")
            return 1

        args = parser.parse_args(list(argv))
        request = CliDispatchService.build_cli_request(args)
        start, music_video, lyric_video = (
            CliDispatchService.resolve_cli_handlers()
        )
        return CliDispatchService.dispatch_request(
            request,
            command_registry=command_registry,
            start=start,
            music_video=music_video,
            lyric_video=lyric_video,
            output=print,
        )


resolve_gui_registry = CliDispatchService.resolve_gui_registry
resolve_cli_command_registry = CliDispatchService.resolve_cli_command_registry
build_parser = CliDispatchService.build_parser
find_unknown_command = CliDispatchService.find_unknown_command
read_lyrics_text = CliDispatchService.read_lyrics_text
build_cli_request = CliDispatchService.build_cli_request
resolve_cli_handlers = CliDispatchService.resolve_cli_handlers
dispatch_request = CliDispatchService.dispatch_request
run_cli = CliDispatchService.run_cli
