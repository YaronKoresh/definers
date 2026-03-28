from definers.presentation.cli_parser import (
    build_cli_request,
    build_parser,
    find_unknown_command,
    read_lyrics_text,
)
from definers.presentation.cli_runtime import (
    resolve_cli_command_registry,
    resolve_cli_handlers,
    resolve_cli_runtime_state,
    resolve_gui_registry,
)


class CliDispatchService:
    resolve_gui_registry = staticmethod(resolve_gui_registry)
    resolve_cli_command_registry = staticmethod(resolve_cli_command_registry)
    build_parser = staticmethod(build_parser)
    find_unknown_command = staticmethod(find_unknown_command)
    read_lyrics_text = staticmethod(read_lyrics_text)
    build_cli_request = staticmethod(build_cli_request)
    resolve_cli_handlers = staticmethod(resolve_cli_handlers)

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
        runtime_state = resolve_cli_runtime_state()
        command_registry = runtime_state.command_registry
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
