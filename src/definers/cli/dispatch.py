from definers.cli.command_dispatcher import CliCommandDispatcher
from definers.cli.command_parser import CliCommandParser
from definers.cli.parser import (
    build_cli_request,
    build_parser,
    find_unknown_command,
    read_lyrics_text,
)
from definers.cli.runtime import (
    resolve_cli_command_registry,
    resolve_cli_handlers,
    resolve_cli_runtime_state,
    resolve_gui_registry,
    resolve_optional_install_handler,
)


class CliDispatchService:
    resolve_gui_registry = staticmethod(resolve_gui_registry)
    resolve_cli_command_registry = staticmethod(resolve_cli_command_registry)
    build_parser = staticmethod(build_parser)
    find_unknown_command = staticmethod(find_unknown_command)
    read_lyrics_text = staticmethod(read_lyrics_text)
    build_cli_request = staticmethod(build_cli_request)
    resolve_cli_handlers = staticmethod(resolve_cli_handlers)
    resolve_optional_install_handler = staticmethod(
        resolve_optional_install_handler
    )

    @staticmethod
    def dispatch_request(
        request,
        *,
        command_registry,
        start,
        music_video,
        lyric_video,
        install,
        output,
    ):
        command = CliCommandParser.parse_cli_command(
            request,
            read_lyrics_text=CliDispatchService.read_lyrics_text,
            command_registry=command_registry,
        )
        return CliCommandDispatcher.dispatch_cli_command(
            command,
            start=start,
            music_video=music_video,
            lyric_video=lyric_video,
            install=install,
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
        install = CliDispatchService.resolve_optional_install_handler()
        return CliDispatchService.dispatch_request(
            request,
            command_registry=command_registry,
            start=start,
            music_video=music_video,
            lyric_video=lyric_video,
            install=install,
            output=print,
        )


resolve_gui_registry = CliDispatchService.resolve_gui_registry
resolve_cli_command_registry = CliDispatchService.resolve_cli_command_registry
build_parser = CliDispatchService.build_parser
find_unknown_command = CliDispatchService.find_unknown_command
read_lyrics_text = CliDispatchService.read_lyrics_text
build_cli_request = CliDispatchService.build_cli_request
resolve_cli_handlers = CliDispatchService.resolve_cli_handlers
resolve_optional_install_handler = (
    CliDispatchService.resolve_optional_install_handler
)
dispatch_request = CliDispatchService.dispatch_request
run_cli = CliDispatchService.run_cli
