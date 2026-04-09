from definers.cli.command_dispatcher import dispatch_cli_command
from definers.cli.command_parser import parse_cli_command
from definers.cli.parser import (
    build_cli_request,
    build_parser,
    find_unknown_command,
    read_lyrics_text,
)
from definers.cli.runtime import (
    resolve_cli_handlers,
    resolve_cli_runtime_state,
    resolve_gui_registry,
    resolve_optional_install_handler,
)


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
    command = parse_cli_command(
        request,
        read_lyrics_text=read_lyrics_text,
        command_registry=command_registry,
    )
    return dispatch_cli_command(
        command,
        start=start,
        music_video=music_video,
        lyric_video=lyric_video,
        install=install,
        output=output,
    )


def run_cli(argv, *, version):
    runtime_state = resolve_cli_runtime_state()
    command_registry = runtime_state.command_registry
    parser = build_parser(
        version,
        command_registry=command_registry,
    )

    unknown_command = find_unknown_command(
        argv,
        command_registry=command_registry,
    )
    if unknown_command is not None:
        print(f"unknown command {unknown_command}")
        return 1

    args = parser.parse_args(list(argv))
    request = build_cli_request(args)
    start, music_video, lyric_video = resolve_cli_handlers()
    install = resolve_optional_install_handler()
    return dispatch_request(
        request,
        command_registry=command_registry,
        start=start,
        music_video=music_video,
        lyric_video=lyric_video,
        install=install,
        output=print,
    )
