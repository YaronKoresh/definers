from definers.cli.application.catalog import (
    create_cli_command_registry,
    get_known_cli_names,
    iter_cli_command_definitions,
    normalize_cli_name,
    normalize_gui_commands,
    resolve_cli_command_definition,
)
from definers.cli.application.parsing import (
    build_cli_request,
    build_parser,
    find_unknown_command,
    read_lyrics_text,
)
from definers.cli.application.runtime import (
    CliRuntimeState,
    resolve_cli_command_registry,
    resolve_cli_handlers,
    resolve_cli_runtime_state,
    resolve_gui_registry,
    resolve_optional_install_handler,
)


def dispatch_request(*args, **kwargs):
    from definers.cli.application.service import (
        dispatch_request as dispatch_request_impl,
    )

    return dispatch_request_impl(*args, **kwargs)


def run_cli(*args, **kwargs):
    from definers.cli.application.service import run_cli as run_cli_impl

    return run_cli_impl(*args, **kwargs)


__all__ = (
    "CliRuntimeState",
    "build_cli_request",
    "build_parser",
    "create_cli_command_registry",
    "dispatch_request",
    "find_unknown_command",
    "get_known_cli_names",
    "iter_cli_command_definitions",
    "normalize_cli_name",
    "normalize_gui_commands",
    "read_lyrics_text",
    "resolve_cli_command_definition",
    "resolve_cli_command_registry",
    "resolve_cli_handlers",
    "resolve_cli_runtime_state",
    "resolve_gui_registry",
    "resolve_optional_install_handler",
    "run_cli",
)
