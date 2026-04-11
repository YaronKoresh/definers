from definers.cli.application import (
    build_parser,
    create_cli_command_registry,
    resolve_cli_runtime_state,
)


def test_cli_application_catalog_builds_registered_commands():
    command_registry = create_cli_command_registry(("chat", "audio"))

    assert "install" in command_registry
    assert command_registry["chat"].project == "chat"
    assert command_registry["start"].kind == "start"


def test_cli_application_parser_uses_catalog_definitions():
    parser = build_parser(
        "1.0.0",
        command_registry=create_cli_command_registry(("chat",)),
    )

    parsed = parser.parse_args(["install", "audio", "--type", "group"])

    assert parsed.command == "install"
    assert parsed.install_target == "audio"
    assert parsed.install_kind == "group"


def test_cli_application_runtime_state_exposes_bound_handlers():
    runtime_state = resolve_cli_runtime_state()

    assert callable(runtime_state.start)
    assert callable(runtime_state.music_video)
    assert callable(runtime_state.lyric_video)
    assert callable(runtime_state.install)
