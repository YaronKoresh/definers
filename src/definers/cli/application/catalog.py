from collections.abc import Iterable, Mapping

from definers.cli.cli_command_definition import (
    DEFAULT_INSTALL_KIND,
    DEFAULT_LYRIC_FADE,
    DEFAULT_LYRIC_FONT_SIZE,
    DEFAULT_LYRIC_MAX_DIM,
    DEFAULT_LYRIC_POSITION,
    DEFAULT_LYRIC_STROKE_COLOR,
    DEFAULT_LYRIC_STROKE_WIDTH,
    DEFAULT_LYRIC_TEXT_COLOR,
    DEFAULT_START_PROJECT,
    INSTALL_KIND_CHOICES,
    CliCommandDefinition,
)


def normalize_cli_name(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip().lower()
    return normalized or None


def normalize_gui_commands(gui_commands: Iterable[str]) -> frozenset[str]:
    return frozenset(
        normalized
        for command in gui_commands
        if (normalized := normalize_cli_name(command)) is not None
    )


def _configure_start_parser(parser) -> None:
    parser.add_argument(
        "project",
        nargs="?",
        default=DEFAULT_START_PROJECT,
        help="project to launch",
    )


def _configure_install_parser(parser) -> None:
    parser.add_argument(
        "install_target",
        nargs="?",
        default="",
        help="group, task, or module target to install",
    )
    parser.add_argument(
        "--type",
        choices=INSTALL_KIND_CHOICES,
        default=DEFAULT_INSTALL_KIND,
        dest="install_kind",
        help="target kind for the install command",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        dest="install_list",
        help="list available install targets",
    )


def _configure_music_video_parser(parser) -> None:
    parser.add_argument("audio", help="input audio file path")
    parser.add_argument("width", type=int, help="video width")
    parser.add_argument("height", type=int, help="video height")
    parser.add_argument("fps", type=int, help="frames per second")


def _configure_lyric_video_parser(parser) -> None:
    parser.add_argument("audio", help="input audio file")
    parser.add_argument("background", help="background video/image")
    parser.add_argument("lyrics", help="lyrics text or file")
    parser.add_argument(
        "position",
        choices=["top", "center", "bottom"],
        default=DEFAULT_LYRIC_POSITION,
    )
    parser.add_argument("--max-dim", type=int, default=DEFAULT_LYRIC_MAX_DIM)
    parser.add_argument(
        "--font-size",
        type=int,
        default=DEFAULT_LYRIC_FONT_SIZE,
    )
    parser.add_argument("--text-color", default=DEFAULT_LYRIC_TEXT_COLOR)
    parser.add_argument(
        "--stroke-color",
        default=DEFAULT_LYRIC_STROKE_COLOR,
    )
    parser.add_argument(
        "--stroke-width",
        type=int,
        default=DEFAULT_LYRIC_STROKE_WIDTH,
    )
    parser.add_argument("--fade", type=float, default=DEFAULT_LYRIC_FADE)


def iter_cli_command_definitions(
    command_registry: Mapping[str, CliCommandDefinition],
) -> tuple[CliCommandDefinition, ...]:
    seen_command_names: set[str] = set()
    ordered_definitions: list[CliCommandDefinition] = []
    for definition in command_registry.values():
        if definition.name in seen_command_names:
            continue
        seen_command_names.add(definition.name)
        ordered_definitions.append(definition)
    return tuple(ordered_definitions)


def resolve_cli_command_definition(
    command_registry: Mapping[str, CliCommandDefinition],
    command_name: str | None,
) -> CliCommandDefinition | None:
    normalized_command_name = normalize_cli_name(command_name)
    if normalized_command_name is None:
        return command_registry.get("start")
    return command_registry.get(normalized_command_name)


def create_cli_command_registry(
    gui_commands: Iterable[str],
) -> dict[str, CliCommandDefinition]:
    registry = {
        "start": CliCommandDefinition(
            name="start",
            kind="start",
            help_text="launch a GUI by name",
            configure_parser=_configure_start_parser,
        ),
        "install": CliCommandDefinition(
            name="install",
            kind="install",
            help_text=(
                "install optional runtime dependencies or prewarm model assets"
            ),
            configure_parser=_configure_install_parser,
        ),
        "music-video": CliCommandDefinition(
            name="music-video",
            kind="music-video",
            help_text="create a music visualizer video",
            configure_parser=_configure_music_video_parser,
        ),
        "lyric-video": CliCommandDefinition(
            name="lyric-video",
            kind="lyric-video",
            help_text="create a lyric video",
            configure_parser=_configure_lyric_video_parser,
        ),
    }
    for command in sorted(normalize_gui_commands(gui_commands)):
        if command in registry:
            raise ValueError(f"CLI command name conflict: {command}")
        registry[command] = CliCommandDefinition(
            name=command,
            kind="start",
            project=command,
            help_text=f"launch the {command} interface",
        )
    return registry


def get_known_cli_names(
    command_registry: Mapping[str, CliCommandDefinition],
    *,
    include_options: bool = True,
) -> tuple[str, ...]:
    known_names = tuple(command_registry)
    if not include_options:
        return known_names
    return known_names + ("--help", "--version")
