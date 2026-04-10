from __future__ import annotations

from definers.cli.application.catalog import (
    get_known_cli_names,
    iter_cli_command_definitions,
    normalize_cli_name,
)
from definers.cli.request_coercer import coerce_cli_request


def build_parser(version, *, command_registry):
    import argparse

    parser = argparse.ArgumentParser(prog="definers")
    parser.add_argument("--version", action="version", version=version)

    subparsers = parser.add_subparsers(dest="command")
    for definition in iter_cli_command_definitions(command_registry):
        command_parser = subparsers.add_parser(
            definition.name,
            help=definition.help_text,
        )
        if definition.configure_parser is not None:
            definition.configure_parser(command_parser)
    return parser


def find_unknown_command(argv, *, command_registry):
    if not argv:
        return None
    first = normalize_cli_name(argv[0])
    if first is None or first.startswith("-"):
        return None
    if first in get_known_cli_names(command_registry, include_options=False):
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
