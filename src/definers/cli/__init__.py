import sys


def main(argv: list[str] | None = None) -> int:
    from definers import __version__

    from .dispatch import run_cli

    return run_cli(sys.argv[1:] if argv is None else argv, version=__version__)


if __name__ == "__main__":
    sys.exit(main())
