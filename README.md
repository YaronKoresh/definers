# Definers

Definers is a Python toolkit for AI workflows, media processing, data operations, and system utilities.

## Core Modules

- `definers._capabilities` provides retry policies and circuit-breaker resilience boundaries.
- `definers._web` contains retrieval and transfer orchestration utilities.
- `definers._audio`, `definers._image`, `definers._video`, and `definers._ml` provide domain processing capabilities.
- `definers._system` and related utility modules provide environment and runtime helpers.

## Audio Dependency

*The `sox` Python package is an optional dependency used by
`load_as_numpy` for audio conversion.  To avoid spurious error messages on
import, `definers` now lazily loads the module and suppresses the
underlying CLI check.  If the `sox` binary is not installed or not on
`PATH`, the module falls back to a proxy that raises `ImportError` when
used; audio features will therefore return `None`.

Users on Windows encountering the message "'sox' is not recognized as an
internal or external command" no longer see it when simply running
`import definers`.

## Development Workflow

- Install development dependencies:
	- `pip install -e ".[dev]"`
- Run full local quality pipeline:
	- `poe check`
- Run tests only:
	- `poe test`

The `poe check` task executes cleanup, compile verification, linting, formatting, code sanitization, pre-commit hooks, and test execution.

## Automation

- Formatting and quote normalization are enforced by Ruff using double quotes.
- Repository hygiene is automated by `scripts/clean_workspace.py`.
- Python source sanitization for comments and docstrings is available in `scripts/sanitize_python.py`.
- CI validation runs via GitHub Actions in `.github/workflows/check.yml`.

## License Summary

This project is licensed under the MIT License.

- Allowed:
	- Private and commercial use.
	- Modification and redistribution.
	- Internal and external deployment.
- Required:
	- Keep copyright and license notices in redistributed copies.
- Not provided:
	- Warranty, liability coverage, or fitness guarantees.

See `LICENSE` for the full legal text.