# Definers

Definers is a Python toolkit for AI workflows, media processing, data operations, and system utilities.

## Core Modules

- `definers._capabilities` provides retry policies and circuit-breaker resilience boundaries.
- `definers._web` contains retrieval and transfer orchestration utilities.
- `definers._audio`, `definers._image`, `definers._video`, and `definers._ml` provide domain processing capabilities.
- `definers._system` and related utility modules provide environment and runtime helpers.

## Security & Performance

Regular-expression operations are centralized in `definers.regex_utils`.
Consumers should never compile patterns containing raw user data directly;
use `regex_utils.escape`, `escape_and_compile`, or the thin wrappers
`sub`/`fullmatch` which enforce a maximum pattern length and reject
nested quantifiers to prevent catastrophic backtracking.  User-facing
textboxes are also guarded by `MAX_INPUT_LENGTH` and maximum consecutive
spaces checks to reduce attack surface.

## Audio Dependency

The `sox` Python package is an optional dependency used by
`load_as_numpy` for audio conversion.  To avoid spurious error messages on
import, `definers` now lazily loads the module and suppresses the
underlying CLI check.  If the `sox` binary is not installed or not on
`PATH`, the module falls back to a proxy that raises `ImportError` when
used; audio features will therefore return `None`.

Users on Windows encountering the message "'sox' is not recognized as an
internal or external command" no longer see it when simply running
`import definers`.

## Available GUIs

The package exposes a simple launcher in `definers._chat.start()` which
brings up a lightweight Gradio interface for various subprojects.  The list
of valid project names is computed dynamically from the available `_gui_`
helpers in the module; new interfaces can be added without any changes to
`start` itself.

At the time of writing, available GUIs include:

- `translate` – text translation and image captioning
- `animation` – manual chunked image-to-animation generator
- `image` – text‑to-image generation and upscaling tools
- `chat` – multimodal AI chatbot
- `faiss` – download a prebuilt FAISS wheel
- `video` – AI video architect (composition/layout engine)
- `audio` – AI-powered audio production (formerly "Audio Studio Pro")
- `train` – train or predict with custom models (formerly "teachless")

Each GUI is loaded lazily; unknown project names are reported via
`definers._system.catch()` which may raise or log depending on configuration.

## Command Execution

All external programs are run through the `definers.run()` helper, which
delegates to `run_linux()` or `run_windows()` depending on the platform.
Calls should **prefer list form** (`["cmd", "arg1", ...]`) to avoid
shell‑injection hazards.  When a multi‑line script or shell features are
needed the list may be `[
    "bash", "-lc", "first && second",
]` on Unix.  ``run_linux`` and ``run_windows`` will raise a
`ValueError` if an unsafe string containing characters such as `;` or `&`
is passed, encouraging correct usage.  Tests in the suite expect
arguments to be lists accordingly, so be sure to update them when
changing invocation style.

## Development Workflow

- Install development dependencies:
	- `pip install -e ".[dev]"`
- Run full local quality pipeline:
	- `poe check`
- Run tests only:
	- `poe test`

The `poe check` task executes cleanup, compile verification, linting, formatting, code sanitization, pre-commit hooks, and test execution.

## Example: automatic data preparation and training

```python
import definers

# prepare CSV features automatically; returns loaders and metadata
data = definers.prepare_data(
    features=["/path/to/f1.csv", "/path/to/f2.csv"],
    drop=["unneeded_column"],
    order_by=lambda x: len(str(x)),    # simple ordering strategy
    stratify="label",
    val_frac=0.1,
    test_frac=0.1,
    batch_size=32,
)
print(data.metadata)

# the train entrypoint can also take the same arguments directly
model_path = definers.train(
    remote_src=None,
    features=["/path/to/f1.csv"],
    dataset_label_columns=["label"],
    order_by="shuffle",
    stratify="label",
    val_frac=0.1,
    test_frac=0.1,
    batch_size=32,
)
```

## Automation

- Formatting and quote normalization are enforced by Ruff using double quotes.
- Repository hygiene is automated by `scripts/clean_workspace.py`.
- Python source sanitization for comments and docstrings is available in `scripts/strip_comments.py`.
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