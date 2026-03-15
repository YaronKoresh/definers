# Definers

Definers is a modular Python framework designed for building and orchestrating AI-driven pipelines, media transformation workflows, and data processing tasks. It provides:

- A consistent API for handling audio, image, video, and machine‑learning workloads.
- Resilient execution primitives with built‑in retry and circuit‑breaker support.
- Safe system and subprocess management with explicit guardrails to prevent injection and platform‑specific pitfalls.
- Extensible utilities for feature extraction, preprocessing, and end‑to‑end model training, with a focus on predictable behavior and reproducible results.

The library emphasizes performance, security, and maintainability, making it suitable for both experimentation and production deployment.

## Core Modules

- `definers` is a minimal package root. It exposes version metadata plus the optional `sox` probe used by audio loading paths, but it no longer mirrors the broader implementation surface.

- `definers.capabilities` defines execution resilience primitives. It exports retry policies, circuit‑breaker guards, and standardized error categories that let higher‑level workflows tolerate transient failures while preserving observability.

- `definers.web` implements safe HTTP retrieval and upload orchestration. It centralizes request retries, rate‑limit backoff, content validation, and proxy/credential handling so data transfer logic is consistent across tools.

- `definers.audio`, `definers.image`, `definers.video`, and `definers.ml` are the domain engines. Each exposes a stable processing pipeline API, including I/O normalization, augmentation utilities, and model inference helpers, while isolating format‑specific dependencies and runtime constraints.

- `definers.system` and its companion utilities provide environment introspection, platform‑aware subprocess execution, and guarded system calls. They supply the shared foundation for safe process spawning (`run`, `run_linux`, `run_windows`), deterministic path handling, and configuration sanitization used throughout the package.

## Import Guidance

Import concrete APIs from their implementation modules:

```python
from definers.data import prepare_data
from definers.ml import train
from definers.system import run
```

The package root remains available for version metadata and the optional `sox` handle:

```python
import definers

if definers.has_sox():
    transformer = definers.sox.Transformer()
```

New imports should target submodules directly.

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
`load_as_numpy` for audio conversion. The package root performs a quiet
import probe during initialization and exposes the resulting module as
`definers.sox`. If the `sox` binary is not installed or not on `PATH`, the
package falls back to an object that raises `ImportError` only when used;
audio features will therefore return `None`.

Users on Windows encountering the message "'sox' is not recognized as an
internal or external command" no longer see it when simply running
`import definers`.

## Available GUIs

The package exposes a shared project launcher through `definers.chat.start()` and `definers.presentation.launchers.launch_installed_project()`. CLI dispatch, docker app entrypoints, and direct launcher calls normalize the project name through the same registry-aware resolution path.

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
`definers.system.catch()` which may raise or log depending on configuration.

## Catalogs

Static catalogs live under `definers.catalogs` as immutable registries. `definers.catalogs` and `definers.catalogs.access` both expose the registry objects directly without getter helpers.

## Command Execution

All external programs are run through the `definers.system.run()` helper, which
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

## Entry Points

`python -m definers` is a thin wrapper over `definers.cli.main()`.
Programmatic callers that need the CLI entrypoint should import it from
`definers.cli`.

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
from definers.data import prepare_data
from definers.ml import train

# prepare CSV features automatically; returns loaders and metadata
data = prepare_data(
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
model_path = train(
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
- Docker helper scripts validate the requested project before invoking Compose or tagging images.
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

### Maintainer

This project is owned and maintained by Yaron Koresh. Contributions are welcome via pull requests or issues.
