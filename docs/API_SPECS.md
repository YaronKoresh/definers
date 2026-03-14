# Definers API Specifications

## System Architecture Overview

Definers provides a modular utility toolkit with extension-oriented boundaries for reliability, data handling, media processing, and system orchestration.

### Capabilities

- `definers.capabilities.CircuitBreaker`
	- Sync and async operation gates.
	- Explicit state machine: `CLOSED`, `OPEN`, `HALF_OPEN`.
	- Snapshot API for runtime diagnostics.
- `definers.capabilities.ExponentialBackoffDelay`
	- Configurable base delay, multiplier, max delay, and jitter.
	- Retry spacing strategy abstraction through `RetryDelayStrategy` protocol.
- `definers.capabilities.with_retry`
	- Async retry decorator.
	- Selective exception retry boundaries with deterministic failure re-raise.
- `definers.web.ResourceRetrievalOrchestrator`
	- Strategy-driven transfer execution with integration error boundaries.

## Execution Patterns

- Resilience boundaries are composed through retry and circuit-breaker primitives.
- Extension points use protocols and strategy implementations instead of direct hard-coupled logic.
- Critical failures are surfaced explicitly through typed exceptions and state snapshots.

### Error Boundaries

- Unavailable downstream integrations can be isolated by opening the circuit after threshold failures.
- Retry logic is constrained by max-attempt policy and retryable exception filtering.
- Transfer orchestration catches integration faults and returns safe status outcomes.

### Data preparation helpers

- `prepare_data(remote_src=None, features=None, labels=None, url_type=None, revision=None, drop=None, order_by=None, stratify=None, val_frac=0.0, test_frac=0.0, batch_size=1)` loads a dataset from a HuggingFace source or local feature/label files, applies optional column drops, orders the examples (shuffle, sort, or user-supplied key), and returns a `TrainingData` object containing one or more `torch.utils.data.DataLoader` instances. Splitting into train/val/ test sets with stratification and configurable fractions is supported.
- `TrainingData` dataclass returned by `prepare_data`; fields are `train`, `val`,
  `test` (each a DataLoader) and `metadata` (a dict recording split parameters).

### Audio analysis contracts

- `analyze_audio(audio_path, hop_length=1024, duration=None, offset=0.0)` returns a dense analysis mapping for renderer-style consumers. The mapping includes waveform and frame-domain metadata: `y`, `sr`, `hop_length`, `duration`, `bpm`, `beat_frames`, `spectral_centroid`, `stft`, `stft_db`, `rms`, `rms_low`, `rms_mid`, `rms_high`, and `normalize`.
- Frame-aligned arrays produced by `analyze_audio` are expected to share a common time basis so consumers can safely convert time to frame indices with `sr / hop_length`.
- `analyze_audio_features(audio_path, txt=True)` is the compact summary contract. It returns a formatted string such as `C major (120 bpm)` when `txt=True`, and `(key, mode, tempo)` when `txt=False`.
- Audio load failures are surfaced as safe null-style outcomes for summary flows, while dense analysis callers are expected to treat analysis errors as hard failures.

### Public import pattern

- Import runtime APIs from their implementation modules such as `definers.data`, `definers.ml`, `definers.system`, and `definers.catalogs`.
- The package root is intentionally narrow and only exposes version metadata plus the optional `sox` module probe through `definers.sox` and `definers.has_sox()`.

### Catalog contracts

- `definers.catalogs.languages.LANGUAGE_CODES` and `definers.catalogs.languages.UNESCO_MAPPING` are immutable registries keyed by normalized language code.
- `definers.catalogs.tasks.TASKS` is an immutable task-to-model registry consumed directly by `definers.constants`.
- `definers.catalogs.references.USER_AGENTS` and `definers.catalogs.references.STYLE_CATALOG` are immutable registries with tuple-backed collections for read-only consumers.
- `definers.catalogs` and `definers.catalogs.access` expose those registries directly as immutable module exports; there is no getter-function compatibility layer.

### CLI and launcher contracts

- `definers.application_shell.commands.parse_cli_command(args, read_lyrics_text, gui_commands)` normalizes command names before routing them to `StartCommand`, `MusicVideoCommand`, `LyricVideoCommand`, or `UnknownCommand`.
- `definers.presentation.cli_dispatch.run_cli(argv, version)` shares the normalized GUI project names used by the launcher registry and only reads lyric text from disk when the provided argument resolves to a file.
- `definers.presentation.launchers.launch_installed_project(project)` is the deployment-facing launcher used by the docker app stubs after package installation.

### Web transfer facades

- `definers.web.download_file(url, destination)` and `definers.web.download_and_unzip(url, extract_to)` are thin wrappers over `definers.media.web_transfer` that inject the async executor and the appropriate orchestrator factory.
