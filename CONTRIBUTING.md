# Contributing To Definers

This guide is the shortest reliable path for contributing without breaking packaging, focused apps, or optional-runtime flows.

## Quick Path

1. Install the editable development environment.
2. Change the smallest feature area that owns the behavior.
3. Run focused tests first.
4. Run `poe check` before finishing.
5. Update docs when behavior, commands, or structure changed.

## Environment Setup

Definers targets Python 3.10 through 3.12.

```bash
pip install -e ".[dev]"
```

Install only the extras your work needs.

```bash
pip install -e ".[dev,audio]"
pip install -e ".[dev,image]"
pip install -e ".[dev,video]"
pip install -e ".[dev,ml]"
pip install -e ".[dev,web]"
```

If you are working on CUDA-specific code, add the CUDA extra only after the CPU path is already working.

## Repo Layout

The codebase is feature-first.

```text
src/definers/
  audio/
  catalogs/
  chat/
  cli/
  data/
    datasets/
    text/
  image/
  media/
  ml/
    answer/
    text/
  system/
  text/
  ui/
    apps/
      train/
  video/
```

Use the most specific package that owns the behavior:

- `audio`, `image`, `video`, `text`: domain-facing capabilities
- `chat`: chat handlers and request shaping
- `data`: preparation, dataset loaders, vectorization
- `ml`: training, inference, answer generation, text ML utilities
- `system`: subprocess, paths, install, filesystem, resilience helpers
- `ui`: launcher surfaces and focused apps
- `catalogs`: registries and reference definitions
- `media`: shared media transfer helpers and compatibility facades

Do not add new layered namespaces when a focused feature package is the correct home.

## Validation Workflow

Run the narrowest validation that proves your change, then run the main project check.

### Main gate

```bash
poe check
```

### Common focused commands

```bash
poe test
poe coverage
poe lint
poe format
poe build
poe cli-health
poe ml-health
```

### Focused pytest examples

```bash
pytest tests/test_cli.py -q
pytest tests/test_application_ml_answer_services.py -q
pytest tests/test_audio_mastering_generation.py -q
```

Rules:

- Prefer focused test slices while iterating.
- Run broader validation before finishing.
- Do not ignore failing tests that are directly connected to your change.
- Tests must not import optional third-party packages directly.
- Tests must not use third-party library output as the oracle for expected results.

## Coding Expectations

- Keep changes small and local to the owning package.
- Preserve optional-dependency behavior. If a dependency is optional, the code must fail cleanly when it is missing.
- Prefer explicit imports from the new feature packages over older compatibility paths.
- Keep public APIs stable unless the change intentionally updates them.
- Remove dead code created by the change.
- Avoid duplicate helpers. Reuse the package that already owns the capability.
- Add or update tests when behavior changes.
- Keep type information intact and avoid widening interfaces without a reason.

## Imports And Compatibility

Several legacy import aliases still exist to reduce breakage during the architecture transition. They are compatibility layers, not the preferred structure.

For new code:

- prefer `definers.ml.answer` over older long-form module paths
- prefer `definers.ml.text` for text ML helpers
- prefer `definers.data.datasets` and `definers.data.text` for moved data utilities
- prefer `definers.ui.apps.train` for train app code

If you move code again, update both the canonical imports and any compatibility alias points that must continue to work.

## Optional Dependencies

A large part of the project is intentionally segmented.

When changing code that touches optional integrations:

- verify import-time behavior without the dependency
- verify the error path is actionable
- avoid importing heavy modules at package import time unless that package already does so by design

Examples include audio toolchains, CUDA-specific paths, web runtimes, and specialized ML packages.

## RVC Policy

RVC support is fork-based.

Required policy:

- use `YaronKoresh/definers-rvc-files`
- keep the bootstrap LFS-aware
- do not reintroduce `lj1995/VoiceConversionWebUI` into `src`
- if bootstrap behavior changes, test the model-installation path directly

## Documentation Policy

Update docs when any of the following changes:

- install commands
- CLI or launcher names
- package layout
- required environment assumptions
- model bootstrap source or runtime behavior

Keep docs short, navigable, and concrete. Prefer a tight command path over long narrative sections.

## Pull Request Checklist

Before finishing, confirm all of the following:

- the change lives in the correct feature package
- imports use the current package structure
- focused tests pass
- `poe check` passes or any unrelated failure is clearly identified
- docs are updated if commands, structure, or behavior changed
- no accidental dependency or bootstrap regression was introduced

## Common Pitfalls

- Importing optional heavy modules at package import time
- Updating the canonical module path but forgetting the compatibility alias
- Changing launcher names without updating CLI or docs
- Reintroducing old layered structure inside new feature packages
- Assuming local environment tools such as FFmpeg, `sox`, or CUDA are always available
