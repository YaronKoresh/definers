# Contributing To Definers

Thank you for contributing to Definers. This repository is a Python project with a strong emphasis on predictable behavior, modular design, and automated quality checks. This guide explains how to set up the project, validate changes locally, and prepare a pull request that aligns with the current toolchain.

## Development Environment

### Prerequisites

| Requirement | Notes |
| --- | --- |
| Python | `>=3.10` |
| Packaging | `pip` with editable-install support |
| Recommended workflow | Virtual environment plus editable install |
| Optional tools | FFmpeg, `sox`, CUDA components, depending on the feature area |

### Local Setup

Create or activate a Python environment, then install contributor dependencies.

```bash
pip install -e ".[dev]"
```

If you are working on feature areas with optional dependencies, install the relevant extras too.

```bash
# For audio-related work
pip install -e ".[dev,audio]"

# For image, or video-related work
pip install -e ".[dev,image,video]"

# For machine learning or web-related work
pip install -e ".[dev,ml,web]"
```

Windows contributors can also use [scripts/install.bat](scripts/install.bat).

## Daily Workflow

### Main Validation Command

Use the full validation pipeline before pushing work.

```bash
poe check
```

This command runs:

1. workspace cleanup
2. compile verification
3. dead-code scanning
4. source sanitization
5. pre-commit checks
6. test execution
7. final cleanup

### Focused Commands

Use the narrower commands when iterating on a change.

```bash
poe test
poe lint
poe format
poe build
poe hook
poe cli-health
poe answer-simulations
poe ml-health
```

| Command | Purpose |
| --- | --- |
| `poe test` | Run the test suite |
| `poe lint` | Run Ruff lint checks |
| `poe format` | Apply Ruff formatting |
| `poe build` | Build the package |
| `poe hook` | Install pre-commit hooks |
| `poe cli-health` | Validate CLI registry, parser, launcher wiring, and command health |
| `poe answer-simulations` | Run mixed-media answer regressions and dependency-fallback simulations |
| `poe ml-health` | Validate AutoTrainer DX, ML health snapshots, and AI workspace bootstrap helpers |

For changes in the multimodal answer path, run the focused regression set first.

```bash
pytest tests/test_application_ml_answer_services.py tests/test_application_ml_answer_history_preparer.py tests/test_answer.py -q
```

For changes in the mastering path, run the focused mastering regression set first.

```bash
pytest tests/test_audio_mastering_generation.py tests/test_audio_mastering_phase.py tests/test_audio_mastering_spectrum.py tests/test_audio_mastering_finalization.py tests/test_audio_mastering_contract.py tests/test_audio_mastering_dynamics.py tests/test_audio_mastering_character.py -q
pytest tests/test_audio_mastering_reporting.py tests/test_audio_mastering_delivery.py tests/test_audio_mastering_reference.py tests/test_audio_mastering_rollout.py tests/test_audio_mastering_loudness_accuracy.py tests/test_audio_mastering_stems.py -q
pytest tests/test_public_module_splits.py tests/test_audio_lazy_exports.py -q
```

Keep the generation-based mastering suites separate from the direct-import public-surface suites. The generation loaders stub `scipy` locally, so mixing them into the same pytest process can contaminate the import-based checks.

For mastering diagnostics, `master()` accepts `report_path="...json"` and writes the returned report directly to disk. Explicit delivery profiles remain authoritative, while runs without an explicit profile still resolve lossy outputs such as `.mp3` and `.m4a` to the default lossy distribution profile.

## Coding Expectations

### General Standards

- Keep changes focused on the problem being solved.
- Preserve existing public APIs unless the change explicitly requires an API update.
- Prefer descriptive names over explanatory comments.
- Add tests when behavior changes or new behavior is introduced.
- Avoid introducing new optional dependencies unless the capability clearly requires them.

### Imports And Module Usage

Prefer importing from concrete public modules rather than from the package root.

```python
from definers.application_data.preparation import prepare_data
from definers.ml import train
from definers.system import run
```

When splitting large modules, prefer keeping the existing public facade stable and moving heavyweight behavior into dedicated focused public modules behind that facade.

In the audio path, keep `audio/mastering.py` as the public facade and move dense spectral, EQ, dynamics, pipeline, contract, finalization, delivery-verification, character, reference, and stem-aware orchestration helpers into narrower public modules such as `audio.mastering_profile`, `audio.mastering_eq`, `audio.mastering_contract`, `audio.mastering_loudness`, `audio.mastering_metrics`, `audio.mastering_presets`, `audio.mastering_finalization`, `audio.mastering_delivery`, `audio.mastering_character`, `audio.mastering_reference`, and `audio.mastering_stems` instead of growing the facade into another monolith. Apply the same pattern in other dense areas such as `ml.py`, where dedicated public modules like `ml_text`, `ml_regression`, and `ml_health` now carry narrower concerns while the original facade stays stable.

For mastering work specifically, keep the contract, report, character, and reference layers in sync: profile changes should update the contract bounds, pipeline changes should preserve the staged report surfaces for post-EQ, post-spatial, post-limiter, post-clamp, final in-memory, and decoded-delivery metrics, and finishing or preset changes should keep the adaptive headroom telemetry, stereo-motion telemetry, reference-assist suggestions, and rollout coverage aligned.

Prefer directly importable public module names for split targets. Avoid introducing new non-magic underscore-prefixed module filenames; keep `__init__.py` and other dunder files as the only naming exception.

### Runtime Safety

- Use `definers.system.run()` with list-form commands where possible.
- Treat retry and circuit-breaker behavior as public capabilities, not internal implementation details.
- Validate downloaded archive member paths before extraction so unzip flows cannot write outside the requested target directory.
- Preserve graceful behavior around optional dependencies such as `sox`.
- Keep optional media dependencies lazy in the answer pipeline so text-only flows do not require image or audio extras.
- Keep facade modules narrow and lazy where practical so importing one public symbol does not drag in an entire heavyweight surface.

### Tests

The repository contains a broad Python test suite under `tests/`. When adding or changing behavior:

1. update tests close to the feature you changed
2. add regression coverage for the specific failure mode you fixed
3. avoid broad unrelated refactors inside the same pull request

Test policy for the default suite:

- Tests must validate Definers behavior with Python data, repository fixtures, and local fakes or stubs.
- Tests must not import optional third-party packages directly.
- Tests must not use third-party library output as the oracle for expected results.
- If a feature is only available with an optional dependency, the default suite should validate Definers fallback or unavailable-feature behavior instead of requiring that package.

## Branch And Pull Request Guidance

### Branch Naming

Use short, descriptive branch names in kebab case.

- `audio-mastering-fix`
- `readme-consolidation`
- `launcher-validation-update`

### Pull Request Checklist

Before opening a pull request:

1. ensure your branch is current with the target branch
2. run `poe check` locally
3. verify any optional dependency changes are reflected in documentation where needed
4. add or update tests for behavioral changes
5. explain the scope clearly in the pull request description

### What Reviewers Will Look For

- correctness
- behavioral regressions
- missing tests
- unnecessary dependency growth
- mismatch between documentation and implementation

## CI And Validation

The repository validates changes through GitHub Actions.

| Workflow | Purpose |
| --- | --- |
| `check.yml` | Pull-request validation across Python `3.10`, `3.11`, and `3.12` using `poe check` |
| `quality.yml` | Quality validation on push and manual runs |

Passing local checks before opening a pull request saves review time and reduces avoidable CI failures.

## Documentation Policy

This repository uses two primary documentation entry points:

1. `README.md` for product overview, architecture, API surface, installation, and operational guidance
2. `CONTRIBUTING.md` for contributor workflow and quality expectations

When implementation changes affect usage, installation, launch behavior, or public contracts, update the README in the same pull request.

For AI trainer workflows, document new planning helpers, health checks, and bootstrap commands when they change the recommended developer path.

## Community And Legal

- Keep discussions technical, respectful, and specific.
- By submitting a contribution, you confirm that you have the right to contribute the code.
- Maintainers retain final judgment over architectural direction, scope, and acceptance.
