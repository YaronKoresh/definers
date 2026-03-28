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
from definers.data import prepare_data
from definers.ml import train
from definers.system import run
```

When splitting large modules, prefer keeping the existing public facade stable and moving heavyweight behavior into dedicated implementation modules behind that facade.

### Runtime Safety

- Use `definers.system.run()` with list-form commands where possible.
- Treat retry and circuit-breaker behavior as public capabilities, not internal implementation details.
- Preserve graceful behavior around optional dependencies such as `sox`.
- Keep optional media dependencies lazy in the answer pipeline so text-only flows do not require image or audio extras.
- Keep facade modules narrow and lazy where practical so importing one public symbol does not drag in an entire heavyweight surface.

### Tests

The repository contains a broad Python test suite under `tests/`. When adding or changing behavior:

1. update tests close to the feature you changed
2. add regression coverage for the specific failure mode you fixed
3. avoid broad unrelated refactors inside the same pull request

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
