# Contributing To Definers

This is the shortest safe contributor path for the current feature-first codebase.

## Setup

Definers targets Python 3.10 through 3.12.

```bash
pip install -e ".[dev]"
```

Add extras only for the domain you are touching.

```bash
pip install -e ".[dev,audio]"
pip install -e ".[dev,image]"
pip install -e ".[dev,video]"
pip install -e ".[dev,ml]"
pip install -e ".[dev,web]"
```

## Code Rules

- Change the smallest feature package that actually owns the behavior.
- Prefer concrete module imports over broad package facades.
- Do not add new wrapper-only services, facades, registries, or proxy layers.
- Preserve optional-dependency behavior. Missing optional packages must fail cleanly.
- Remove dead code and old routing layers created by your change.
- Update tests and docs when commands, imports, runtime behavior, or package layout change.

## Validation

Run the narrowest proof first, then the main gate.

```bash
poe check
poe cli-health
poe ml-health
pytest tests/test_cli.py -q
pytest tests/test_application_ml_answer_services.py tests/test_application_ml_answer_history_preparer.py tests/test_answer.py -q
```

Rules:

- Prefer focused pytest slices while iterating.
- Do not ignore failures directly caused by your change.
- Tests must not depend on optional third-party packages being installed.
- Tests must not use third-party library output as the expected-value oracle.

## Import Rules

Use the concrete owner module whenever possible.

Preferred patterns:

- `definers.ml.answer.service`
- `definers.ml.text.generation`
- `definers.data.datasets`
- `definers.data.text.vectorizer`
- `definers.system.paths`
- `definers.ui.apps.train`

## RVC Policy

RVC support is fork-only.

- Use only `YaronKoresh/definers-rvc-files`.
- Keep the bootstrap LFS-aware.
- Treat the fork-owned `assets`, `configs`, `docs`, `i18n`, `infer`, `logs`, and `tools` folders as the readiness contract.
- Do not introduce alternate repositories, upstream GUI projects, or fallback sources into the bootstrap path.
- If bootstrap behavior changes, validate the model-installation path directly.

## Checklist

Before finishing, confirm all of the following:

- the change lives in the correct package
- imports use the concrete module structure
- focused tests pass
- `poe check` passes, or any unrelated failure is clearly identified
- docs are updated if commands, structure, or runtime behavior changed
- no dependency, install, or bootstrap regression was introduced
