# Contributing To Definers

This is the shortest safe contributor path for the current feature-first codebase.

## Setup

Definers targets Python 3.10 through 3.14.

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

## Architecture Rules

- Keep the repository feature-owned. Move logic toward the real owner package instead of adding another shared gateway.
- Treat `definers` as a stable lazy public API, not as a place to mirror the entire source tree.
- Use package facades for discovery and compatibility, but put behavior changes in direct owner modules.
- Temporary compatibility shims are allowed only when they protect a real public import or patch-target path during migration.
- Remove routing-only wrappers after consumers move. Do not keep them once they stop protecting compatibility.

## CLI And Transfer Rules

- CLI changes should preserve the single authoritative command-definition source in `definers.cli.command_registry` instead of creating a second parser or registry path.
- Preserve stable command names and defaults unless the change also updates docs, tests, and migration notes.
- Download changes should keep runtime policy separate from concrete transport strategy.
- Restricted runtimes such as daemon hosts must remain safe by default.

## GUI And Launcher Rules

- Treat `docs/capabilities/ui/launchers.md` as the owner guide for GUI contribution policy and keep it current when launcher taxonomy, registration, shared UI rules, or testing expectations change.
- Use the existing launcher taxonomy instead of inventing a parallel surface model: workbench for the full domain cockpit, focused surface for a narrow task family, and app-only launcher for a small self-contained workflow.
- Register official launcher names in `definers.ui.gui_entrypoints` and keep CLI docs, launcher docs, and tests aligned with those names.
- Reuse `definers.ui.gradio_shared` for progress and outputs-folder affordances instead of duplicating per-domain UI shells.
- Every official GUI should expose a progress tracker, an outputs-folder affordance, named activity stages, and clean optional-dependency failure behavior.
- Persistent job folders, resumable steps, intermediate artifact views, and manifest-based debugging are opt-in patterns for expensive or multi-step workflows, not a requirement for every launcher.
- GUI changes need focused tests for launcher registration, UI binding, and any touched service orchestration.

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

- `definers.cli.application.catalog`
- `definers.media.transfer`
- `definers.ml.answer.service`
- `definers.ml.text.generation`
- `definers.data.datasets`
- `definers.data.text.vectorizer`
- `definers.system.paths`
- `definers.ui.apps.train`

## Documentation Rules

When you touch a maintained docs page, keep it useful enough for the next reader to act without reverse-engineering the code.

At minimum, the owning page should make clear:

- what the area owns
- which entry points are stable
- the normal workflow or usage path
- important defaults, configuration values, or environment variables
- the main runtime limits or failure modes

If launcher names, GUI taxonomy, or cross-domain surface rules change, update `docs/capabilities/ui/launchers.md` and any affected CLI or capability pages in the same change.

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
- GUI changes also satisfy the launcher rules in `docs/capabilities/ui/launchers.md`
