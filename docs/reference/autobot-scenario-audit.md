# Autobot Scenario Audit

This page owns the seeded scenario engine that audits Autobot, Smart Link, and unified automation behavior under fixed test coverage plus optional exploratory replay.

## Stable Entry Points

- `poe check` runs the fixed-seed scenario coverage together with the rest of the repository test and lint gate.
- `poe autobot-random` runs a fixed-size exploratory scenario pack locally.
- `node scripts/run_autobot_scenario_lanes.cjs` is the underlying runner if you need direct control over `SCENARIO_LANE` or `SCENARIO_ROTATION_KEY`.
- `node scripts/print_autobot_regression_fixture.cjs` previews the current local git diff as Autobot labels, semver, and summary before you push. If you pass a saved capture bundle through `SCENARIO_CAPTURE_FILE`, an explicit file argument, or stdin, it still formats that bundle into a registry entry for `tests/deterministic_scenario_regression_fixtures.cjs`.

## Lane Model

- Fixed-seed baseline and regression fixture coverage runs inside the normal JavaScript tests, so `poe check` and the Autobot workflow already exercise the deterministic gate.
- `exploratory` runs a fixed-size random pack across the Autobot suites, keeps committed regression fixtures active, and emits promotion-ready capture candidates for new warnings or failures.
- `capture` is reserved for local promotion tooling and uses the same candidate format as the exploratory lane.

## Important Inputs

- `SCENARIO_LANE` selects `deterministic`, `exploratory`, or `capture`. The manual runner defaults to `exploratory`.
- `SCENARIO_ROTATION_KEY` pins the exploratory seed pack. If omitted, the manual runner generates a fresh rotation key per invocation.
- `SCENARIO_INCLUDE_REGRESSION_FIXTURES=false` disables the committed replay pack if you explicitly want random coverage without the current fixture registry in the same report.
- `SCENARIO_PRINT_CAPTURE_BUNDLE=true` prints the normalized candidate bundle after a lane run.
- `AUTOBOT_LOCAL_BASE_REF` changes the git comparison base for `scripts/print_autobot_regression_fixture.cjs`. The default is `HEAD`, so the preview covers staged and unstaged local changes plus untracked files.
- `AUTOBOT_LOCAL_TITLE` and `AUTOBOT_LOCAL_BODY` optionally inject the planned PR title and body into the local preview when you want title-sensitive labeling before opening the PR.
- `SCENARIO_CAPTURE_FILE` points `scripts/print_autobot_regression_fixture.cjs` at a saved capture bundle.
- `SCENARIO_CAPTURE_ID` selects one candidate from a multi-candidate bundle when promoting a fixture.

## Normal Workflow

1. Run `poe check` for the normal fixed-seed gate when you change Autobot GitHub automation, deterministic scenario generation, or regression fixtures.
2. Run `node scripts/print_autobot_regression_fixture.cjs` before you push when you want a local Autobot preview from the current git diff instead of waiting for the public workflow to classify the PR online.
3. Run `poe autobot-random` when you want extra randomized coverage beyond the normal test suite.
4. If you explicitly want exploratory random coverage without the committed replay pack, add `SCENARIO_INCLUDE_REGRESSION_FIXTURES=false`.
5. If the exploratory report surfaces a promotion-ready candidate, copy the JSON bundle or pipe it into `node scripts/print_autobot_regression_fixture.cjs`.
6. Add the formatted entry to `tests/deterministic_scenario_regression_fixtures.cjs` once the captured case is worth keeping as a permanent regression.
7. Re-run `poe check` to confirm the new fixture still passes under fixed seeds.

## Failure Modes And Boundaries

- If a committed fixture no longer materializes for its recorded `seed` and `count`, the fixed-seed test coverage fails because the regression pack drifted.
- New warnings in the fixed-seed coverage should be treated as contract drift and resolved before merge.
- Exploratory runs are intentionally non-gating. Their job is to surface new replay keys and candidate fixture entries without blocking merges directly, while still replaying the committed regression fixture kinds against rotating exploratory seeds.
- The exploratory lane report now prints full problem blocks for warning or deviation scenarios so you can inspect the exact replay key, scenario payload, optimal result, actual result, and raw result without re-reading the generators.
- The committed fixture registry is intentionally small. Only promote failures or warnings that represent durable workflow risk.

## Owning Files

- `tests/deterministic_scenario_engine.cjs` owns seeded scenario generation, execution, contract validation, and suite reports.
- `tests/deterministic_scenario_regression_fixtures.cjs` owns the committed regression fixture registry.
- `tests/deterministic_scenario_lanes.cjs` owns lane planning, rotating seeds, fixture materialization, and capture candidate generation.
- `scripts/run_autobot_scenario_lanes.cjs` is the CLI runner behind the manual exploratory command.
- `scripts/print_autobot_regression_fixture.cjs` previews the current local git diff and still formats a capture candidate into a registry entry when you feed it a saved bundle.
- `.github/workflows/autobot.yml` installs Node and runs the fixed-seed JavaScript scenario tests before it applies live Autobot automation.