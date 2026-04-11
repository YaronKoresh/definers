# Launchers And GUI Contribution Rules

Definers exposes official GUI launchers as user-facing workflow surfaces. This page is the owner guide for how those surfaces are classified, registered, documented, and tested across every current domain.

## What This Area Owns

- `definers.ui` owns launcher registration, focused-surface composition, shared Gradio primitives, and user-facing UI routing.
- `definers.ui.gui_entrypoints` owns the official launcher registry and the public `start(project)` entry point.
- `definers.ui.launchers` owns launcher-name resolution and fallback behavior.
- `definers.ui.apps.focused_surfaces` owns the split between full workbenches and narrower task surfaces.
- `definers.ui.apps.surface_hub` owns domain landing hubs that list focused surfaces without opening a full workbench first.
- `definers.ui.gradio_shared` owns cross-domain progress, output-folder, and shared Gradio shell primitives.
- `definers.ui.apps.*` owns the concrete surface implementations for each domain.

## Stable Entry Points

- Python entry point: `definers.ui.gui_entrypoints.start(project)`
- CLI entry point: `definers start <project>`
- Official launcher names are the keys registered in `definers.ui.gui_entrypoints.GUI_LAUNCHERS`.

Do not treat a surface as an official launcher until it is registered in `GUI_LAUNCHERS`, covered by focused tests, and documented anywhere user-facing launcher names are listed.

## GUI Taxonomy

### Workbench

A workbench is the full domain cockpit. It keeps multiple related workflows together for expert users who want one advanced surface.

Use a workbench when a domain already has several mature workflows that share context or controls.

### Focused Surface

A focused surface is a narrow launcher for one task family inside a domain, such as mastering, video composition, or image upscaling.

Use a focused surface when users benefit from reduced scope, cleaner copy, and a direct CLI-addressable entry point.

### App-Only Launcher

An app-only launcher is a direct launcher that does not need an additional focused-surface wrapper or domain hub.

Use an app-only launcher for smaller or self-contained workflows where extra routing layers would add complexity without improving the UX.

### Guided-Job Mode

Guided-job mode is an optional pattern for expensive or multi-step workflows. It adds persistent job state, resumable steps, intermediate artifacts, and step-specific status inside an existing surface.

Guided-job behavior is not mandatory for every launcher. It is an opt-in second-tier pattern for workflows that justify persistence and resume behavior, but it does not require a second standalone GUI.

## Current Official Launcher Inventory

### Audio

- Workbench: `audio`
- Focused surfaces: `audio-mastering`, `audio-vocals`, `audio-cleanup`, `audio-stems`, `audio-analysis`, `audio-create`, `audio-midi`
- Guided-job capable surface: `audio-mastering`

### Video

- Workbench: `video`
- Focused surfaces: `video-composer`, `video-lyrics`, `video-visualizer`

### Image

- Workbench: `image`
- Focused surfaces: `image-generate`, `image-upscale`, `image-title`
- Guided-job capable surface: `image-generate`

### ML And Training

- Workbench: `train`
- Guided and advanced training modes live inside the `train` workbench instead of separate launcher names, with quick decisions, managed manifests, and hosted-runtime-safe budgets staying inside that same surface.

### Direct App Launchers

- `chat`
- `translate`
- `animation`
- `faiss`

## Launcher Architecture

- Register official launcher names only in `definers.ui.gui_entrypoints`.
- Keep launcher resolution logic in `definers.ui.launchers` instead of duplicating name handling in app modules.
- Keep focused-surface composition in `definers.ui.apps.focused_surfaces` instead of re-implementing domain routing inside each app.
- Keep shared progress and output affordances in `definers.ui.gradio_shared`.
- Keep concrete UI behavior inside the real owner app module for the domain.

Do not add a second launcher registry, a second name-normalization path, or a wrapper-only UI gateway when a concrete owner module can be called directly.

## Contributor Rules

- Choose the smallest surface type that solves the UX problem: app-only launcher first, focused surface second, full workbench only when a domain truly needs the whole cockpit.
- Keep launcher names stable unless the change also updates CLI docs, launcher docs, and tests.
- Reuse `definers.ui.gradio_shared` for progress and output patterns instead of cloning per-domain shells.
- Keep imports lazy inside launcher functions and handlers so missing optional dependencies fail cleanly.
- Keep domain-specific orchestration in the owning app or service module, not in shared UI helpers.
- Do not mark conceptual surfaces as official until they are registered in `GUI_LAUNCHERS`.
- When a domain already has a workbench and focused surfaces, add new narrow workflows to that structure instead of inventing a parallel routing model.

## Mandatory Baseline Contract For Every Official GUI

Every official launcher, whether it is a workbench, a focused surface, or an app-only launcher, should meet this baseline:

- Show a progress tracker with named user-readable stages.
- Expose an outputs-folder affordance so users can find generated artifacts.
- Report activity through the shared progress and download-activity path instead of emitting silent long-running work.
- Preserve clean optional-dependency behavior when Gradio or heavier runtime packages are unavailable.
- Keep user-facing workflow framing clear enough that the launcher goal is obvious from the page itself.
- Have focused structural tests that assert the launcher is bound and exposes its main actions.
- Be documented anywhere stable launcher names or GUI contribution rules are maintained.

This baseline is mandatory across the repo.

## Optional Guided-Job Contract

Use guided-job mode only when a workflow is expensive, multi-step, interruption-prone, or artifact-heavy.

If you opt in, the surface should:

- persist job state under managed output paths
- keep a small manifest with input, settings, state, artifacts, and next-step data
- allow refresh or resume from an existing job folder
- expose intermediate artifacts when they are materially useful for QA or continuation
- communicate which stages are expensive or hardware-heavy
- keep advanced manifest/debug views behind a collapsed section

Do not force guided-job behavior onto simple utilities such as translation, chat, or FAISS tooling unless the workflow actually needs persisted state.

## Current Cross-Domain Matrix

### Guided-Job Capable Domains

- Audio: built into `audio-mastering`
- Image: built into `image-generate`

### Workbench Plus Focused-Surface Domains

- Audio
- Video
- Image

### Workbench-Only Domains

- ML and training, with guided and advanced modes inside `train`, plus guided session manifests, artifact sidecars, rollout metrics, and hosted-runtime-safe beginner routing inside the same workbench

### Focused-Surface Capable Domains

- Audio
- Video
- Image

### Baseline-Only Direct Launchers

- `chat`
- `translate`
- `animation`
- `faiss`


## Testing Rules

- Launcher registration and routing changes should extend `tests/test_launchers.py`.
- Shared progress-shell behavior should be covered in `tests/test_gradio_shared.py` and `tests/test_gui_activity_reporting.py`.
- Structural launcher-binding behavior should be covered in the nearest UI suite, such as `tests/test_surface_apps_ui.py`, `tests/test_audio_app.py`, `tests/test_train_ui.py`, or `tests/test_video_gui.py`.
- Service orchestration that sits behind a GUI should be covered in the domain service suite, such as `tests/test_audio_app_services.py`.
- Tests must not depend on optional third-party runtime packages being installed.

## Optional-Dependency Safety

- Import Gradio and heavier optional stacks inside launcher functions or action handlers when possible.
- Keep missing extras as clean failures with actionable errors instead of import-time crashes from top-level module loading.
- Prefer test doubles and module stubs over real optional runtimes in GUI tests.

## Documentation Requirements

When you add or change an official launcher, update the owning docs in the same change:

- this page for launcher taxonomy, contribution rules, or official launcher inventory changes
- `docs/reference/cli.md` when stable launcher names or CLI examples change
- the owning capability guide when the user workflow or output contract changes
- `docs/README.md` when the docs index or contributor path should change

Use launchers when the goal is a user-facing workflow surface rather than a low-level library call.