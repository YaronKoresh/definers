# Package Map

Import concrete owner modules when possible.

The package map below is the target operating model for Definers. Package facades remain useful for discovery and stable imports, but new implementation work belongs in the owner module for the feature being changed.

## Root Package Policy

- `import definers` is a stable lazy public API for discovery and high-value runtime helpers.
- The root package must not eagerly import heavy optional stacks such as full audio, ML, or UI runtimes.
- Root lazy exposure currently covers package discovery surfaces such as `audio`, `chat`, `data`, `media`, `ml`, `system`, and `text` without forcing their internal submodules to load.
- New feature behavior does not belong in the root package.
- Root-level exports must stay intentionally curated and compatible rather than growing into a mirror of the entire source tree.

## Primary Packages

- `definers.audio` owns mastering, stems, analysis, generation, editing, and audio I/O workflows.
- `definers.cli` owns command catalog, parser assembly, runtime binding, command parsing, and CLI dispatch flow.
- `definers.data` owns preparation, loaders, tokenization, vectorization, lightweight dataset helpers, and data runtime helpers.
- `definers.image` owns image workflows, generation helpers, and image-facing app logic.
- `definers.media` owns network transfer, shared artifact movement, archive handling, and media-neutral download support.
- `definers.media.web_transfer` is the direct owner module for transfer policy, download entrypoints, runtime-aware orchestration, and artifact extraction.
- `definers.ml` owns answer, inference, training, retrieval, text-generation, regression, and health-facing ML APIs.
- `definers.ml.answer.service` is the direct owner module for answer dependency loading, history preparation, media attachment loading, and response generation.
- `definers.ml.repository_sync` is the direct owner module for model-source resolution, Hugging Face and remote artifact routing, shard discovery, and runtime model loading.
- `definers.runtime_numpy` owns NumPy and CuPy backend selection and compatibility helpers.
- `definers.system` owns installation, paths, process control, runtime state, threads, output paths, and download activity.
- `definers.system.services` is the direct owner module for infrastructure override context and the injectable runtime, filesystem, and process callable ports.
- `definers.text` owns translation, normalization, and text transformation utilities.
- `definers.ui` owns launcher registration, focused-surface definitions, shared Gradio primitives, workbench composition, and user-facing UI routing.
- `definers.ui.apps.audio` is the direct owner module for the audio workbench, staged mastering flow, workspace bootstrap, and audio tool handlers.
- `definers.ui.apps.image` is the direct owner module for image generation, upscale and title actions, and staged image-job launcher wiring.
- `definers.ui.apps.video` is the direct owner module for the video composer, lyric video, visualizer surfaces, and their launcher wiring.
- `definers.video` owns rendering, composition, lyric-video, visualizer, and video helpers.

## Supporting Packages

- `definers.chat` is the chat feature surface and should not become a generic application routing layer.
- `definers.catalogs` owns static or semi-static catalog access.
- `definers.internal_compat` is the only accepted home for third-party fallback shims and compatibility adapters.
- `definers.ui.apps` is the direct owner package for concrete launcher implementations and domain UI surfaces.

## Feature Ownership Rules

- Package facades are stable discovery surfaces and compatibility anchors, not dumping grounds for new logic.
- When a package grows beyond a coherent surface, split work into direct owner modules under that package instead of adding another routing layer above it.
- Prefer imports such as `definers.ml.answer.service` or `definers.audio.mastering.pipeline` when changing behavior.
- Keep cross-cutting runtime policy in `definers.system` or `definers.runtime_numpy`, not scattered across feature packages.

## Facade And Compatibility Rules

- Keep package facades small, lazy where needed, and patch-target stable during migrations.
- Use a temporary compatibility shim only when it protects a real public import or test patch path.
- Do not add new wrapper-only gateway modules, registries, or proxy packages when a direct owner module can be imported instead.
- Remove routing-only layers after consumers move, but not before.

## Current Facade Map

- `definers.cli.command_registry`, `definers.cli.parser`, `definers.cli.runtime`, and `definers.cli.dispatch` are direct owner modules for CLI assembly and execution.
- `definers.media.web_transfer` is the owner module for transfer policy, artifact download entrypoints, and orchestration.
- `definers.data`, `definers.chat`, `definers.text`, and `definers.media` are lazy facades over narrower owner modules.
- `definers.ui.gui_entrypoints` is the stable launcher-registry surface, while `definers.ui.gradio_shared` owns shared cross-domain progress and outputs helpers.

Use package-level facades for discovery, then move to the concrete owner module for implementation work.