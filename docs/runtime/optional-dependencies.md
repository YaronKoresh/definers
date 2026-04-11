# Optional Dependencies

Definers keeps the base package small and treats optional domains as explicit install boundaries.

## Extras

- `audio`
- `image`
- `video`
- `ml`
- `nlp`
- `web`
- `cuda`
- `dev`

## Runtime Policy

- Missing optional packages must fail cleanly.
- CPU paths stay valid even when CUDA packages are absent.
- Runtime install flows can target groups, tasks, modules, model domains, and model tasks.
- Core spreadsheet-backed `xlsx` support is not optional; it is shipped in the base install through `openpyxl`.

## Import Policy

- `import definers` must stay lightweight and safe under the base install.
- Heavy optional stacks should load when a concrete feature module is used, not because the root package was imported.
- Package facades may expose stable names lazily, but new behavior should live in concrete owner modules.
- Optional-dependency shims belong under `definers.internal_compat`, not as public root-level compatibility modules.

## Compatibility Policy

- A missing optional package should fail at first real use with a clear message.
- Tests and base development workflows must keep working without optional extras installed.
- Compatibility shims may exist during migrations, but they must protect a real import surface rather than act as permanent routing layers.

## Install Surface

```bash
definers install --list
definers install audio
definers install answer --type task
definers install image --type model-domain
```

## System Bootstrap

- Linux hosts can use `apt_install()` from `definers.system` to provision the shared system packages behind the official audio, image, video, and hosted UI workflows.
- That bootstrap includes `ffmpeg`, `libsndfile1`, `libgl1`, and `libglib2.0-0` in addition to the broader audio and visual stack.