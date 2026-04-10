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

## Install Surface

```bash
definers install --list
definers install audio
definers install answer --type task
definers install image --type model-domain
```