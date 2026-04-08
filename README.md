# Definers

Definers is a modular Python toolkit for AI, media, data preparation, runtime safety, and app launchers.

It is organized around focused feature packages instead of a layered architecture, so you can use one narrow slice or grow into broader workflows without changing toolsets.

## Quick Navigation

1. [Install](#install)
2. [Quick Start](#quick-start)
3. [Package Layout](#package-layout)
4. [Apps And Launchers](#apps-and-launchers)
5. [Development](#development)
6. [Troubleshooting](#troubleshooting)
7. [Contributing](#contributing)

## What Definers Covers

- Audio workflows: analysis, DSP, preview, stems, mastering, generation, transcription
- Image and video workflows: generation helpers, feature extraction, composition, rendering
- ML and data workflows: dataset preparation, vectorization, training, inference, health checks
- Runtime and integration utilities: safer command execution, filesystem helpers, retries, transfers
- App surfaces: CLI launchers, focused GUI flows, Docker app folders

## Install

Definers targets Python 3.10 through 3.12.

| Goal | Command |
| --- | --- |
| Base install | `pip install .` |
| Audio workflows | `pip install ".[audio]"` |
| Image workflows | `pip install ".[image]"` |
| Video workflows | `pip install ".[video]"` |
| ML workflows | `pip install ".[ml]"` |
| Web and UI workflows | `pip install ".[web]"` |
| Contributor setup | `pip install -e ".[dev]"` |
| Full optional stack | `pip install ".[all]"` |
| CUDA extras | `pip install ".[cuda]" --extra-index-url https://pypi.nvidia.com` |

Notes:

- Optional dependencies can still be installed lazily at runtime for supported flows.
- On Windows, `stopes` is intentionally excluded from published extras.
- `madmom` and `basic-pitch` are installed through the runtime or CLI installer path rather than the published extras.
- RVC bootstrap is fork-only: Definers clones `YaronKoresh/definers-rvc-files` with LFS-aware logic and does not use `lj1995/VoiceConversionWebUI`.

Useful install commands:

```bash
definers install --list
definers install audio
definers install translate --type task
definers install audio --type model-domain
definers install rvc --type model-task
```

## Quick Start

### Python

```python
from definers.data.preparation import prepare_data
from definers.ml import train
from definers.ml.text import summarize
from definers.system import run
from definers.ui.launchers import launch_installed_project

dataset = prepare_data(features=["./features.csv"], batch_size=32)
summary = summarize("A long text that needs a short summary.")

run(["ffmpeg", "-i", "input.mp4", "output.wav"])
launch_installed_project("chat")
```

### CLI

```bash
python -m definers --help
definers --version
definers start chat
definers start audio-mastering
definers start image-generate
definers start video-composer
definers start train-workbench
definers train
```

## Package Layout

The codebase now follows a feature-first layout.

```text
src/definers/
  audio/
    effects/
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

Practical navigation guide:

- `definers.audio`: audio-facing public functionality
- `definers.chat`: chat request normalization and handling
- `definers.cli`: command parsing and dispatch
- `definers.data`: preparation, loaders, datasets, vectorization
- `definers.ml`: training and inference facade plus `ml.answer` and `ml.text`
- `definers.system`: runtime, path, process, filesystem, install helpers
- `definers.ui`: launcher and app surfaces
- `definers.catalogs`: immutable registries and references
- `definers.media`: transfer helpers and compatibility media aliases

Compatibility aliases still exist for several older import paths, but new code should prefer the concrete package layout above.

## Apps And Launchers

Definers exposes both broad workbenches and narrower focused surfaces.

### Main app entry points

| Surface | Purpose |
| --- | --- |
| `chat` | multimodal chat |
| `audio` | audio task hub |
| `image` | image task hub |
| `video` | video task hub |
| `train` | ML studio |
| `translate` | translation UI |
| `animation` | animation flow |
| `faiss` | FAISS utilities |

### Focused surfaces

- Audio: `audio-mastering`, `audio-vocals`, `audio-cleanup`, `audio-stems`, `audio-analysis`, `audio-create`, `audio-midi`, `audio-support`
- Image: `image-generate`, `image-upscale`, `image-title`
- Video: `video-composer`, `video-lyrics`, `video-visualizer`
- ML: `ml-health`, `ml-train`, `ml-run`, `ml-text`, `ml-ops`

The broader surfaces remain available as `audio-workbench`, `image-workbench`, `video-workbench`, and `train-workbench`.

### Docker

Each app under `docker/` contains its own `Dockerfile`, `docker-compose.yml`, and `app.py` entrypoint.

```bash
cd docker/chat
docker compose up --build
```

## Development

### Local setup

```bash
pip install -e ".[dev]"
```

Install feature extras only when you need them.

```bash
pip install -e ".[dev,audio]"
pip install -e ".[dev,image,video]"
pip install -e ".[dev,ml,web]"
```

### Main validation

```bash
poe check
```

### Focused commands

| Command | Purpose |
| --- | --- |
| `poe test` | run tests |
| `poe coverage` | run coverage |
| `poe lint` | Ruff lint |
| `poe format` | Ruff format |
| `poe build` | build package |
| `poe cli-health` | validate CLI routing and launcher wiring |
| `poe ml-health` | validate ML health and DX flows |

Focused answer-path regression:

```bash
pytest tests/test_application_ml_answer_services.py tests/test_application_ml_answer_history_preparer.py tests/test_answer.py -q
```

## Troubleshooting

### FFmpeg or `sox`

- Many audio and video flows need FFmpeg on `PATH`.
- `sox` is optional, but some audio paths degrade or return `None` when it is unavailable.

### CUDA

Treat CUDA as an advanced install. Get CPU flows working first, then add the CUDA extra once the host environment is proven.

### Heavy installs

Install only the extras you need. The package is intentionally segmented so narrow use does not require the full stack.

### Remote model sources

Prefer Hugging Face repo ids or raw artifact URLs. Definers rejects obvious HTML responses and Git LFS pointer files during guarded model loading.

### RVC

RVC bootstrap is based on the maintained fork in `YaronKoresh/definers-rvc-files`. If you change this flow, keep it fork-based and LFS-aware.

## Contributing

Contributor workflow now lives in [CONTRIBUTING.md](CONTRIBUTING.md). Use it for setup, validation, code-layout expectations, and pull request rules.

## License

Definers is licensed under the MIT License. See [LICENSE](LICENSE).

## Maintainer

Definers is maintained by Yaron Koresh.
