# Definers

Definers is a feature-first Python toolkit for ML, media workflows, data preparation, runtime utilities, and launcher surfaces.

## Install

Definers targets Python 3.10 through 3.12.

```bash
pip install .
pip install ".[audio]"
pip install ".[ml]"
pip install -e ".[dev]"
```

Optional runtime targets can also be installed explicitly through the CLI.

```bash
definers install --list
definers install audio
definers install translate --type task
definers install rvc --type model-task
```

RVC uses only the `YaronKoresh/definers-rvc-files` fork. The runtime bootstrap expects the fork-owned `assets`, `configs`, `docs`, `i18n`, `infer`, `logs`, and `tools` folders. No alternate repository or GUI project is used as an RVC source.

## Quick Start

### Python

```python
from definers.data.preparation import prepare_data
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
definers start video-composer
```

## Package Guide

Import concrete modules instead of broad compatibility facades.

- `definers.audio`: audio analysis, generation, stems, mastering, preview
- `definers.chat`: request normalization and chat handlers
- `definers.cli`: command parsing, routing, install commands
- `definers.data`: preparation, dataset helpers, vectorization, tokenization
- `definers.image`: image utilities and image workflows
- `definers.media`: transfer helpers and shared media utilities
- `definers.ml`: training, inference, answer runtime, text ML helpers
- `definers.system`: filesystem, process, installation, path, runtime helpers
- `definers.text`: translation, normalization, hashing, system messages
- `definers.ui`: launcher entrypoints and app surfaces
- `definers.video`: rendering and video helpers

Preferred import style:

```python
from definers.data.preparation import prepare_data
from definers.ml.answer.service import AnswerService
from definers.ml.text.generation import summarize
from definers.system.paths import normalize_path
from definers.ui.gui_entrypoints import start
```

## Development

```bash
pip install -e ".[dev]"
poe check
```

Useful focused commands:

```bash
poe cli-health
poe ml-health
pytest tests/test_cli.py -q
pytest tests/test_application_ml_answer_services.py tests/test_application_ml_answer_history_preparer.py tests/test_answer.py -q
```

## Runtime Notes

- Optional dependencies must fail cleanly when missing.
- FFmpeg is required for many audio and video paths.
- `sox` is optional and some flows degrade when it is unavailable.
- CUDA should be added only after the CPU path works.
- Guarded model loading accepts Hugging Face references and direct artifact URLs, and rejects obvious HTML or Git LFS pointer responses.

## Contributing

Contributor workflow and validation guidance live in [CONTRIBUTING.md](CONTRIBUTING.md).

### RVC

RVC bootstrap is based on the maintained fork in `YaronKoresh/definers-rvc-files`.

The bootstrap restores the fork folders directly into the Definers package root so the imported RVC modules run as ordinary project modules after initialization.

## Contributing

Contributor workflow now lives in [CONTRIBUTING.md](CONTRIBUTING.md). Use it for setup, validation, code-layout expectations, and pull request rules.

## License

Definers is licensed under the MIT License. See [LICENSE](LICENSE).

## Maintainer

Definers is owned and maintained by Yaron Koresh.
