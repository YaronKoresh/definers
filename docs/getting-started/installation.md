# Installation

Definers targets Python 3.10 through 3.14.

## Base Install

```bash
pip install .
```

The base install includes spreadsheet-backed `xlsx` support through `openpyxl`, so tabular training and data loading do not need a separate Excel extra.

## Domain Extras

```bash
pip install ".[audio]"
pip install ".[image]"
pip install ".[video]"
pip install ".[ml]"
pip install ".[nlp]"
pip install ".[web]"
pip install ".[cuda]"
pip install ".[dev]"
```

## Runtime Install Command

```bash
definers install --list
definers install audio
definers install translate --type task
definers install rvc --type model-task
```

## Linux Bootstrap

For Linux hosts that need the common system packages behind the official audio, image, video, and hosted UI paths, Definers also exposes a bootstrap helper:

```bash
python -c "from definers.system import apt_install; apt_install()"
```

That bootstrap covers `ffmpeg`, `libsndfile1`, `libgl1`, and `libglib2.0-0` alongside the existing audio and visual toolchain.

See [Optional Dependencies](../runtime/optional-dependencies.md) for how extras and runtime installs are intended to work.