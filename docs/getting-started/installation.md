# Installation

Definers targets Python 3.10 through 3.14.

## Base Install

```bash
pip install .
```

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

See [Optional Dependencies](../runtime/optional-dependencies.md) for how extras and runtime installs are intended to work.