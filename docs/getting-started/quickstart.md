# Quickstart

## Python

```python
from definers.data.preparation import prepare_data
from definers.ml.text import summarize
from definers.system import run

dataset = prepare_data(features=["./features.csv"], batch_size=32)
summary = summarize("A long text that needs a short summary.")
run(["ffmpeg", "-i", "input.mp4", "output.wav"])
```

## CLI

```bash
python -m definers --help
definers --version
definers start chat
definers start audio-mastering
definers start video-composer
```

## Next Step

Choose a focused guide under [Capabilities](../README.md) based on the workflow you actually want to build.