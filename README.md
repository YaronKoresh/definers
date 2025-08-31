# Definers: Your All-in-One Python Toolkit

[![PyPI version](https://badge.fury.io/py/definers.svg)](https://badge.fury.io/py/definers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Definers** is a powerful and versatile Python library designed to streamline a wide range of tasks, from machine learning and AI-powered media generation to everyday file and system operations. Whether you're a data scientist, a developer, or an AI enthusiast, Definers provides the tools you need to get the job done efficiently.

## Features

-   **AI Media Generation**: Create images, videos, music, and summaries using state-of-the-art models.
-   **Machine Learning**: Train and predict with hybrid models, perform linear regression, and get K-means suggestions.
-   **Data Processing**: Effortlessly extract features from audio, video, images, and text.
-   **File & System Utilities**: A robust set of tools for file manipulation, command execution, and environment management.
-   **Web Scraping & Translation**: Extract text from web pages and translate it with ease.
-   **GPU Acceleration**: Built-in support for CUDA via CuPy and RAPIDS for high-performance computing.

## Installation

You can install the lightweight core components of **Definers** directly from its GitHub repository:

```bash
pip install git+[https://github.com/YaronKoresh/definers.git](https://github.com/YaronKoresh/definers.git)
```

## Optional Dependencies

Definers is modular. You can install extra functionality based on your needs directly from the repository. For example, to install the packages required for all AI and deep learning features, use the following format:

```bash
pip install "definers[ai] @ git+[https://github.com/YaronKoresh/definers.git](https://github.com/YaronKoresh/definers.git)"
```

You can install multiple groups at once:

```bash
pip install "definers[audio,visual] @ git+[https://github.com/YaronKoresh/definers.git](https://github.com/YaronKoresh/definers.git)"
```

Here are the available optional groups:

- ai: For deep learning, transformers, and diffusion models.

- audio: For advanced audio processing.

- visual: For image and video manipulation.

- ml: For traditional machine learning tasks.

- web: For web scraping.

- ui: For running Gradio interfaces.

- gpu: For NVIDIA GPU acceleration with RAPIDS.

- all: To install everything.

To install the complete library with all features, use:

```bash
pip install "definers[all] @ git+[https://github.com/YaronKoresh/definers.git](https://github.com/YaronKoresh/definers.git)"
```

## Quick Start

Here’s a quick example of how to use Definers to summarize a piece of text (requires the ai extra):

```python
from definers import summary, init_pretrained_model

init_pretrained_model("summary",turbo=True)

long_text = """
The burgeoning field of artificial intelligence is rapidly transforming industries worldwide.
From healthcare to finance, AI algorithms are optimizing processes, uncovering insights from vast datasets,
and enabling innovations that were previously unimaginable. As the technology matures, it promises to
tackle even more complex challenges, heralding a new era of technological advancement.
"""
short_summary = summary(long_text, max_words=20)

print("Original Text Length:", len(long_text.split()))
print("Summary Text Length:", len(short_summary.split()))
print("\nSummary:")
print(short_summary)
```

## Contributing

Contributions are welcome! If you have a feature request, bug report, or want to contribute to the code, please feel free to open an issue or submit a pull request on our [GitHub repository](https://github.com/YaronKoresh/definers).

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/YaronKoresh/definers/LICENSE) file for details.