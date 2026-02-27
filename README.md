# Definers: Your All-in-One Python Toolkit

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Definers** is a powerful and versatile Python library designed to streamline a wide range of tasks, from machine learning and AI-powered media generation to everyday file and system operations. Whether you're a data scientist, a developer, or an AI enthusiast, Definers provides the tools you need to get the job done efficiently.

## Features

-   **AI Media Generation**: Create images, videos, music, and summaries using state-of-the-art models.
-   **Machine Learning**: Train and predict with hybrid models, perform linear regression, and get K-means suggestions.
-   **Data Processing**: Effortlessly extract features from audio, video, images, and text.
-   **File & System Utilities**: A robust set of tools for file manipulation, command execution, and environment management.
-   **Web Scraping & Translation**: Extract text from web pages and translate it with ease.
-   **GPU Acceleration**: Built-in support for CUDA via CuPy and RAPIDS for high-performance computing.

## Modular Architecture

The package is organized into focused submodules for clean, modular access:

| Submodule | Description |
|-----------|-------------|
| `definers._constants` | Constants, mappings, and configuration data |
| `definers._logger` | Logger initialization |
| `definers._system` | OS, filesystem, process, and package utilities |
| `definers._cuda` | CUDA and GPU utilities |
| `definers._data` | NumPy and data processing utilities |
| `definers._text` | NLP, text processing, translation, and cryptographic utilities |
| `definers._audio` | Audio processing functions |
| `definers._image` | Image processing functions |
| `definers._video` | Video processing functions |
| `definers._ml` | Machine learning, training, and model management |
| `definers._web` | Web, network, and download utilities |
| `definers._chat` | Chat, UI, and Gradio functions |

All symbols are also available directly from the top-level `definers` package for backward compatibility.

## Installation

### Core Only (minimal dependencies)

```bash
pip install git+https://github.com/YaronKoresh/definers.git
```

### With Optional Groups

Install only the components you need:

```bash
# Audio processing
pip install "definers[audio] @ git+https://github.com/YaronKoresh/definers.git"

# Image processing
pip install "definers[image] @ git+https://github.com/YaronKoresh/definers.git"

# Video processing
pip install "definers[video] @ git+https://github.com/YaronKoresh/definers.git"

# Machine learning
pip install "definers[ml] @ git+https://github.com/YaronKoresh/definers.git"

# Natural language processing
pip install "definers[nlp] @ git+https://github.com/YaronKoresh/definers.git"

# Web scraping and UI
pip install "definers[web] @ git+https://github.com/YaronKoresh/definers.git"

# GPU support
pip install "definers[gpu] @ git+https://github.com/YaronKoresh/definers.git"

# Everything (all optional groups)
pip install "definers[all] @ git+https://github.com/YaronKoresh/definers.git"

# Multiple groups at once
pip install "definers[audio,ml,web] @ git+https://github.com/YaronKoresh/definers.git"
```

### With CUDA Support

If you don't have CUDA installed, and you are using linux with bash, you can install CUDA Toolkit, by using the following format:

```bash
wget https://raw.githubusercontent.com/YaronKoresh/definers/refs/heads/main/scripts/setup_cuda.sh
chmod +x setup_cuda.sh
./setup_cuda.sh
```

Now, after you have CUDA installed, you can install definers with CUDA acceleration:

```bash
pip install "definers[all,cuda] @ git+https://github.com/YaronKoresh/definers.git" --extra-index-url https://pypi.nvidia.com
```

## Quick Start

Here's a quick example of how to use Definers to summarize a piece of text:

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
