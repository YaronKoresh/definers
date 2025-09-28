# Definers: Your All-in-One Python Toolkit

[![CI Tests](https://github.com/YaronKoresh/definers/actions/workflows/tests-status.yml/badge.svg?branch=main&cache_bust=1)](https://github.com/YaronKoresh/definers/actions/workflows/tests-status.yml)
[![codecov](https://codecov.io/gh/YaronKoresh/definers/graph/badge.svg?token=QRZ33FO9N3&cache_bust=1)](https://codecov.io/gh/YaronKoresh/definers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Definers** is a powerful and versatile Python library designed to streamline a wide range of tasks, from machine learning and AI-powered media generation to everyday file and system operations. Whether you're a data scientist, a developer, or an AI enthusiast, Definers provides the tools you need to get the job done efficiently.

## Features

-   **AI Media Generation**: Create images, videos, music, and summaries using state-of-the-art models.
-   **Machine Learning**: Train and predict with hybrid models, perform linear regression, and get K-means suggestions.
-   **Data Processing**: Effortlessly extract features from audio, video, images, and text.
-   **File & System Utilities**: A robust set of tools for file manipulation, command execution, and environment management.
-   **Web Scraping & Translation**: Extract text from web pages and translate it with ease.
-   **GPU Acceleration**: Built-in support for CUDA via CuPy and RAPIDS for high-performance computing.

## Installation without CUDA Support

You can install the core components of **Definers** directly from its GitHub repository:

```bash
pip install git+https://github.com/YaronKoresh/definers.git
```

### Installation with CUDA Support

If you don't have CUDA installed, and you are using linux with bash, you can install CUDA Toolkit, by using the following format:

```bash
wget https://raw.githubusercontent.com/YaronKoresh/definers/refs/heads/main/scripts/setup_cuda.sh
chmod +x setup_cuda.sh
./setup_cuda.sh
```

Now, after you have CUDA installed, you can install definers with CUDA acceleration:

```bash
pip install "definers[cuda] @ git+https://github.com/YaronKoresh/definers.git" --extra-index-url https://pypi.nvidia.com
```

## Quick Start

Here’s a quick example of how to use Definers to summarize a piece of text:

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


