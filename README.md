# Definers

Definers is a modular Python platform for teams that build AI, media, and data products and want one serious codebase instead of a pile of disconnected utilities.

It brings together workflow-grade audio, text, image, video, data preparation, runtime compatibility, and launcher surfaces behind a concrete module structure that is designed to stay credible under real project pressure.

## What Definers Is Built For

- Shipping AI and media workflows without rewriting the surrounding infrastructure every time the stack changes.
- Keeping CPU-only environments usable while still unlocking GPU-backed acceleration where the runtime can support it.
- Giving teams one place for preparation, inference, automation, and runtime.

## Capability Areas

- Audio workflows for mastering, stems, analysis, cleanup, and generation.
- Data workflows for preparation, tokenization, vectorization, and dataset assembly.
- ML workflows for text processing, training, inference, and retrieval-oriented features.
- Image and video workflows for generation, composition, enhancement, and rendering.
- System and runtime workflows for installation, process control, download handling, and compatibility.
- Focused launcher surfaces for domain-specific applications instead of a single overloaded interface.

## Documentation

- Start with [docs/README.md](docs/README.md).
- Installation and first-run guidance lives in [docs/getting-started/installation.md](docs/getting-started/installation.md) and [docs/getting-started/quickstart.md](docs/getting-started/quickstart.md).
- Runtime compatibility guidance lives in [docs/runtime/numpy-cupy-compat.md](docs/runtime/numpy-cupy-compat.md).
- Capability guides are grouped under [docs/capabilities](docs/capabilities).
- CLI and package reference notes live under [docs/reference](docs/reference).

## Project Standards

- Python support targets 3.10 through 3.14.
- The package is structured to keep runtime policy, optional dependencies, and user-facing launchers explicit.
- Contributor workflow and validation guidance live in [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Definers is licensed under the MIT License. See [LICENSE](LICENSE).
