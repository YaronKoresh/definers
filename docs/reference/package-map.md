# Package Map

Import concrete owner modules when possible.

## Primary Packages

- `definers.audio` for audio production, analysis, stems, and mastering
- `definers.data` for preparation, tokenization, datasets, and vectorization
- `definers.image` for image workflows and helpers
- `definers.media` for transfer and shared media helpers
- `definers.ml` for training, inference, text, and health APIs
- `definers.runtime_numpy` for NumPy and CuPy runtime compatibility
- `definers.system` for installation, paths, processes, and runtime utilities
- `definers.text` for translation and text normalization
- `definers.ui` for launcher entrypoints and app surfaces
- `definers.video` for rendering and video helpers

Use package-level facades for discovery, then move to the concrete owner module for implementation work.