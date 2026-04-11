# Definers Documentation

This documentation is organized by intent: start-up, runtime policy, capability guides, and reference material.

## Documentation Standard

Every maintained documentation area should answer the same practical questions:

- what this area owns
- which public entry points are stable
- what the normal workflow looks like
- which defaults, configuration values, or environment variables matter
- which failure modes or runtime boundaries users should expect
- where contributors should work when behavior changes

The documentation tree is intentionally split by reader intent rather than by source directory.

## Getting Started

- [Installation](getting-started/installation.md)
- [Quickstart](getting-started/quickstart.md)

## Runtime

- [NumPy and CuPy Compatibility](runtime/numpy-cupy-compat.md)
- [Optional Dependencies](runtime/optional-dependencies.md)

## Capabilities

### Audio

- [Mastering](capabilities/audio/mastering.md)
- [Stems](capabilities/audio/stems.md)
- [Analysis](capabilities/audio/analysis.md)

### Data

- [Preparation](capabilities/data/preparation.md)

### ML

- [Text Workflows](capabilities/ml/text.md)
- [Training and Retrieval](capabilities/ml/training.md)

### Media

- [Image Workflows](capabilities/media/image.md)
- [Video Workflows](capabilities/media/video.md)
- [Web Transfer](capabilities/media/web-transfer.md)

### System

- [Installation and Runtime](capabilities/system/installation-and-runtime.md)

### UI

- [Launchers](capabilities/ui/launchers.md)

The UI launcher guide also owns the contributor rules for workbenches, focused surfaces, app-only launchers, shared progress and output contracts, and optional guided-job patterns.

## Reference

- [CLI Reference](reference/cli.md)
- [Package Map](reference/package-map.md)

## How To Use The Docs

- Start in Getting Started if you are installing or running Definers for the first time.
- Use Runtime when behavior depends on extras, CUDA, download policy, or other environment boundaries.
- Use Capabilities when you need workflow guidance for a feature area.
- Use Reference when you need stable command, import, or package ownership guidance.

## Maintainer Rules

- Put end-user workflows in capability guides, not in reference pages.
- Put stable command and import contracts in reference pages, not in capability guides.
- When a page becomes the owner of a domain, keep it current instead of adding a second overlapping note elsewhere.