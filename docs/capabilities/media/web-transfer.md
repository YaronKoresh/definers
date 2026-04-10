# Web Transfer

Definers includes resilient network and artifact-transfer utilities for download-heavy workflows.

## Start Here

- `definers.media.transfer`
- `definers.media.web_transfer`
- `definers.model_installation`
- `definers.system.download_activity`

## What This Area Owns

- staged HTTP and archive downloads
- runtime-aware concurrency policy
- install-time artifact acquisition
- extraction safety and target-path safety
- download activity reporting

## Best Fit

- Google Drive artifact download
- guarded HTTP transfer
- staged extraction and file movement
- text extraction from remote pages when the required web stack is installed

Use this capability when downloads and transfer integrity are part of the product path, not an afterthought.

## Target Transfer Architecture

Definers is standardizing web transfer around four layers:

1. policy selection based on runtime class, protocol support, and configured limits
2. orchestration for retries, failure boundaries, and staging lifecycle
3. concrete transport strategies for HTTP chunked, HTTP range, HTTP/2 multiplexed, and HTTP/3-capable paths
4. install integrations that consume the transfer layer instead of re-implementing download policy

The main goal is to keep fast paths available without hiding behavior behind a maze of incidental fallbacks.

The current owner package is `definers.media.transfer`. The legacy `definers.media.web_transfer` module remains as a compatibility facade and public patch-target seam for existing imports.

The current policy surface is implemented around `http_transfer_capabilities()` and `http_transfer_policy()` in `definers.media.transfer`.

## Runtime Classes

### Restricted Runtime

Use this for daemon processes, embedded event-loop hosts, and other environments where process pools are unsafe.

- default to thread-first behavior
- keep event-loop bridging explicit
- prefer predictable completion and clear failure reporting over maximum parallelism
- use the threaded HTTP range base strategy instead of process workers

### Standard Runtime

Use this for normal local development and CI.

- allow adaptive protocol selection
- keep worker and stream limits bounded by explicit policy
- preserve staged writes and atomic completion

### High-Throughput Runtime

Use this for large artifacts and high-volume model staging.

- favor range downloads, mmap-backed merge paths, and bounded parallelism
- keep activity reporting and cleanup behavior intact under load

### Vendor-Integrated Runtime

Use this when a provider-specific fast path exists, such as model installation over Hugging Face.

- keep provider shortcuts behind the same transfer policy surface where possible
- do not bypass staging, failure reporting, or runtime-class safety rules

## Contributor Rules

- Keep environment-variable parsing centralized rather than scattering it across call sites.
- Do not add a new fallback branch unless it has a clear runtime class and a measurable reason to exist.
- Preserve staged writes, atomic rename, and path-safety checks.
- Keep daemon-safe download behavior explicit.
- Surface install and transfer failures with enough detail that operators can distinguish unsupported fast paths from ordinary network failure.
- Route strategy selection through the transfer policy layer instead of hard-coding new precedence chains inside call sites.