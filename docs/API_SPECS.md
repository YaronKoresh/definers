# Definers API Specifications

## System Architecture Overview

Definers provides a modular utility toolkit with extension-oriented boundaries for reliability, data handling, media processing, and system orchestration.

### Capabilities

- `definers.capabilities.CircuitBreaker`
	- Sync and async operation gates.
	- Explicit state machine: `CLOSED`, `OPEN`, `HALF_OPEN`.
	- Snapshot API for runtime diagnostics.
- `definers.capabilities.ExponentialBackoffDelay`
	- Configurable base delay, multiplier, max delay, and jitter.
	- Retry spacing strategy abstraction through `RetryDelayStrategy` protocol.
- `definers.capabilities.with_retry`
	- Async retry decorator.
	- Selective exception retry boundaries with deterministic failure re-raise.
- `definers.web.ResourceRetrievalOrchestrator`
	- Strategy-driven transfer execution with integration error boundaries.

## Execution Patterns

- Resilience boundaries are composed through retry and circuit-breaker primitives.
- Extension points use protocols and strategy implementations instead of direct hard-coupled logic.
- Critical failures are surfaced explicitly through typed exceptions and state snapshots.

### Error Boundaries

- Unavailable downstream integrations can be isolated by opening the circuit after threshold failures.
- Retry logic is constrained by max-attempt policy and retryable exception filtering.
- Transfer orchestration catches integration faults and returns safe status outcomes.

### Data preparation helpers

- `prepare_data(remote_src=None, features=None, labels=None, url_type=None, revision=None, drop=None, order_by=None, stratify=None, val_frac=0.0, test_frac=0.0, batch_size=1)` loads a dataset from a HuggingFace source or local feature/label files, applies optional column drops, orders the examples (shuffle, sort, or user-supplied key), and returns a `TrainingData` object containing one or more `torch.utils.data.DataLoader` instances. Splitting into train/val/ test sets with stratification and configurable fractions is supported.
- `TrainingData` dataclass returned by `prepare_data`; fields are `train`, `val`,
  `test` (each a DataLoader) and `metadata` (a dict recording split parameters).
