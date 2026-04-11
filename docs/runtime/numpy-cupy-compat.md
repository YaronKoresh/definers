# NumPy and CuPy Compatibility

Definers exposes a public runtime layer for projects that need one code path across NumPy 1.x, NumPy 2.x, and optional CuPy acceleration.

## Public API

- `bootstrap_runtime_numpy()` or `patch_numpy_runtime()` initializes the runtime policy.
- `get_numpy_module()` returns patched real NumPy for interoperability-sensitive code.
- `get_array_module()` returns the active array backend, which may be CuPy when the runtime can support it.
- `is_cupy_backend()`, `runtime_backend_name()`, and `runtime_backend_info()` report the selected backend.

## Usage Rule

- Use `get_numpy_module()` when a dependency expects real `numpy.ndarray` behavior.
- Use `get_array_module()` when the code is backend-safe and can benefit from CuPy.

## Important Boundary

The compatibility layer patches missing runtime symbols, but it does not spoof `numpy.__version__`.

Install-time dependency resolution and version-gated third-party behavior still depend on the real installed NumPy version.