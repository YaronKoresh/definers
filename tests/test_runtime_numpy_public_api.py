from pathlib import Path

import definers
import definers.runtime_numpy as runtime_numpy


def test_root_exports_runtime_numpy_public_api():
    assert (
        definers.bootstrap_runtime_numpy
        is runtime_numpy.bootstrap_runtime_numpy
    )
    assert definers.patch_numpy_runtime is runtime_numpy.patch_numpy_runtime
    assert definers.get_array_module is runtime_numpy.get_array_module
    assert definers.get_numpy_module is runtime_numpy.get_numpy_module
    assert definers.is_cupy_backend is runtime_numpy.is_cupy_backend
    assert definers.runtime_backend_info is runtime_numpy.runtime_backend_info
    assert definers.runtime_backend_name is runtime_numpy.runtime_backend_name


def test_runtime_backend_info_matches_helpers():
    info = definers.runtime_backend_info()

    assert info["array_module"] == definers.runtime_backend_name()
    assert info["numpy_version"] == definers.get_numpy_module().__version__
    assert info["cupy_enabled"] == definers.is_cupy_backend()
    if info["cupy_enabled"]:
        assert info["array_module"].split(".", 1)[0] == "cupy"
        assert info["cupy_version"] is not None
    else:
        assert info["array_module"] == "numpy"
        assert info["cupy_version"] is None


def test_runtime_backend_selection_stays_centralized():
    src_root = Path(__file__).resolve().parents[1] / "src" / "definers"
    offenders: list[str] = []

    for path in src_root.rglob("*.py"):
        if path.name == "runtime_numpy.py":
            continue
        text = path.read_text(encoding="utf-8")
        if 'import_module("cupy")' in text or 'import_module("numpy")' in text:
            offenders.append(path.relative_to(src_root).as_posix())

    assert offenders == []


def test_direct_numpy_imports_stay_inside_runtime_module_only():
    src_root = Path(__file__).resolve().parents[1] / "src" / "definers"
    offenders: list[str] = []
    allowed_files = {
        "runtime_numpy.py",
    }

    for path in src_root.rglob("*.py"):
        if path.name in allowed_files:
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped in {
                "import numpy as np",
                "import numpy as _np",
                "import numpy",
            } or stripped.startswith("from numpy import "):
                offenders.append(path.relative_to(src_root).as_posix())
                break

    assert offenders == []
