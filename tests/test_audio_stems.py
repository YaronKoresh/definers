import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
AUDIO_ROOT = ROOT / "src" / "definers" / "audio"


def _stub_delete(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
        return None
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
    return None


def _stub_tmp(suffix: str | None = None, keep: bool = True, dir: bool = False):
    if dir:
        return tempfile.mkdtemp()
    normalized_suffix = (suffix or "wav").strip(".")
    handle = tempfile.NamedTemporaryFile(
        suffix=f".{normalized_suffix}", delete=False
    )
    handle.close()
    if not keep:
        _stub_delete(handle.name)
    return handle.name


def _load_stems_module(package_name: str):
    for name in list(sys.modules):
        if name == package_name or name.startswith(f"{package_name}."):
            del sys.modules[name]

    saved_modules = {
        name: sys.modules.get(name)
        for name in (
            "librosa",
            "definers.constants",
            "definers.logger",
            "definers.system",
            "definers.text",
        )
    }
    sys.modules["librosa"] = types.SimpleNamespace(
        resample=lambda y, orig_sr, target_sr: np.asarray(y, dtype=np.float32)
    )
    sys.modules["definers.constants"] = types.SimpleNamespace(MODELS={})
    sys.modules["definers.logger"] = types.SimpleNamespace(
        init_logger=lambda: types.SimpleNamespace(
            warning=lambda *args, **kwargs: None,
            info=lambda *args, **kwargs: None,
            exception=lambda *args, **kwargs: None,
        )
    )
    sys.modules["definers.system"] = types.SimpleNamespace(
        catch=lambda *args, **kwargs: None,
        delete=_stub_delete,
        tmp=_stub_tmp,
    )
    sys.modules["definers.text"] = types.SimpleNamespace(
        random_string=lambda: "abc123"
    )

    parent_name, _, _ = package_name.rpartition(".")
    if parent_name:
        parent_package = types.ModuleType(parent_name)
        parent_package.__path__ = [str(ROOT / "src" / "definers")]
        sys.modules[parent_name] = parent_package

    package = types.ModuleType(package_name)
    package.__path__ = [str(AUDIO_ROOT)]
    sys.modules[package_name] = package

    io_module = types.ModuleType(f"{package_name}.io")
    io_module.read_audio = lambda path: (
        44100,
        np.zeros((2, 32), dtype=np.float32),
    )
    io_module.save_audio = lambda **kwargs: kwargs["destination_path"]
    sys.modules[f"{package_name}.io"] = io_module

    utils_module = types.ModuleType(f"{package_name}.utils")
    utils_module.normalize_audio_to_peak = lambda path: path
    sys.modules[f"{package_name}.utils"] = utils_module

    try:
        module = _load_module(
            f"{package_name}.stems",
            AUDIO_ROOT / "stems.py",
        )
    finally:
        for name, module_value in saved_modules.items():
            if module_value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module_value

    return module


STEMS_MODULE = _load_stems_module("_test_audio_stems_pkg.audio")


def test_build_separator_kwargs_includes_demucs_shifts():
    kwargs = STEMS_MODULE._build_separator_kwargs(
        "output",
        44100,
        shifts=5,
    )

    assert kwargs["demucs_params"] == {
        "shifts": 5,
        "overlap": 0.25,
        "segments_enabled": True,
    }
    assert kwargs["mdxc_params"] == {
        "segment_size": 256,
        "overlap": 4,
    }


def test_run_separator_stage_batch_reuses_loaded_model_for_cleanup(
    tmp_path: Path,
    monkeypatch,
):
    created_kwargs: list[dict[str, object]] = []
    loaded_models: list[str] = []
    separate_calls: list[list[str]] = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            created_kwargs.append(dict(kwargs))
            self.output_dir = kwargs["output_dir"]

        def load_model(self, model_filename):
            loaded_models.append(model_filename)

        def separate(self, input_paths):
            if isinstance(input_paths, str):
                input_paths = [input_paths]
            resolved_inputs = [str(path) for path in input_paths]
            separate_calls.append(resolved_inputs)
            return [
                str(
                    Path(self.output_dir)
                    / f"{Path(input_path).stem}_(No_Bleed)_cleanup.wav"
                )
                for input_path in resolved_inputs
            ]

    monkeypatch.setattr(
        STEMS_MODULE,
        "_load_audio_separator_class",
        lambda: FakeSeparator,
    )

    selected_outputs = STEMS_MODULE._run_separator_stage_batch(
        {
            "drums": "drums.wav",
            "bass": "bass.wav",
            "other": "other.wav",
        },
        STEMS_MODULE.SeparatorModelStage(
            model_candidates=("cleanup.ckpt",),
            preferred_stems=("no bleed",),
            required=False,
        ),
        str(tmp_path / "cleanup"),
        44100,
        shifts=4,
    )

    assert selected_outputs == {
        "drums": str(tmp_path / "cleanup" / "drums_(No_Bleed)_cleanup.wav"),
        "bass": str(tmp_path / "cleanup" / "bass_(No_Bleed)_cleanup.wav"),
        "other": str(tmp_path / "cleanup" / "other_(No_Bleed)_cleanup.wav"),
    }
    assert loaded_models == ["cleanup.ckpt"]
    assert separate_calls == [["drums.wav", "bass.wav", "other.wav"]]
    assert created_kwargs[0]["demucs_params"] == {
        "shifts": 4,
        "overlap": 0.25,
        "segments_enabled": True,
    }
