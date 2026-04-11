import importlib.util
import os
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest


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


def test_stem_mixer_saves_float_mix_without_int16_roundtrip(monkeypatch):
    saved_call = {}

    def fake_load(file_path, sr=None):
        if str(file_path).endswith("first.wav"):
            return np.array([0.5, -0.5], dtype=np.float32), 44100
        return np.array([0.25, -0.25], dtype=np.float32), 44100

    monkeypatch.setattr(
        STEMS_MODULE,
        "librosa_module",
        lambda: types.SimpleNamespace(
            load=fake_load,
            resample=lambda y, orig_sr, target_sr: np.asarray(
                y, dtype=np.float32
            ),
        ),
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "save_audio",
        lambda **kwargs: (
            saved_call.update(kwargs) or kwargs["destination_path"]
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "definers.system.output_paths",
        types.SimpleNamespace(
            managed_output_path=lambda extension, section, stem: (
                f"{stem}.{extension}"
            )
        ),
    )

    result = STEMS_MODULE.stem_mixer(["first.wav", "second.wav"], "wav")

    assert result == "stem_mix_abc123.wav"
    assert saved_call["sample_rate"] == 44100
    assert saved_call["bit_depth"] == 32
    assert saved_call["destination_path"] == "stem_mix_abc123.wav"
    assert saved_call["audio_signal"].dtype == np.float32
    assert np.allclose(
        saved_call["audio_signal"],
        np.array([0.99, -0.99], dtype=np.float32),
    )


def test_build_mastering_separator_plan_skips_optional_repair_stages_for_clean_material():
    plan = STEMS_MODULE.build_mastering_separator_plan(44100)

    assert plan.preprocess_stages == ()
    assert plan.vocal_restoration_stage is None
    assert plan.instrumental_cleanup_stage is None


def test_build_mastering_separator_plan_enables_repair_stages_for_flagged_material():
    plan = STEMS_MODULE.build_mastering_separator_plan(
        44100,
        quality_flags=("Low-Quality", "Old-Recording"),
    )

    assert len(plan.preprocess_stages) == 2
    assert plan.vocal_restoration_stage is not None
    assert plan.instrumental_cleanup_stage is not None
    assert plan.instrumental_cleanup_stage.model_candidates == (
        "mel_band_roformer_bleed_suppressor_v1.ckpt",
        "mel_band_roformer_instrumental_bleedless_v2_gabox.ckpt",
    )


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


def test_load_audio_separator_class_installs_runtime_hooks(monkeypatch):
    hook_calls: list[bool] = []

    class FakeSeparator:
        pass

    fake_audio_separator = types.ModuleType("audio_separator")
    fake_separator_module = types.ModuleType("audio_separator.separator")
    fake_separator_module.Separator = FakeSeparator

    monkeypatch.setattr(
        "definers.model_installation.install_audio_separator_runtime_hooks",
        lambda: hook_calls.append(True) or True,
    )
    monkeypatch.setitem(sys.modules, "audio_separator", fake_audio_separator)
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator",
        fake_separator_module,
    )

    assert STEMS_MODULE._load_audio_separator_class() is FakeSeparator
    assert hook_calls == [True]


def test_has_local_stem_model_requires_all_companion_artifacts(monkeypatch):
    checked_models = []

    monkeypatch.setattr(
        "definers.model_installation.stem_model_artifacts_ready",
        lambda model_name: checked_models.append(model_name) or True,
    )

    assert STEMS_MODULE._has_local_stem_model("htdemucs_ft.yaml") is True
    assert checked_models == ["htdemucs_ft.yaml"]


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
            output_paths = []
            for input_path in resolved_inputs:
                output_path = (
                    Path(self.output_dir)
                    / f"{Path(input_path).stem}_(No_Bleed)_cleanup.wav"
                )
                output_path.write_text("cleanup", encoding="utf-8")
                output_paths.append(str(output_path))
            return output_paths

    monkeypatch.setattr(
        STEMS_MODULE,
        "_load_audio_separator_class",
        lambda: FakeSeparator,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: True,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_download_runtime_stage_models",
        lambda model_candidates: tuple(model_candidates),
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


def test_run_separator_stage_batch_falls_back_to_single_input_calls(
    tmp_path: Path,
    monkeypatch,
):
    loaded_models: list[str] = []
    separate_calls: list[object] = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]

        def load_model(self, model_filename):
            loaded_models.append(model_filename)

        def separate(self, input_paths):
            separate_calls.append(input_paths)
            if isinstance(input_paths, list):
                raise TypeError(
                    "expected str, bytes or os.PathLike object, not list"
                )
            output_path = (
                Path(self.output_dir)
                / f"{Path(str(input_paths)).stem}_(No_Bleed)_cleanup.wav"
            )
            output_path.write_text("cleanup", encoding="utf-8")
            return [str(output_path)]

    monkeypatch.setattr(
        STEMS_MODULE,
        "_load_audio_separator_class",
        lambda: FakeSeparator,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: True,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_download_runtime_stage_models",
        lambda model_candidates: tuple(model_candidates),
    )

    selected_outputs = STEMS_MODULE._run_separator_stage_batch(
        {
            "drums": "drums.wav",
            "bass": "bass.wav",
        },
        STEMS_MODULE.SeparatorModelStage(
            model_candidates=("cleanup.ckpt",),
            preferred_stems=("no bleed",),
            required=False,
        ),
        str(tmp_path / "cleanup"),
        44100,
    )

    assert loaded_models == ["cleanup.ckpt"]
    assert separate_calls == [
        ["drums.wav", "bass.wav"],
        "drums.wav",
        "bass.wav",
    ]
    assert selected_outputs == {
        "drums": str(tmp_path / "cleanup" / "drums_(No_Bleed)_cleanup.wav"),
        "bass": str(tmp_path / "cleanup" / "bass_(No_Bleed)_cleanup.wav"),
    }


def test_run_separator_stage_skips_remote_fallbacks_for_optional_stage(
    tmp_path: Path,
    monkeypatch,
):
    loaded_models: list[str] = []
    downloaded_models: list[tuple[str, ...]] = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]
            self.loaded_model = ""

        def load_model(self, model_filename):
            self.loaded_model = str(model_filename)
            loaded_models.append(self.loaded_model)

        def separate(self, input_path):
            output_path = (
                Path(self.output_dir)
                / f"{Path(str(input_path)).stem}_(Vocals)_{self.loaded_model}.wav"
            )
            return [str(output_path)]

    monkeypatch.setattr(
        STEMS_MODULE,
        "_load_audio_separator_class",
        lambda: FakeSeparator,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: False,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_download_runtime_stage_models",
        lambda model_candidates: (
            downloaded_models.append(tuple(model_candidates))
            or tuple(model_candidates)
        ),
    )

    model_name, output_files = STEMS_MODULE._run_separator_stage(
        "song.wav",
        STEMS_MODULE.SeparatorModelStage(
            model_candidates=("ghost.ckpt", "good.ckpt"),
            preferred_stems=("vocals",),
            required=False,
        ),
        str(tmp_path / "stage"),
        44100,
        shifts=2,
    )

    assert downloaded_models == [("ghost.ckpt",)]
    assert loaded_models == ["ghost.ckpt"]
    assert model_name == ""
    assert output_files == ()


def test_run_separator_stage_prefers_cached_candidates(
    tmp_path: Path,
    monkeypatch,
):
    loaded_models: list[str] = []
    downloaded_models: list[tuple[str, ...]] = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]
            self.loaded_model = ""

        def load_model(self, model_filename):
            self.loaded_model = str(model_filename)
            loaded_models.append(self.loaded_model)

        def separate(self, input_path):
            output_path = (
                Path(self.output_dir)
                / f"{Path(str(input_path)).stem}_(Vocals)_{self.loaded_model}.wav"
            )
            output_path.write_text("vocals", encoding="utf-8")
            return [str(output_path)]

    monkeypatch.setattr(
        STEMS_MODULE,
        "_load_audio_separator_class",
        lambda: FakeSeparator,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: model_name == "good.ckpt",
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_download_runtime_stage_models",
        lambda model_candidates: (
            downloaded_models.append(tuple(model_candidates))
            or tuple(model_candidates)
        ),
    )

    model_name, output_files = STEMS_MODULE._run_separator_stage(
        "song.wav",
        STEMS_MODULE.SeparatorModelStage(
            model_candidates=("ghost.ckpt", "good.ckpt"),
            preferred_stems=("vocals",),
            required=True,
        ),
        str(tmp_path / "stage"),
        44100,
        shifts=2,
    )

    assert downloaded_models == [("good.ckpt",)]
    assert loaded_models == ["good.ckpt"]
    assert model_name == "good.ckpt"
    assert output_files == (
        str(tmp_path / "stage" / "song_(Vocals)_good.ckpt.wav"),
    )


def test_run_separator_stage_reports_unmaterialized_output_paths(
    tmp_path: Path,
    monkeypatch,
):
    class FakeSeparator:
        def __init__(self, **kwargs):
            self.output_dir = kwargs["output_dir"]

        def load_model(self, model_filename):
            return None

        def separate(self, input_path):
            return [f"{Path(str(input_path)).stem}_(Vocals)_ghost.wav"]

    monkeypatch.setattr(
        STEMS_MODULE,
        "_load_audio_separator_class",
        lambda: FakeSeparator,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: True,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_download_runtime_stage_models",
        lambda model_candidates: tuple(model_candidates),
    )

    with pytest.raises(RuntimeError) as error_info:
        STEMS_MODULE._run_separator_stage(
            "song.wav",
            STEMS_MODULE.SeparatorModelStage(
                model_candidates=("ghost.ckpt",),
                preferred_stems=("vocals",),
                required=True,
            ),
            str(tmp_path / "stage"),
            44100,
            shifts=2,
        )

    assert "not created under" in str(error_info.value)
    assert "song_(Vocals)_ghost.wav" in str(error_info.value)


def test_download_runtime_stage_models_resolves_legacy_aliases(monkeypatch):
    downloaded_models = []

    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: False,
    )
    monkeypatch.setattr(
        "definers.model_installation.resolve_stem_model_filename",
        lambda model_name: {
            "bs_roformer_vocals_resurrection_unwa.ckpt": "bs_roformer_vocals_gabox.ckpt"
        }.get(model_name, model_name),
    )
    monkeypatch.setattr(
        "definers.model_installation.download_stem_models",
        lambda model_names: (
            downloaded_models.append(tuple(model_names)) or tuple(model_names)
        ),
    )

    resolved_models = STEMS_MODULE._download_runtime_stage_models(
        ("bs_roformer_vocals_resurrection_unwa.ckpt",)
    )

    assert resolved_models == ("bs_roformer_vocals_gabox.ckpt",)
    assert downloaded_models == [("bs_roformer_vocals_gabox.ckpt",)]


def test_build_mastering_separator_plan_automatic_prefers_bs_roformer_defaults():
    plan = STEMS_MODULE.build_mastering_separator_plan(
        44100,
        model_name="mastering",
    )

    assert plan.vocal_pair_stage is not None
    assert plan.vocal_pair_stage.model_candidates[0] == (
        "bs_roformer_vocals_resurrection_unwa.ckpt"
    )
    assert plan.reference_split_stage.model_candidates[0] == (
        "bs_roformer_instrumental_resurrection_unwa.ckpt"
    )
    assert plan.four_stem_stage.model_candidates[:2] == (
        "htdemucs_ft.yaml",
        "hdemucs_mmi.yaml",
    )


def test_build_mastering_separator_plan_explicit_demucs_skips_vocal_pair_stage():
    plan = STEMS_MODULE.build_mastering_separator_plan(
        44100,
        model_name="htdemucs_ft.yaml",
    )

    assert plan.vocal_pair_stage is None
    assert plan.reference_split_stage.model_candidates[0] == (
        "MDX23C-8KFFT-InstVoc_HQ.ckpt"
    )


def test_prefetch_mastering_plan_models_prefers_cached_stage_candidates(
    monkeypatch,
):
    downloaded_models = []
    plan = STEMS_MODULE.MasteringSeparatorPlan(
        target_sample_rate=44100,
        quality_flags=(),
        preprocess_stages=(
            STEMS_MODULE.SeparatorModelStage(
                model_candidates=("ghost.ckpt", "good.ckpt"),
                preferred_stems=("vocals",),
                required=False,
            ),
        ),
        vocal_pair_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("pair.ckpt",),
            preferred_stems=("vocals",),
            required=True,
        ),
        reference_split_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("reference.ckpt",),
            preferred_stems=("other",),
            required=False,
        ),
        four_stem_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("four.ckpt",),
            preferred_stems=("drums",),
            required=True,
        ),
        vocal_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("lead.ckpt",),
            preferred_stems=("vocals",),
            required=True,
        ),
        vocal_restoration_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("pair.ckpt",),
            preferred_stems=("vocals",),
            required=False,
        ),
        instrumental_cleanup_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("cleanup.ckpt",),
            preferred_stems=("other",),
            required=False,
        ),
    )

    monkeypatch.setattr(
        STEMS_MODULE,
        "_has_local_stem_model",
        lambda model_name: model_name in {"good.ckpt", "pair.ckpt"},
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_download_runtime_stage_models",
        lambda model_candidates: (
            downloaded_models.append(tuple(model_candidates))
            or tuple(model_candidates)
        ),
    )

    resolved_models = STEMS_MODULE._prefetch_mastering_plan_models(plan)

    assert downloaded_models == [
        (
            "good.ckpt",
            "pair.ckpt",
            "reference.ckpt",
            "four.ckpt",
            "lead.ckpt",
            "cleanup.ckpt",
        )
    ]
    assert resolved_models == downloaded_models[0]


def test_separate_stem_layers_prefetches_mastering_plan_models(
    monkeypatch,
    tmp_path: Path,
):
    fake_plan = STEMS_MODULE.MasteringSeparatorPlan(
        target_sample_rate=44100,
        quality_flags=(),
        preprocess_stages=(),
        vocal_pair_stage=None,
        reference_split_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("reference.ckpt",),
            preferred_stems=("other",),
            required=False,
        ),
        four_stem_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("four.ckpt",),
            preferred_stems=("drums",),
            required=True,
        ),
        vocal_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("lead.ckpt",),
            preferred_stems=("vocals",),
            required=True,
        ),
        vocal_restoration_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("restore.ckpt",),
            preferred_stems=("vocals",),
            required=False,
        ),
        instrumental_cleanup_stage=STEMS_MODULE.SeparatorModelStage(
            model_candidates=("cleanup.ckpt",),
            preferred_stems=("other",),
            required=False,
        ),
    )
    prefetched_plans = []

    monkeypatch.setattr(
        STEMS_MODULE,
        "read_audio",
        lambda audio_path: (44100, np.zeros((2, 32), dtype=np.float32)),
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "build_mastering_separator_plan",
        lambda *args, **kwargs: fake_plan,
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_prefetch_mastering_plan_models",
        lambda plan: prefetched_plans.append(plan) or (),
    )
    monkeypatch.setattr(
        STEMS_MODULE,
        "_run_mastering_separator_pipeline",
        lambda audio_path, output_root, plan, shifts=2: {
            "vocals": str(Path(output_root) / "vocals.wav")
        },
    )

    stem_paths, resolved_output_dir = STEMS_MODULE.separate_stem_layers(
        "song.wav",
        output_dir=str(tmp_path / "stage"),
    )

    assert prefetched_plans == [fake_plan]
    assert stem_paths == {"vocals": str(tmp_path / "stage" / "vocals.wav")}
    assert resolved_output_dir == str(tmp_path / "stage")
