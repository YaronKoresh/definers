import sys
from pathlib import Path
from types import ModuleType

from definers import model_installation
from definers.system.download_activity import (
    bind_download_activity_scope,
    clear_download_activity_scope,
    create_download_activity_scope,
    get_download_activity_snapshot,
)


def test_model_runtime_targets_list_domains_and_tasks():
    targets = model_installation.model_runtime_targets()

    assert targets["model-domains"][-1] == "all"
    assert "audio" in targets["model-domains"]
    assert "text" in targets["model-domains"]
    assert "answer" in targets["model-tasks"]
    assert "stems" in targets["model-tasks"]
    assert "rvc" in targets["model-tasks"]


def test_install_model_target_dispatches_domain_targets_in_order():
    installed = []

    result = model_installation.install_model_target(
        "audio",
        kind="model-domain",
        installer=installed.append,
    )

    assert result is True
    assert installed == [
        "music",
        "speech-recognition",
        "audio-classification",
        "tts",
        "stable-whisper",
        "stems",
        "rvc",
    ]


def test_install_model_target_supports_domain_aliases():
    installed = []

    result = model_installation.install_model_target(
        "language",
        kind="model-domain",
        installer=installed.append,
    )

    assert result is True
    assert installed == ["answer", "summary", "translate"]


def test_install_model_target_rejects_unknown_model_target():
    assert (
        model_installation.install_model_target(
            "unknown",
            kind="model-task",
            installer=lambda target: None,
        )
        is False
    )


def test_download_enhanced_rvc_fork_folders_restores_expected_directories(
    monkeypatch, tmp_path
):
    def fake_clone_enhanced_rvc_fork(target_root):
        extracted_repo_root = Path(target_root)
        extracted_repo_root.mkdir(parents=True, exist_ok=True)
        for folder_name in model_installation.ENHANCED_RVC_FORK_FOLDERS:
            source_folder = extracted_repo_root / folder_name
            source_folder.mkdir(parents=True, exist_ok=True)
            (source_folder / "marker.txt").write_text(folder_name)
        return extracted_repo_root

    monkeypatch.setattr(
        "definers.model_installation._clone_enhanced_rvc_fork",
        fake_clone_enhanced_rvc_fork,
    )

    restored_paths = model_installation.download_enhanced_rvc_fork_folders(
        tmp_path
    )

    assert len(restored_paths) == 7
    assert {
        Path(restored_path).name for restored_path in restored_paths
    } == set(model_installation.ENHANCED_RVC_FORK_FOLDERS)
    assert model_installation.has_enhanced_rvc_fork_folders(tmp_path) is True


def test_download_rvc_assets_uses_only_enhanced_fork_folders(
    monkeypatch, tmp_path
):
    expected_paths = tuple(
        str(tmp_path / folder_name)
        for folder_name in model_installation.ENHANCED_RVC_FORK_FOLDERS
    )
    captured_targets = []

    def fake_ensure_enhanced_rvc_fork_folders(target_root="."):
        captured_targets.append(Path(target_root))
        return expected_paths

    monkeypatch.setattr(
        "definers.model_installation.ensure_enhanced_rvc_fork_folders",
        fake_ensure_enhanced_rvc_fork_folders,
    )

    restored_paths = model_installation.download_rvc_assets(tmp_path)

    assert restored_paths == expected_paths
    assert captured_targets == [tmp_path]


def test_enhanced_rvc_fork_folder_paths_default_to_package_root():
    package_root = Path(model_installation.__file__).resolve().parent

    paths = model_installation.enhanced_rvc_fork_folder_paths()

    assert {path.parent for path in paths} == {package_root}
    assert {path.name for path in paths} == set(
        model_installation.ENHANCED_RVC_FORK_FOLDERS
    )


def test_download_stem_models_targets_only_requested_files(
    monkeypatch, tmp_path
):
    created_kwargs = []
    downloaded_models = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            created_kwargs.append(dict(kwargs))

        def download_model_files(self, model_name):
            downloaded_models.append(model_name)

    fake_audio_separator = ModuleType("audio_separator")
    fake_separator_module = ModuleType("audio_separator.separator")
    fake_separator_module.Separator = FakeSeparator

    monkeypatch.setitem(sys.modules, "audio_separator", fake_audio_separator)
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator",
        fake_separator_module,
    )
    monkeypatch.setattr(
        "definers.model_installation.ensure_module_runtime",
        lambda name: name == "audio_separator",
    )
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))

    downloaded = model_installation.download_stem_models(
        ["deverb.ckpt", "deverb.ckpt", "denoise.ckpt"]
    )

    assert downloaded == ("deverb.ckpt", "denoise.ckpt")
    assert downloaded_models == ["deverb.ckpt", "denoise.ckpt"]
    assert created_kwargs == [
        {
            "output_dir": str(tmp_path.resolve()),
            "output_format": "WAV",
            "sample_rate": 44100,
            "use_soundfile": True,
            "log_level": 40,
            "model_file_dir": str(tmp_path.resolve()),
            "demucs_params": {
                "shifts": 2,
                "overlap": 0.25,
                "segments_enabled": True,
            },
            "mdxc_params": {
                "segment_size": 256,
                "overlap": 4,
            },
        }
    ]


def test_hf_snapshot_download_enables_transfer_acceleration_and_reports_activity(
    monkeypatch,
):
    captured_kwargs = {}

    def fake_snapshot_download(**kwargs):
        captured_kwargs.update(kwargs)
        return "snapshot-dir"

    fake_hf_module = ModuleType("huggingface_hub")
    fake_hf_module.snapshot_download = fake_snapshot_download
    fake_hf_module.hf_hub_download = lambda **kwargs: "file.bin"

    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setattr(
        "definers.model_installation.ensure_module_runtime",
        lambda name: name == "huggingface_hub",
    )
    monkeypatch.setattr(
        "definers.model_installation.importlib.util.find_spec",
        lambda name: object() if name == "hf_transfer" else None,
    )
    monkeypatch.delenv("HF_HUB_ENABLE_HF_TRANSFER", raising=False)
    monkeypatch.delenv("HF_XET_HIGH_PERFORMANCE", raising=False)
    monkeypatch.setenv("DEFINERS_HF_MAX_WORKERS", "13")

    scope_id = create_download_activity_scope()
    with bind_download_activity_scope(scope_id):
        result = model_installation.hf_snapshot_download(
            "repo/model",
            allow_patterns=["*.json"],
            item_label="Answer model",
            detail="Downloading answer model source files.",
        )
    activity_snapshot = get_download_activity_snapshot(scope_id)
    clear_download_activity_scope(scope_id)

    assert result == "snapshot-dir"
    assert captured_kwargs == {
        "repo_id": "repo/model",
        "max_workers": 13,
        "allow_patterns": ["*.json"],
    }
    assert activity_snapshot is not None
    assert activity_snapshot.phase == "download"
    assert activity_snapshot.item_label == "Answer model"
    assert "Downloading answer model source files." in (
        activity_snapshot.message
    )
    assert model_installation.os.environ["HF_HUB_ENABLE_HF_TRANSFER"] == "1"
    assert model_installation.os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"
