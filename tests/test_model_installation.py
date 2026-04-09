import sys
from pathlib import Path
from types import ModuleType

from definers import model_installation
from definers.system.download_activity import (
    bind_download_activity_scope,
    clear_download_activity_scope,
    create_activity_reporter,
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
    monkeypatch.setattr(
        model_installation,
        "install_audio_separator_runtime_hooks",
        lambda: True,
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


def test_download_stem_models_prefers_direct_separator_artifacts(
    monkeypatch, tmp_path
):
    direct_downloads = []
    fallback_downloads = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.model_file_dir = kwargs["model_file_dir"]

        def list_supported_model_files(self):
            return {
                "MDXC": {
                    "Direct model": {
                        "filename": "deverb.ckpt",
                        "download_files": (
                            "deverb.ckpt",
                            "config.yaml",
                        ),
                    }
                }
            }

        def download_model_files(self, model_name):
            fallback_downloads.append(model_name)

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
    monkeypatch.setattr(
        model_installation,
        "install_audio_separator_runtime_hooks",
        lambda: True,
    )
    monkeypatch.setattr(
        model_installation,
        "supported_audio_separator_model_files",
        lambda: frozenset({"deverb.ckpt"}),
    )
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_supported_model_catalog",
        lambda model_root: {
            "MDXC": {
                "Direct model": {
                    "filename": "deverb.ckpt",
                    "download_files": (
                        "deverb.ckpt",
                        "config.yaml",
                    ),
                }
            }
        },
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        lambda source_url, target_path, **kwargs: (
            direct_downloads.append(
                (
                    source_url,
                    Path(target_path).name,
                    kwargs["completed"],
                    kwargs["total"],
                )
            )
            or str(target_path)
        ),
    )
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))

    downloaded = model_installation.download_stem_models(["deverb.ckpt"])

    assert downloaded == ("deverb.ckpt",)
    assert fallback_downloads == []
    assert direct_downloads == [
        (
            f"{model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/deverb.ckpt",
            "deverb.ckpt",
            1,
            2,
        ),
        (
            f"{model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/config.yaml",
            "config.yaml",
            2,
            2,
        ),
    ]


def test_patched_audio_separator_download_prefers_nomadkaraoke_mirror(
    monkeypatch, tmp_path
):
    download_calls = []
    source_url = (
        model_installation._AUDIO_SEPARATOR_PUBLIC_REPO_URL_PREFIX
        + "/mel_band_roformer_instrumental_instv7_gabox.ckpt"
    )
    fallback_url = (
        model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX
        + "/mel_band_roformer_instrumental_instv7_gabox.ckpt"
    )

    def fake_direct_download_artifact(candidate_url, target_path, **kwargs):
        download_calls.append((candidate_url, Path(target_path).name))
        if candidate_url == source_url:
            raise FileNotFoundError("404")
        return str(target_path)

    monkeypatch.setattr(
        model_installation,
        "_artifact_is_ready",
        lambda target_path: False,
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        fake_direct_download_artifact,
    )

    model_installation._patched_audio_separator_download_file_if_not_exists(
        object(),
        source_url,
        str(tmp_path / "mel_band_roformer_instrumental_instv7_gabox.ckpt"),
    )

    assert download_calls == [
        (fallback_url, "mel_band_roformer_instrumental_instv7_gabox.ckpt"),
    ]


def test_patched_audio_separator_download_prefers_nomadkaraoke_demucs_mirror(
    monkeypatch, tmp_path
):
    download_calls = []
    source_url = (
        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
        "f7e0c4bc-ba3fe64a.th"
    )
    fallback_url = (
        model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX
        + "/f7e0c4bc-ba3fe64a.th"
    )

    monkeypatch.setattr(
        model_installation,
        "_artifact_is_ready",
        lambda target_path: False,
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        lambda candidate_url, target_path, **kwargs: (
            download_calls.append((candidate_url, Path(target_path).name))
            or str(target_path)
        ),
    )

    model_installation._patched_audio_separator_download_file_if_not_exists(
        object(),
        source_url,
        str(tmp_path / "f7e0c4bc-ba3fe64a.th"),
    )

    assert download_calls == [
        (fallback_url, "f7e0c4bc-ba3fe64a.th"),
    ]


def test_patched_audio_separator_download_raises_runtime_error_when_all_urls_fail(
    monkeypatch, tmp_path
):
    def fake_direct_download_artifact(candidate_url, target_path, **kwargs):
        raise FileNotFoundError("missing")

    monkeypatch.setattr(
        model_installation,
        "_artifact_is_ready",
        lambda target_path: False,
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        fake_direct_download_artifact,
    )

    try:
        model_installation._patched_audio_separator_download_file_if_not_exists(
            object(),
            "https://example.com/missing.ckpt",
            str(tmp_path / "missing.ckpt"),
        )
    except RuntimeError as error:
        assert "missing" in str(error)
    else:
        assert False


def test_download_audio_separator_model_direct_tries_secondary_url(
    monkeypatch, tmp_path
):
    download_calls = []
    primary_url = "https://example.com/primary.ckpt"
    secondary_url = "https://example.com/secondary.ckpt"

    class FakeSeparator:
        model_file_dir = str(tmp_path)

    def fake_direct_download_artifact(candidate_url, target_path, **kwargs):
        download_calls.append((candidate_url, Path(target_path).name))
        if candidate_url == primary_url:
            raise FileNotFoundError("primary failed")
        return str(target_path)

    monkeypatch.setattr(
        model_installation,
        "_audio_separator_model_targets",
        lambda model_name, model_root=None: (
            ("model.ckpt", (primary_url, secondary_url)),
        ),
    )
    monkeypatch.setattr(
        model_installation,
        "_artifact_is_ready",
        lambda target_path: False,
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        fake_direct_download_artifact,
    )

    model_installation._download_audio_separator_model_direct(
        FakeSeparator(),
        "model.ckpt",
    )

    assert download_calls == [
        (primary_url, "model.ckpt"),
        (secondary_url, "model.ckpt"),
    ]


def test_supported_audio_separator_model_files_includes_remote_demucs_models(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        "definers.model_installation.ensure_module_runtime",
        lambda name: name == "audio_separator",
    )
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_packaged_models_payload",
        lambda: {},
    )
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_model_scores_payload",
        lambda: {},
    )
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_download_checks_payload",
        lambda model_root: {
            "demucs_download_list": {
                "Demucs v4: htdemucs_ft": {
                    "f7e0c4bc-ba3fe64a.th": "https://example.com/f7e0c4bc-ba3fe64a.th",
                    "d12395a8-e57c48e6.th": "https://example.com/d12395a8-e57c48e6.th",
                    "htdemucs_ft.yaml": "https://example.com/htdemucs_ft.yaml",
                }
            }
        },
    )
    model_installation.supported_audio_separator_model_files.cache_clear()

    supported_files = model_installation.supported_audio_separator_model_files()

    assert "htdemucs_ft.yaml" in supported_files


def test_stem_model_artifacts_ready_requires_all_companion_files(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_supported_model_catalog",
        lambda model_root: {
            "Demucs": {
                "Demucs v4: htdemucs_ft": {
                    "filename": "htdemucs_ft.yaml",
                    "download_files": (
                        "https://example.com/f7e0c4bc-ba3fe64a.th",
                        "https://example.com/d12395a8-e57c48e6.th",
                        "https://example.com/htdemucs_ft.yaml",
                    ),
                }
            }
        },
    )

    (tmp_path / "htdemucs_ft.yaml").write_text("yaml", encoding="utf-8")

    assert (
        model_installation.stem_model_artifacts_ready(
            "htdemucs_ft.yaml",
            model_root=str(tmp_path),
        )
        is False
    )

    (tmp_path / "f7e0c4bc-ba3fe64a.th").write_text(
        "weights-1",
        encoding="utf-8",
    )
    (tmp_path / "d12395a8-e57c48e6.th").write_text(
        "weights-2",
        encoding="utf-8",
    )

    assert (
        model_installation.stem_model_artifacts_ready(
            "htdemucs_ft.yaml",
            model_root=str(tmp_path),
        )
        is True
    )


def test_download_stem_models_prefers_direct_demucs_artifacts(
    monkeypatch, tmp_path
):
    direct_downloads = []
    fallback_downloads = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.model_file_dir = kwargs["model_file_dir"]

        def download_model_files(self, model_name):
            fallback_downloads.append(model_name)

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
    monkeypatch.setattr(
        model_installation,
        "install_audio_separator_runtime_hooks",
        lambda: True,
    )
    monkeypatch.setattr(
        model_installation,
        "supported_audio_separator_model_files",
        lambda: frozenset({"htdemucs_ft.yaml"}),
    )
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_supported_model_catalog",
        lambda model_root: {
            "Demucs": {
                "Demucs v4: htdemucs_ft": {
                    "filename": "htdemucs_ft.yaml",
                    "download_files": (
                        "https://example.com/f7e0c4bc-ba3fe64a.th",
                        "https://example.com/d12395a8-e57c48e6.th",
                        "https://example.com/htdemucs_ft.yaml",
                    ),
                }
            }
        },
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        lambda source_url, target_path, **kwargs: (
            direct_downloads.append(
                (
                    source_url,
                    Path(target_path).name,
                    kwargs["completed"],
                    kwargs["total"],
                )
            )
            or str(target_path)
        ),
    )
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))

    downloaded = model_installation.download_stem_models(["htdemucs_ft.yaml"])

    assert downloaded == ("htdemucs_ft.yaml",)
    assert fallback_downloads == []
    assert direct_downloads == [
        (
            "https://example.com/f7e0c4bc-ba3fe64a.th",
            "f7e0c4bc-ba3fe64a.th",
            1,
            3,
        ),
        (
            "https://example.com/d12395a8-e57c48e6.th",
            "d12395a8-e57c48e6.th",
            2,
            3,
        ),
        (
            "https://example.com/htdemucs_ft.yaml",
            "htdemucs_ft.yaml",
            3,
            3,
        ),
    ]


def test_download_stem_models_prefers_nomadkaraoke_demucs_artifacts(
    monkeypatch, tmp_path
):
    direct_downloads = []
    fallback_downloads = []

    class FakeSeparator:
        def __init__(self, **kwargs):
            self.model_file_dir = kwargs["model_file_dir"]

        def download_model_files(self, model_name):
            fallback_downloads.append(model_name)

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
    monkeypatch.setattr(
        model_installation,
        "install_audio_separator_runtime_hooks",
        lambda: True,
    )
    monkeypatch.setattr(
        model_installation,
        "supported_audio_separator_model_files",
        lambda: frozenset({"htdemucs_ft.yaml"}),
    )
    monkeypatch.setattr(
        model_installation,
        "_audio_separator_supported_model_catalog",
        lambda model_root: {
            "Demucs": {
                "Demucs v4: htdemucs_ft": {
                    "filename": "htdemucs_ft.yaml",
                    "download_files": (
                        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/f7e0c4bc-ba3fe64a.th",
                        "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/d12395a8-e57c48e6.th",
                        "https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs_ft.yaml",
                    ),
                }
            }
        },
    )
    monkeypatch.setattr(
        model_installation,
        "_direct_download_artifact",
        lambda source_url, target_path, **kwargs: (
            direct_downloads.append(
                (
                    source_url,
                    Path(target_path).name,
                    kwargs["completed"],
                    kwargs["total"],
                )
            )
            or str(target_path)
        ),
    )
    monkeypatch.setenv("AUDIO_SEPARATOR_MODEL_DIR", str(tmp_path))

    downloaded = model_installation.download_stem_models(["htdemucs_ft.yaml"])

    assert downloaded == ("htdemucs_ft.yaml",)
    assert fallback_downloads == []
    assert direct_downloads == [
        (
            f"{model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/f7e0c4bc-ba3fe64a.th",
            "f7e0c4bc-ba3fe64a.th",
            1,
            3,
        ),
        (
            f"{model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/d12395a8-e57c48e6.th",
            "d12395a8-e57c48e6.th",
            2,
            3,
        ),
        (
            f"{model_installation._AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/htdemucs_ft.yaml",
            "htdemucs_ft.yaml",
            3,
            3,
        ),
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
    monkeypatch.setenv("DEFINERS_FAST_HF_DOWNLOADS", "0")
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
    assert model_installation.os.environ["HF_TRANSFER_CONCURRENCY"] == "13"
    assert model_installation.os.environ["HF_XET_HIGH_PERFORMANCE"] == "1"


def test_hf_file_download_prefers_fast_direct_download(monkeypatch, tmp_path):
    download_calls = []

    def fake_download_file(source_url, destination_path):
        download_calls.append((source_url, destination_path))
        return destination_path

    monkeypatch.setenv("DEFINERS_FAST_HF_DOWNLOADS", "1")
    monkeypatch.setenv("DEFINERS_HF_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(
        "definers.media.web_transfer.download_file",
        fake_download_file,
    )

    result = model_installation.hf_file_download(
        "owner/repo",
        "weights/model.safetensors",
        revision="rev-1",
    )

    expected_path = (
        tmp_path / "owner" / "repo" / "rev-1" / "weights" / "model.safetensors"
    )
    assert result == str(expected_path.resolve())
    assert download_calls == [
        (
            "https://huggingface.co/owner/repo/resolve/rev-1/weights/model.safetensors?download=1",
            str(expected_path.resolve()),
        )
    ]


def test_hf_snapshot_download_prefers_fast_direct_snapshot_download(
    monkeypatch, tmp_path
):
    download_calls = []

    def fake_download_file(source_url, destination_path):
        download_calls.append((source_url, destination_path))
        return destination_path

    monkeypatch.setenv("DEFINERS_FAST_HF_DOWNLOADS", "1")
    monkeypatch.setenv("DEFINERS_HF_MAX_WORKERS", "1")
    monkeypatch.setattr(
        model_installation,
        "_huggingface_repo_files",
        lambda repo_id, revision=None: (
            "config.json",
            "weights/model.safetensors",
            "notes.md",
        ),
    )
    monkeypatch.setattr(
        "definers.media.web_transfer.download_file",
        fake_download_file,
    )

    snapshot_root = tmp_path / "snapshot"
    result = model_installation.hf_snapshot_download(
        "owner/repo",
        revision="rev-2",
        allow_patterns=["*.json", "*.safetensors"],
        local_dir=str(snapshot_root),
    )

    assert result == str(snapshot_root.resolve())
    assert sorted(download_calls) == sorted(
        [
            (
                "https://huggingface.co/owner/repo/resolve/rev-2/config.json?download=1",
                str((snapshot_root / "config.json").resolve()),
            ),
            (
                "https://huggingface.co/owner/repo/resolve/rev-2/weights/model.safetensors?download=1",
                str(
                    (snapshot_root / "weights" / "model.safetensors").resolve()
                ),
            ),
        ]
    )


def test_hf_max_workers_defaults_to_higher_parallel_budget(monkeypatch):
    monkeypatch.delenv("DEFINERS_HF_MAX_WORKERS", raising=False)
    monkeypatch.setattr(
        model_installation.os,
        "cpu_count",
        lambda: 8,
    )

    assert model_installation._huggingface_max_workers() == 96


def test_fast_hf_snapshot_download_uses_full_worker_budget(monkeypatch):
    captured = {}

    class FakeFuture:
        def __init__(self, value):
            self._value = value

        def result(self):
            return self._value

    class FakeExecutor:
        def __init__(self, *, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def submit(self, function, *args, **kwargs):
            return FakeFuture(function(*args, **kwargs))

    monkeypatch.setattr(
        model_installation,
        "_huggingface_repo_files",
        lambda repo_id, revision=None: tuple(
            f"weights/model_{index}.bin" for index in range(200)
        ),
    )
    monkeypatch.setattr(
        model_installation,
        "_fast_hf_file_download",
        lambda repo_id, filename, **kwargs: filename,
    )
    monkeypatch.setattr(
        model_installation,
        "ThreadPoolExecutor",
        FakeExecutor,
    )
    monkeypatch.setattr(
        model_installation.os,
        "cpu_count",
        lambda: 8,
    )
    monkeypatch.delenv("DEFINERS_HF_MAX_WORKERS", raising=False)

    result = model_installation._fast_hf_snapshot_download("owner/repo")

    assert result
    assert captured["max_workers"] == 96


def test_install_fast_huggingface_download_hooks_patches_imported_aliases(
    monkeypatch, tmp_path
):
    download_calls = []
    fake_hf_module = ModuleType("huggingface_hub")
    fake_transformers_hub = ModuleType("transformers.utils.hub")

    def original_hf_download(*args, **kwargs):
        raise AssertionError("original hf_hub_download should not be used")

    def original_snapshot_download(*args, **kwargs):
        raise AssertionError("original snapshot_download should not be used")

    fake_hf_module.hf_hub_download = original_hf_download
    fake_hf_module.snapshot_download = original_snapshot_download
    fake_transformers_hub.hf_hub_download = original_hf_download
    fake_transformers_hub.snapshot_download = original_snapshot_download

    monkeypatch.setattr(
        model_installation,
        "_HUGGINGFACE_ORIGINAL_HF_HUB_DOWNLOAD",
        None,
    )
    monkeypatch.setattr(
        model_installation,
        "_HUGGINGFACE_ORIGINAL_SNAPSHOT_DOWNLOAD",
        None,
    )
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf_module)
    monkeypatch.setitem(
        sys.modules,
        "transformers.utils.hub",
        fake_transformers_hub,
    )
    monkeypatch.setattr(
        "definers.model_installation.ensure_module_runtime",
        lambda name: name == "huggingface_hub",
    )
    monkeypatch.setattr(
        model_installation,
        "_huggingface_repo_files",
        lambda repo_id, revision=None: ("config.json",),
    )
    monkeypatch.setattr(
        "definers.media.web_transfer.download_file",
        lambda source_url, destination_path: (
            download_calls.append((source_url, destination_path))
            or destination_path
        ),
    )
    monkeypatch.setenv("DEFINERS_FAST_HF_DOWNLOADS", "1")
    monkeypatch.setenv("DEFINERS_HF_MODEL_DIR", str(tmp_path))

    assert model_installation.install_fast_huggingface_download_hooks() is True

    result = fake_transformers_hub.hf_hub_download(
        repo_id="owner/repo",
        filename="config.json",
        revision="rev-3",
    )

    expected_path = tmp_path / "owner" / "repo" / "rev-3" / "config.json"
    assert result == str(expected_path.resolve())
    assert download_calls == [
        (
            "https://huggingface.co/owner/repo/resolve/rev-3/config.json?download=1",
            str(expected_path.resolve()),
        )
    ]


def test_load_stable_whisper_model_uses_custom_download_root(
    monkeypatch, tmp_path
):
    fake_model = object()
    load_calls = {}
    fake_whisper_module = ModuleType("whisper")

    def original_download(url, root, in_memory):
        return "original-download"

    fake_whisper_module._download = original_download
    fake_stable_whisper = ModuleType("stable_whisper")

    def fake_load_model(name, device=None, download_root=None, **kwargs):
        load_calls.update(
            {
                "name": name,
                "device": device,
                "download_root": download_root,
                "download_hook": fake_whisper_module._download,
            }
        )
        return fake_model

    fake_stable_whisper.load_model = fake_load_model

    monkeypatch.setattr(model_installation, "_WHISPER_ORIGINAL_DOWNLOAD", None)
    monkeypatch.setitem(sys.modules, "whisper", fake_whisper_module)
    monkeypatch.setitem(sys.modules, "stable_whisper", fake_stable_whisper)
    monkeypatch.setattr(
        "definers.model_installation.ensure_module_runtime",
        lambda name: name == "stable_whisper",
    )
    monkeypatch.setenv("DEFINERS_WHISPER_MODEL_DIR", str(tmp_path))

    result = model_installation.load_stable_whisper_model(device_name="cpu")

    assert result is fake_model
    assert load_calls["name"] == model_installation.STABLE_WHISPER_MODEL_NAME
    assert load_calls["device"] == "cpu"
    assert load_calls["download_root"] == str(tmp_path.resolve())
    assert load_calls["download_hook"] is not original_download


def test_create_activity_reporter_records_requested_progress():
    scope_id = create_download_activity_scope()

    with bind_download_activity_scope(scope_id):
        reporter = create_activity_reporter(4)
        reporter(
            2,
            "Render artifact",
            detail="Halfway through the artifact workflow.",
        )

    snapshot = get_download_activity_snapshot(scope_id)
    clear_download_activity_scope(scope_id)

    assert snapshot is not None
    assert snapshot.completed == 2
    assert snapshot.total == 4
    assert snapshot.item_label == "Render artifact"
    assert "Halfway through the artifact workflow." in snapshot.message


def test_patched_audio_separator_tqdm_reports_activity_progress():
    scope_id = create_download_activity_scope()

    with bind_download_activity_scope(scope_id):
        progress = model_installation._patched_audio_separator_tqdm(range(3))
        assert list(progress) == [0, 1, 2]

    snapshot = get_download_activity_snapshot(scope_id)
    clear_download_activity_scope(scope_id)

    assert snapshot is not None
    assert snapshot.phase == "step"
    assert snapshot.completed == 3
    assert snapshot.total == 3


def test_patched_audio_separator_write_audio_soundfile_uses_output_dir(
    monkeypatch, tmp_path
):
    written_paths = []

    monkeypatch.setattr(
        model_installation,
        "_AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE",
        lambda separator, stem_path, stem_source: (
            written_paths.append((stem_path, stem_source)) or stem_path
        ),
    )

    separator = ModuleType("separator_instance")
    separator.output_dir = str(tmp_path / "stage")
    stem_source = object()

    written_path = (
        model_installation._patched_audio_separator_write_audio_soundfile(
            separator,
            "song_(Vocals)_demo.wav",
            stem_source,
        )
    )

    assert written_path == str(tmp_path / "stage" / "song_(Vocals)_demo.wav")
    assert written_paths == [
        (
            str(tmp_path / "stage" / "song_(Vocals)_demo.wav"),
            stem_source,
        )
    ]
    assert (tmp_path / "stage").is_dir()


def test_patched_audio_separator_write_audio_pydub_falls_back_to_managed_root(
    monkeypatch, tmp_path
):
    written_paths = []

    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        model_installation,
        "_AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_PYDUB",
        lambda separator, stem_path, stem_source: (
            written_paths.append((stem_path, stem_source)) or stem_path
        ),
    )

    separator = ModuleType("separator_instance")
    separator.output_dir = ""
    stem_source = object()

    written_path = (
        model_installation._patched_audio_separator_write_audio_pydub(
            separator,
            "song_(Vocals)_demo.wav",
            stem_source,
        )
    )

    assert Path(written_path) == (
        tmp_path / "audio" / "stems" / "song_(Vocals)_demo.wav"
    )
    assert written_paths == [
        (
            str(tmp_path / "audio" / "stems" / "song_(Vocals)_demo.wav"),
            stem_source,
        )
    ]


def test_patched_audio_separator_write_audio_soundfile_falls_back_to_managed_root(
    monkeypatch, tmp_path
):
    written_paths = []

    monkeypatch.setenv("DEFINERS_GUI_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setattr(
        model_installation,
        "_AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE",
        lambda separator, stem_path, stem_source: (
            written_paths.append((stem_path, stem_source)) or stem_path
        ),
    )

    separator = ModuleType("separator_instance")
    separator.output_dir = ""
    stem_source = object()

    written_path = (
        model_installation._patched_audio_separator_write_audio_soundfile(
            separator,
            "song_(Vocals)_demo.wav",
            stem_source,
        )
    )

    assert Path(written_path) == (
        tmp_path / "audio" / "stems" / "song_(Vocals)_demo.wav"
    )
    assert written_paths == [
        (
            str(tmp_path / "audio" / "stems" / "song_(Vocals)_demo.wav"),
            stem_source,
        )
    ]


def test_resolve_stem_model_filename_maps_legacy_model_names(monkeypatch):
    monkeypatch.setattr(
        model_installation,
        "supported_audio_separator_model_files",
        lambda: frozenset(
            {
                "bs_roformer_vocals_gabox.ckpt",
                "mel_band_roformer_instrumental_instv7_gabox.ckpt",
            }
        ),
    )

    assert (
        model_installation.resolve_stem_model_filename(
            "bs_roformer_vocals_resurrection_unwa.ckpt"
        )
        == "bs_roformer_vocals_gabox.ckpt"
    )
    assert (
        model_installation.resolve_stem_model_filename(
            "bs_roformer_instrumental_resurrection_unwa.ckpt"
        )
        == "mel_band_roformer_instrumental_instv7_gabox.ckpt"
    )
