from pathlib import Path

from definers import model_installation


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
