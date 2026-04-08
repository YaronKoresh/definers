from __future__ import annotations

import gc
import shutil
import tempfile
import threading
from collections.abc import Callable, Iterable
from pathlib import Path

from definers.catalogs.tasks import TASKS
from definers.optional_dependencies import ensure_module_runtime

ANSWER_MODEL_REVISION = "33e62acdd07cd7d6635badd529aa0a3467bb9c6a"
TEXT_GENERATION_ALLOW_PATTERNS = (
    "*.bin",
    "*.json",
    "*.model",
    "*.py",
    "*.safetensors",
    "*.vocab",
)
STABLE_WHISPER_MODEL_NAME = "tiny"
TTS_MODEL_NAME = "facebook/mms-tts-eng"
ASSET_ROOT = Path(__file__).resolve().parent / "assets"
ENHANCED_RVC_FORK_OWNER = "YaronKoresh"
ENHANCED_RVC_FORK_REPO = "definers-rvc-files"
ENHANCED_RVC_FORK_BRANCH = "main"
ENHANCED_RVC_FORK_FOLDERS: tuple[str, ...] = (
    "assets",
    "configs",
    "docs",
    "i18n",
    "infer",
    "logs",
    "tools",
)
STEM_MODEL_FILES = (
    "bs_roformer_vocals_resurrection_unwa.ckpt",
    "MelBandRoformerSYHFTV2.5.ckpt",
    "MelBandRoformerSYHFT.ckpt",
    "vocals_mel_band_roformer.ckpt",
    "bs_roformer_instrumental_resurrection_unwa.ckpt",
    "htdemucs_ft.yaml",
    "hdemucs_mmi.yaml",
    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    "deverb_bs_roformer_8_384dim_10depth.ckpt",
    "deverb_bs_roformer_8_256dim_8depth.ckpt",
    "dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt",
    "denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt",
    "denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt",
    "mel_band_roformer_bleed_suppressor_v1.ckpt",
    "mel_band_roformer_instrumental_fv7z_gabox.ckpt",
)
UPSCALE_FILES: tuple[tuple[str, str, str], ...] = (
    (
        "refiners/juggernaut.reborn.sd1_5.unet",
        "model.safetensors",
        "347d14c3c782c4959cc4d1bb1e336d19f7dda4d2",
    ),
    (
        "refiners/juggernaut.reborn.sd1_5.text_encoder",
        "model.safetensors",
        "744ad6a5c0437ec02ad826df9f6ede102bb27481",
    ),
    (
        "refiners/juggernaut.reborn.sd1_5.autoencoder",
        "model.safetensors",
        "3c1aae3fc3e03e4a2b7e0fa42b62ebb64f1a4c19",
    ),
    (
        "refiners/controlnet.sd1_5.tile",
        "model.safetensors",
        "48ced6ff8bfa873a8976fa467c3629a240643387",
    ),
    (
        "philz1337x/upscaler",
        "4x-UltraSharp.pth",
        "011deacac8270114eb7d2eeff4fe6fa9a837be70",
    ),
    (
        "philz1337x/embeddings",
        "JuggernautNegative-neg.pt",
        "203caa7e9cc2bc225031a4021f6ab1ded283454a",
    ),
    (
        "philz1337x/loras",
        "more_details.safetensors",
        "a3802c0280c0d00c2ab18d37454a8744c44e474e",
    ),
    (
        "philz1337x/loras",
        "SDXLrender_v2.0.safetensors",
        "a3802c0280c0d00c2ab18d37454a8744c44e474e",
    ),
)
MODEL_TASKS = (
    "answer",
    "summary",
    "translate",
    "music",
    "speech-recognition",
    "audio-classification",
    "tts",
    "stable-whisper",
    "stems",
    "rvc",
    "image",
    "detect",
    "upscale",
    "video",
)
MODEL_DOMAIN_ALIASES = {
    "language": "text",
    "nlp": "text",
}
MODEL_DOMAIN_TASKS: dict[str, tuple[str, ...]] = {
    "audio": (
        "music",
        "speech-recognition",
        "audio-classification",
        "tts",
        "stable-whisper",
        "stems",
        "rvc",
    ),
    "text": (
        "answer",
        "summary",
        "translate",
    ),
    "image": (
        "image",
        "detect",
        "upscale",
    ),
    "video": ("video",),
}

_MODEL_INSTALL_LOCK = threading.RLock()
_COMPLETED_MODEL_INSTALLS: set[str] = set()
_FAILED_MODEL_INSTALLS: set[str] = set()


def _normalize_model_task_name(value: str | None) -> str:
    return str(value or "").strip().lower()


def _normalize_model_domain_name(value: str | None) -> str:
    normalized_name = _normalize_model_task_name(value)
    return MODEL_DOMAIN_ALIASES.get(normalized_name, normalized_name)


def model_domain_names() -> tuple[str, ...]:
    return (*MODEL_DOMAIN_TASKS.keys(), "all")


def model_task_names() -> tuple[str, ...]:
    return MODEL_TASKS


def model_runtime_targets() -> dict[str, tuple[str, ...]]:
    return {
        "model-domains": model_domain_names(),
        "model-tasks": model_task_names(),
    }


def _model_targets_for_domain(domain: str) -> tuple[str, ...]:
    normalized_domain = _normalize_model_domain_name(domain)
    if normalized_domain == "all":
        ordered_targets: list[str] = []
        for domain_name in MODEL_DOMAIN_TASKS:
            for target_name in MODEL_DOMAIN_TASKS[domain_name]:
                if target_name not in ordered_targets:
                    ordered_targets.append(target_name)
        return tuple(ordered_targets)
    return MODEL_DOMAIN_TASKS.get(normalized_domain, ())


def _ensure_huggingface_hub() -> None:
    if not ensure_module_runtime("huggingface_hub"):
        raise RuntimeError("huggingface_hub is required to download models")


def _snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: Iterable[str] | None = None,
) -> str:
    _ensure_huggingface_hub()
    from huggingface_hub import snapshot_download

    kwargs: dict[str, object] = {"repo_id": repo_id}
    if revision is not None:
        kwargs["revision"] = revision
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    return str(snapshot_download(**kwargs))


def _hf_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
) -> str:
    _ensure_huggingface_hub()
    from huggingface_hub import hf_hub_download

    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "filename": filename,
    }
    if revision is not None:
        kwargs["revision"] = revision
    return str(hf_hub_download(**kwargs))


def _clone_enhanced_rvc_fork(target_root: str | Path) -> Path:
    from definers.ml import git as git_clone

    resolved_target_root = Path(target_root).resolve()
    resolved_target_root.mkdir(parents=True, exist_ok=True)
    git_clone(
        ENHANCED_RVC_FORK_OWNER,
        ENHANCED_RVC_FORK_REPO,
        branch=ENHANCED_RVC_FORK_BRANCH,
        parent=str(resolved_target_root),
    )
    return resolved_target_root


def enhanced_rvc_fork_folder_paths(
    target_root: str | Path = ".",
) -> tuple[Path, ...]:
    resolved_target_root = Path(target_root).resolve()
    return tuple(
        resolved_target_root / folder_name
        for folder_name in ENHANCED_RVC_FORK_FOLDERS
    )


def has_enhanced_rvc_fork_folders(target_root: str | Path = ".") -> bool:
    return all(
        folder_path.is_dir()
        for folder_path in enhanced_rvc_fork_folder_paths(target_root)
    )


def download_enhanced_rvc_fork_folders(
    target_root: str | Path = ".",
) -> tuple[str, ...]:
    resolved_target_root = Path(target_root).resolve()
    resolved_target_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temporary_directory:
        extracted_repo_root = _clone_enhanced_rvc_fork(
            Path(temporary_directory) / ENHANCED_RVC_FORK_REPO
        )
        restored_paths: list[str] = []
        for folder_name in ENHANCED_RVC_FORK_FOLDERS:
            source_folder = extracted_repo_root / folder_name
            if not source_folder.is_dir():
                raise FileNotFoundError(
                    f"Enhanced RVC fork folder '{folder_name}' is missing"
                )
            destination_folder = resolved_target_root / folder_name
            shutil.copytree(
                source_folder,
                destination_folder,
                dirs_exist_ok=True,
            )
            restored_paths.append(str(destination_folder))
    return tuple(restored_paths)


def ensure_enhanced_rvc_fork_folders(
    target_root: str | Path = ".",
) -> tuple[str, ...]:
    if has_enhanced_rvc_fork_folders(target_root):
        return tuple(
            str(folder_path)
            for folder_path in enhanced_rvc_fork_folder_paths(target_root)
        )
    return download_enhanced_rvc_fork_folders(target_root)


def download_rvc_assets(target_root: str | Path = ".") -> tuple[str, ...]:
    return ensure_enhanced_rvc_fork_folders(target_root)


def _download_stem_models() -> None:
    if not ensure_module_runtime("audio_separator"):
        raise RuntimeError(
            "audio-separator is required to download stem models"
        )
    from audio_separator.separator import Separator

    separator = Separator(
        output_dir=str(ASSET_ROOT),
        output_format="WAV",
        sample_rate=44100,
        use_soundfile=True,
        log_level=40,
        demucs_params={
            "shifts": 2,
            "overlap": 0.25,
            "segments_enabled": True,
        },
        mdxc_params={
            "segment_size": 256,
            "overlap": 4,
        },
    )
    for model_name in STEM_MODEL_FILES:
        separator.download_model_files(model_name)
    gc.collect()


def _download_stable_whisper_model() -> None:
    if not ensure_module_runtime("stable_whisper"):
        raise RuntimeError(
            "stable-ts is required to download the lyric sync model"
        )
    import stable_whisper

    model = stable_whisper.load_model(STABLE_WHISPER_MODEL_NAME, device="cpu")
    del model
    gc.collect()


def _download_upscale_models() -> None:
    for repo_id, filename, revision in UPSCALE_FILES:
        _hf_download(repo_id, filename, revision=revision)


def _download_answer_model() -> None:
    _snapshot_download(
        str(TASKS["answer"]),
        revision=ANSWER_MODEL_REVISION,
        allow_patterns=TEXT_GENERATION_ALLOW_PATTERNS,
    )


def _download_summary_model() -> None:
    _snapshot_download(str(TASKS["summary"]))


def _download_translate_model() -> None:
    _snapshot_download(str(TASKS["translate"]))


def _download_music_model() -> None:
    _snapshot_download(str(TASKS["music"]))


def _download_speech_recognition_model() -> None:
    _snapshot_download(str(TASKS["speech-recognition"]))


def _download_audio_classification_model() -> None:
    _snapshot_download(str(TASKS["audio-classification"]))


def _download_tts_model() -> None:
    _snapshot_download(TTS_MODEL_NAME)


def _download_image_models() -> None:
    _snapshot_download(str(TASKS["image"]))
    _hf_download(
        str(TASKS["image-spro"]),
        "diffusion_pytorch_model.safetensors",
    )


def _download_detect_model() -> None:
    _snapshot_download(str(TASKS["detect"]))


def _download_video_model() -> None:
    _snapshot_download(str(TASKS["video"]))


MODEL_TASK_DOWNLOADERS: dict[str, Callable[[], None]] = {
    "answer": _download_answer_model,
    "summary": _download_summary_model,
    "translate": _download_translate_model,
    "music": _download_music_model,
    "speech-recognition": _download_speech_recognition_model,
    "audio-classification": _download_audio_classification_model,
    "tts": _download_tts_model,
    "stable-whisper": _download_stable_whisper_model,
    "stems": _download_stem_models,
    "rvc": download_rvc_assets,
    "image": _download_image_models,
    "detect": _download_detect_model,
    "upscale": _download_upscale_models,
    "video": _download_video_model,
}


def _install_model_task(task_name: str) -> None:
    MODEL_TASK_DOWNLOADERS[task_name]()


def _run_model_task_install(
    task_name: str,
    *,
    installer: Callable[[str], None],
) -> bool:
    with _MODEL_INSTALL_LOCK:
        if task_name in _COMPLETED_MODEL_INSTALLS:
            return True
        if task_name in _FAILED_MODEL_INSTALLS:
            return False
    try:
        installer(task_name)
    except Exception:
        with _MODEL_INSTALL_LOCK:
            _FAILED_MODEL_INSTALLS.add(task_name)
        return False
    with _MODEL_INSTALL_LOCK:
        _COMPLETED_MODEL_INSTALLS.add(task_name)
    return True


def install_model_target(
    target: str,
    *,
    kind: str = "model-domain",
    installer: Callable[[str], None] | None = None,
) -> bool:
    normalized_kind = _normalize_model_task_name(kind)
    if normalized_kind == "model-domain":
        resolved_targets = _model_targets_for_domain(target)
    elif normalized_kind == "model-task":
        normalized_target = _normalize_model_task_name(target)
        resolved_targets = (
            (normalized_target,)
            if normalized_target in MODEL_TASK_DOWNLOADERS
            else ()
        )
    else:
        return False
    if not resolved_targets:
        return False
    active_installer = _install_model_task if installer is None else installer
    return all(
        _run_model_task_install(task_name, installer=active_installer)
        for task_name in resolved_targets
    )


__all__ = [
    "ANSWER_MODEL_REVISION",
    "ENHANCED_RVC_FORK_BRANCH",
    "ENHANCED_RVC_FORK_FOLDERS",
    "ENHANCED_RVC_FORK_OWNER",
    "ENHANCED_RVC_FORK_REPO",
    "MODEL_DOMAIN_TASKS",
    "MODEL_TASK_DOWNLOADERS",
    "STEM_MODEL_FILES",
    "TEXT_GENERATION_ALLOW_PATTERNS",
    "UPSCALE_FILES",
    "download_rvc_assets",
    "download_enhanced_rvc_fork_folders",
    "enhanced_rvc_fork_folder_paths",
    "ensure_enhanced_rvc_fork_folders",
    "has_enhanced_rvc_fork_folders",
    "install_model_target",
    "model_domain_names",
    "model_runtime_targets",
    "model_task_names",
]
