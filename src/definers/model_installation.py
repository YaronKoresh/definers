from __future__ import annotations

import contextlib
import fnmatch
import gc
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from importlib import resources
from pathlib import Path
from urllib.parse import quote, urlsplit

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
_LEGACY_STEM_MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "bs_roformer_vocals_resurrection_unwa.ckpt": (
        "bs_roformer_vocals_gabox.ckpt",
        "mel_band_roformer_vocals_fv4_gabox.ckpt",
        "vocals_mel_band_roformer.ckpt",
    ),
    "bs_roformer_instrumental_resurrection_unwa.ckpt": (
        "mel_band_roformer_instrumental_instv7_gabox.ckpt",
        "mel_band_roformer_instrumental_gabox.ckpt",
        "mel_band_roformer_instrumental_becruily.ckpt",
        "MDX23C-8KFFT-InstVoc_HQ.ckpt",
    ),
}
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
_HUGGINGFACE_PATCH_LOCK = threading.RLock()
_WHISPER_PATCH_LOCK = threading.RLock()
_HUGGINGFACE_ORIGINAL_HF_HUB_DOWNLOAD: Callable[..., object] | None = None
_HUGGINGFACE_ORIGINAL_SNAPSHOT_DOWNLOAD: Callable[..., object] | None = None
_WHISPER_ORIGINAL_DOWNLOAD: Callable[..., object] | None = None
_HUGGINGFACE_ALIAS_MODULES = (
    "transformers.utils.hub",
    "diffusers.utils.dynamic_modules_utils",
    "diffusers.utils.hub_utils",
)
_AUDIO_SEPARATOR_PUBLIC_REPO_URL_PREFIX = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models"
_AUDIO_SEPARATOR_VIP_REPO_URL_PREFIX = (
    "https://github.com/Anjok0109/ai_magic/releases/download/v5"
)
_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs"
_AUDIO_SEPARATOR_RELEASE_MIRRORABLE_SUFFIXES = frozenset(
    {
        ".ckpt",
        ".onnx",
        ".pth",
        ".th",
        ".yaml",
        ".yml",
    }
)
_AUDIO_SEPARATOR_DOWNLOAD_CHECKS_URL = "https://raw.githubusercontent.com/TRvlvr/application_data/main/filelists/download_checks.json"
_AUDIO_SEPARATOR_PATCH_LOCK = threading.RLock()
_AUDIO_SEPARATOR_ORIGINAL_DOWNLOAD_FILE_IF_NOT_EXISTS: (
    Callable[..., object] | None
) = None
_AUDIO_SEPARATOR_ORIGINAL_LIST_SUPPORTED_MODEL_FILES: (
    Callable[..., object] | None
) = None
_AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_PYDUB: Callable[..., object] | None = None
_AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE: (
    Callable[..., object] | None
) = None
_TQDM_ORIGINAL_TQDM: Callable[..., object] | None = None
_AUDIO_SEPARATOR_TQDM_MODULES = (
    "audio_separator.separator.separator",
    "audio_separator.separator.architectures.mdx_separator",
    "audio_separator.separator.architectures.mdxc_separator",
    "audio_separator.separator.architectures.vr_separator",
)


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


def _report_download_activity(
    item_label: str | None = None,
    *,
    detail: str | None = None,
    phase: str = "download",
    completed: int | None = None,
    total: int | None = None,
) -> None:
    try:
        from definers.system.download_activity import report_download_activity
    except Exception:
        return
    report_download_activity(
        item_label,
        detail=detail,
        phase=phase,
        completed=completed,
        total=total,
    )


def _huggingface_max_workers() -> int:
    configured_workers = os.environ.get("DEFINERS_HF_MAX_WORKERS", "").strip()
    if configured_workers:
        try:
            resolved_workers = int(configured_workers)
        except ValueError:
            resolved_workers = 0
        if resolved_workers > 0:
            return resolved_workers
    cpu_count = os.cpu_count() or 8
    return max(32, min(96, cpu_count * 12))


def _prepare_huggingface_runtime() -> None:
    hf_transfer_available = importlib.util.find_spec("hf_transfer") is not None
    if (
        not os.environ.get("HF_HUB_ENABLE_HF_TRANSFER", "").strip()
        and hf_transfer_available
    ):
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    if (
        hf_transfer_available
        and not os.environ.get("HF_TRANSFER_CONCURRENCY", "").strip()
    ):
        os.environ["HF_TRANSFER_CONCURRENCY"] = str(_huggingface_max_workers())
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def _prefer_fast_huggingface_downloads() -> bool:
    configured_value = os.environ.get("DEFINERS_FAST_HF_DOWNLOADS", "1").strip()
    return configured_value.lower() not in {"0", "false", "off", "no"}


def _supports_fast_huggingface_download(
    extra_kwargs: dict[str, object],
) -> bool:
    repo_type = str(extra_kwargs.get("repo_type") or "").strip().lower()
    if repo_type and repo_type != "model":
        return False
    if bool(extra_kwargs.get("local_files_only")):
        return False
    if extra_kwargs.get("token") is not None:
        return False
    if extra_kwargs.get("use_auth_token") is not None:
        return False
    if extra_kwargs.get("headers"):
        return False
    endpoint = str(extra_kwargs.get("endpoint") or "").strip().rstrip("/")
    if endpoint and endpoint != "https://huggingface.co":
        return False
    return True


def _normalize_cache_segment(value: str | None, *, default: str) -> str:
    normalized_text = str(value or "").strip()
    if not normalized_text:
        return default
    resolved = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in normalized_text
    ).strip("._-")
    return resolved or default


def huggingface_model_dir() -> str:
    configured_model_dir = os.environ.get("DEFINERS_HF_MODEL_DIR", "").strip()
    configured_data_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
    if configured_model_dir:
        target_root = Path(configured_model_dir).expanduser()
    elif configured_data_root:
        target_root = (
            Path(configured_data_root).expanduser() / "models" / "huggingface"
        )
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        target_root = (
            Path(local_app_data).expanduser() / "definers" / "huggingface"
            if local_app_data
            else Path.home() / "AppData" / "Local" / "definers" / "huggingface"
        )
    else:
        target_root = Path.home() / ".cache" / "definers" / "huggingface"
    target_root.mkdir(parents=True, exist_ok=True)
    return str(target_root.resolve())


def whisper_model_dir() -> str:
    configured_model_dir = os.environ.get(
        "DEFINERS_WHISPER_MODEL_DIR", ""
    ).strip()
    configured_data_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
    if configured_model_dir:
        target_root = Path(configured_model_dir).expanduser()
    elif configured_data_root:
        target_root = (
            Path(configured_data_root).expanduser() / "models" / "whisper"
        )
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        target_root = (
            Path(local_app_data).expanduser() / "definers" / "whisper"
            if local_app_data
            else Path.home() / "AppData" / "Local" / "definers" / "whisper"
        )
    else:
        target_root = Path.home() / ".cache" / "definers" / "whisper"
    target_root.mkdir(parents=True, exist_ok=True)
    return str(target_root.resolve())


def _huggingface_snapshot_root(
    repo_id: str,
    *,
    revision: str | None = None,
    local_dir: str | None = None,
) -> Path:
    if local_dir is not None and str(local_dir).strip():
        target_root = Path(str(local_dir)).expanduser()
        target_root.mkdir(parents=True, exist_ok=True)
        return target_root.resolve()
    try:
        owner_name, repo_name = str(repo_id).strip().split("/", 1)
    except ValueError:
        owner_name, repo_name = ("repo", str(repo_id).strip() or "unknown")
    target_root = (
        Path(huggingface_model_dir())
        / _normalize_cache_segment(owner_name, default="owner")
        / _normalize_cache_segment(repo_name, default="repo")
        / _normalize_cache_segment(revision, default="main")
    )
    target_root.mkdir(parents=True, exist_ok=True)
    return target_root.resolve()


def _huggingface_file_url(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
) -> str:
    owner_name, repo_name = str(repo_id).strip().split("/", 1)
    resolved_revision = (
        str(revision).strip() if revision is not None else "main"
    )
    encoded_owner = quote(owner_name, safe="")
    encoded_repo = quote(repo_name, safe="")
    encoded_revision = quote(resolved_revision, safe="")
    encoded_filename = "/".join(
        quote(path_segment, safe="")
        for path_segment in str(filename).split("/")
        if path_segment
    )
    return (
        f"https://huggingface.co/{encoded_owner}/{encoded_repo}/resolve/"
        f"{encoded_revision}/{encoded_filename}?download=1"
    )


@lru_cache(maxsize=64)
def _huggingface_repo_files(
    repo_id: str,
    revision: str | None = None,
) -> tuple[str, ...]:
    _prepare_huggingface_runtime()
    _ensure_huggingface_hub()
    from huggingface_hub import HfApi

    return tuple(
        sorted(HfApi().list_repo_files(repo_id=repo_id, revision=revision))
    )


def _filter_huggingface_repo_files(
    repo_files: Iterable[str],
    allow_patterns: Iterable[str] | None,
    ignore_patterns: Iterable[str] | None = None,
) -> tuple[str, ...]:
    normalized_repo_files = tuple(
        str(repo_file).strip()
        for repo_file in repo_files
        if str(repo_file).strip()
    )
    if allow_patterns is None:
        return normalized_repo_files
    normalized_patterns = tuple(
        str(pattern).strip()
        for pattern in allow_patterns
        if str(pattern).strip()
    )
    if not normalized_patterns:
        filtered_repo_files = normalized_repo_files
    else:
        filtered_repo_files = tuple(
            repo_file
            for repo_file in normalized_repo_files
            if any(
                fnmatch.fnmatch(repo_file, pattern)
                for pattern in normalized_patterns
            )
        )
    normalized_ignore_patterns = tuple(
        str(pattern).strip()
        for pattern in tuple(ignore_patterns or ())
        if str(pattern).strip()
    )
    if not normalized_ignore_patterns:
        return filtered_repo_files
    return tuple(
        repo_file
        for repo_file in filtered_repo_files
        if not any(
            fnmatch.fnmatch(repo_file, pattern)
            for pattern in normalized_ignore_patterns
        )
    )


def _artifact_is_ready(target_path: Path) -> bool:
    try:
        return target_path.is_file() and target_path.stat().st_size > 0
    except Exception:
        return False


def _direct_download_artifact(
    source_url: str,
    target_path: Path,
    *,
    item_label: str,
    detail: str,
    phase: str,
    completed: int | None = None,
    total: int | None = None,
    force_download: bool = False,
) -> str:
    from definers.media.web_transfer import download_file

    target_path.parent.mkdir(parents=True, exist_ok=True)
    if not force_download and _artifact_is_ready(target_path):
        _report_download_activity(
            item_label,
            detail=f"Using cached artifact at {target_path.name}.",
            phase=phase,
            completed=completed,
            total=total,
        )
        return str(target_path)
    _report_download_activity(
        item_label,
        detail=detail,
        phase=phase,
        completed=completed,
        total=total,
    )
    downloaded_path = download_file(source_url, str(target_path))
    if downloaded_path is None:
        raise FileNotFoundError(
            f"Could not download artifact from '{source_url}'."
        )
    return downloaded_path


def _fast_hf_file_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    item_label: str | None = None,
    detail: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    local_dir: str | None = None,
    force_download: bool = False,
) -> str:
    snapshot_root = _huggingface_snapshot_root(
        repo_id,
        revision=revision,
        local_dir=local_dir,
    )
    target_path = snapshot_root.joinpath(
        *[segment for segment in str(filename).split("/") if segment]
    )
    return _direct_download_artifact(
        _huggingface_file_url(repo_id, filename, revision=revision),
        target_path,
        item_label=item_label or f"{repo_id}/{filename}",
        detail=detail
        or "Downloading model artifact directly from Hugging Face.",
        phase="artifact",
        completed=completed,
        total=total,
        force_download=force_download,
    )


def _fast_hf_snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
    item_label: str | None = None,
    detail: str | None = None,
    local_dir: str | None = None,
    max_workers: int | None = None,
    force_download: bool = False,
) -> str:
    from definers.system.download_activity import (
        bind_download_activity_scope,
        current_download_activity_scope,
    )

    snapshot_root = _huggingface_snapshot_root(
        repo_id,
        revision=revision,
        local_dir=local_dir,
    )
    repo_files = _huggingface_repo_files(repo_id, revision)
    selected_files = _filter_huggingface_repo_files(
        repo_files,
        allow_patterns,
        ignore_patterns,
    )
    if not selected_files:
        return str(snapshot_root)

    activity_scope_id = current_download_activity_scope()

    def download_one(download_index: int, repo_file: str) -> str:
        with (
            bind_download_activity_scope(activity_scope_id)
            if activity_scope_id is not None
            else contextlib.nullcontext()
        ):
            return _fast_hf_file_download(
                repo_id,
                repo_file,
                revision=revision,
                item_label=repo_file,
                detail=f"Downloading {repo_file} directly from Hugging Face.",
                completed=download_index,
                total=len(selected_files),
                local_dir=str(snapshot_root),
                force_download=force_download,
            )

    resolved_max_workers = (
        _huggingface_max_workers()
        if max_workers is None
        else max(int(max_workers), 1)
    )
    worker_count = max(1, min(resolved_max_workers, len(selected_files)))
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(download_one, download_index, repo_file)
            for download_index, repo_file in enumerate(selected_files, start=1)
        ]
        for future in futures:
            future.result()
    _report_download_activity(
        item_label or repo_id,
        detail=detail or "Model snapshot is ready in the local cache.",
        phase="download",
        completed=len(selected_files),
        total=len(selected_files),
    )
    return str(snapshot_root)


def stem_model_dir() -> str:
    configured_model_dir = os.environ.get(
        "AUDIO_SEPARATOR_MODEL_DIR", ""
    ).strip()
    configured_data_root = os.environ.get("DEFINERS_DATA_ROOT", "").strip()
    if configured_model_dir:
        target_root = Path(configured_model_dir).expanduser()
    elif configured_data_root:
        target_root = (
            Path(configured_data_root).expanduser()
            / "models"
            / "audio_separator"
        )
    elif os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        target_root = (
            Path(local_app_data).expanduser() / "definers" / "audio_separator"
            if local_app_data
            else Path.home()
            / "AppData"
            / "Local"
            / "definers"
            / "audio_separator"
        )
    else:
        target_root = Path.home() / ".cache" / "definers" / "audio_separator"
    target_root.mkdir(parents=True, exist_ok=True)
    return str(target_root.resolve())


def _audio_separator_resource_payload(resource_name: str) -> dict[str, object]:
    if not ensure_module_runtime("audio_separator"):
        return {}
    try:
        payload = json.loads(
            resources.files("audio_separator")
            .joinpath(resource_name)
            .read_text(encoding="utf-8")
        )
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=1)
def _audio_separator_packaged_models_payload() -> dict[str, object]:
    return _audio_separator_resource_payload("models.json")


@lru_cache(maxsize=1)
def _audio_separator_model_scores_payload() -> dict[str, object]:
    return _audio_separator_resource_payload("models-scores.json")


def _read_json_payload(source_path: Path) -> dict[str, object]:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


@lru_cache(maxsize=8)
def _audio_separator_download_checks_payload(
    model_root: str,
) -> dict[str, object]:
    target_path = Path(str(model_root)).resolve() / "download_checks.json"
    if not _artifact_is_ready(target_path):
        try:
            _direct_download_artifact(
                _AUDIO_SEPARATOR_DOWNLOAD_CHECKS_URL,
                target_path,
                item_label=target_path.name,
                detail="Downloading audio-separator model index.",
                phase="index",
            )
        except Exception:
            return {}
    try:
        return _read_json_payload(target_path)
    except Exception:
        with contextlib.suppress(Exception):
            target_path.unlink()
        try:
            _direct_download_artifact(
                _AUDIO_SEPARATOR_DOWNLOAD_CHECKS_URL,
                target_path,
                item_label=target_path.name,
                detail="Refreshing audio-separator model index.",
                phase="index",
                force_download=True,
            )
        except Exception:
            return {}
    try:
        return _read_json_payload(target_path)
    except Exception:
        return {}


def _audio_separator_payload_section(
    payload: dict[str, object],
    section_name: str,
) -> dict[str, object]:
    section = payload.get(section_name)
    return dict(section) if isinstance(section, dict) else {}


def _audio_separator_score_entry(
    model_scores: dict[str, object],
    filename: str,
) -> dict[str, object]:
    score_entry = model_scores.get(str(filename).strip())
    return dict(score_entry) if isinstance(score_entry, dict) else {}


def _audio_separator_basic_model_entries(
    model_entries: dict[str, object],
    model_scores: dict[str, object],
) -> dict[str, dict[str, object]]:
    resolved_entries: dict[str, dict[str, object]] = {}
    for model_name, model_filename in model_entries.items():
        normalized_filename = str(model_filename).strip()
        if not normalized_filename:
            continue
        score_entry = _audio_separator_score_entry(
            model_scores,
            normalized_filename,
        )
        resolved_entries[str(model_name)] = {
            "filename": normalized_filename,
            "scores": dict(score_entry.get("median_scores", {})),
            "stems": tuple(score_entry.get("stems", ())),
            "target_stem": score_entry.get("target_stem"),
            "download_files": (normalized_filename,),
        }
    return resolved_entries


@lru_cache(maxsize=8)
def _audio_separator_supported_model_catalog(
    model_root: str,
) -> dict[str, dict[str, dict[str, object]]]:
    packaged_models = _audio_separator_packaged_models_payload()
    model_scores = _audio_separator_model_scores_payload()
    download_checks = _audio_separator_download_checks_payload(model_root)

    vr_models = {
        **_audio_separator_payload_section(download_checks, "vr_download_list"),
        **_audio_separator_payload_section(packaged_models, "vr_download_list"),
    }
    mdx_models = {
        **_audio_separator_payload_section(
            download_checks, "mdx_download_list"
        ),
        **_audio_separator_payload_section(
            download_checks,
            "mdx_download_vip_list",
        ),
        **_audio_separator_payload_section(
            packaged_models, "mdx_download_list"
        ),
    }

    demucs_models: dict[str, dict[str, object]] = {}
    for model_name, model_files in _audio_separator_payload_section(
        download_checks,
        "demucs_download_list",
    ).items():
        if not str(model_name).startswith("Demucs v4"):
            continue
        if not isinstance(model_files, dict):
            continue
        yaml_filename = next(
            (
                str(filename).strip()
                for filename in model_files
                if str(filename).strip().endswith(".yaml")
            ),
            "",
        )
        if not yaml_filename:
            continue
        score_entry = _audio_separator_score_entry(model_scores, yaml_filename)
        download_files = tuple(
            str(file_value).strip()
            for file_value in model_files.values()
            if str(file_value).strip()
        )
        demucs_models[str(model_name)] = {
            "filename": yaml_filename,
            "scores": dict(score_entry.get("median_scores", {})),
            "stems": tuple(score_entry.get("stems", ())),
            "target_stem": score_entry.get("target_stem"),
            "download_files": download_files,
        }

    mdxc_model_sources = {
        **_audio_separator_payload_section(
            download_checks,
            "mdx23c_download_list",
        ),
        **_audio_separator_payload_section(
            download_checks,
            "mdx23c_download_vip_list",
        ),
        **_audio_separator_payload_section(
            download_checks,
            "roformer_download_list",
        ),
        **_audio_separator_payload_section(
            packaged_models,
            "mdx23c_download_list",
        ),
        **_audio_separator_payload_section(
            packaged_models,
            "roformer_download_list",
        ),
    }
    mdxc_models: dict[str, dict[str, object]] = {}
    for model_name, model_files in mdxc_model_sources.items():
        if not isinstance(model_files, dict) or not model_files:
            continue
        primary_files = tuple(
            str(filename).strip()
            for filename in model_files.keys()
            if str(filename).strip()
        )
        auxiliary_files = tuple(
            str(filename).strip()
            for filename in model_files.values()
            if str(filename).strip()
        )
        if not primary_files:
            continue
        score_entry = _audio_separator_score_entry(
            model_scores,
            primary_files[0],
        )
        mdxc_models[str(model_name)] = {
            "filename": primary_files[0],
            "scores": dict(score_entry.get("median_scores", {})),
            "stems": tuple(score_entry.get("stems", ())),
            "target_stem": score_entry.get("target_stem"),
            "download_files": primary_files + auxiliary_files,
        }

    return {
        "VR": _audio_separator_basic_model_entries(vr_models, model_scores),
        "MDX": _audio_separator_basic_model_entries(mdx_models, model_scores),
        "Demucs": demucs_models,
        "MDXC": mdxc_models,
    }


def _audio_separator_download_filename(file_to_download: str) -> str:
    normalized_name = str(file_to_download).strip()
    if normalized_name.startswith("http"):
        return normalized_name.rstrip("/").split("/")[-1]
    return normalized_name


def _audio_separator_release_mirror_url(source_url: str) -> str:
    normalized_source_url = str(source_url).strip()
    if not normalized_source_url:
        return ""
    if normalized_source_url.startswith(
        f"{_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/"
    ):
        return ""
    parsed_source_url = urlsplit(normalized_source_url)
    fallback_filename = Path(parsed_source_url.path).name.strip()
    if not fallback_filename:
        return ""
    if (
        Path(fallback_filename).suffix.lower()
        not in _AUDIO_SEPARATOR_RELEASE_MIRRORABLE_SUFFIXES
    ):
        return ""
    if normalized_source_url.startswith(
        f"{_AUDIO_SEPARATOR_PUBLIC_REPO_URL_PREFIX}/"
    ):
        return f"{_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/{fallback_filename}"
    normalized_source_host = parsed_source_url.netloc.lower()
    normalized_source_path = parsed_source_url.path.lower()
    if (
        normalized_source_host == "dl.fbaipublicfiles.com"
        and "/demucs/" in normalized_source_path
    ):
        return f"{_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/{fallback_filename}"
    if (
        normalized_source_host == "raw.githubusercontent.com"
        and "/demucs/" in normalized_source_path
    ):
        return f"{_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/{fallback_filename}"
    return ""


def _audio_separator_download_candidate_urls(
    source_url: str,
) -> tuple[str, ...]:
    normalized_source_url = str(source_url).strip()
    if not normalized_source_url:
        return ()
    candidate_urls: list[str] = []
    release_mirror_url = _audio_separator_release_mirror_url(
        normalized_source_url
    )
    if release_mirror_url:
        candidate_urls.append(release_mirror_url)
    candidate_urls.append(normalized_source_url)
    return tuple(dict.fromkeys(candidate_urls))


def _audio_separator_model_targets(
    model_name: str,
    *,
    model_root: str | None = None,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    resolved_model_root = stem_model_dir() if model_root is None else model_root
    catalog = _audio_separator_supported_model_catalog(str(resolved_model_root))
    for model_type, models in catalog.items():
        for model_friendly_name, model_info in models.items():
            model_is_uvr_vip = "VIP" in model_friendly_name
            model_repo_url_prefix = (
                _AUDIO_SEPARATOR_VIP_REPO_URL_PREFIX
                if model_is_uvr_vip
                else _AUDIO_SEPARATOR_PUBLIC_REPO_URL_PREFIX
            )
            model_filename = str(model_info.get("filename", "")).strip()
            download_files = tuple(
                str(file_to_download).strip()
                for file_to_download in tuple(
                    model_info.get("download_files", ())
                )
                if str(file_to_download).strip()
            )
            if (
                model_filename != model_name
                and model_name not in download_files
            ):
                continue
            targets: list[tuple[str, tuple[str, ...]]] = []
            for file_to_download in download_files:
                if file_to_download.startswith("http"):
                    filename = _audio_separator_download_filename(
                        file_to_download
                    )
                    targets.append(
                        (
                            filename,
                            _audio_separator_download_candidate_urls(
                                file_to_download
                            ),
                        )
                    )
                    continue
                normalized_filename = str(file_to_download).strip()
                if not normalized_filename:
                    continue
                if model_type == "MDXC" and normalized_filename.endswith(
                    ".yaml"
                ):
                    primary_url = (
                        f"{model_repo_url_prefix}/mdx_model_data/"
                        f"mdx_c_configs/{normalized_filename}"
                    )
                    if (
                        model_repo_url_prefix
                        == _AUDIO_SEPARATOR_PUBLIC_REPO_URL_PREFIX
                    ):
                        candidate_urls = (
                            _audio_separator_download_candidate_urls(
                                primary_url
                            )
                        )
                    else:
                        candidate_urls = (
                            primary_url,
                            f"{_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/{normalized_filename}",
                        )
                    targets.append(
                        (
                            normalized_filename,
                            candidate_urls,
                        )
                    )
                    continue
                primary_url = f"{model_repo_url_prefix}/{normalized_filename}"
                if (
                    model_repo_url_prefix
                    == _AUDIO_SEPARATOR_PUBLIC_REPO_URL_PREFIX
                ):
                    candidate_urls = _audio_separator_download_candidate_urls(
                        primary_url
                    )
                else:
                    candidate_urls = (
                        primary_url,
                        f"{_AUDIO_SEPARATOR_CONFIG_REPO_URL_PREFIX}/{normalized_filename}",
                    )
                targets.append(
                    (
                        normalized_filename,
                        candidate_urls,
                    )
                )
            return tuple(targets)
    return ()


def stem_model_artifacts_ready(
    model_name: str,
    *,
    model_root: str | None = None,
) -> bool:
    resolved_model_root = Path(
        stem_model_dir() if model_root is None else model_root
    ).resolve()
    resolved_model_name = resolve_stem_model_filename(model_name)
    model_targets = _audio_separator_model_targets(
        resolved_model_name,
        model_root=str(resolved_model_root),
    )
    if not model_targets:
        return _artifact_is_ready(
            resolved_model_root / Path(resolved_model_name).name
        )
    return all(
        _artifact_is_ready(resolved_model_root / filename)
        for filename, _candidate_urls in model_targets
    )


@lru_cache(maxsize=1)
def supported_audio_separator_model_files() -> frozenset[str]:
    if not ensure_module_runtime("audio_separator"):
        return frozenset()
    supported_files: set[str] = set()
    for model_group in _audio_separator_supported_model_catalog(
        stem_model_dir()
    ).values():
        if not isinstance(model_group, dict):
            continue
        for model_info in model_group.values():
            if not isinstance(model_info, dict):
                continue
            normalized_filename = str(model_info.get("filename", "")).strip()
            if normalized_filename:
                supported_files.add(normalized_filename)
    return frozenset(supported_files)


def resolve_stem_model_filename(model_name: str) -> str:
    normalized_model_name = str(model_name).strip()
    if not normalized_model_name:
        return normalized_model_name
    supported_files = supported_audio_separator_model_files()
    if not supported_files or normalized_model_name in supported_files:
        return normalized_model_name
    for alias_name in _LEGACY_STEM_MODEL_ALIASES.get(
        normalized_model_name,
        (),
    ):
        if alias_name in supported_files:
            return alias_name
    return normalized_model_name


def _snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
    item_label: str | None = None,
    detail: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    **extra_kwargs: object,
) -> str:
    local_dir = extra_kwargs.get("local_dir") or extra_kwargs.get("cache_dir")
    if (
        _prefer_fast_huggingface_downloads()
        and _supports_fast_huggingface_download(dict(extra_kwargs))
    ):
        try:
            return _fast_hf_snapshot_download(
                repo_id,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                item_label=item_label,
                detail=detail,
                local_dir=None if local_dir is None else str(local_dir),
                max_workers=extra_kwargs.get("max_workers"),
                force_download=bool(extra_kwargs.get("force_download", False)),
            )
        except Exception as error:
            _report_download_activity(
                item_label or repo_id,
                detail=(
                    f"Fast direct download failed, falling back to huggingface_hub: {error}"
                ),
                phase="download",
                completed=completed,
                total=total,
            )
    _prepare_huggingface_runtime()
    _ensure_huggingface_hub()

    _report_download_activity(
        item_label or repo_id,
        detail=detail or "Preparing Hugging Face snapshot download.",
        phase="download",
        completed=completed,
        total=total,
    )
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "max_workers": _huggingface_max_workers(),
    }
    if revision is not None:
        kwargs["revision"] = revision
    if allow_patterns is not None:
        kwargs["allow_patterns"] = list(allow_patterns)
    if ignore_patterns is not None:
        kwargs["ignore_patterns"] = list(ignore_patterns)
    kwargs.update(extra_kwargs)
    return str(_call_original_snapshot_download(**kwargs))


def _hf_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    item_label: str | None = None,
    detail: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    **extra_kwargs: object,
) -> str:
    resolved_item_label = item_label or f"{repo_id}/{filename}"
    local_dir = extra_kwargs.get("local_dir") or extra_kwargs.get("cache_dir")
    if (
        _prefer_fast_huggingface_downloads()
        and _supports_fast_huggingface_download(dict(extra_kwargs))
    ):
        try:
            return _fast_hf_file_download(
                repo_id,
                filename,
                revision=revision,
                item_label=item_label,
                detail=detail,
                completed=completed,
                total=total,
                local_dir=None if local_dir is None else str(local_dir),
                force_download=bool(extra_kwargs.get("force_download", False)),
            )
        except Exception as error:
            _report_download_activity(
                resolved_item_label,
                detail=(
                    f"Fast direct download failed, falling back to huggingface_hub: {error}"
                ),
                phase="artifact",
                completed=completed,
                total=total,
            )
    _prepare_huggingface_runtime()
    _ensure_huggingface_hub()

    _report_download_activity(
        resolved_item_label,
        detail=detail or "Preparing Hugging Face file download.",
        phase="artifact",
        completed=completed,
        total=total,
    )
    kwargs: dict[str, object] = {
        "repo_id": repo_id,
        "filename": filename,
    }
    if revision is not None:
        kwargs["revision"] = revision
    kwargs.update(extra_kwargs)
    return str(_call_original_hf_hub_download(**kwargs))


def hf_snapshot_download(
    repo_id: str,
    *,
    revision: str | None = None,
    allow_patterns: Iterable[str] | None = None,
    ignore_patterns: Iterable[str] | None = None,
    item_label: str | None = None,
    detail: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    **extra_kwargs: object,
) -> str:
    return _snapshot_download(
        repo_id,
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        item_label=item_label,
        detail=detail,
        completed=completed,
        total=total,
        **extra_kwargs,
    )


def hf_file_download(
    repo_id: str,
    filename: str,
    *,
    revision: str | None = None,
    item_label: str | None = None,
    detail: str | None = None,
    completed: int | None = None,
    total: int | None = None,
    **extra_kwargs: object,
) -> str:
    return _hf_download(
        repo_id,
        filename,
        revision=revision,
        item_label=item_label,
        detail=detail,
        completed=completed,
        total=total,
        **extra_kwargs,
    )


def _call_original_hf_hub_download(*args: object, **kwargs: object) -> object:
    original_download = _HUGGINGFACE_ORIGINAL_HF_HUB_DOWNLOAD
    if original_download is None:
        from huggingface_hub import hf_hub_download as original_download

    return original_download(*args, **kwargs)


def _call_original_snapshot_download(*args: object, **kwargs: object) -> object:
    original_download = _HUGGINGFACE_ORIGINAL_SNAPSHOT_DOWNLOAD
    if original_download is None:
        from huggingface_hub import snapshot_download as original_download

    return original_download(*args, **kwargs)


def _join_huggingface_subfolder(filename: str, subfolder: str | None) -> str:
    normalized_filename = str(filename).strip().lstrip("/")
    normalized_subfolder = str(subfolder or "").strip().strip("/")
    if not normalized_subfolder:
        return normalized_filename
    return f"{normalized_subfolder}/{normalized_filename}"


def _patched_hf_hub_download(*args: object, **kwargs: object) -> object:
    if len(args) > 2:
        return _call_original_hf_hub_download(*args, **kwargs)
    patched_kwargs = dict(kwargs)
    repo_id = patched_kwargs.pop("repo_id", args[0] if args else None)
    filename = patched_kwargs.pop(
        "filename",
        args[1] if len(args) > 1 else None,
    )
    if repo_id is None or filename is None:
        return _call_original_hf_hub_download(*args, **kwargs)
    revision = patched_kwargs.pop("revision", None)
    item_label = patched_kwargs.pop("item_label", None)
    detail = patched_kwargs.pop("detail", None)
    completed = patched_kwargs.pop("completed", None)
    total = patched_kwargs.pop("total", None)
    resolved_filename = _join_huggingface_subfolder(
        str(filename),
        str(patched_kwargs.pop("subfolder", "") or ""),
    )
    return _hf_download(
        str(repo_id),
        resolved_filename,
        revision=revision,
        item_label=item_label,
        detail=detail,
        completed=completed,
        total=total,
        **patched_kwargs,
    )


def _patched_snapshot_download(*args: object, **kwargs: object) -> object:
    if len(args) > 1:
        return _call_original_snapshot_download(*args, **kwargs)
    patched_kwargs = dict(kwargs)
    repo_id = patched_kwargs.pop("repo_id", args[0] if args else None)
    if repo_id is None:
        return _call_original_snapshot_download(*args, **kwargs)
    revision = patched_kwargs.pop("revision", None)
    allow_patterns = patched_kwargs.pop("allow_patterns", None)
    ignore_patterns = patched_kwargs.pop("ignore_patterns", None)
    item_label = patched_kwargs.pop("item_label", None)
    detail = patched_kwargs.pop("detail", None)
    completed = patched_kwargs.pop("completed", None)
    total = patched_kwargs.pop("total", None)
    return _snapshot_download(
        str(repo_id),
        revision=revision,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        item_label=item_label,
        detail=detail,
        completed=completed,
        total=total,
        **patched_kwargs,
    )


def _patch_huggingface_alias_module(module: object | None) -> None:
    if module is None:
        return
    if hasattr(module, "hf_hub_download"):
        setattr(module, "hf_hub_download", _patched_hf_hub_download)
    if hasattr(module, "snapshot_download"):
        setattr(module, "snapshot_download", _patched_snapshot_download)


def install_fast_huggingface_download_hooks() -> bool:
    _prepare_huggingface_runtime()
    if not ensure_module_runtime("huggingface_hub"):
        return False
    import huggingface_hub

    with _HUGGINGFACE_PATCH_LOCK:
        global _HUGGINGFACE_ORIGINAL_HF_HUB_DOWNLOAD
        global _HUGGINGFACE_ORIGINAL_SNAPSHOT_DOWNLOAD

        current_hf_hub_download = getattr(huggingface_hub, "hf_hub_download")
        current_snapshot_download = getattr(
            huggingface_hub, "snapshot_download"
        )
        if (
            _HUGGINGFACE_ORIGINAL_HF_HUB_DOWNLOAD is None
            and current_hf_hub_download is not _patched_hf_hub_download
        ):
            _HUGGINGFACE_ORIGINAL_HF_HUB_DOWNLOAD = current_hf_hub_download
        if (
            _HUGGINGFACE_ORIGINAL_SNAPSHOT_DOWNLOAD is None
            and current_snapshot_download is not _patched_snapshot_download
        ):
            _HUGGINGFACE_ORIGINAL_SNAPSHOT_DOWNLOAD = current_snapshot_download

        huggingface_hub.hf_hub_download = _patched_hf_hub_download
        huggingface_hub.snapshot_download = _patched_snapshot_download

        file_download_module = sys.modules.get("huggingface_hub.file_download")
        if file_download_module is not None and hasattr(
            file_download_module,
            "hf_hub_download",
        ):
            file_download_module.hf_hub_download = _patched_hf_hub_download

        for module_name in _HUGGINGFACE_ALIAS_MODULES:
            _patch_huggingface_alias_module(sys.modules.get(module_name))
    return True


class _AudioSeparatorActivityTqdm:
    def __init__(
        self,
        iterable: Iterable[object] | None = None,
        *,
        total: object = None,
        description: object = None,
        unit: object = None,
    ) -> None:
        self._iterable = iterable
        self._description = str(description or "").strip()
        self._uses_byte_progress = str(unit or "").strip().lower() in {
            "ib",
            "b",
            "byte",
            "bytes",
        }
        self._item_label = self._description or "audio-separator"
        self._detail = self._description or (
            "Downloading audio-separator artifact."
            if self._uses_byte_progress
            else "Running audio-separator task."
        )
        self.n = 0
        self.total = self._resolve_total(total, iterable)
        self._publish()

    @staticmethod
    def _resolve_total(
        total: object,
        iterable: Iterable[object] | None,
    ) -> int | None:
        try:
            normalized_total = int(total) if total is not None else None
        except Exception:
            normalized_total = None
        if normalized_total is not None:
            return normalized_total if normalized_total >= 0 else None
        if iterable is None:
            return None
        try:
            resolved_total = len(iterable)
        except Exception:
            return None
        return max(int(resolved_total), 0)

    def _publish(self) -> None:
        try:
            from definers.system.download_activity import (
                report_download_activity,
            )
        except Exception:
            return
        if self._uses_byte_progress:
            report_download_activity(
                self._item_label,
                detail=self._detail,
                phase="transfer",
                bytes_downloaded=self.n,
                bytes_total=self.total,
            )
            return
        report_download_activity(
            self._item_label,
            detail=self._detail,
            phase="step",
            completed=self.n,
            total=self.total,
        )

    def __iter__(self):
        if self._iterable is None:
            return
        for item in self._iterable:
            yield item
            self.update(1)

    def update(self, value: object = 1) -> None:
        try:
            delta = int(value)
        except Exception:
            delta = 0
        self.n = max(self.n + max(delta, 0), 0)
        self._publish()

    def close(self) -> None:
        return None

    def refresh(self) -> None:
        return None

    def set_description(
        self, description: object, refresh: bool = True
    ) -> None:
        self._description = str(description or "").strip()
        self._item_label = self._description or "audio-separator"
        self._detail = self._description or self._detail
        if refresh:
            self._publish()


def _patched_audio_separator_tqdm(*args: object, **kwargs: object) -> object:
    try:
        from definers.system.download_activity import (
            current_download_activity_scope,
        )
    except Exception:
        current_download_activity_scope = None

    original_tqdm = _TQDM_ORIGINAL_TQDM
    if (
        not callable(current_download_activity_scope)
        or current_download_activity_scope() is None
    ):
        if original_tqdm is None:
            raise RuntimeError("tqdm is not available")
        return original_tqdm(*args, **kwargs)

    iterable = args[0] if args else kwargs.get("iterable")
    return _AudioSeparatorActivityTqdm(
        iterable=iterable,
        total=kwargs.get("total"),
        description=kwargs.get("desc"),
        unit=kwargs.get("unit"),
    )


def _patched_audio_separator_download_file_if_not_exists(
    separator,
    url: str,
    output_path: str,
) -> None:
    target_path = Path(str(output_path)).expanduser().resolve()
    if _artifact_is_ready(target_path):
        return
    last_error: Exception | None = None
    for candidate_url in _audio_separator_download_candidate_urls(str(url)):
        try:
            _direct_download_artifact(
                candidate_url,
                target_path,
                item_label=target_path.name,
                detail=f"Downloading separator artifact {target_path.name}.",
                phase="artifact",
            )
            return
        except Exception as error:
            last_error = error
    if last_error is not None:
        raise RuntimeError(str(last_error)) from last_error
    raise RuntimeError("audio-separator download URL is empty")


def _patched_audio_separator_list_supported_model_files(separator):
    model_root = Path(str(separator.model_file_dir)).resolve()
    catalog = _audio_separator_supported_model_catalog(str(model_root))
    if catalog:
        return catalog
    original_loader = _AUDIO_SEPARATOR_ORIGINAL_LIST_SUPPORTED_MODEL_FILES
    if original_loader is None:
        raise RuntimeError("audio-separator model catalog is unavailable")
    return original_loader(separator)


def _audio_separator_output_base_path(output_dir: object) -> Path:
    from definers.system.output_paths import managed_output_dir

    resolved_output_dir = str(output_dir or "").strip()
    if resolved_output_dir:
        candidate_path = Path(resolved_output_dir).expanduser()
        if candidate_path.is_absolute():
            candidate_path.mkdir(parents=True, exist_ok=True)
            return candidate_path
        return Path(managed_output_dir("audio/stems", resolved_output_dir))
    return Path(managed_output_dir("audio/stems"))


def _patched_audio_separator_write_audio_pydub(
    separator,
    stem_path: str,
    stem_source,
) -> object:
    original_writer = _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_PYDUB
    if original_writer is None:
        raise RuntimeError("audio-separator pydub writer is unavailable")

    resolved_stem_path = Path(str(stem_path)).expanduser()
    if not resolved_stem_path.is_absolute():
        resolved_stem_path = (
            _audio_separator_output_base_path(
                getattr(separator, "output_dir", "")
            )
            / resolved_stem_path
        )
    resolved_stem_path.parent.mkdir(parents=True, exist_ok=True)
    return original_writer(separator, str(resolved_stem_path), stem_source)


def _patched_audio_separator_write_audio_soundfile(
    separator,
    stem_path: str,
    stem_source,
) -> object:
    original_writer = _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE
    if original_writer is None:
        raise RuntimeError("audio-separator soundfile writer is unavailable")

    resolved_stem_path = Path(str(stem_path)).expanduser()
    if not resolved_stem_path.is_absolute():
        resolved_stem_path = (
            _audio_separator_output_base_path(
                getattr(separator, "output_dir", "")
            )
            / resolved_stem_path
        )
    resolved_stem_path.parent.mkdir(parents=True, exist_ok=True)
    return original_writer(separator, str(resolved_stem_path), stem_source)


def _patch_audio_separator_tqdm_modules() -> None:
    for module_name in _AUDIO_SEPARATOR_TQDM_MODULES:
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "tqdm"):
            setattr(module, "tqdm", _patched_audio_separator_tqdm)


def install_audio_separator_runtime_hooks() -> bool:
    if not ensure_module_runtime("audio_separator"):
        return False
    import tqdm as tqdm_module
    from audio_separator.separator import Separator
    from audio_separator.separator.common_separator import CommonSeparator

    with _AUDIO_SEPARATOR_PATCH_LOCK:
        global _AUDIO_SEPARATOR_ORIGINAL_DOWNLOAD_FILE_IF_NOT_EXISTS
        global _AUDIO_SEPARATOR_ORIGINAL_LIST_SUPPORTED_MODEL_FILES
        global _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_PYDUB
        global _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE
        global _TQDM_ORIGINAL_TQDM

        current_tqdm = getattr(tqdm_module, "tqdm", None)
        if (
            _TQDM_ORIGINAL_TQDM is None
            and callable(current_tqdm)
            and current_tqdm is not _patched_audio_separator_tqdm
        ):
            _TQDM_ORIGINAL_TQDM = current_tqdm
        if (
            getattr(tqdm_module, "tqdm", None)
            is not _patched_audio_separator_tqdm
        ):
            tqdm_module.tqdm = _patched_audio_separator_tqdm

        current_download_method = getattr(
            Separator,
            "download_file_if_not_exists",
            None,
        )
        current_catalog_method = getattr(
            Separator,
            "list_supported_model_files",
            None,
        )
        current_write_audio_soundfile = getattr(
            CommonSeparator,
            "write_audio_soundfile",
            None,
        )
        current_write_audio_pydub = getattr(
            CommonSeparator,
            "write_audio_pydub",
            None,
        )
        if (
            _AUDIO_SEPARATOR_ORIGINAL_DOWNLOAD_FILE_IF_NOT_EXISTS is None
            and current_download_method
            is not _patched_audio_separator_download_file_if_not_exists
        ):
            _AUDIO_SEPARATOR_ORIGINAL_DOWNLOAD_FILE_IF_NOT_EXISTS = (
                current_download_method
            )
        if (
            _AUDIO_SEPARATOR_ORIGINAL_LIST_SUPPORTED_MODEL_FILES is None
            and current_catalog_method
            is not _patched_audio_separator_list_supported_model_files
        ):
            _AUDIO_SEPARATOR_ORIGINAL_LIST_SUPPORTED_MODEL_FILES = (
                current_catalog_method
            )
        if (
            _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_PYDUB is None
            and current_write_audio_pydub
            is not _patched_audio_separator_write_audio_pydub
        ):
            _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_PYDUB = (
                current_write_audio_pydub
            )
        if (
            _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE is None
            and current_write_audio_soundfile
            is not _patched_audio_separator_write_audio_soundfile
        ):
            _AUDIO_SEPARATOR_ORIGINAL_WRITE_AUDIO_SOUNDFILE = (
                current_write_audio_soundfile
            )

        Separator.download_file_if_not_exists = (
            _patched_audio_separator_download_file_if_not_exists
        )
        Separator.list_supported_model_files = (
            _patched_audio_separator_list_supported_model_files
        )
        CommonSeparator.write_audio_pydub = (
            _patched_audio_separator_write_audio_pydub
        )
        CommonSeparator.write_audio_soundfile = (
            _patched_audio_separator_write_audio_soundfile
        )
        _patch_audio_separator_tqdm_modules()
    return True


def _download_whisper_model_artifact(
    url: str,
    root: str,
    in_memory: bool,
) -> bytes | str:
    from definers.media.web_transfer import download_file

    download_root = Path(root).expanduser().resolve()
    download_root.mkdir(parents=True, exist_ok=True)
    expected_sha256 = str(url).rstrip("/").split("/")[-2]
    target_path = download_root / Path(urlsplit(url).path).name
    if target_path.exists() and not target_path.is_file():
        raise RuntimeError(f"{target_path} exists and is not a regular file")
    if target_path.is_file():
        model_bytes = target_path.read_bytes()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            _report_download_activity(
                target_path.name,
                detail=f"Using cached artifact at {target_path.name}.",
                phase="artifact",
            )
            return model_bytes if in_memory else str(target_path)
    downloaded_path = download_file(url, str(target_path))
    if downloaded_path is None:
        raise FileNotFoundError(f"Could not download artifact from '{url}'.")
    model_bytes = target_path.read_bytes()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError(
            "Model has been downloaded but the SHA256 checksum does not match."
        )
    return model_bytes if in_memory else str(target_path)


def _patched_whisper_download(
    url: str, root: str, in_memory: bool
) -> bytes | str:
    try:
        return _download_whisper_model_artifact(url, root, in_memory)
    except Exception as error:
        _report_download_activity(
            Path(urlsplit(url).path).name,
            detail=(
                f"Fast direct download failed, falling back to whisper downloader: {error}"
            ),
            phase="artifact",
        )
        if _WHISPER_ORIGINAL_DOWNLOAD is None:
            raise
        return _WHISPER_ORIGINAL_DOWNLOAD(url, root, in_memory)


def install_fast_whisper_download_hooks() -> bool:
    if not ensure_module_runtime("stable_whisper"):
        return False
    import whisper

    with _WHISPER_PATCH_LOCK:
        global _WHISPER_ORIGINAL_DOWNLOAD

        current_download = getattr(whisper, "_download", None)
        if current_download is None:
            return False
        if (
            _WHISPER_ORIGINAL_DOWNLOAD is None
            and current_download is not _patched_whisper_download
        ):
            _WHISPER_ORIGINAL_DOWNLOAD = current_download
        whisper._download = _patched_whisper_download
    return True


def load_stable_whisper_model(
    *,
    device_name: str = "cpu",
    model_name: str = STABLE_WHISPER_MODEL_NAME,
):
    if not ensure_module_runtime("stable_whisper"):
        raise RuntimeError(
            "stable-ts is required to download the lyric sync model"
        )
    install_fast_whisper_download_hooks()
    import stable_whisper

    _report_download_activity(
        model_name,
        detail="Initializing the transcription runtime.",
        phase="model",
    )
    model = stable_whisper.load_model(
        model_name,
        device=device_name,
        download_root=whisper_model_dir(),
    )
    _report_download_activity(
        model_name,
        detail="The transcription runtime is ready.",
        phase="model",
        completed=1,
        total=1,
    )
    return model


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
    target_root: str | Path | None = None,
) -> tuple[Path, ...]:
    resolved_target_root = (
        Path(__file__).resolve().parent
        if target_root is None
        else Path(target_root).resolve()
    )
    return tuple(
        resolved_target_root / folder_name
        for folder_name in ENHANCED_RVC_FORK_FOLDERS
    )


def has_enhanced_rvc_fork_folders(
    target_root: str | Path | None = None,
) -> bool:
    return all(
        folder_path.is_dir()
        for folder_path in enhanced_rvc_fork_folder_paths(target_root)
    )


def download_enhanced_rvc_fork_folders(
    target_root: str | Path | None = None,
) -> tuple[str, ...]:
    resolved_target_root = (
        Path(__file__).resolve().parent
        if target_root is None
        else Path(target_root).resolve()
    )
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
    target_root: str | Path | None = None,
) -> tuple[str, ...]:
    if has_enhanced_rvc_fork_folders(target_root):
        return tuple(
            str(folder_path)
            for folder_path in enhanced_rvc_fork_folder_paths(target_root)
        )
    return download_enhanced_rvc_fork_folders(target_root)


def download_rvc_assets(
    target_root: str | Path | None = None,
) -> tuple[str, ...]:
    return ensure_enhanced_rvc_fork_folders(target_root)


def _download_audio_separator_model_direct(
    separator,
    model_name: str,
) -> None:
    resolved_model_dir = Path(str(separator.model_file_dir)).resolve()
    download_targets = _audio_separator_model_targets(
        model_name,
        model_root=str(resolved_model_dir),
    )
    if not download_targets:
        raise ValueError(
            f"Model file {model_name} not found in supported model files"
        )
    for download_index, (filename, candidate_urls) in enumerate(
        download_targets, start=1
    ):
        target_path = resolved_model_dir / filename
        if _artifact_is_ready(target_path):
            _report_download_activity(
                filename,
                detail=f"Using cached separator artifact for {model_name}.",
                phase="artifact",
                completed=download_index,
                total=len(download_targets),
            )
            continue
        last_error: Exception | None = None
        for candidate_url in candidate_urls:
            try:
                _direct_download_artifact(
                    candidate_url,
                    target_path,
                    item_label=filename,
                    detail=f"Downloading separator artifact for {model_name}.",
                    phase="artifact",
                    completed=download_index,
                    total=len(download_targets),
                )
                last_error = None
                break
            except Exception as error:
                last_error = error
        if last_error is not None:
            raise last_error


def download_stem_models(
    model_names: Iterable[str] | None = None,
) -> tuple[str, ...]:
    if not ensure_module_runtime("audio_separator"):
        raise RuntimeError(
            "audio-separator is required to download stem models"
        )
    install_audio_separator_runtime_hooks()
    from audio_separator.separator import Separator

    resolved_model_names = tuple(
        dict.fromkeys(
            str(model_name).strip()
            for model_name in (
                STEM_MODEL_FILES if model_names is None else tuple(model_names)
            )
            if str(model_name).strip()
        )
    )
    supported_files = supported_audio_separator_model_files()
    requested_to_download = tuple(
        dict.fromkeys(
            resolve_stem_model_filename(model_name)
            for model_name in resolved_model_names
        )
    )
    resolved_model_dir = stem_model_dir()

    separator = Separator(
        output_dir=resolved_model_dir,
        output_format="WAV",
        sample_rate=44100,
        use_soundfile=True,
        log_level=40,
        model_file_dir=resolved_model_dir,
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
    for index, model_name in enumerate(requested_to_download, start=1):
        if supported_files and model_name not in supported_files:
            raise ValueError(f"Unsupported audio-separator model: {model_name}")
        _report_download_activity(
            model_name,
            detail="Preparing separator checkpoint.",
            phase="download",
            completed=index,
            total=len(requested_to_download),
        )
        try:
            _download_audio_separator_model_direct(separator, model_name)
        except Exception:
            separator.download_model_files(model_name)
    gc.collect()
    return requested_to_download


def _download_stem_models() -> None:
    download_stem_models()


def _download_stable_whisper_model() -> None:
    model = load_stable_whisper_model(device_name="cpu")
    del model
    gc.collect()


def _download_upscale_models() -> None:
    for index, (repo_id, filename, revision) in enumerate(
        UPSCALE_FILES,
        start=1,
    ):
        _hf_download(
            repo_id,
            filename,
            revision=revision,
            item_label=filename,
            detail=f"Fetching {repo_id}",
            completed=index,
            total=len(UPSCALE_FILES),
        )


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
    _report_download_activity(
        task_name,
        detail="Preparing model assets.",
        phase="model",
    )
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
    "download_stem_models",
    "enhanced_rvc_fork_folder_paths",
    "ensure_enhanced_rvc_fork_folders",
    "hf_file_download",
    "hf_snapshot_download",
    "huggingface_model_dir",
    "has_enhanced_rvc_fork_folders",
    "install_audio_separator_runtime_hooks",
    "install_fast_huggingface_download_hooks",
    "install_model_target",
    "load_stable_whisper_model",
    "model_domain_names",
    "model_runtime_targets",
    "model_task_names",
    "resolve_stem_model_filename",
    "stem_model_dir",
    "stem_model_artifacts_ready",
    "supported_audio_separator_model_files",
    "whisper_model_dir",
]
