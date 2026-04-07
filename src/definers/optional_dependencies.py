from __future__ import annotations

import builtins
import importlib
import os
import subprocess
import sys
import threading
from collections.abc import Callable, Iterable
from typing import Any

AUTO_INSTALL_ENV_VAR = "DEFINERS_AUTO_INSTALL_OPTIONAL"
MADMOM_GITHUB_COMMIT = "27f032e8947204902c675e5e341a3faf5dc86dae"
MADMOM_GITHUB_ARCHIVE_URL = (
    f"https://github.com/CPJKU/madmom/archive/{MADMOM_GITHUB_COMMIT}.tar.gz"
)

MODULE_PACKAGE_SPECS: dict[str, tuple[str, ...]] = {
    "aiofiles": ("aiofiles",),
    "aiohttp": ("aiohttp",),
    "audio_separator": ("audio-separator>=0.30.0",),
    "basic_pitch": ("basic-pitch>=0.4.0",),
    "chatterbox": ("chatterbox-tts>=0.1.4",),
    "cssselect": ("cssselect>=1.2.0",),
    "cv2": ("opencv-contrib-python-headless>=4.8.0",),
    "datasets": ("datasets>=2.14.0",),
    "diffusers": (
        "diffusers>=0.35.0",
        "accelerate>=1.10.0",
        "huggingface-hub>=0.20.0",
        "safetensors>=0.4.0",
        "torch>=2.1.0",
    ),
    "edlib": ("edlib",),
    "faiss": ("faiss-cpu>=1.7.4",),
    "fastapi": ("fastapi>=0.100.0",),
    "googledrivedownloader": ("googledrivedownloader>=1.1.0",),
    "gradio": ("gradio>=6.9.0",),
    "huggingface_hub": ("huggingface-hub>=0.20.0",),
    "imageio": ("imageio>=2.30.0",),
    "imageio_ffmpeg": ("imageio-ffmpeg>=0.4.0",),
    "langdetect": ("langdetect>=1.0.9",),
    "librosa": (
        "librosa>=0.10.0",
        "numba>=0.57.0",
        "resampy>=0.4.2",
        "soundfile>=0.12.0",
    ),
    "lxml": (
        "lxml[html_clean]>=5.2.0",
        "cssselect>=1.2.0",
    ),
    "madmom": ("madmom>=0.16.1",),
    "matplotlib": ("matplotlib>=3.7.0",),
    "midi2audio": ("midi2audio",),
    "moviepy": (
        "moviepy>=1.0.3",
        "imageio>=2.30.0",
        "imageio-ffmpeg>=0.4.0",
    ),
    "nltk": ("nltk>=3.8.0",),
    "onnx": ("onnx>=1.14.0",),
    "onnxruntime": ("onnxruntime",),
    "pillow_heif": ("pillow-heif>=0.13.0",),
    "playwright": ("playwright>=1.40.0",),
    "pydub": ("pydub>=0.25.1",),
    "refiners": ("refiners>=0.4.0",),
    "sacremoses": ("sacremoses>=0.0.53",),
    "safetensors": ("safetensors>=0.4.0",),
    "sklearn": ("scikit-learn>=1.3.0",),
    "skimage": ("scikit-image>=0.21.0",),
    "soundfile": ("soundfile>=0.12.0",),
    "sox": ("sox>=1.4.1",),
    "sentencepiece": ("sentencepiece>=0.1.99",),
    "stable_whisper": (
        "stable-ts>=2.19.1",
        "torch>=2.1.0",
    ),
    "stopes": ("stopes>=2.2.1",),
    "tensorflow": ("tensorflow>=2.15.0",),
    "tf_keras": ("tf-keras>=2.15.0",),
    "tokenizers": ("tokenizers>=0.15.0",),
    "torch": ("torch>=2.1.0",),
    "torchaudio": (
        "torchaudio>=2.1.0",
        "torch>=2.1.0",
    ),
    "torchvision": (
        "torchvision>=0.16.0",
        "torch>=2.1.0",
    ),
    "transformers": (
        "transformers>=4.36.0",
        "tokenizers>=0.15.0",
        "sentencepiece>=0.1.99",
        "torch>=2.1.0",
    ),
}

MODULE_INSTALL_SPEC_OVERRIDES: dict[str, tuple[str, ...]] = {
    "madmom": (f"madmom @ {MADMOM_GITHUB_ARCHIVE_URL}",),
}

OPTIONAL_DEPENDENCY_GROUP_MODULES: dict[str, tuple[str, ...]] = {
    "audio": (
        "audio_separator",
        "librosa",
        "midi2audio",
        "pydub",
        "soundfile",
        "sox",
        "torchaudio",
        "basic_pitch",
        "chatterbox",
        "stable_whisper",
    ),
    "image": (
        "imageio",
        "imageio_ffmpeg",
        "cv2",
        "pillow_heif",
        "skimage",
        "refiners",
    ),
    "video": (
        "edlib",
        "imageio",
        "imageio_ffmpeg",
        "moviepy",
        "cv2",
        "skimage",
    ),
    "ml": (
        "diffusers",
        "datasets",
        "faiss",
        "huggingface_hub",
        "onnx",
        "safetensors",
        "sklearn",
        "sentencepiece",
        "tensorflow",
        "tf_keras",
        "tokenizers",
        "torchvision",
        "transformers",
    ),
    "nlp": (
        "langdetect",
        "nltk",
        "sacremoses",
        "stopes",
    ),
    "web": (
        "fastapi",
        "googledrivedownloader",
        "gradio",
        "lxml",
        "matplotlib",
        "playwright",
    ),
}

ML_TASK_MODULES: dict[str, tuple[str, ...]] = {
    "answer": ("transformers", "huggingface_hub"),
    "audio-classification": ("transformers",),
    "image": ("diffusers", "refiners", "pillow_heif"),
    "music": ("transformers", "pydub", "soundfile"),
    "speech-recognition": ("transformers",),
    "summary": ("transformers",),
    "translate": (
        "transformers",
        "nltk",
        "sacremoses",
        "langdetect",
    ),
    "tts": ("chatterbox",),
    "video": ("diffusers",),
}

_ORIGINAL_IMPORT = builtins.__import__
_HOOK_IMPORT = builtins.__import__
_INSTALL_LOCK = threading.RLock()
_HOOK_INSTALLED = False
_ACTIVE_STATE = threading.local()
_COMPLETED_INSTALLS: set[tuple[str, ...]] = set()
_FAILED_INSTALLS: set[tuple[str, ...]] = set()


def auto_install_enabled() -> bool:
    value = os.environ.get(AUTO_INSTALL_ENV_VAR, "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def normalize_module_name(module_name: str | None) -> str:
    if module_name is None:
        return ""
    return str(module_name).strip().split(".", 1)[0].replace("-", "_")


def package_specs_for_module(module_name: str | None) -> tuple[str, ...]:
    normalized_name = normalize_module_name(module_name)
    return MODULE_PACKAGE_SPECS.get(normalized_name, ())


def install_specs_for_module(module_name: str | None) -> tuple[str, ...]:
    normalized_name = normalize_module_name(module_name)
    if normalized_name in MODULE_INSTALL_SPEC_OVERRIDES:
        return MODULE_INSTALL_SPEC_OVERRIDES[normalized_name]
    return MODULE_PACKAGE_SPECS.get(normalized_name, ())


def package_specs_for_modules(
    module_names: Iterable[str],
) -> tuple[str, ...]:
    specs: list[str] = []
    for module_name in module_names:
        for spec in package_specs_for_module(module_name):
            if spec not in specs:
                specs.append(spec)
    return tuple(specs)


def install_specs_for_modules(
    module_names: Iterable[str],
) -> tuple[str, ...]:
    specs: list[str] = []
    for module_name in module_names:
        for spec in install_specs_for_module(module_name):
            if spec not in specs:
                specs.append(spec)
    return tuple(specs)


def package_specs_for_group(group: str) -> tuple[str, ...]:
    normalized_group = str(group or "").strip().lower()
    if normalized_group == "all":
        return package_specs_for_modules(
            module_name
            for group_modules in OPTIONAL_DEPENDENCY_GROUP_MODULES.values()
            for module_name in group_modules
        )
    return package_specs_for_modules(
        OPTIONAL_DEPENDENCY_GROUP_MODULES.get(normalized_group, ())
    )


def install_specs_for_group(group: str) -> tuple[str, ...]:
    normalized_group = str(group or "").strip().lower()
    if normalized_group == "all":
        return install_specs_for_modules(
            module_name
            for group_modules in OPTIONAL_DEPENDENCY_GROUP_MODULES.values()
            for module_name in group_modules
        )
    return install_specs_for_modules(
        OPTIONAL_DEPENDENCY_GROUP_MODULES.get(normalized_group, ())
    )


def package_specs_for_task(task: str) -> tuple[str, ...]:
    return package_specs_for_modules(ML_TASK_MODULES.get(str(task).strip(), ()))


def install_specs_for_task(task: str) -> tuple[str, ...]:
    return install_specs_for_modules(ML_TASK_MODULES.get(str(task).strip(), ()))


def group_target_names() -> tuple[str, ...]:
    return (*OPTIONAL_DEPENDENCY_GROUP_MODULES.keys(), "all")


def task_target_names() -> tuple[str, ...]:
    return tuple(ML_TASK_MODULES.keys())


def module_target_names() -> tuple[str, ...]:
    return tuple(sorted(MODULE_PACKAGE_SPECS))


def optional_runtime_targets() -> dict[str, tuple[str, ...]]:
    return {
        "groups": group_target_names(),
        "tasks": task_target_names(),
        "modules": module_target_names(),
    }


def package_specs_for_target(
    target: str,
    *,
    kind: str = "group",
) -> tuple[str, ...]:
    normalized_kind = str(kind or "group").strip().lower()
    if normalized_kind == "group":
        return package_specs_for_group(target)
    if normalized_kind == "task":
        return package_specs_for_task(str(target or "").strip())
    if normalized_kind == "module":
        return package_specs_for_module(target)
    return ()


def install_specs_for_target(
    target: str,
    *,
    kind: str = "group",
) -> tuple[str, ...]:
    normalized_kind = str(kind or "group").strip().lower()
    if normalized_kind == "group":
        return install_specs_for_group(target)
    if normalized_kind == "task":
        return install_specs_for_task(str(target or "").strip())
    if normalized_kind == "module":
        return install_specs_for_module(target)
    return ()


def _set_install_active(value: bool) -> None:
    _ACTIVE_STATE.install_active = value


def _install_active() -> bool:
    return bool(getattr(_ACTIVE_STATE, "install_active", False))


def _run_pip_install(package_specs: tuple[str, ...]) -> None:
    print(
        "[definers] Installing optional dependencies: "
        + ", ".join(package_specs)
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--prefer-binary",
            "--disable-pip-version-check",
            *package_specs,
        ]
    )


def install_package_specs(
    package_specs: Iterable[str],
    *,
    installer: Callable[[tuple[str, ...]], None] | None = None,
) -> bool:
    normalized_specs = tuple(
        spec
        for spec in dict.fromkeys(
            str(value).strip() for value in package_specs if str(value).strip()
        )
    )
    if not normalized_specs:
        return False
    cache_key = tuple(sorted(normalized_specs))
    with _INSTALL_LOCK:
        if cache_key in _COMPLETED_INSTALLS:
            return True
        if cache_key in _FAILED_INSTALLS:
            return False
        _set_install_active(True)
        try:
            active_installer = (
                _run_pip_install if installer is None else installer
            )
            active_installer(normalized_specs)
        except Exception:
            _FAILED_INSTALLS.add(cache_key)
            return False
        finally:
            _set_install_active(False)
        _COMPLETED_INSTALLS.add(cache_key)
        return True


def ensure_module_runtime(
    module_name: str | None,
    *,
    installer: Callable[[tuple[str, ...]], None] | None = None,
) -> bool:
    return install_package_specs(
        install_specs_for_module(module_name),
        installer=installer,
    )


def ensure_ml_task_runtime(
    task: str,
    *,
    installer: Callable[[tuple[str, ...]], None] | None = None,
) -> bool:
    return install_package_specs(
        install_specs_for_task(task),
        installer=installer,
    )


def ensure_group_runtime(
    group: str,
    *,
    installer: Callable[[tuple[str, ...]], None] | None = None,
) -> bool:
    return install_package_specs(
        install_specs_for_group(group),
        installer=installer,
    )


def install_optional_target(
    target: str,
    *,
    kind: str = "group",
    installer: Callable[[tuple[str, ...]], None] | None = None,
) -> bool:
    return install_package_specs(
        install_specs_for_target(target, kind=kind),
        installer=installer,
    )


def _candidate_module_names(
    module_name: str, error: BaseException
) -> tuple[str, ...]:
    candidates: list[str] = []
    error_name = normalize_module_name(getattr(error, "name", None))
    if error_name:
        candidates.append(error_name)
    requested_name = normalize_module_name(module_name)
    if requested_name and requested_name not in candidates:
        candidates.append(requested_name)
    return tuple(candidates)


def import_optional_module(
    module_name: str,
    *,
    installer: Callable[[tuple[str, ...]], None] | None = None,
):
    attempted: set[str] = set()
    while True:
        try:
            return importlib.import_module(module_name)
        except (ImportError, ModuleNotFoundError) as error:
            if not auto_install_enabled():
                raise
            installed = False
            for candidate_name in _candidate_module_names(module_name, error):
                if candidate_name in attempted:
                    continue
                attempted.add(candidate_name)
                if ensure_module_runtime(candidate_name, installer=installer):
                    installed = True
                    break
            if not installed:
                raise


def _import_with_auto_install(
    importer: Callable[..., Any],
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    try:
        return importer(name, globals, locals, fromlist, level)
    except (ImportError, ModuleNotFoundError) as error:
        if level != 0 or _install_active() or not auto_install_enabled():
            raise
        for candidate_name in _candidate_module_names(name, error):
            if ensure_module_runtime(candidate_name):
                return importer(name, globals, locals, fromlist, level)
        raise


def _auto_install_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    return _import_with_auto_install(
        _ORIGINAL_IMPORT,
        name,
        globals,
        locals,
        fromlist,
        level,
    )


def _hooked_auto_install_import(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
):
    return _import_with_auto_install(
        _HOOK_IMPORT,
        name,
        globals,
        locals,
        fromlist,
        level,
    )


def install_import_hook() -> None:
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    builtins.__import__ = _hooked_auto_install_import
    _HOOK_INSTALLED = True


__all__ = [
    "AUTO_INSTALL_ENV_VAR",
    "MADMOM_GITHUB_ARCHIVE_URL",
    "MADMOM_GITHUB_COMMIT",
    "MODULE_INSTALL_SPEC_OVERRIDES",
    "MODULE_PACKAGE_SPECS",
    "ML_TASK_MODULES",
    "OPTIONAL_DEPENDENCY_GROUP_MODULES",
    "auto_install_enabled",
    "ensure_group_runtime",
    "ensure_ml_task_runtime",
    "ensure_module_runtime",
    "group_target_names",
    "import_optional_module",
    "install_specs_for_group",
    "install_specs_for_module",
    "install_specs_for_modules",
    "install_specs_for_target",
    "install_specs_for_task",
    "install_optional_target",
    "install_import_hook",
    "install_package_specs",
    "module_target_names",
    "normalize_module_name",
    "optional_runtime_targets",
    "package_specs_for_group",
    "package_specs_for_module",
    "package_specs_for_modules",
    "package_specs_for_task",
    "package_specs_for_target",
    "task_target_names",
]
