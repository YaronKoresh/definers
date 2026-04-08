import importlib
from typing import Any

_LAZY_SUBMODULES = {
    "apps",
    "chat_handlers",
    "gradio_shared",
    "gui_entrypoints",
    "gui_registry",
    "launchers",
    "lyric_video_service",
    "music_video_service",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(name)


__all__ = (
    "apps",
    "chat_handlers",
    "gradio_shared",
    "gui_entrypoints",
    "gui_registry",
    "launchers",
    "lyric_video_service",
    "music_video_service",
)
