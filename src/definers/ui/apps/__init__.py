import sys

from . import (
    animation,
    audio,
    audio_app_services,
    audio_workspace,
    chat_app,
    faiss,
    focused_surfaces,
    image,
    surface_hub,
    train,
    translate,
    video,
)
from .train import handlers as train_handlers, ui as train_ui

sys.modules.setdefault(f"{__name__}.train_handlers", train_handlers)
sys.modules.setdefault(f"{__name__}.train_ui", train_ui)

__all__ = (
    "animation",
    "audio",
    "audio_app_services",
    "audio_workspace",
    "chat_app",
    "faiss",
    "focused_surfaces",
    "image",
    "surface_hub",
    "train",
    "train_handlers",
    "train_ui",
    "translate",
    "video",
)
