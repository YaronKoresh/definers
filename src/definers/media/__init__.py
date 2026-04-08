import sys

from definers.image import helpers as image_helpers
from definers.video import helpers as video_helpers

from . import web_transfer

sys.modules.setdefault(f"{__name__}.image_helpers", image_helpers)
sys.modules.setdefault(f"{__name__}.video_helpers", video_helpers)

__all__ = (
    "image_helpers",
    "video_helpers",
    "web_transfer",
)
