from typing import Any, Protocol


class LyricVideoPort(Protocol):
    def __call__(
        self,
        audio: str,
        background: str,
        lyrics: str,
        position: str,
        *,
        max_dim: int,
        font_size: int,
        text_color: str,
        stroke_color: str,
        stroke_width: int,
        fade_duration: float,
    ) -> Any: ...
