from typing import Any, Protocol


class MusicVideoPort(Protocol):
    def __call__(self, audio: str, width: int, height: int, fps: int) -> Any: ...