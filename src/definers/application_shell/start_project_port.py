from typing import Any, Protocol


class StartProjectPort(Protocol):
    def __call__(self, project: str) -> Any: ...
