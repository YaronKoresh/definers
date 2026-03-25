from typing import Protocol


class OutputPort(Protocol):
    def __call__(self, value: object) -> None: ...