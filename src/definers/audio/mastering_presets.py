from __future__ import annotations

from .config import SmartMasteringConfig


class MasteringPresets:
    edm = staticmethod(SmartMasteringConfig.edm)
    pop = staticmethod(SmartMasteringConfig.pop)
    flat = staticmethod(SmartMasteringConfig.flat)
    safe = staticmethod(SmartMasteringConfig.safe)

    @staticmethod
    def names() -> tuple[str, ...]:
        return SmartMasteringConfig.preset_names()


def mastering_preset(name: str) -> SmartMasteringConfig:
    return SmartMasteringConfig.from_preset(name)


def edm() -> SmartMasteringConfig:
    return SmartMasteringConfig.edm()


def pop() -> SmartMasteringConfig:
    return SmartMasteringConfig.pop()


def flat() -> SmartMasteringConfig:
    return SmartMasteringConfig.flat()


def safe() -> SmartMasteringConfig:
    return SmartMasteringConfig.safe()


__all__ = [
    "MasteringPresets",
    "edm",
    "mastering_preset",
    "pop",
    "flat",
    "safe",
]
